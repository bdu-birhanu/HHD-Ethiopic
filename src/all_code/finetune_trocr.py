import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import tensorflow as tf
from torch.utils.data import Dataset
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

df = pd.read_fwf('./train/train_raw/image_text_pairs_train.csv', header=None)
df.rename(columns={0: "file_name", 1: "text"}, inplace=True)
#sdel df[2]
df = df.applymap(lambda x: x.replace(',', ''))
df.head()

#just to splite the dat to train and test
#train_df, test_df = train_test_split(df, test_size=0.3) 0.35, 0.45 by 2 epoch
train_df, test_df = train_test_split(df, test_size=0.2)
#train_df, test_df = train_test_split(test_df, test_size=0.2)

# we reset the indices to start from zero
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)


class ethiopicDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text 
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]
        # prepare image (i.e. resize + normalize)
        image = Image.open(self.root_dir + file_name).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text, 
                                          padding="max_length", 
                                          max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding



processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
#processor = TrOCRProcessor.from_pretrained("./ethiopic")

train_dataset = ethiopicDataset(root_dir='./new_hhd/train/train_raw/image_train/',
                           df=train_df,
                           processor=processor)
eval_dataset = ethiopicDataset(root_dir='./new_hhd/train/train_raw/image_train/',
                           df=test_df,
                           processor=processor)

print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(eval_dataset))

encoding = train_dataset[0]
for k,v in encoding.items():
  print(k, v.shape)

labels = encoding['labels']
labels[labels == -100] = processor.tokenizer.pad_token_id
label_str = processor.decode(labels, skip_special_tokens=True)
print(labels)
print(label_str)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
model.to(device)

# set special tokens used for creating the decoder_input_ids from the labels
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
# make sure vocab size is set correctly
model.config.vocab_size = model.config.decoder.vocab_size

# set beam search parameters
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = 64
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 2



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    num_train_epochs=2,
    predict_with_generate=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    fp16=True, 
    output_dir="./trocr_check_point/",
    logging_steps=2,
    save_steps=500,
    eval_steps=1000,
)


from datasets import load_metric

cer_metric = load_metric("cer")
'''
def compute_cer(pred_ids, label_ids):
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return cer
'''
def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}

from transformers import default_data_collator

# instantiate trainer
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=processor.feature_extractor,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=default_data_collator,
)
trainer.train()

'''
from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=4)


from transformers import AdamW
from tqdm.notebook import tqdm

optimizer = AdamW(model.parameters(), lr=5e-5)

# loop over the dataset multiple times
for epoch in range(2):  
   # train
   model.train()
   train_loss = 0.0
   for batch in tqdm(train_dataloader):
      # get the inputs
      for k,v in batch.items():
        batch[k] = v.to(device)

      # forward + backward + optimize
      outputs = model(**batch)
      loss = outputs.loss
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      train_loss += loss.item()

   print(f"Loss after epoch {epoch}:", train_loss/len(train_dataloader))
    
   # evaluate
   model.eval()
   valid_cer = 0.0
   with torch.no_grad():
     for batch in tqdm(eval_dataloader):
       # run batch generation
       outputs = model.generate(batch["pixel_values"].to(device))
       # compute metrics
       cer = compute_cer(pred_ids=outputs, label_ids=batch["labels"])
       valid_cer += cer 

   print("Validation CER:", valid_cer / len(eval_dataloader))

model.save_pretrained('./trocr')
'''
trainer.save_model('./trocr_full')
processor.save_pretrained('./ethiopic_full')

print("============you model is saved==========================")
print("now we are going to test the model performacen with test set")
image = Image.open('./new_hhd/sample_test_rand_raw/sample_im_rand/test_rand_00302.png').convert("RGB")

p = processor(image, return_tensors="pt").pixel_values
#processor_new= TrOCRProcessor.from_pretrained("./ethiopic")
model = VisionEncoderDecoderModel.from_pretrained("./trocr_fulls")

generated_ids = model.generate(p, max_length=128)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_ids)
print(generated_text)

#test_dataset = ethiopicDataset(root_dir='./new_hhd/sample_test_rand_raw/sample_im_rand/',
                          # df=test_df,
                          # processor=processor)




