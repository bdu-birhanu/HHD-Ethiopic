import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import tensorflow as tf
from torch.utils.data import Dataset
from PIL import Image

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


df = pd.read_fwf('./test/test_rand/test_rand_raw/image_text_pairs_test_rand.csv', header=None)
#df = pd.read_fwf('./new_hhd/test/test_18th/test_18th_raw/image_text_pairs_test_18th.csv', header=None)
df.rename(columns={0: "file_name", 1: "text"}, inplace=True)
#del df[2]
df = df.applymap(lambda x: x.replace(',', ''))
df.head()

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
#processor = TrOCRProcessor.from_pretrained("./ethiopic")




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



test_dataset = ethiopicDataset(root_dir='./new_hhd/test/test_rand/test_rand_raw/image_rand/',
                           df=df,
                           processor=processor)
#test_dataset = ethiopicDataset(root_dir='./new_hhd/test/test_18th/test_18th_raw/image_18th/',
#                           df=df,
#                           processor=processor)

# load the tokenizer

from torch.utils.data import DataLoader

test_dataloader = DataLoader(test_dataset, batch_size=8)

batch = next(iter(test_dataloader))
for k,v in batch.items():
  print(k, v.shape)


labels = batch["labels"]
labels[labels == -100] = processor.tokenizer.pad_token_id
label_str = processor.batch_decode(labels, skip_special_tokens=True)
print(label_str[:5])

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VisionEncoderDecoderModel.from_pretrained("./trocr_full")
model.to(device)


from datasets import load_metric

cer = load_metric("cer")

from tqdm.notebook import tqdm

print("Running evaluation...")

pred_18th=[]
truth_18th=[]
for batch in tqdm(test_dataloader):
    # predict using generate
    pixel_values = batch["pixel_values"].to(device)
    outputs = model.generate(pixel_values)

    # decode
    pred_str = processor.batch_decode(outputs, skip_special_tokens=True)
    pred_18th.append(pred_str)
    labels = batch["labels"]
    labels[labels == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels, skip_special_tokens=True)
    truth_18th.append(label_str)
    # add batch to metric
    cer.add_batch(predictions=pred_str, references=label_str)
print(pred_18th[:5])
print(truth_18th[:5])
final_score = cer.compute()
print("CER_Rand:", final_score)

#print("now we are going to test the model performacen with test set")

#image = Image.open('./new_hhd/sample_test_rand_raw/sample_im_rand/test_rand_00302.png').convert("RGB")

#p = processor(image, return_tensors="pt").pixel_values

#model = VisionEncoderDecoderModel.from_pretrained("./trocr")
#generated_ids = model.generate(p, max_length=128)
#generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

#with open("generated_text.txt", "w", encoding="utf-8") as f:
    #f.write(generated_text)

#print(generated_ids)
#print(generated_text)

#test_dataset = ethiopicDataset(root_dir='./new_hhd/sample_test_rand_raw/sample_im_rand/',
                          # df=test_df,
                          # processor=processor)




