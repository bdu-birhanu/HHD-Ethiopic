# -*- coding: utf-8 -*-
import sys
import tensorflow as tf
sys.path.append('..')
tf.get_logger().setLevel('ERROR')
APPROACH_NAME = 'CNNxTransformer'

"""# Check GPU working"""


"""# Data input pipeline"""

#DATASET_DIR = './new_hhd/sample_test_rand_raw'
#DATASET_DIR1 = './new_hhd/train/train_raw/sample_im_rand'
#ALL_TRANSCRIPTS_PATH = '{}{}'.format(DATASET_DIR, '/image_text_pairs_test_rand.csv')

DATASET_DIR = r'./new_hhd/train/train_raw'
DATASET_DIR1 = r'./new_hhd/train/train_raw/image_train'
#ALL_TRANSCRIPTS_PATH = f'{DATASET_DIR}/image_text_pairs_train.csv'
ALL_TRANSCRIPTS_PATH = '{}{}'.format(DATASET_DIR, '/image_text_pairs_train.csv')

#testset
DATASET_DIR_rand = r'./new_hhd/test/test_rand/test_rand_raw/'
DATASET_DIR_rand1 = r'./new_hhd/test/test_rand/test_rand_raw/image_rand'
ALL_TRANSCRIPTS_PATH_rand = '{}{}'.format(DATASET_DIR_rand, '/image_text_pairs_test_rand.csv')
#18th
DATASET_DIR_18th = r'./new_hhd/test/test_18th/test_18th_raw/'
DATASET_DIR_18th1 = r'./new_hhd/test/test_18th/test_18th_raw/image_18th'
ALL_TRANSCRIPTS_PATH_18th = '{}{}'.format(DATASET_DIR_18th, '/image_text_pairs_test_18th.csv')

#import sys
#sys.path.insert(0, '/content/drive/MyDrive/Code_2023/Text recognition/')

"""## Load and remove records with rare characters"""


from loader import DataImporter
dataset = DataImporter(DATASET_DIR1, ALL_TRANSCRIPTS_PATH, min_length=1)
print(dataset)

#rand test
dataset_rand = DataImporter(DATASET_DIR_rand1, ALL_TRANSCRIPTS_PATH_rand, min_length=1)
print("rand test")
print(dataset_rand)

#18th test
dataset_18th = DataImporter(DATASET_DIR_18th1, ALL_TRANSCRIPTS_PATH_18th, min_length=1)
print("18th test")
print(dataset_18th)

"""## Data constants and input pipeline"""

HEIGHT, WIDTH = 48,432
PADDING_CHAR = '[PAD]' 
START_CHAR = '[START]'
END_CHAR = '[END]'

from loader import DataHandler
data_handler = DataHandler(
    dataset, 
    img_size = (HEIGHT, WIDTH), 
    padding_char = PADDING_CHAR,
    start_char = START_CHAR,
    end_char = END_CHAR
)

NUM_VALIDATE = DataImporter(DATASET_DIR1, ALL_TRANSCRIPTS_PATH, min_length=1).size
#number of test rand
NUM_rand = DataImporter(DATASET_DIR_rand1, ALL_TRANSCRIPTS_PATH_rand, min_length=1).size
#number of 18th rand
NUM_18th = DataImporter(DATASET_DIR_18th1, ALL_TRANSCRIPTS_PATH_18th, min_length=1).size

MAX_LENGTH = data_handler.max_length
START_TOKEN = data_handler.start_token
END_TOKEN = data_handler.end_token
VOCAB_SIZE = data_handler.char2num.vocab_size()
BATCH_SIZE = 32
print("train:")
print(NUM_VALIDATE)
print(VOCAB_SIZE)
print(MAX_LENGTH)
print("rand test:")
print(NUM_rand)
print("18th test:")
print(NUM_18th)

"""## Visualize the data"""

import importlib
import visualizer
importlib.reload(visualizer)

from visualizer import visualize_images_labels
visualize_images_labels(
    dataset.img_paths, 
    dataset.labels, 
    figsize = (18, 3),
    subplot_size = (3, 3),
    #font_path = FONT_PATH
)

"""# Define model components"""

from tensorflow.keras.layers import Input
from layers import custom_cnn, reshape_features
from transformer import TransformerEncoderBlock, TransformerDecoderBlock

"""## Features extraction"""

def get_cnn_model(imagenet_model=None, imagenet_output_layer=None, name='CNN_model'):
    if imagenet_model: # Use Imagenet model as CNN layers
        image_input = imagenet_model.input
        imagenet_model.layers[0]._name = 'image'
        x = imagenet_model.get_layer(imagenet_output_layer).output
    else: 
        image_input = Input(shape=(HEIGHT, WIDTH, 3), dtype='float32', name='image')
        conv_blocks_config = {
            'block1': {'num_conv': 1, 'filters':  64, 'pool_size': (2, 2)}, 
            'block2': {'num_conv': 1, 'filters': 128, 'pool_size': (2, 2)}, 
            'block3': {'num_conv': 2, 'filters': 256, 'pool_size': (2, 2)}, 
            'block4': {'num_conv': 2, 'filters': 512, 'pool_size': (2, 1)}, 
            
            # Last Conv blocks with 2x2 kernel but without no padding and pooling layer
            'block5': {'num_conv': 2, 'filters': 512, 'pool_size': None}, 
        }
        x = custom_cnn(conv_blocks_config, image_input)
    
    # Reshape accordingly before passing output to the Transformer encoder
    feature_maps = reshape_features(x, dim_to_keep=2)
    return tf.keras.Model(inputs=image_input, outputs=feature_maps, name=name)

from models import get_imagenet_model
imagenet_model, imagenet_output_layer = None, None
"""## Transformer encoder and decoder"""

NUM_LAYERS = 2
NUM_HEADS = 1
EMBEDDING_DIM = 512 # d_model
FEED_FORWARD_UNITS = 512 # dff
DROPOUT_RATE = 0.1

cnn_model = get_cnn_model(imagenet_model, imagenet_output_layer)
encoder = TransformerEncoderBlock(
    receptive_size = cnn_model.layers[-1].output_shape[-2],
    num_layers = NUM_LAYERS, # N encoder layers
    num_heads = NUM_HEADS,
    embedding_dim = EMBEDDING_DIM, # d_model
    feed_forward_units = FEED_FORWARD_UNITS, # dff
    dropout_rate = DROPOUT_RATE,
)
decoder = TransformerDecoderBlock(
    enc_shape = encoder.output_shape[1:], # (receptive_size, enc_channels)
    seq_length = MAX_LENGTH - 1, # The inputs is shifted by 1
    vocab_size = VOCAB_SIZE,
    num_layers = NUM_LAYERS, # N decoder layers
    num_heads = NUM_HEADS,
    embedding_dim = EMBEDDING_DIM, # d_model
    feed_forward_units = FEED_FORWARD_UNITS, # dff
    dropout_rate = DROPOUT_RATE,
)


from transformer import TransformerOCR
model = TransformerOCR(cnn_model, encoder, decoder, data_handler)
cnn_model.summary(line_length=90)
print()
encoder.summary(line_length=110)
print()
decoder.summary(line_length=110)
print()
model.summary()

"""# Training"""

import random

# Generate a list of indices from 0 to 1000
indices = list(range(NUM_VALIDATE))
random.shuffle(indices)

#  10% of the total for val
num_validation_indices = int(0.1 * len(indices))

# Select the first num_validation_indices indices for validation
valid_idxs = indices[:num_validation_indices]

# The remaining indices will be used for training
train_idxs = indices[num_validation_indices:]

# Print the validation and training indices to verify
#print(f"Validation indices: {valid_idxs}")
#print(f"Training indices: {train_idxs}")
print('Number of training samples:', len(train_idxs))
print('Number of validate samples:', len(valid_idxs))
#======================================================================================
#test sets
#indices_rand = list(range(NUM_rand))
#idxs_rand = indices_rand

#indices_18th = list(range(NUM_18th))
#idxs_18th = indices_18th

offset_rand = 0
offset_18th = NUM_rand

idxs_rand = list(range(offset_rand, offset_rand + NUM_rand))
idxs_18th = list(range(offset_18th, offset_18th + NUM_18th))
# Print the validation and training indices to verify
#print(f"Validation indices: {valid_idxs}")
#print(f"Training indices: {train_idxs}")
print('Number of rand test samples:', len(idxs_rand))
print('Number of 18th test samples:', len(idxs_18th))
#======================================================================================
import random
random.seed(2022)
random.shuffle(train_idxs)

# When run on a small RAM machine, you can set use_cache=False to 
# not run out of memory but it will slow down the training speed
#drop_remainder=True indicates that any remaining samples that cannot form a complete batch will be droppedssss
train_tf_dataset = data_handler.prepare_tf_dataset(train_idxs, BATCH_SIZE, drop_remainder=True)
valid_tf_dataset = data_handler.prepare_tf_dataset(valid_idxs, BATCH_SIZE, drop_remainder=True)

#=================================================
#it ensures that all batches have the same size and avoids potential issues with incompatible tensor shapes
#drop_remainder=True
tf_dataset_rand = data_handler.prepare_tf_dataset(idxs_rand, BATCH_SIZE, drop_remainder=False)
tf_dataset_18th = data_handler.prepare_tf_dataset(idxs_18th, BATCH_SIZE, drop_remainder=False)

"""## Callbacks"""

from tensorflow.keras.callbacks import EarlyStopping
early_stopping_callback = EarlyStopping(
    monitor = 'val_loss', 
    min_delta = 1e-3, # Change that less than 1e-3, will count as no improvement
    patience = 7, # Stop if no improvement after 7 epochs
    restore_best_weights = True, # Restore weights from the epoch with the best value
    verbose = 1
)

"""## Fine-tuning the dataset"""

from losses import MaskedLoss
from metrics import SequenceAccuracy, CharacterAccuracy, LevenshteinDistance
from tensorflow.keras.optimizers import Adadelta

# Adadelta tends to benefit from higher initial learning rate values compared to
# other optimizers. Here use 1.0 to match the exact form in the original paper
LEARNING_RATE = 1.0
EPOCHS = 2

model.compile(
    optimizer = Adadelta(LEARNING_RATE), 
    loss = MaskedLoss(), 
    metrics = [
        SequenceAccuracy(),
        CharacterAccuracy(),
        LevenshteinDistance(normalize=True)
    ]
)

# Commented out IPython magic to ensure Python compatibility.
 #%%time
history = model.fit(
     train_tf_dataset,
     validation_data = valid_tf_dataset,
     epochs = EPOCHS,
     callbacks = [early_stopping_callback],
     verbose = 1
 )
 
path= './model/'
best_epoch = early_stopping_callback.best_epoch
from tensorflow.keras.models import save_model, load_model
model.cnn_model.save_weights('./model/model_cnn_transformer5_cnn.h5')
model.encoder.save_weights('./model/model_cnn_transformer5_enc.h5')
model.decoder.save_weights('./model/model_cnn_transformer5_dec.h5')

model.save('./model/model_cnn_transformer_test',save_format='tf')

import editdistance
"""## On test dataset"""
'''
true=[]
pred=[]
batch_results = []
#The valid_tf_dataset.take(1) limits the loop to only one iteration, processing a single batch.
for idx, (batch_images, batch_tokens) in enumerate(tf_dataset_rand):
    idxs_in_batch = idxs_rand[idx * BATCH_SIZE: (idx + 1) * BATCH_SIZE]
    labels = data_handler.tokens2texts(batch_tokens)
    pred_tokens, attentions = model.predict(batch_images, return_attention=True)
    pred_labels = data_handler.tokens2texts(pred_tokens)
    true.append(labels)
    pred.append(pred_labels)
    batch_results.append({'true': labels, 'pred': pred_labels, 'attentions': attentions})

print(
f'Batch {idx + 1:02d}:\n'
f'- True: {dict(enumerate(labels[:10], start=1))}\n'
f'- Pred: {dict(enumerate(pred_labels[:10], start=1))}\n'
)


print("true and pred")
print(true[100])
print(pred[100])


total_edits = 0
true_len = 0

for true_sampler, pred_sampler in zip(true, pred):
    true_len += sum(len(s) for s in true_sampler)
    num_true = len(true_sampler)
    num_pred = len(pred_sampler)
    
    for i in range(max(num_true, num_pred)):
        if i < num_true and i < num_pred:
            true_str = true_sampler[i]
            pred_str = pred_sampler[i]
            total_edits += editdistance.eval(true_str, pred_str)
        elif i < num_true_18th:
            total_edits += len(true_sampler[i])
        else:
            total_edits += len(pred_sampler[i])

CER = total_edits / true_len

print(f"Total edit distance rand: {total_edits}")
print(f"Number of rand_true batch samples: {len(true)}")
print(f"Total length of true samples: {true_len}")
print(f"CER of rand: {CER}")

'''
#trial
print("===========================trial===============================================")

true_18th=[]
pred_18th=[]
batch_results_18th = []
#The valid_tf_dataset.take(1) limits the loop to only one iteration, processing a single batch.
for idx, (batch_images, batch_tokens) in enumerate(tf_dataset_18th):
    idxs_in_batch = idxs_18th[idx * BATCH_SIZE: (idx + 1) * BATCH_SIZE]
    labels = data_handler.tokens2texts(batch_tokens)
    pred_tokens, attentions = model.predict(batch_images, return_attention=True)
    pred_labels = data_handler.tokens2texts(pred_tokens)
    true_18th.append(labels)
    pred_18th.append(pred_labels)
    batch_results_18th.append({'true': labels, 'pred': pred_labels, 'attentions': attentions})
    '''
    print(
        f'Batch {idx + 1:02d}:\n'
        f'- True_18th: {dict(enumerate(labels[:10], start=1))}\n'
        f'- Pred_18th: {dict(enumerate(pred_labels[:10], start=1))}\n'
    )
    '''
print("true_18th and pred_18th")
print("index_18t")
print(true_18th[40])
print(pred_18th[40])

total_edits_18th = 0
true_len_18th = 0

for true_sample, pred_sample in zip(true_18th, pred_18th):
    true_len_18th += sum(len(s) for s in true_sample)
    num_true_18th = len(true_sample)
    num_pred_18th = len(pred_sample)
    
    for i in range(max(num_true_18th, num_pred_18th)):
        if i < num_true_18th and i < num_pred_18th:
            true_str_18th = true_sample[i]
            pred_str_18th = pred_sample[i]
            total_edits_18th += editdistance.eval(true_str_18th, pred_str_18th)
        elif i < num_true_18th:
            total_edits_18th += len(true_sample[i])
        else:
            total_edits_18th += len(pred_sample[i])

CER_18th = total_edits_18th / true_len_18th

print(f"Total edit distance: {total_edits_18th}")
print(f"Number of true samples: {len(true_18th)}")
print(f"Total length of true chracter: {true_len_18th}")
print(f"CER of 18th: {CER_18th}")



