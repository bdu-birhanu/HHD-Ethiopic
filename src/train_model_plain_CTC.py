from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Permute, Flatten, Masking, \
    GaussianNoise, Reshape, Lambda, TimeDistributed, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import GRU, LSTM, Bidirectional, MaxPool2D
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
import numpy as np
from data_loader import num_class, maxlen
from data_loader import x_train, y_train, x_train_length, y_train_length, x_val, y_val, x_val_length, y_val_length


# import matplotlib.pyplot as plt

def ctc_lambda_func(args):
    '''

    '''
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def hist_ocr_model(batch_size=64, epoch=25, rnn_size=256, img_row=32, img_col=200):
    '''
    if you use the full datset you could increase the batch_size and epo
    '''
    inputs_data = Input(shape=(img_row, img_col, 1))
    conv_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs_data)
    # poolig layer with kernel size (2,2)
    pool_1 = MaxPool2D(pool_size=(2, 1), strides=2)(conv_1)
    conv_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool_1)
    pool_2 = MaxPool2D(pool_size=(2, 1))(conv_2)  # we remove the strides here
    conv_3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool_2)
    conv_4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv_3)
    # poolig layer with kernel size (2,1)
    pool_3 = MaxPool2D(pool_size=(2, 1))(conv_4)
    # Batch normalization layer
    batch_norm_5 = BatchNormalization()(pool_3)
    conv_5 = Conv2D(128, (3, 3), activation='relu', padding='same')(batch_norm_5)

    conv_6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv_5)
    batch_norm_6 = BatchNormalization()(conv_6)
    conv_7 = Conv2D(512, (2, 2), activation='relu')(batch_norm_6)

    # conv2=(int(conv1[2]),int(conv1[1]*int(conv1[3])))
    r = Reshape((int(conv_7.shape[2]), int(conv_7.shape[1]) * int(conv_7.shape[3])))(conv_7)
    # [ sample, timesteps, features]
    # bidirectional LSTM layers with units=128
    blstm_1 = Bidirectional(LSTM(rnn_size, return_sequences=True, dropout=0.25))(r)
    blstm_2 = Bidirectional(LSTM(rnn_size, return_sequences=True, dropout=0.25))(blstm_1)
    outputs = Dense(num_class + 1, activation='softmax')(blstm_2)

    pred_model = Model(inputs_data, outputs)

    labels = Input(name='the_labels', shape=[46], dtype='float32')  # 46 is the max size of text length
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([outputs, labels, input_length, label_length])

    model = Model(inputs=[inputs_data, labels, input_length, label_length], outputs=loss_out)
    # lrate = 0.01
    # decay = lrate / epoch
    # sgd = SGD(learning_rate=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')

 #to same models with better prformace during training at each epoch
    filepath = "tr12_check.hdf5"
    checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    early_stop = EarlyStopping(monitor='val_loss', patience=15)
    callbacks_list = [checkpoint, early_stop]

    hist = model.fit(x=[x_train, y_train, x_train_length, y_train_length], y=np.zeros(len(y_train)),
                     batch_size=batch_size, epochs=epoch,
                     validation_data=([x_val, y_val, x_val_length, y_val_length], [np.zeros(len(y_val))]),
                     verbose=1, callbacks=callbacks_list)

    return pred_model, hist


if __name__ == "__main__":
    model_train = hist_ocr_model()
    model = model_train[0]
    hist = model_train[1]
    model.summary()
    model.save('./model/model_CTC_0.hdf5')
    np.save('./model/plain_CTC.npy', hist.history)
    print("Training is successfully completed and now your model  and history are stored to your disk")



