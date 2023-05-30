import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import RepeatVector, Dot, Bidirectional,LSTM,Input, Dense, Activation,Lambda,Concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Permute, Flatten, Masking, \
    GaussianNoise, Reshape, Lambda, TimeDistributed, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import GRU, LSTM, Bidirectional, MaxPool2D
import tensorflow as tf

#from keras.models import load_model
from keras import backend as K
#import matplotlib.pyplot as plt
import editdistance

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
import numpy as np
import cv2
from data_loader_hhd import num_class, maxlen,im_col, im_row
from data_loader_hhd  import x_train, y_train, x_train_length, y_train_length, x_val, y_val, x_val_length, y_val_length

x_test_rand_var = np.load('./test/test_rand/test_rand_numpy/x_test_rand.npy', allow_pickle=True)
x_test_18th_var = np.load('./test/test_18th/test_18th_numpy/x_test_18th.npy', allow_pickle=True)
y_test_rand = np.load('./test/test_rand/test_rand_numpy/y_test_rand.npy', allow_pickle=True)
y_test_18th = np.load('./test/test_18th/test_18th_numpy/y_test_18th.npy', allow_pickle=True)


dot_pro = Dot(axes = 1)
concatenat = Concatenate(axis=-1)
# import matplotlib.pyplot as plt
def resize(x_image):


    resized_x_rand = np.zeros((len(x_image), im_row, im_col), dtype=np.uint8)

    # loop over the input images
    for i, image in enumerate(x_image):
        # resize the image to 48 by 368 using padding
        current_height, current_width = image.shape[:2]
        aspect_ratio_current = current_width / current_height
        aspect_ratio_target = im_col / im_row
        if aspect_ratio_current != aspect_ratio_target:
            if aspect_ratio_current > aspect_ratio_target:
                new_height = int(current_width / aspect_ratio_target)
                top_padding = (new_height - current_height) // 2
                bottom_padding = new_height - current_height - top_padding
                padded_image = cv2.copyMakeBorder(image, top_padding, bottom_padding, 0, 0, cv2.BORDER_CONSTANT, value=255)
            else:
                new_width = int(current_height * aspect_ratio_target)
                left_padding = (new_width - current_width) // 2
                right_padding = new_width - current_width - left_padding
                padded_image = cv2.copyMakeBorder(image, 0, 0, 0, right_padding+left_padding, cv2.BORDER_CONSTANT, value=255)
        else:
            padded_image = image
        resized_image = cv2.resize(padded_image, (im_col, im_row))

        resized_x_rand[i] = resized_image

    return resized_x_rand


x_test_rand= resize(x_test_rand_var)
x_test_18th= resize(x_test_18th_var)


dot_pro = Dot(axes = 1)
def attention(blstm_2):
    score = Dense(91, activation='tanh', name='attention_score_vec')(blstm_2)

    attention_weights = Activation('softmax', name='attention_weight')(score)

    # (batch_size, hidden_size, time_steps) dot (batch_size, time_steps, 1) => (batch_size, hidden_size, 1)
    context_vector = dot_pro([attention_weights,blstm_2])
    #context_vector_conb = Concatenate(axis=-1)([context_vector, output1])

    return  context_vector


def ctc_lambda_func(args):
    '''

    '''
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def hist_ocr_model(batch_size=32, epoch=50, rnn_size=128, img_row=im_row, img_col=im_col):
    # '''
    # if you use the full datset you could increase the batch_size and epo
    # '''
    inputs_data = Input(shape=(img_row, img_col, 1))
    conv_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs_data)
    # poolig layer with kernel size (2,2)
    pool_1 = MaxPool2D(pool_size=(2, 1), strides=2)(conv_1)
    conv_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool_1)
    pool_2 = MaxPool2D(pool_size=(2, 2))(conv_2)  # we remove the strides here
    conv_3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool_2)
    conv_4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv_3)
    # poolig layer with kernel size (2,1)
    pool_3 = MaxPool2D(pool_size=(2, 1))(conv_4)
    # Batch normalization layer
    batch_norm_5 = BatchNormalization()(pool_3)
    conv_5 = Conv2D(128, (3, 3), activation='relu', padding='same')(batch_norm_5)

    conv_6 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv_5)
    batch_norm_6 = BatchNormalization()(conv_6)
    conv_7 = Conv2D(128, (2, 2), activation='relu')(batch_norm_6)

    # conv2=(int(conv1[2]),int(conv1[1]*int(conv1[3])))
    r = Reshape((int(conv_7.shape[2]), int(conv_7.shape[1]) * int(conv_7.shape[3])))(conv_7)
    # [ sample, timesteps, features]
    # bidirectional LSTM layers with units=128
    blstm_1 = Bidirectional(LSTM(rnn_size, return_sequences=True, dropout=0.25))(r)
    blstm_2 = Bidirectional(LSTM(rnn_size, return_sequences=True, dropout=0.25))(blstm_1)
    #pretrained_modelctc = tf.keras.models.load_model(
    #     './model/tr1_25_BBO_6k.hdf5',
    #     custom_objects={'Functional': tf.keras.models.Model})
    #
    # for layer in pretrained_modelctc.layers[:17]:  # this is used to freeze the lower layers of the MDPI paper up to 16th
    #     layer.trainable = False
    #
    
    #context = attention(output1,encoder_out)
    context = attention(blstm_2)
    outputs = Dense(num_class + 1, activation='softmax')(context)

    #pred_model = Model(pretrained_modelctc.input, outputs)
    pred_model = Model(inputs_data, outputs)
    # context = attention(blstm_2)
    # outputs = Dense(num_class + 1, activation='softmax')(context)

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
    filepath = "tr12_check_belend.hdf5"
    checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    early_stop = EarlyStopping(monitor='val_loss', patience=15)
    callbacks_list = [checkpoint, early_stop]

    hist = model.fit(x=[x_train, y_train, x_train_length, y_train_length], y=np.zeros(len(y_train)),
                     batch_size=batch_size, epochs=epoch,
                     validation_data=([x_val, y_val, x_val_length, y_val_length], [np.zeros(len(y_val))]),
                     verbose=1, callbacks=callbacks_list)

    return pred_model, hist


if __name__ == "__main__":

    y_pred_all_rand=[]
    y_pred_all_18th= []
    path = './model/'
    rounds=10

    for i in range(rounds):
        model_train = hist_ocr_model()
        model = model_train[0]
        model.save(path + f'model_belened_CTC_{i}.hdf5')
        y_pred = model.predict(x_test_rand)
        y_pred_18 = model.predict(x_test_18th)
        y_pred_all_rand.append(y_pred)
        y_pred_all_18th.append(y_pred_18)
        print(f'this is training belended-CTC round_{i}')

    y_decode_all_rand=[]
    y_decode_all_18th=[]
    for j in range(rounds):
        y_decode = K.get_value(
            K.ctc_decode(y_pred_all_rand[j][:, :, :], input_length=np.ones(y_pred_all_rand[j].shape[0]) * y_pred_all_rand[j].shape[1])[0][0])
        y_decod_18 = K.get_value(
            K.ctc_decode(y_pred_all_18th[j][:, :, :], input_length=np.ones(y_pred_all_18th[j].shape[0]) * y_pred_all_18th[j].shape[1])[0][0])
        y_decode_all_rand.append(y_decode)
        y_decode_all_18th.append(y_decod_18)

    CER_all_rand=[]
    CER_all_18th = []
    for k in range(rounds):

        true = []  # to store value of character by removing zero which was padded previously and also this is the value of newline in the test label
        for i in range(len(y_test_rand)):
            x = [j for j in y_test_rand[i] if j != 0]
            true.append(x)

        pred = []
        # to stor the predicted charcter except zerro and -1 which are padded value nad blank space predicted during testing
        # for i in range(len(y_decode_all_rand[k])):
        #     x = [j for j in y_decode_all_rand[k][i] if j not in (0, -1)]
        #     pred.append(x)
        for i in range(len(y_decode_all_rand[k])):
            kk = []
            for j in y_decode_all_rand[k][i]:
                if j == 0:
                    break
                if j not in (0, -1):
                    kk.append(j)
            pred.append(kk)


        cer = 0
        for (i, j) in zip(true, pred):
            x = editdistance.eval(i, j)
            cer = cer + x
        err = cer
        x = 0
        for i in range(len(true)):
            x = x + len(true[i])
        totalchar = x
        cerp = (float(err) / totalchar) * 100

        CER_all_rand.append(cerp)



        # 18th century test set

        true_18 = []  # to stor value of character by removing zero which was padded previously and also this is the value of newline in the test label
        for i in range(len(y_test_18th)):
            x = [j for j in y_test_18th[i] if j != 0]
            true_18.append(x)

        pred_18 = []
        # to stor the pdicted charcter except zerro and -1 which are padded value nad blank space predicted during testing
        # for i in range(len(y_decode_all_18th[k])):
        #     x = [j for j in y_decode_all_18th[k][i] if j not in (0, -1)]
        #     pred_18.append(x)
        for i in range(len(y_decode_all_18th[k])):
            kkk = []
            for j in y_decode_all_18th[k][i]:
                if j == 0:
                    break
                if j not in (0, -1):
                    kkk.append(j)
            pred_18.append(kkk)

        cer_18 = 0
        for (i, j) in zip(true_18, pred_18):
            x = editdistance.eval(i, j)
            cer_18 = cer_18 + x

        err_18 = cer_18

        x_18 = 0
        for i in range(len(true_18)):
            x_18 = x_18 + len(true_18[i])

        totalchar_18 = x_18

        cerp_18 = (float(err_18) / totalchar_18) * 100
        CER_all_18th.append(cerp_18)


    print("the CER of random test set_with blended_CTC:")
    print(CER_all_rand)

    print("the CER of 18th century test set_with blended_CTC:")
    print(CER_all_18th)
