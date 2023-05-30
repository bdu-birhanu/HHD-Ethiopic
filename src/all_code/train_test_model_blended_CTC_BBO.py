#new deephyper to select the best hyper-parameter with 10% of my test data
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Permute, Flatten, Masking, \
    GaussianNoise, Reshape, Lambda, TimeDistributed, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import GRU, LSTM, Bidirectional, MaxPool2D
from tensorflow.keras.layers import RepeatVector, Multiply, Dot, Bidirectional,LSTM, Input, Dense, Activation,Lambda,Concatenate
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
import numpy as np
from data_loader import num_class, maxlen
from keras import backend as K
#import matplotlib.pyplot as plt
import editdistance


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
# import matplotlib.pyplot as plts
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


def hist_ocr_model(config: dict, n_components: int = 5, verbose: bool = 0, num_class=num_class, img_row=im_row, img_col=im_col):
    tf.keras.utils.set_random_seed(2)

    default_config = {
        "rnn_size": 128,
        "feature_map_1": 64,
        "feature_map_2": 128,
        "activation": "relu",
        "batch_size": 32,
        "epoch": 10,
        "kernel": 3,
        "dropout": 0.25,

    }
    default_config.update(config)
    '''
    if you use the full datset you could increase the batch_size and epo
    '''
    
    inputs_data = Input(shape=(img_row, img_col, 1))
    conv_1 = Conv2D(default_config["feature_map_1"], (default_config["kernel"], default_config["kernel"]),
                    activation=default_config["activation"], padding='same')(inputs_data)
    # poolig layer with kernel size (2,2)
    pool_1 = MaxPool2D(pool_size=(2, 1), strides=2)(conv_1)
    conv_2 = Conv2D(default_config["feature_map_1"], (default_config["kernel"], default_config["kernel"]),
                    activation=default_config["activation"], padding='same')(pool_1)
    pool_2 = MaxPool2D(pool_size=(2, 2))(conv_2)  # we remove the strides here
    conv_3 = Conv2D(default_config["feature_map_1"], (default_config["kernel"], default_config["kernel"]),
                    activation=default_config["activation"], padding='same')(pool_2)
    conv_4 = Conv2D(default_config["feature_map_1"], (default_config["kernel"], default_config["kernel"]),
                    activation=default_config["activation"], padding='same')(conv_3)
    # poolig layer with kernel size (2,1)
    pool_3 = MaxPool2D(pool_size=(2, 1))(conv_4)
    # Batch normalization layer
    batch_norm_5 = BatchNormalization()(pool_3)
    conv_5 = Conv2D(default_config["feature_map_1"], (default_config["kernel"], default_config["kernel"]),
                    activation=default_config["activation"], padding='same')(batch_norm_5)

    conv_6 = Conv2D(default_config["feature_map_1"], (default_config["kernel"], default_config["kernel"]),
                    activation=default_config["activation"], padding='same')(conv_5)
    batch_norm_6 = BatchNormalization()(conv_6)
    conv_7 = Conv2D(default_config["feature_map_1"], (default_config["kernel"], default_config["kernel"]),
                    activation=default_config["activation"], padding='same')(batch_norm_6)
    #conv_7 = Conv2D(default_config["feature_map_1"], (default_config["kernel"], default_config["kernel"]),
                    #activation=default_config["activation"])(batch_norm_6)

    # conv2=(int(conv1[2]),int(conv1[1]*int(cosnv1[3])))
    r = Reshape((int(conv_7.shape[2]), int(conv_7.shape[1]) * int(conv_7.shape[3])))(conv_7)
    # [ sample, timesteps, features]
    # bidirectional LSTM layers with units=128
    blstm_1 = Bidirectional(LSTM(default_config["rnn_size"], return_sequences=True, dropout=default_config["dropout"]))(r)
    blstm_2 = Bidirectional(LSTM(default_config["rnn_size"], return_sequences=True, dropout=default_config["dropout"]))(blstm_1)

    #output1 = Dense(num_class + 1, activation='softmax')(blstm_2)
    #context = attention(output1,encoder_out)
    context = attention(blstm_2)
    outputs = Dense(num_class + 1, activation='softmax')(context)

    #pred_model = Model(pretrained_modelctc.input, outputs)
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

    hist = model.fit(x=[x_train, y_train, x_train_length, y_train_length], y=np.zeros(len(y_train)),
                     batch_size=default_config["batch_size"], epochs=100,
                     validation_data=([x_val, y_val, x_val_length, y_val_length], [np.zeros(len(y_val))]),
                     verbose=1)

    return model, pred_model, hist

# from deephyper.problem import HpProblem
#
# problem = HpProblem()
# problem.add_hyperparameter((10, 256), "rnn_size", default_value=128)
# problem.add_hyperparameter(["sigmoid", "tanh", "relu"], "activation", default_value="relu")
# problem.add_hyperparameter((2, 4), "batch_size", default_value=2)
# problem.add_hyperparameter((10, 100), "epochs", default_value=20)
# problem


# def run(config):
#     # important to avoid memory exploision
#     tf.keras.backend.clear_session()
#
#     _, _, hist = hist_ocr_model(config, n_components=5, verbose=0)
#
#     return -hist.history["val_loss"][-1]
#
#
# from deephyper.search.hps import CBO
#
# search = CBO(problem, run, initial_points=[problem.default_configuration], log_dir="cbo-results", random_state=2)
# results = search.search(max_evals=4)
#
# i_max = results.objective.argmax()
# best_config = results.iloc[i_max][:-4].to_dict()
# best_config
with open('./model/best_hyp_train_attention.txt') as f:
    data = f.read()
best_config = eval(data)# toget dictionary from the string which is saved in the disk

#best_model,best_pred_model, best_history = hist_ocr_model(best_config, n_components=5, verbose=1)

# best_model,best_pred_model, best_history = hist_ocr_model(best_config, n_components=5, verbose=1)
#
# best_pred_model.save('./model/tr1_25_attention_opt.hdf5')
# np.save('./model/tr1_history1_25_BBO_615k.npy', best_history.history)
# print("Training is successfully completed and now your model  and history are stored to your disk")
# print("optimized and best model is successfully completed and now your models are stored to your disk")

y_pred_all_rand=[]
y_pred_all_18th= []
path = './model/'
rounds=10
for i in range(rounds):
    #model_train = hist_ocr_model()
    best_model, best_pred_model, best_history = hist_ocr_model(best_config, n_components=5, verbose=1)
    #model = model_train[0]
    best_pred_model.save(path + f'model_belened_CTC_deephyper_{i}.hdf5')
    y_pred = best_pred_model.predict(x_test_rand)
    y_pred_18 = best_pred_model.predict(x_test_18th)
    y_pred_all_rand.append(y_pred)
    y_pred_all_18th.append(y_pred_18)
    print(f'this is training belended-CTC_deephyper round_{i}')

y_decode_all_rand=[]
y_decode_all_18th=[]
for  j in range(rounds):
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


print("the CER of random test set_with blended_CTC_deephyper:")
print(CER_all_rand)

print("the CER of 18th century test set_with blended_CTC_deephyper:")
print(CER_all_18th)
