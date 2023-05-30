#new deephyper to select the best hyper-parameter with 10% of my test data
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Permute, Flatten, Masking, \
    GaussianNoise, Reshape, Lambda, TimeDistributed, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GRU, LSTM, Bidirectional, MaxPool2D
import tensorflow as tf
from tensorflow.keras.layers import RepeatVector, Dot, Bidirectional,LSTM,Input, Dense, Activation,Lambda,Concatenate
from sklearn.model_selection import train_test_split

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
import numpy as np
import cv2
from data_loader_hhd import num_class, maxlen,im_col, im_row
from data_loader_hhd  import x_train, y_train, x_val, y_val, x_val_length, y_val_length

#let't take randomly select(10%) validation data as a training se--> just for hyperparameter selection
x_train_new = x_val
y_train_new = y_val
x_train_len_new = x_val_length
y_train_len_new = y_val_length

#split the training data to 90:10 and take new 10% (x_valn and y_valn) as a validation data for hyperparameter selection
x_trainn, x_valn, y_trainn, y_valn = train_test_split(x_train, y_train, test_size=0.1)

# the new validation dataset
x_val_new = xvaln
y_val_new = y_valn
nb_val = len(xvaln)
x_val_length_new = np.array([len(xvaln[i]) for i in range(nb_val)]) 
y_val_length_new = np.array([len(y_valn[i]) for i in range(nb_val)])


dot_pro = Dot(axes = 1)
concatenat = Concatenate(axis=-1)



def ctc_lambda_func(args):
    '''

    '''
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def hist_ocr_model(config: dict, n_components: int = 5, verbose: bool = 0, num_class=310, img_row=32, img_col=200):
    tf.keras.utils.set_random_seed(3)

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
    if you use the full dataset you could increase the batch_size and epo
    '''

    inputs_data = Input(shape=(img_row, img_col, 1))
    conv_1 = Conv2D(default_config["feature_map_1"], (default_config["kernel"], default_config["kernel"]),
                    activation=default_config["activation"], padding='same')(inputs_data)
    # poolig layer with kernel size (2,2)
    pool_1 = MaxPool2D(pool_size=(2, 1), strides=2)(conv_1)
    conv_2 = Conv2D(default_config["feature_map_1"], (default_config["kernel"], default_config["kernel"]),
                    activation=default_config["activation"], padding='same')(pool_1)
    pool_2 = MaxPool2D(pool_size=(2, 1))(conv_2)  # we remove the strides here
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
    conv_7=batch_norm_6
    #conv_7 = Conv2D(default_config["feature_map_1"], (default_config["kernel"], default_config["kernel"]),
                    #activation=default_config["activation"])(batch_norm_6)

    # conv2=(int(conv1[2]),int(conv1[1]*int(conv1[3])))
    r = Reshape((int(conv_7.shape[2]), int(conv_7.shape[1]) * int(conv_7.shape[3])))(conv_7)
    # [ sample, timesteps, features]
    # bidirectional LSTM layers with units=128
    blstm_1 = Bidirectional(LSTM(default_config["rnn_size"], return_sequences=True, dropout=0.25))(r)
    blstm_2 = Bidirectional(LSTM(default_config["rnn_size"], return_sequences=True, dropout=0.25))(blstm_1)
    outputs = Dense(num_class + 1, activation='softmax')(blstm_2)

    pred_model = Model(inputs_data, outputs)

    labels = Input(name='the_labels', shape=[46], dtype='float32')  # 46 is the max size of GT text length
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([outputs, labels, input_length, label_length])

    model = Model(inputs=[inputs_data, labels, input_length, label_length], outputs=loss_out)
    # lrate = 0.01
    # decay = lrate / epoch
    # sgd = SGD(learning_rate=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')

    hist = model.fit(x=[x_train_new, y_train_new, x_train_len_new, y_train_len_new], y=np.zeros(len(y_train_new)),
                     batch_size=default_config["batch_size"], epochs=default_config["epoch"],
                     validation_data=([x_val_new, y_val_new, x_val_length_new, y_val_length_new], [np.zeros(len(y_val_new))]),
                     verbose=1) #verbos zero is not tol see the progress bar

    return model, pred_model, hist

_, model,  history = hist_ocr_model(config={}, n_components=5, verbose=1)

from deephyper.problem import HpProblem

problem = HpProblem()
problem.add_hyperparameter((10, 256), "rnn_size", default_value=128)
problem.add_hyperparameter((16, 512), "feature_map_1", default_value=64)
problem.add_hyperparameter((16, 512), "feature_map_2", default_value=128)
problem.add_hyperparameter(["sigmoid", "tanh", "relu"], "activation", default_value="relu")
#problem.add_hyperparameter((16, 64), "batch_size", default_value=32)
problem.add_hyperparameter((16, 128), "batch_size", default_value=32)
#problem.add_hyperparameter((5, 25), "epoch", default_value=10)
problem.add_hyperparameter((10, 25), "epoch", default_value=10)
problem.add_hyperparameter((2, 3), "kernel", default_value=3)
problem.add_hyperparameter((0.0, 0.5), "dropout", default_value=0.25)
problem


def run(config):
    # important to avoid memory exploision
    tf.keras.backend.clear_session()

    _, _, hist = hist_ocr_model(config, n_components=5, verbose=0)

    return -hist.history["val_loss"][-1]


from deephyper.search.hps import CBO

search = CBO(problem, run, initial_points=[problem.default_configuration], log_dir="cbo-results", random_state=2)
results = search.search(max_evals=10)

i_max = results.objective.argmax()
best_config = results.iloc[i_max][:-4].to_dict()
best_config

#best_model,best_pred_model, best_history = hist_ocr_model(best_config, n_components=5, verbose=1)
#f = open('./model/best_hyp_615k_attention.txt',"w")
f = open('./model/best_hyp_615k_attention.txt',"w")
f.write( str(best_config) )
f.close()

print("optomization is successfully completed and now your best hyperparameters are stored to your disk")
