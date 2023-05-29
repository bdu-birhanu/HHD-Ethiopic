#from keras.models import load_model
from keras import backend as K
#import matplotlib.pyplot as plt
import editdistance
import tensorflow as tf
import numpy as np

'''
from data_loader import x_test_rand, y_test_rand, x_test_18th,y_test_18th


The following program returns the CER of printed text-line image onl,y and then
 you can follow the same steps for the synthetic images
'''
x_test_rand_var = np.load('./new_hhd/test/test_rand/test_rand_numpy/x_test_rand.npy', allow_pickle=True)
x_test_18th_var = np.load('./new_hhd/test/test_18th/test_18th_numpy/x_test_18th.npy', allow_pickle=True)
y_test_rand = np.load('./new_hhd/test/test_rand/test_rand_numpy/y_test_rand.npy', allow_pickle=True)
y_test_18th = np.load('./new_hhd/test/test_18th/test_18th_numpy/y_test_18th.npy', allow_pickle=True)
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

model= tf.keras.models.load_model(
    './model/model_CTC_0.hdf5',
    custom_objects={'Functional':tf.keras.models.Model})

y_pred=model.predict(x_test_rand)

#the CTC decoer
y_decode = K.get_value(K.ctc_decode(y_pred[:, :, :], input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1])[0][0])

for i in range(5):
    # print the first 10 predictions
    print("Prediction :", [j for j in y_decode[i] if j!=-1], " -- Label : ", y_test_rand[i])

#=========== compute editdistance and returne CER ====================================

true=[]# to stor value of character by removing zero which was padded previously and also this is the value of newline in the test label
for i in range(len(y_test_rand)):
    x=[j for j in y_test_rand[i] if j!=0]
    true.append(x)

pred=[]
# to stor the predicted charcter except zerro and -1 which are padded value nad blank space predicted during testing
for i in range(len(y_decode)):
    x=[j for j in y_decode[i] if j not in(0,-1)]
    pred.append(x)

cer=0
for(i,j) in zip(true,pred):
    x=editdistance.eval(i,j)
    cer=cer+x
err=cer
x=0
for i in range(len(true)):
    x=x+len(true[i])
totalchar=x
cerp=(float(err)/totalchar)*100
print("the CER of random test set:")
print(cerp)
'''
his= np.load('./model/tr1_history12_25.npy', allow_pickle=True)
print("history of the trained model")
#print(his)
#to visualiza sample text-line image( the image at index 1
# x_testp_orig=x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2])
# fig, ax = plt.subplots()
# i=ax.imshow(x_testp_orig[1],cmap='Greys_r')# todisply the backgrouund to be white
# plt.show()
'''
y_pred_18th=model.predict(x_test_18th)

#the CTC decoer
y_decode_18th = K.get_value(K.ctc_decode(y_pred_18th[:, :, :], input_length=np.ones(y_pred_18th.shape[0]) * y_pred_18th.shape[1])[0][0])

for i in range(3):
    # print the first 10 predictions
    print("Prediction :", [j for j in y_decode_18th[i] if j!=-1], " -- Label : ", y_test_18th[i])

#=========== compute editdistance and returne CER ====================================

true_18=[]# to stor value of character by removing zero which was padded previously and also this is the value of newline in the test label
for i in range(len(y_test_18th)):
    x=[j for j in y_test_18th[i] if j!=0]
    true_18.append(x)

pred_18=[]
# to stor the pdicted charcter except zerro and -1 which are padded value nad blank space predicted during testing
for i in range(len(y_decode_18th)):
    x=[j for j in y_decode_18th[i] if j not in(0,-1)]
    pred_18.append(x)

cer_18=0
for(i,j) in zip(true_18,pred_18):
    x=editdistance.eval(i,j)
    cer_18=cer_18+x

err_18=cer_18

x_18=0
for i in range(len(true_18)):
    x_18=x_18+len(true_18[i])

totalchar_18=x_18

cerp_18=(float(err_18)/totalchar_18)*100
print("The CER of 18th century book is:")
print(cerp_18)
