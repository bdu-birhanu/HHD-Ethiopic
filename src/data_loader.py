import cv2
from sklearn.model_selection import train_test_split
import numpy as np
num_class=307
maxlen=46
im_row=48
im_col=368


def load_dataset():
    '''

    '''
    x_train_real = np.load('./train/train_numpy/x_train.npy' ,allow_pickle=True)
 
    y_train_padded= np.load('./train/train_numpy/y_train.npy', allow_pickle=True)
    

    #x_val=np.load('./hist_all_resize/x_val_all.npy', allow_pickle=True)
    #y_val = np.load('./hist_all_resize/y_val_all.npy', allow_pickle=True)



    return x_train_real, y_train_padded #, x_val,y_val
    #return x_train_real_resize,  y_train_real_resize
#the following two functions are employed for test sets and trainsets separetly just for simplcity

def resize():
    data_loaded=load_dataset()
    x_train_real=data_loaded[0]
    y_train_real = data_loaded[1]

    resized_x_rand = np.zeros((len(x_train_real), im_row, im_col), dtype=np.uint8)

    # loop over the input images
    for i, image in enumerate(x_train_real):
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

    return resized_x_rand, y_train_real

def preprocess_train_val_test_data():
    ''' 
    input: a 2D shape text-line image (h,w)
    output:  returns 3D shape image format (h,w,1)

    Plus this function randomly splits the training and validation set
    This function also computes list of length for both training and validation images and GT
      '''
    #training
    x_tr=resize()
    x_train_real_resize=x_tr[0]
    y_train_real = x_tr[1]
  
    #x_val=x_tr[4]
   # y_val=x_tr[5]





    x_train, x_val, y_train, y_val = train_test_split(x_train_real_resize, y_train_real, test_size=0.1)


    # reshape the size of the image from 3D to 4D so as to make the input size is similar with it.
    x_train_r = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)  # [samplesize,32,128,1]
    x_val_r = x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], 1)
    y_train = y_train
    y_val = y_val

    nb_train = len(x_train_r)
    nb_val = len(x_val_r)


    x_train_len = np.array([len(x_train_r[i]) + 43 for i in range(nb_train)])
    # the + 43 here is just to make the balance size ((2*46-1=91) then 48+43=91)of the image equal to the input of LSTMlayer
    x_val_len = np.array([len(x_val_r[i]) + 43 for i in range(nb_val)])  
    
    y_train_len = np.array([len(y_train[i]) for i in range(nb_train)])
    y_val_len = np.array([len(y_val[i]) for i in range(nb_val)])


    return x_train_r, y_train, x_train_len, y_train_len, x_val, y_val, x_val_len, y_val_len
'''
all set of text images and GT
'''
train=preprocess_train_val_test_data()
x_train=train[0]
y_train=train[1]
x_train_length=train[2]
y_train_length=train[3]

x_val=train[4]
y_val=train[5]
x_val_length=train[6]
y_val_length=train[7]


print("data_loading is compeletd")
print("===============================")
print(str(len(x_train))+ " train image and "+ str(len(y_train))+" labels")
print(str(len(x_val))+ "valid image and "+ str(len(y_val))+ "labels")

