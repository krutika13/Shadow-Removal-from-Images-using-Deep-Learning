import os
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
# from google.colab.patches import cv2_imshow
from glob import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
from sklearn.model_selection import train_test_split
# from model import build_unet
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from model_archi import build_unet
import tensorflow as tf
import pickle
import tensorflow.keras.backend as K
from skimage.metrics import structural_similarity as ssim
from tensorflow.keras.models import load_model
import pickle
#




def load_data(path, split=0.1):
    W = 256
    H = 256
    images = sorted(glob(os.path.join(path, "train_A", "*.png")))
    #mask = sorted(glob(os.path.join(path, "train_B", "*.png")))
    sf1 = sorted(glob(os.path.join(path, "train_C", "*.png")))
    a = []
    b=[]
    for i in range(len(images)):
        x = cv2.imread(images[i], cv2.IMREAD_UNCHANGED)
        x = cv2.resize(x, (W, H))
        x = x/255.0

        # shadow_mask = cv2.imread(mask[i], cv2.IMREAD_GRAYSCALE)
        # shadow_mask = cv2.resize(shadow_mask, (x.shape[1], x.shape[0]))

        # train_x1 = np.dstack((x, shadow_mask))
        # a.append(train_x1)
        a.append(x)
        x1 = cv2.imread(sf1[i], cv2.IMREAD_UNCHANGED)
        x1 = cv2.resize(x1, (W, H))
        x1 = x1/255.0
        b.append(x1)
        
    #x_train_tensor = tf.convert_to_tensor(a)
    #sf_train = tf.convert_to_tensor(sf1)
    split_size = split
    train_x=a
    train_y1=b
    path1= r"C:\Users\Ahlad Kumar\Desktop\ISTD_Dataset\test"
    test_x = sorted(glob(os.path.join(path1, "test_A", "*.png")))
    test_y1 = sorted(glob(os.path.join(path1, "test_C", "*.png")))
    # Split the data into train, validation, and test sets
     # Split the data into train, validation, and test sets
    #train_x, test_x, train_y1, test_y1 = train_test_split(a, b, test_size=split_size, random_state=42)
    train_x, valid_x, train_y1, valid_y1 = train_test_split(train_x, train_y1, test_size=split_size, random_state=42)

    return tf.convert_to_tensor(train_x), tf.convert_to_tensor(valid_x), tf.convert_to_tensor(test_x), tf.convert_to_tensor(train_y1), tf.convert_to_tensor(valid_y1), tf.convert_to_tensor(test_y1)
# dataset_path = r"C:\Users\Ahlad Kumar\Desktop\ISTD_Dataset\train"
# train_x, valid_x, test_x, train_y1, valid_y1, test_y1 = load_data(dataset_path)

# path=r"\content\drive\MyDrive\Thesis_new\ISTD_Dataset\train"
# load_data(path)

def create_dir(path):
    """ Create a directory. """
    if not os.path.exists(path):
        os.makedirs(path)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try: 
       
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
       
    except RuntimeError as e:
        print(e)
if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("files")

    """ Hyperparameters """
    # batch_size = 2
    # lr = 1e-5
    # num_epochs = 20
    model_path = os.path.join(r"D:\Krutika\Cascade_Unet", "model_cascade_highParam.h5")
    

    """ Dataset """
    dataset_path = r"C:\Users\Ahlad Kumar\Desktop\ISTD_Dataset\train"
    # test_x="/content/drive/MyDrive/Thesis_new/ISTD_Dataset/test/test_A"
    # test_y="/content/drive/MyDrive/Thesis_new/ISTD_Dataset/test/test_C"
                                           
    #train_x, valid_x, test_x, train_y1, valid_y1, test_y1 = load_data(dataset_path)
    # with open("D:/Krutika/Unet/pickles_files/ISTD_AsItIs/train_x", "wb") as fp:   #Pickling
    #  pickle.dump(train_x, fp)
    # with open("D:/Krutika/Unet/pickles_files/ISTD_AsItIs/valid_x", "wb") as fp1:   #Pickling
    #  pickle.dump(valid_x, fp1)
    
    # with open("D:/Krutika/Unet/pickles_files/ISTD_AsItIs/train_y1", "wb") as fp3:   #Pickling
    #  pickle.dump( train_y1, fp3)
    # with open("D:/Krutika/Unet/pickles_files/ISTD_AsItIs/valid_y1", "wb") as fp4:   #Pickling
    #  pickle.dump(valid_y1, fp4)  
    pickle_file0 = open("D:/Krutika/Unet/pickles_files/ISTD_AsItIs/train_x", 'rb')
    pickle_file10 = open("D:/Krutika/Unet/pickles_files/ISTD_AsItIs/train_y1", 'rb')  
    train_x = pickle.load(pickle_file0)
    train_y1 = pickle.load(pickle_file10)
    pickle_file3 = open("D:/Krutika/Unet/pickles_files/ISTD_AsItIs/valid_x", 'rb')
    pickle_file4 = open("D:/Krutika/Unet/pickles_files/ISTD_AsItIs/valid_y1", 'rb')

    valid_x = pickle.load(pickle_file3)
    valid_y1 = pickle.load(pickle_file4)
    pickle_file = open("D:/Krutika/Unet/pickles_files/ISTD_AsItIs/test_x", 'rb')
    pickle_file1 = open("D:/Krutika/Unet/pickles_files/ISTD_AsItIs/test_y1", 'rb')

    test_x = pickle.load(pickle_file)
    test_y1 = pickle.load(pickle_file1)
   # print("tarin_y1",train_y1)
    print(train_y1.shape)
    print(valid_y1.shape)
    print(test_y1.shape)
    print(f"Train: {len(train_x)} - {len(train_y1)} ")
    print(f"Valid: {len(valid_x)} - {len(valid_y1)} ")
    print(f"Test: {len(test_x)} - {len(test_y1)} ")

    # train_dataset = tf_dataset(train_x, train_y1, batch=batch_size)
    # valid_dataset = tf_dataset(valid_x, valid_y1, batch=batch_size)

    """ Model """
#     model = build_unet((256, 256, 3))
#    # metrics = [dice_coef, iou, Recall(), Precision()]
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mean_squared_error'])
    #model_path1 = os.path.join(r"D:\Krutika\Cascade_Unet", "model_cascade1.h5")
    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.0001, verbose=1)
        ]
#    # model.fit(train_x, train_y1, epochs=100, batch_size=32, validation_data=(valid_x, valid_y1),callbacks=callbacks)
    # def ssim_metric(y_true, y_pred):
    #   return K.mean(tf.py_function(ssim, [y_true, y_pred],tf.float32))

    def mse_metric(y_true, y_pred):
       return K.mean(K.square(y_pred - y_true))


#     model1 =build_unet((256, 256, 3))
#     model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mean_squared_error'])

# # train the first U-Net model
#     model1.fit(train_x, train_y1, epochs=30, batch_size=32,validation_data=(valid_x, valid_y1),callbacks=callbacks)
    model1 = tf.keras.models.load_model(r"D:\Krutika\Cascade_Unet\model_cascade.h5")
    # pickle_file = open("D:/Krutika/Unet/pickles_files/ISTD_AsItIs/test_x", 'rb')
    # pickle_file1 = open("D:/Krutika/Unet/pickles_files/ISTD_AsItIs/test_y1", 'rb')

    # test_x = pickle.load(pickle_file)
    # test_y1 = pickle.load(pickle_file1)
    # pickle_file3 = open("D:/Krutika/Unet/pickles_files/ISTD_AsItIs/valid_x", 'rb')
    # pickle_file4 = open("D:/Krutika/Unet/pickles_files/ISTD_AsItIs/valid_y1", 'rb')

    # valid_x = pickle.load(pickle_file3)
    # valid_y1 = pickle.load(pickle_file4)
    res=[]
    res1=[]
    for i in range(len(test_x)) :
# use the first U-Net model to predict the shadow-free image
      img=test_x[i]
        
      img=img.numpy().astype(np.float64)
      x = np.expand_dims(img, axis=0)
      predicted_y1 = model1.predict(x)
      #predicted_y11=np.reshape(predicted_y1, (128,128, 3))
      predicted_y1=np.reshape(predicted_y1, (256, 256, 3))
      res.append(predicted_y1)
      #test_y1=np.reshape(test_y1[i], (256, 256, 3))
      
      cv2.imwrite(r"D:/Krutika/Cascade_Unet/Results/new"+f'image_sdfree{i}.png',predicted_y1*255)
      squared_diff = np.square(test_y1[i] -predicted_y1)
      mean_squared_diff = np.mean(squared_diff)
      rmse = np.sqrt(mean_squared_diff)
      i+=1 
      print("mean_squared_diff",rmse)
      #res.append(predicted_y11)
      #test_y11=np.reshape(test_y1[i], (128,128, 3))
      res1.append(test_y1)
# concatenate the predicted shadow-free image and the original shadow-free image
    res_arr = np.array(res)
    print(res_arr.shape)
   
    test_y1_arr = np.array(test_y1)
    # print(test_y1_arr)
    # print(test_y1_arr.shape)
    input_x = np.concatenate((res_arr, test_y1_arr), axis=-1)
    
    # model_path1 = os.path.join(r"D:\Krutika\Cascade_Unet", "model_cascade1_highParam.h5")
    # # callbacks1 = [
    # #     ModelCheckpoint(model_path1, verbose=1, save_best_only=True),
    # #     ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.0001, verbose=1)
    # #     ]
    # callbacks1 = [
    #     ModelCheckpoint(model_path1, verbose=1, save_best_only=True)
    #    ]
# define the second U-Net model
    # model2 = build_unet((256, 256, 6))
    # model2.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy',  metrics=['mean_squared_error'])
    
        # train the second U-Net model
    # model2.fit(input_x, test_y1, epochs=30, batch_size=32)
    # model2.save('D:\Krutika\Cascade_Unet\model_cascade1.h5')
    model2=load_model('D:\Krutika\Cascade_Unet\model_cascade1.h5')
    for i in range(len(test_x)) :
      img1=input_x[i]
        
      img1=img1.astype(np.float64)
      x = np.expand_dims(img1, axis=0)
      #predicted_y1 = model1.predict(x)
# use the first U-Net model to predict the shadow-free image
      predicted_y2 = model2.predict(x)
      predicted_y2=np.reshape(predicted_y2, (256, 256, 3))
      #cv2.imwrite(r"D:/Krutika/Cascade_Unet/Results/gt/"+f'gt{i}.png', test_y1_arr[i]*255)
     # cv2.imwrite(r"D:/Krutika/Cascade_Unet/Results/new/sec_unet/"+f'sf{i}.png',predicted_y2*255)
      test_x_array=np.array(test_x)
      cv2.imwrite(r"D:/Krutika/Cascade_Unet/Results/orignal/"+f'Ori{i}.png', test_x_array[i]*255)
      diff = test_y1_arr[i].flatten() - predicted_y2.flatten()
      difff = diff*255
      #print(difff)
      diffff= np.sqrt(difff*difff)
      #print(diffff)
      su=np.sum(diffff)
        # print(len(difff))
      su = su/len(difff)
      print(su)
      
          # use the second U-Net model to refine the predicted shadow-free image
    #   squared_diff = np.square(res[i]*255 -predicted_y2*255)
    #   mean_squared_diff = np.mean(squared_diff)
    #   rmse = np.sqrt(mean_squared_diff)
    #   i+=1 
    #   print("mean_squared_diff1",rmse)
