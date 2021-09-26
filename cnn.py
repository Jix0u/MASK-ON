import numpy as np
from numpy import std
from matplotlib import pyplot
from sklearn.model_selection import KFold
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import keras.backend as k
import mediapipe
import cv2

import datetime
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
# Function for transforming PIL Image to Numpy Array
from tensorflow.keras.preprocessing.image import img_to_array
# Preprocessing Function for Input Image before passing into MobileNet Model (Mask Detector Model)
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# # load train and test dataset
# def load_dataset():
#     # load dataset
#     (tx, ty)= mnist.load_data()
#     trainX=tx[:55000]
#     # one hot encode target values
#     return trainX
 
# # scale pixels
# def prep_pixels(train, test):
#     # convert from integers to floats
#     train_norm = train.astype('float32')
#     test_norm = test.astype('float32')
#     # normalize to range 0-1
#     train_norm = train_norm / 255.0
#     test_norm = test_norm / 255.0
#     # return normalized images
#     return train_norm, test_norm
 
# define cnn model
# def define_model():
    

 
# evaluate a model using k-fold cross-validation
# def evaluate_model(dataX, dataY, n_folds=5):
    # scores, histories = list(), list()
    # # prepare cross validation
    # kfold = KFold(n_folds, shuffle=True, random_state=1)
    # # enumerate splits
    # for train_ix, test_ix in kfold.split(dataX):
    #     # define model
    #     model = define_model()
    #     # select rows for train and test
    #     trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
    #     # fit model
    #     start = time.time()
    #     history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
    #     end = time.time()-start
    #     print("training time: ",end)
    #     # evaluate model
    #     _, acc = model.evaluate(testX, testY, verbose=0)
    #     print('> %.4f' % (acc))
    #     # stores scores
    #     scores.append(acc)
    #     histories.append(history)
    # return scores, histories
    # model = define_model()
    # trainX, trainY = dataX[train_ix], dataY[train_ix]
 
 
# run the test harness for evaluating a model
# def run_test_harness():
    # load dataset
    # trainX, trainY, testX, testY = load_dataset()
    # prepare pixel data
    # trainX, testX = prep_pixels(trainX, testX)
    # evaluate model
    # scores, histories = evaluate_model(trainX, trainY)
    


    
# entry point, run the test harness
# run_test_harness()
# model = Sequential()
# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
# model.add(MaxPooling2D())
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(MaxPooling2D())
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(MaxPooling2D())
# model.add(Flatten())
# model.add(Dense(100, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# # compile model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# from keras.preprocessing.image import ImageDataGenerator
# train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True)

# test_datagen = ImageDataGenerator(rescale=1./255)

# training_set = train_datagen.flow_from_directory(
#         'train',
#         target_size=(150,150),
#         batch_size=16 ,
#         class_mode='binary')

# test_set = test_datagen.flow_from_directory(
#         'test',
#         target_size=(150,150),
#         batch_size=16,
#         class_mode='binary')

# model_saved=model.fit_generator(
#         training_set,
#         epochs=10,
#         validation_data=test_set,
#     )
# model.save('theneverbeforeseenbrownscreen.h5',model_saved)
# mymodel=load_model('theneverbeforeseenbrownscreen.h5')

# test_image=image.load_img("/Users/leonachen/Downloads/flappy-mediapipe-main/test/with_mask/1-with-mask.jpg",
#                           target_size=(150,150,3))
# test_image=image.img_to_array(test_image)
# test_image=np.expand_dims(test_image,axis=0)
# mymodel.predict(test_image)[0][0]

mask_detector = load_model("mask_detector.model")

# Mediapipe's LightWeight Face Detection Model
face_detector = mediapipe.solutions.face_detection.FaceDetection()
# try:
capture = cv2.VideoCapture(0)

while True:
    success,img=capture.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Pass RGB frame to Face Detector Model and get multiple detections of faces if exists
    results = face_detector.process(imgRGB)
    # if not success:
    #     print("Ignoring empty camera frame.")
    #     continue
    if results.detections:
    # Looping over each detection in RGB Frame
        for detection in results.detections:

            # Get relative bounding box of that detection
            boxR = detection.location_data.relative_bounding_box
            ih, iw, _ = img.shape
            

            # Extracting the face from the RGB Frame to pass into Mask Detection Model
            (startX, startY, endX, endY) = (boxR.xmin, boxR.ymin, boxR.width, boxR.height) * np.array([iw, ih, iw, ih])
            startX = max(0, int(startX))
            startY = max(0, int(startY))
            endX = min(iw - 1, int(startX + endX))
            endY = min(ih - 1, int(startY + endY))

            # Extracting the face from the RGB Frame to pass into Mask Detection Model
            face = imgRGB[startY:endY, startX:endX]
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.array([face], dtype='float32')

            # Predicting Mask or No Mask on the extracted RGB Face
            preds = mask_detector.predict(face, batch_size=32)[0][0]
            if preds < 0.5:
                label = "Mask"
                color = (0, 255, 0)
                cv2.putText(img, label, (startX, startY - 10), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
                cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)
            else:
                label = "No Mask"
                color = (0, 0, 255)
                cv2.putText(img, label, (startX, startY - 10), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
                cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)

            # Drawing Bounding Box and Putting Text on the BGR frame
        
            
    cv2.imshow('img',img)
    cv2.waitKey(1)
# except:
#     cap.release()


 




