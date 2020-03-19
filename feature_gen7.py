# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 11:59:53 2020

@author: cenic
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np 
import PIL.Image as Image
import glob
import random
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#-------------------------------------------
#block 0. read and process images


def process_resized_img(imgpath):
    imagerun = Image.open(imgpath).convert('RGB').resize((224, 224))
    imagerun =  np.array(imagerun)/255.0
    return imagerun[np.newaxis, ...]




#-------------------------------------------
#block 1: calculate distances between images


def metric_calculator(feature_query, feature_catalog, image_x, image_y):
  #TODO: turn this into a delayed tf computation i.e. use merge layers
  #in order to turn it into a difference-of-layers 
  # diff = layers.diff(feature_query,feature_catalog)
  #NOTE: due to the large number of dimensions, unless we learn a metric
  #we run into the risk of the curse of dimensionality with kNN
  #NOTE: normalizing the feature vectors before computing the distance
  # seems to help, equivalent to 2*(1-cos_similarity).
  f_q = feature_query.predict(image_x)
  f_q = f_q/np.linalg.norm(f_q)
  f_c = feature_catalog.predict(image_y)
  f_c =f_c/np.linalg.norm(f_c)
   
  return np.linalg.norm(f_q-f_c)


def model_cosine(feature_query, feature_catalog):
  img_a = keras.Input(shape=(224, 224, 3))
  img_b = keras.Input(shape=(224, 224, 3))
  f_a = feature_query(img_a)
  f_b = feature_generator_catalog(img_b)
  outsim = layers.dot([f_a, f_b], axes=-1, normalize=True)
  return tf.keras.models.Model(inputs=[img_a, img_b],outputs=[outsim])





#-------------------------------------------
#block 2: test reanking precision


def ranking_precision_score(img_q, img_p, img_n, dist_fun):
    #function marks a success (1) if positive image is closer to query than 
    #negative image
    if dist_fun(img_q, img_n)> dist_fun(img_q, img_p):
        return 1
    else:
        return 0

def total_folder_precision(path, dist_fun):
    list_subfolders_with_paths = [f.path for f in os.scandir(path) if f.is_dir()]
    total_count = 0
    for subfldr in list_subfolders_with_paths:
        print(subfldr)
        imgpaths=glob.glob(os.path.join(subfldr, '*.jpg'))
        imgpath_q = next((s for s in imgpaths if "q.jpg" in s), None)
        img_q = process_resized_img(imgpath_q )
        imgpath_p = next((s for s in imgpaths if "p.jpg" in s), None)
        img_p = process_resized_img(imgpath_p)
        imgpath_n = next((s for s in imgpaths if "n.jpg" in s), None)
        img_n = process_resized_img(imgpath_n)
        print(ranking_precision_score(img_q, img_p, img_n, dist_fun))
        total_count+= ranking_precision_score(img_q, img_p, img_n, dist_fun)
    return total_count/len(list_subfolders_with_paths)
 




#-------------------------------------------
#block 3: create data generation flow to create a set of images and
#a label  if they are "close"


def gen_train_test_folders(path):
    # function returns two lists with the paths for the train and test set
    list_subfolders_with_paths = [f.path for f in os.scandir(path) if f.is_dir()]
    n = len(list_subfolders_with_paths)
    random.shuffle(list_subfolders_with_paths)
    return list_subfolders_with_paths[:3*n//4], list_subfolders_with_paths[3*n//4:]

def gen_dataframe_for_flow(folders):
    # create dataframes to be used in generate_generator_multiple
    recordvals=[]
    for subfldr in folders:
        imgpaths = glob.glob(os.path.join(subfldr, '*.jpg'))
        imgpath_q = next((s for s in imgpaths if "q.jpg" in s), None)
        imgpath_p = next((s for s in imgpaths if "p.jpg" in s), None)
        imgpath_n = next((s for s in imgpaths if "n.jpg" in s), None)
        #positive example
        recordvals.append([imgpath_q, imgpath_p, "1"])
        #negative example
        recordvals.append([imgpath_q, imgpath_n, "0"])
    return pd.DataFrame.from_records(recordvals, columns=["x_q_col", "q_t_col", "label"])
train, test = gen_train_test_folders("./testprecision/Imagenes")


def generate_generator_multiple(generator,df, batch_size, img_height,img_width):
    #custom generator to flow the two images and their label from the df dataframe
    generaX1 = generator.flow_from_dataframe(df,
                                             x_col="x_q_col",
                                             y_col="label",
                                             target_size = (img_height,img_width),
                                             class_mode = 'categorical',
                                             batch_size = batch_size,
                                             shuffle=False, 
                                             seed=7)

    
    generaX2 = generator.flow_from_dataframe(df,
                                             x_col="q_t_col",
                                             y_col="label",
                                             target_size = (img_height,img_width),
                                             class_mode = 'categorical',
                                             batch_size = batch_size,
                                             shuffle=False, 
                                             seed=7)
    while True:
            X1i = generaX1.next()
            X2i = generaX2.next()
            yield [X1i[0], X2i[0]], X2i[1]  #Yield both images and their mutual label





#-------------------------------------------
#TODO block 4: model for predicting similarity

def metric_shallow_model(feature_query, feature_catalog):
  img_a = keras.Input(shape=(224, 224, 3))
  img_b = keras.Input(shape=(224, 224, 3))
  f_a = feature_query(img_a)
  f_b = feature_generator_catalog(img_b)
  #TODO: test normalization before subtraction
  diff  = layers.subtract([f_a, f_b])
  dsq = layers.multiply([diff, diff])
  main_output = layers.Dense(1, activation='sigmoid', kernel_initializer = keras.initializers.Ones(), name='main_output')(dsq)
  return tf.keras.models.Model(inputs=[img_a, img_b],outputs=[main_output])





    





# currently, the feature extractors for the query image and the catalog set
# are the same convnet model.  
feature_generator_query = tf.keras.Sequential([tf.keras.models.load_model("/home/analytics/source/128392891238t"),
                                              layers.GlobalAveragePooling2D()])

print(feature_generator_query.summary())
feature_generator_query._name="fet_gen_query"

feature_generator_catalog = tf.keras.Sequential([tf.keras.models.load_model("/home/analytics/source/128392891238t"),
                                              layers.GlobalAveragePooling2D()])



#freeze weights, currently it doesn't make a difference
feature_generator_query.trainable = False
feature_generator_catalog.trainable = False

newmodel = model_cosine(feature_generator_query, feature_generator_catalog)

print(newmodel.summary())
#metric model
metric_model= metric_shallow_model(feature_generator_query, feature_generator_catalog)
#compile the metric model as a binary classification problem
metric_model.compile(optimizer='rmsprop',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
#create train, test folders 
train_folders, test_folders = gen_train_test_folders("./testprecision/Imagenes")
#train and test dataframes for the flow_from generator
df_train = gen_dataframe_for_flow(train_folders)
df_test = gen_dataframe_for_flow(test_folders)

input_imgen = ImageDataGenerator(rescale = 1./255)

test_imgen = ImageDataGenerator(rescale = 1./255)
# creating generator flows
BATCH_SIZE = 32
IMG_HEIGHT = 224

inputgenerator = generate_generator_multiple(generator=input_imgen,
                                           df=df_train,
                                           batch_size=BATCH_SIZE,
                                           img_height=IMG_HEIGHT,
                                           img_width=IMG_HEIGHT)       
     
testgenerator = generate_generator_multiple(generator=test_imgen,
                                          df=df_test,
                                          batch_size=BATCH_SIZE,
                                          img_height=IMG_HEIGHT,
                                          img_width=IMG_HEIGHT)





#TODO: fitting model from inputgenerator
#print("now fitting")
#history=metric_model.fit_generator(inputgenerator,
#                        steps_per_epoch=len(df_train)/BATCH_SIZE,
#                        epochs = 8,
#                        validation_data = testgenerator,
#                        validation_steps = len(df_test)/BATCH_SIZE,
#                        use_multiprocessing=False,
#                        shuffle=False)



imagerun = Image.open("/home/analytics/source/imgfolder/2092.jpg").resize((224, 224))
imagerun =  np.array(imagerun)/255.0
jjj = newmodel.predict([imagerun[np.newaxis, ...], imagerun[np.newaxis, ...]])
print(jjj)


#sanity check: should be 0.
metval = metric_calculator(feature_generator_query, feature_generator_catalog,imagerun[np.newaxis, ...], imagerun[np.newaxis, ...])
print(metval)
#find closest
imagerun_i = Image.open('away.jpg').resize((224, 224))
imagerun_i =  np.array(imagerun_i)/255.0

#row =[]
#searchlist = list(glob.iglob('/home/analytics/source/imgfolder/*.jpg'))
#for filepath in searchlist:
#  imagerun_j = Image.open(filepath).resize((224, 224))
#  imagerun_j =  np.array(imagerun_j)/255.0
#  metval = metric_calculator(feature_generator_query, feature_generator_catalog,imagerun_i[np.newaxis, ...], imagerun_j[np.newaxis, ...])
#  row.append(metval)
#print(row)
#print(searchlist[row.index(min(row))])

#metric matrix
simmat=[]
print(list(glob.iglob('/home/analytics/source/imgfolder/*.jpg')))
#for filepath in glob.iglob('/home/analytics/source/imgfolder/*.jpg'):
#  row = []
#  for filepath2 in glob.iglob('/home/analytics/source/imgfolder/*.jpg'):
#    imagerun_i = Image.open(filepath).resize((224, 224))
#    imagerun_i =  np.array(imagerun_i)/255.0
#    imagerun_j = Image.open(filepath2).resize((224, 224))
#    imagerun_j =  np.array(imagerun_j)/255.0
#    metval = metric_calculator(feature_generator_query, feature_generator_catalog,imagerun_i[np.newaxis, ...], imagerun_j[np.newaxis, ...])
#    row.append(metval)
#  simmat.append(row)
#print(simmat)
print("now testing for precision")
disti = lambda imga, imgb : metric_calculator(feature_generator_query, feature_generator_catalog, imga, imgb)
print(total_folder_precision("./testprecision/Imagenes", disti))
