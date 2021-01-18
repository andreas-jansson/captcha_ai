import sys
import tensorflow as tf
from PIL import Image
from tensorflow import keras
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import PIL
import PIL.Image
import glob
#import tensorflow_datasets as tfds




########## import dataset ##############


# *.jpg wildcard everything
dataset_url = glob.glob('./dataset/*.jpg')
i = 0
train_data = []

#read each file into list
for filename in dataset_url:
    image = Image.open(filename)
    imagearray = np.asarray(image)
    text = np.asarray(filename[10:-4])               #splice image name from filepath for labeling
    train_data.append((imagearray, text))           #(())tupile
    i = i + 1


#print(train_images)
#print(train_text[20])
#plt.imshow(train_images[20])
#plt.show()




######## splits dataset into training/testing ##########

#total nr of images 7329, 5863 for training and 1466 for testing

'''

#(train_images, train_lables) = np.array([item[0] for item in train_data]), np.array([item[1] for item in train_data])
#(test_images, test_lables) = np.array([item[0] for item in train_data]), np.array([item[1] for item in train_data])

'''

train_images = []
train_lables = []
test_images = []
test_lables = []


#uggly code, 80% into train, rest into test
train_size = 5863
total_size = 7329

#read data to train set
index = 0
for item in train_data:
    if index < train_size:
        train_images.append(item[0])
        train_lables.append(item[1])
    else:
        break
    index = index + 1

print('index: ', index)

#read the rest into test set
for item in train_data:
    if index < total_size:
        test_images.append(item[0])
        test_lables.append(item[1])
    else:
        break
    index = index + 1




#convert to numpy array
train_images = np.array(train_images)
train_lables = np.array(train_lables)
test_images = np.array(test_images)
test_lables = np.array(test_lables)

#normaliserar
train_images = train_images / 255.0
test_images = test_images / 255.0


print('train:', len(train_images), len(train_lables), type(train_images))
print('test', len(test_images), len(test_lables), type(test_images))


# shape 90x282 3 colours (90, 282, 3) ()
print(train_images[30].shape, train_lables[30])





############# Model ##################

model = keras.Sequential([
    keras.layers.Dense(64, input_shape=(90, 282, 3)),
    keras.layers.Conv2D(64,(3, 3), activation='relu'),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.Dense(64),
    keras.layers.Flatten(),
    keras.layers.Dense(36, activation=tf.nn.softmax)
])



############# optimize #################

#model.summary()

# optimerar modellen
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

#Tränar modellen 50 ggr
model.fit(train_images, train_lables, epochs=5, batch_size = 100)


############## Test ####################

#testar modellen med testdata
test_loss, test_acc = model.evaluate(test_images, test_lables)
print(f'Test Accuracy: {test_acc}')






'''
listoftuples = [('hejsan', 'hoppsan'), ('ciao', 'ciao'), ('detvar', 'engång')]

#firstoftuples = []
#firstoftuples = [firstoftuples[0] for firstoftuples in listoftuples]
#print(firstoftuples)
array = []

index = 0
for firstoftuples in listoftuples:
    if index <2:
        #print('firstoftuples: ', firstoftuples[0])
        array.append(firstoftuples[0])
    else:
        break
    index = index +1

print(array)
'''