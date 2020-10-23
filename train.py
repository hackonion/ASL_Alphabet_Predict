from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout,Flatten, Dense, Activation
from tensorflow.keras.layers import Convolution2D , MaxPool2D
from tensorflow.keras import backend as k
from tensorflow.keras import metrics
import matplotlib.pyplot as plt
k.clear_session()

data_training = '<Path of data for training>'

epochs = 5
size = (64,64)
inp_shape = (64,64,3)
bach_size = 64 
steps = 1000
validation_steps = 200
filtrosConv1 = 32
filtrosConv2 = 64
filtrosConv3 = 128
size_filter1 =(3,3)
size_filter2 =(2,2)
size_pool =(2,2)
nums_class = 29
lr = 0.0009
val_split = 0.1

#preprocessing data
data_gen = ImageDataGenerator(
        rescale=1./255,
        samplewise_center=True,
        samplewise_std_normalization=True,
        validation_split=val_split,
        )

train_gen = data_gen.flow_from_directory(
    data_training,
    target_size=size,
    batch_size=bach_size,
    shuffle=True,
    class_mode='categorical',
    subset= "training"

)

val_gen = data_gen.flow_from_directory(
    data_training,
    target_size=size,
    batch_size=bach_size,
    shuffle=True,
    class_mode='categorical',
    subset= "validation"
)

#Model
cnn = Sequential()
cnn.add(Convolution2D(filtrosConv1,size_filter1,padding='same',input_shape =inp_shape,activation='relu'))
cnn.add(Convolution2D(filtrosConv1,size_filter1,padding='same',activation='relu'))
cnn.add(MaxPool2D(pool_size = size_pool))
cnn.add(Dropout(0.25))

cnn.add(Convolution2D(filtrosConv2,size_filter2,padding='same',activation='relu'))
cnn.add(Convolution2D(filtrosConv2,size_filter2,padding='same',activation='relu'))
cnn.add(MaxPool2D(pool_size = size_pool))
cnn.add(Dropout(0.25))

cnn.add(Convolution2D(filtrosConv3,size_filter2,padding='same',activation='relu'))
cnn.add(Convolution2D(filtrosConv3,size_filter2,padding='same',activation='relu'))
cnn.add(MaxPool2D(pool_size = size_pool))
cnn.add(Flatten())
cnn.add(Dense(256,activation='relu'))
cnn.add(Dropout(0.5))

cnn.add(Dense(nums_class,activation='softmax'))
cnn.compile(loss='categorical_crossentropy',optimizer = optimizers.Adam(lr = lr),
            metrics = ['accuracy'])

history = cnn.fit_generator(train_gen,epochs=epochs,validation_data=val_gen, use_multiprocessing=True)



#Visualization of results
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()