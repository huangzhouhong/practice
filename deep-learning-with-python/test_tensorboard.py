from keras.datasets import mnist
import numpy as np
from keras.utils import to_categorical
from keras import *

((train_data,train_labels),(test_data,test_labels))=mnist.load_data()
train_data=train_data.reshape((train_data.shape[0],28,28,1))
train_data=train_data.astype('float32')/255
train_labels=to_categorical(train_labels)
test_data=test_data.reshape((test_data.shape[0],28,28,1))
test_data=test_data.astype('float32')/255
test_labels=to_categorical(test_labels)


network=models.Sequential()
network.add(layers.Conv2D(32,(3,3),activation='relu', input_shape=(28,28,1)))
network.add(layers.MaxPool2D(2,2))
network.add(layers.Conv2D(64,(3,3),activation='relu'))
network.add(layers.MaxPool2D(2,2))
network.add(layers.Conv2D(64,(3,3),activation='relu'))
network.add(layers.Flatten())
network.add(layers.Dense(64,activation='relu'))
network.add(layers.Dense(10,activation='softmax'))

network.compile(optimizer=optimizers.rmsprop(),loss=losses.categorical_crossentropy,metrics=['acc'])

callbacks = [
    callbacks.TensorBoard(
        log_dir='/Users/huangzhouhong/Downloads/logs',   #日志文件将被写入这个位置
        histogram_freq=1,       # 每一轮之后记录激活直方图
        # embeddings_freq=1,      #每一轮之后记录嵌入数据
    )
]

network.fit(train_data,train_labels,batch_size=100,epochs=5,validation_data=(test_data,test_labels),callbacks=callbacks)

# run: tensorboard --logdir=/Users/huangzhoong/Downloads/logs
