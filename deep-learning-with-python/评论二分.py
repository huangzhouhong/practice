
#存在过拟合问题
#binary_crossentropy适合二分问题
#sigmoid将输出压缩到0-1

from keras.datasets import imdb
from keras import *
import numpy as np
def to_vector(datas):
    vectors=np.zeros((len(datas),10000))
    for i in range(len(datas)):
        vectors[i,datas[i]]=1
    return vectors

def load_data():
    (train_data,train_label),(test_data,test_label)=imdb.load_data(num_words=10000)
    return (to_vector(train_data),train_label.astype('float32')),(to_vector(test_data),test_label.astype('float32'))

(train_data,train_label),(test_data,test_label)=load_data()

network=models.Sequential()
network.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
network.add(layers.Dense(16,activation='relu'))
network.add(layers.Dense(1,activation='sigmoid'))

network.compile(optimizer=optimizers.rmsprop(),loss=losses.binary_crossentropy,metrics=['acc'])
history = network.fit(train_data,train_label,batch_size=512,epochs=10,validation_data=(test_data,test_label))
acc=history.history['acc']
val_acc=history.history['val_acc']
x=range(1,len(acc)+1)

import matplotlib.pyplot as plt
plt.clf()
plt.plot(x,acc,color='green')
plt.plot(x,val_acc,'r+',color='red')
plt.show()