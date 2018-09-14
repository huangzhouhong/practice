
#存在过拟合问题

from keras.datasets import imdb
from keras import *
import numpy as np

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

word_index=imdb.get_word_index()
reverse_work_index=dict([(index,word) for (word,index) in word_index.items()])
sentense=[reverse_work_index.get(i-3,'?') for i in train_data[0]]

train_vectors=np.zeros((len(train_data),10000))
for i in range(len(train_data)):
    train_vectors[i,train_data[i]]=1
train_result=train_labels.astype('float32')

test_vectors=np.zeros((len(test_data),10000))
for i in range(len(test_data)):
    test_vectors[i,test_data[i]]=1
test_result=test_labels.astype('float32')

network=models.Sequential()
network.add(layers.Dense(32,activation='tanh',input_shape=(10000,)))
network.add(layers.Dense(32,activation='tanh'))
network.add(layers.Dense(1,activation='sigmoid'))
network.compile(optimizer=optimizers.rmsprop(),loss=losses.mse,metrics=['accuracy'])
history = network.fit(train_vectors,train_result,batch_size=512,epochs=5,validation_data=(test_vectors,test_result))

print(history)

import matplotlib.pyplot as plt

y1=history.history['acc']
x=range(1,len(y1)+1)
plt.clf()
plt.plot(x,y1,'bo',label='acc')
plt.plot(x,history.history['val_acc'],'b',label='val_acc')
plt.legend()
plt.show()