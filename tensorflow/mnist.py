import tensorflow as tf
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

((train_data,train_label),(test_data,test_label))=mnist.load_data()

train_data=train_data.reshape(60000,784).astype('float32')/255
test_data=test_data.reshape(test_data.shape[0],784).astype('float32')/255

test_label=to_categorical(test_label)
train_label=to_categorical(train_label)

x=tf.placeholder('float32',shape=[None,784])
y_=tf.placeholder('float32',shape=[None,10])

W1=tf.Variable(tf.random_normal([784,20]))
b1=tf.Variable(tf.random_normal([20]))

W2=tf.Variable(tf.random_normal([20,10]))
b2=tf.Variable(tf.random_normal([10]))

a=tf.nn.relu(tf.matmul(x,W1)+b1)

y=tf.nn.softmax(tf.matmul(a,W2)+b2)

cross_error=-tf.reduce_mean(tf.reduce_sum(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)),1))
train=tf.train.RMSPropOptimizer(0.1).minimize(cross_error)

correct_predict=tf.equal(tf.argmax(y_,1),tf.argmax(y,1))
accuracy= tf.reduce_mean(tf.cast(correct_predict,'float32'))

batch_size=1000
train_data_size=train_data.shape[0]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100000):
        start=i*batch_size%train_data_size
        end=min(start+batch_size,train_data_size)
        sess.run(train,feed_dict={x:train_data[start:end],y_:train_label[start:end]})
        if i%1000 == 0:
            print(sess.run(accuracy,feed_dict={x:test_data,y_:test_label}))