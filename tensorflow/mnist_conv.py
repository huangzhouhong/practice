import tensorflow as tf
from keras.datasets import mnist
from keras.utils import to_categorical

((train_data,train_label),(test_data,test_label))=mnist.load_data()

train_data=train_data.reshape([-1,28,28,1]).astype('float32')/255
test_data=test_data.reshape([-1,28,28,1]).astype('float32')/255

train_label=to_categorical(train_label)
test_label=to_categorical(test_label)

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1,shape=shape))

def conv2d(input,input_channel,output_channel,activation=None):
    weight=weight_variable([3,3,input_channel,output_channel])
    a=tf.nn.conv2d(input,filter=weight,strides=[1,1,1,1],padding='VALID')
    b=bias_variable([output_channel])
    output=tf.nn.bias_add(a,b)
    if activation!=None:
        output=activation(output)
    return output

def max_pool(input):
    return tf.nn.max_pool(input,(1,2,2,1),(1,2,2,1),padding='VALID')

def dense(input,input_size,output_size,activation=None):
    print("input:",input_size,"output:",output_size)
    weight=weight_variable([input_size,output_size])
    bias=bias_variable([output_size])
    a=tf.matmul(input,weight)+bias
    if activation!=None:
        a=activation(a)
    return a

def flattern(input):
    shape=input.get_shape().as_list()
    print(shape)
    nodes=shape[1]*shape[2]*shape[3]
    print(nodes)
    return tf.reshape(input,[-1,nodes])

x=tf.placeholder(dtype='float32',shape=[None,28,28,1])
y_=tf.placeholder(dtype='float32',shape=[None,10])

layer1=conv2d(x,1,16,activation=tf.nn.relu)
layer2=max_pool(layer1)
layer3=conv2d(layer2,16,32,activation=tf.nn.relu)
layer4=max_pool(layer3)
reshaped=flattern(layer4)
layer5=dense(reshaped,800,512,activation=tf.nn.relu)
y=dense(layer5,512,10)

cross_entropy=tf.nn.softmax_cross_entropy_with_logits_v2(logits=y,labels=y_)
train=tf.train.RMSPropOptimizer(0.01).minimize(cross_entropy)

correct_predict=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_predict,'float32'))

batch_size=200
train_data_size=train_data.shape[0]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        start=i*batch_size%train_data_size
        end=min(start+batch_size,train_data_size)
        sess.run(train,feed_dict={x:train_data[start:end],y_:train_label[start:end]})
        if i%100 == 0:
            print("acc:",sess.run(accuracy,feed_dict={x:train_data[start:end],y_:train_label[start:end]}))
            print("valid:", sess.run(accuracy, feed_dict={x: test_data, y_: test_label}))





