import tensorflow as tf
from tensorflow.python.framework import graph_util

# a=tf.Variable([1,2,3])
# b=tf.Variable([1,2,2])
#
# c=a+b
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     graph_def=tf.get_default_graph().as_graph_def()
#     output_graph_def=graph_util.convert_variables_to_constants(sess,graph_def,['add'])
#     with tf.gfile.GFile("/Users/huangzhouhong/Downloads/test/test.pb","wb") as f:
#         f.write(output_graph_def.SerializeToString())

with tf.Session() as sess:
    with tf.gfile.FastGFile('/Users/huangzhouhong/Downloads/test/test.pb','rb') as f:
        graph_def=tf.GraphDef()
        graph_def.ParseFromString(f.read())
    result=tf.import_graph_def(graph_def,return_elements=['add:0'])
    print(sess.run(result))
