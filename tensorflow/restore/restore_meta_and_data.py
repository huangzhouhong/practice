import tensorflow as tf

# a=tf.Variable([1,2,3])
# b=tf.Variable([1,2,2])
#
# c=a+b
#
# saver=tf.train.Saver()
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     saver.save(sess,'/Users/huangzhouhong/Downloads/test/test')

saver=tf.train.import_meta_graph('/Users/huangzhouhong/Downloads/test/test.meta')
with tf.Session() as sess:
    saver.restore(sess,tf.train.latest_checkpoint('/Users/huangzhouhong/Downloads/test'))
    c=tf.get_default_graph().get_tensor_by_name('add:0')
    print(sess.run(c))
