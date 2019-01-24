import tensorflow as tf

sess = tf.InteractiveSession()

matrix1 = tf.Variable([[ 3. , 3.]])
matrix2 = tf.constant([[2.],[2.]])

matrix1.initializer.run()

product = tf.matmul(matrix2,matrix1)

print(product.eval())

sess.close()
