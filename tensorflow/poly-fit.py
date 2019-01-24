
# coding: utf-8

# ### Using Tensorflow as a generalized compute graph
# 
# Fit a high order polynomial , defined by poly order.
# 
# In this experiment, we generate a random polynomial function of the given order.
# 
# Using the sampled data, we want to see if tensorflow's gradient descent can determine
# 
# the random weights (and fit the same curve to the data)
# 
# Trend observed is that the higher the polynomial order, the lower the learning rate needs to be.   
# (as low as 0.00001) for poly order of 10, otherwise gradient descent diverges (nan values)
# 
# For lower order (say 3), need to increase the learning rate, else need a lot more iterations.

# In[59]:

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#get_ipython().magic(u'matplotlib inline')

poly_order = 3

x = np.arange(-2,2,0.01)
x = x[:,None]
X_ = x
for i in range(2,poly_order+1,1):
    X_ = np.hstack((X_,x**i))
#print(X_)

W_ = np.random.uniform(-0.5,0.5,size=poly_order)
W_ = W_[:,None]
b_ = np.random.randn(1)
print('W_ shape:',W_.shape)
print('W_:',W_)
print('b_:',b_)

Y_ = np.dot(X_,W_) + b_

#y_  = 0.01*x1_**5 -0.4*x1_**2+0.3*x1_**2-0.8*x1_-0.55


# In[60]:

plt.plot(X_[:,0],Y_[:,0])
plt.show()

# In[61]:

# Create the model
X = tf.placeholder(tf.float32, [None, poly_order])

W = tf.Variable(tf.zeros([poly_order, 1]))
b = tf.Variable(tf.zeros([1]))

Y = tf.matmul(X, W) + b


# In[62]:

loss = tf.reduce_mean(tf.square(Y - Y_))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session()
sess.run(init)

# Fit the line.
for step in range(2000):
    sess.run(train, feed_dict={X: X_})
    #if step % 200 == 0:

#print(step, sess.run(W), sess.run(b))
#print('W_:',W_)
#print('b_:',b_)
#get_ipython().magic(u'matplotlib inline')
print('loss = ' , sess.run(loss, feed_dict={X: X_}))
Wpred = sess.run(W)
bpred = sess.run(b)
Ypred = np.dot(X_,Wpred) + bpred

plt.plot(X_[:,0],Ypred[:,0], 'r' , X_[:,0],Y_[:,0], 'b')
plt.show()
