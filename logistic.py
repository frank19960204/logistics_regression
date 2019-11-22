import tensorflow as tf
import numpy as np

def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random.normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

#Training data
x_data = np.linspace(-1,1,300).reshape(150,2)
noise = np.random.normal(0, 0.05,(150,1))
y_data = np.zeros((150,2))
for i in range(0,150):
    if(x_data[i][0] + x_data[i][1])>0:
        y_data[i][0] = 1
        y_data[i][1] = 0
    else:
        y_data[i][0] = 0
        y_data[i][1] = 1

#Testing data
x_test = np.linspace(-1,1,300).reshape(150,2)
noise = np.random.normal(0, 0.05,(150,1))
y_test = np.zeros((150,2))
for i in range(0,150):
    if(x_test[i][0] + x_test[i][1])>0:
        y_test[i][0] = 1
        y_test[i][1] = 0
    else:
        y_test[i][0] = 0
        y_test[i][1] = 1

# define placeholder for inputs to network
xs = tf.compat.v1.placeholder(tf.float32, [None, 2])
ys = tf.compat.v1.placeholder(tf.float32, [None, 2])
# add hidden layer
l1 = add_layer(xs, 2, 5, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 5, 2, activation_function=None)

# the error between prediction and real data(cross entropy)
loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits = prediction,labels = ys)
train_step = tf.compat.v1.train.AdamOptimizer(0.1).minimize(loss)


init = tf.compat.v1.global_variables_initializer()
sess = tf.compat.v1.Session()
sess.run(init)

for i in range(3000):
    # training
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        #show accuracy
        y_pre = sess.run(prediction, feed_dict={xs: x_test})
        correct = tf.equal(tf.argmax(y_pre,1), tf.argmax(y_test,1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        result = sess.run(accuracy)
        print(result)
  
