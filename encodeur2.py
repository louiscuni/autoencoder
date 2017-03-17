# coding=utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

#parametre
learning_rate=0.2

iteration=5

batch_size=256
examples_to_show= 10

#network parameters
n_h1=500 #nombre de neurone sur la couche 1
n_h2=350
n_h3=160
n_h4=100
n_input=784 #nombre de neurone en entrée 28*28

#couche d'entrée
#A=tf.placeholder(tf.float32, n_input)

#reseau

weights = {
	'encodeur_h1' : tf.Variable(tf.random_normal([n_input, n_h1])),
	'encodeur_h2' : tf.Variable(tf.random_normal([n_h1, n_h2])),
    'encodeur_h3' : tf.Variable(tf.random_normal([n_h2, n_h3])),
    'encodeur_h4' : tf.Variable(tf.random_normal([n_h3, n_h4])),
	'decodeur_h1' : tf.Variable(tf.random_normal([n_h4, n_h3])),
	'decodeur_h2' : tf.Variable(tf.random_normal([n_h3, n_h2])),
    'decodeur_h3' : tf.Variable(tf.random_normal([n_h2, n_h1])),
    'decodeur_h4' : tf.Variable(tf.random_normal([n_h1, n_input])),
}
	
biases ={
	'encodeur_b1': tf.Variable(tf.random_normal([n_h1])),
	'encodeur_b2': tf.Variable(tf.random_normal([n_h2])),
    'encodeur_b3': tf.Variable(tf.random_normal([n_h3])),
    'encodeur_b4': tf.Variable(tf.random_normal([n_h4])),
	'decodeur_b1': tf.Variable(tf.random_normal([n_h3])),
    'decodeur_b2': tf.Variable(tf.random_normal([n_h2])),
    'decodeur_b3': tf.Variable(tf.random_normal([n_h1])),
	'decodeur_b4': tf.Variable(tf.random_normal([n_input])),

}

def encoder(x):
    l1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encodeur_h1']),biases['encodeur_b1']))
    lem2= tf.matmul(l1, weights['encodeur_h2'])
    l2 = tf.nn.sigmoid(tf.add(lem2 ,biases['encodeur_b2']))
    l3 = tf.nn.sigmoid(tf.add(tf.matmul(l2, weights['encodeur_h3']),biases['encodeur_b3']))
    l4 = tf.nn.sigmoid(tf.add(tf.matmul(l3, weights['encodeur_h4']),biases['encodeur_b4']))
    return l4

def decoder(x):
    l1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decodeur_h1']),biases['decodeur_b1']))
    l2 = tf.nn.sigmoid(tf.add(tf.matmul(l1, weights['decodeur_h2']),biases['decodeur_b2']))
    l3 = tf.nn.sigmoid(tf.add(tf.matmul(l2, weights['decodeur_h3']),biases['decodeur_b3']))
    l4 = tf.nn.sigmoid(tf.add(tf.matmul(l3, weights['decodeur_h4']),biases['decodeur_b4']))
    return l4



##MAIN##

X=tf.placeholder("float32", [None, n_input])

encoder_op = encoder(X)
decoder_op= decoder(encoder_op)

y_pred=decoder_op
y_true=X

cost = tf.reduce_mean(tf.pow(y_true - decoder_op, 2))
#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

#launcher#

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    total_batch = int(mnist.train.num_examples/batch_size)
    #i=0
    c=1
    cc=0
    # Training cycle
    for i in range(iteration):
    #while abs(c-cc)>epsilon:
        #print c
        #print cc 
        
        # Loop over all batches
        for j in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            cc=c
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        # Display logs per i step
        if i % 1 == 0:
            #i+=1
            print("i:", '%04d' % (i+1),
                  "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")
    print(abs(c-cc))

    # Applying encode and decode over test set
    encode_decode = sess.run(
        y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})
    # Compare original images with their reconstructions
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    f.show()
    plt.draw()
plt.waitforbuttonpress()
########
