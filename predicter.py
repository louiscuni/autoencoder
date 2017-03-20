import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

class Predict:
    def __init__(self, learning_rate,  nbp_input, time_training):
        init = tf.global_variables_initializer()
        self.learning_rate=learning_rate
        self.nbp_input=nbp_input
        self.time_training=time_training
        self.image = None
        self.X=tf.placeholder("float32", [None, nbp_input])
        n_h1=256 
        n_h2=128
        self.weights={
                    'encodeur_h1' : tf.Variable(tf.random_normal([nbp_input, n_h1])),
                    'encodeur_h2' : tf.Variable(tf.random_normal([n_h1, n_h2])),
                    'decodeur_h1' : tf.Variable(tf.random_normal([n_h2, n_h1])),
                    'decodeur_h2' : tf.Variable(tf.random_normal([n_h1, nbp_input])),
                    }
        self.biases ={
                    'encodeur_b1': tf.Variable(tf.random_normal([n_h1])),
                    'encodeur_b2': tf.Variable(tf.random_normal([n_h2])),
                    'decodeur_b1': tf.Variable(tf.random_normal([n_h1])),
                    'decodeur_b2': tf.Variable(tf.random_normal([nbp_input])),

                    }

    def encodeur(self, x):
        self.el1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['encodeur_h1']),self.biases['encodeur_b1']))
        self.el2 = tf.nn.sigmoid(tf.add(tf.matmul(self.el1, self.weights['encodeur_h2']) ,self.biases['encodeur_b2']))
        return self.el2

    def decoder(self, x):
        self.dl1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['decodeur_h1']),self.biases['decodeur_b1']))
        self.dl2 = tf.nn.sigmoid(tf.add(tf.matmul(self.dl1, self.weights['decodeur_h2']),self.biases['decodeur_b2']))
        return self.dl2

    def train(self, image, image_next):
        y_pred=self.predict(image)
        y_true=image_next

        self.cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
        #print("entrainement pas termine ")
        tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

        #print("entrainement done")
        return self.cost


    def predict(self, image):
        encode_op=self.encodeur(image)
        pred=self.decoder(encode_op)
        return pred

    def accuracy(self, image, label):
        pred = self.predict(image)
        return cost

    def act(self, observation, reward, done, info):
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            total_batch = int(mnist.train.num_examples/256)
            for j in range(total_batch):
                x_batch, y_batch=mnist.train.next_batch(256)
                c=sess.run(self.train(x_batch, x_batch))
                print(c)

        if self.image is None:
            self.image = info['cursor']
            return (10, (0,1))

        print(self.accuracy(self.image, info['cursor']))
        self.image = info['cursor']

        # Return action (don't try to predict digit, move in the same direction)
        return (10, (0,1))

    def same(self, tf1, tf2):
        if tf1==tf2:
            print ("pas d'evolution")
            return tf1
        else:
            print("evolue de")
            return tf1-tf2

x_batch, y_batch= mnist.train.next_batch(1)
#print(x_batch)

P=Predict(0.02, 784, 2)
P.act(0,0,0,0)