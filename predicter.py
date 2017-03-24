import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class Predicter:
    def __init__(self, learning_rate,  nbp_input, time_training):
        self.X = tf.placeholder("float32", [None, nbp_input])
        self.Y = tf.placeholder("float32", [None, nbp_input])
        n_l1 = 256 
        n_l2 = 128
        weights = {
            'encoder_l1': tf.Variable(tf.random_normal([nbp_input, n_l1])),
            'encoder_l2': tf.Variable(tf.random_normal([n_l1, n_l2])),
            'decoder_l1': tf.Variable(tf.random_normal([n_l2, n_l1])),
            'decoder_l2': tf.Variable(tf.random_normal([n_l1, nbp_input])),
        }
        biases = {
            'encoder_l1': tf.Variable(tf.random_normal([n_l1])),
            'encoder_l2': tf.Variable(tf.random_normal([n_l2])),
            'decoder_l1': tf.Variable(tf.random_normal([n_l1])),
            'decoder_l2': tf.Variable(tf.random_normal([nbp_input])),
        }

        el1 = tf.nn.sigmoid(tf.add(tf.matmul(self.X, weights['encoder_l1']), biases['encoder_l1']))
        el2 = tf.nn.sigmoid(tf.add(tf.matmul(el1, weights['encoder_l2']), biases['encoder_l2']))
        self.encoder = el2

        dl1 = tf.nn.sigmoid(tf.add(tf.matmul(el2, weights['decoder_l1']), biases['decoder_l1']))
        dl2 = tf.nn.sigmoid(tf.add(tf.matmul(dl1, weights['decoder_l2']), biases['decoder_l2']))
        self.decoder = dl2

        self.cost = tf.reduce_mean(tf.pow(self.Y - dl2, 2))
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.image = None

    def train(self, image, next_image):
        cost, _ = self.sess.run([self.cost, self.optimizer], feed_dict={self.X: image, self.Y: next_image})
        return cost

    def predict(self, image):
        return self.sess.run([self.decoder], feed_dict={self.X: image})

    def accuracy(self, image, next_image):
        cost = self.sess.run([self.cost], feed_dict={self.X: image, self.Y: next_image})
        return 1 - cost

    def act(self, observation, reward, done, info):
        if done:
            self.image = None
            return

        next_image = info['cursor'].reshape(1,-1).astype(np.float32) / 255

        if self.image is None:
            self.image = next_image
            return (10, (0, 1))

        cost = self.train(self.image, next_image)
        print(1 - cost)

        self.image = next_image

        # Return action (don't try to predict digit, move in the same direction)
        return (10, (0, 1))
