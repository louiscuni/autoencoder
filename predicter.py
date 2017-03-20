import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class Predicter:
    def __init__(self, learning_rate,  nbp_input, time_training):
        init = tf.global_variables_initializer()
        self.learning_rate=learning_rate
        self.nbp_input=nbp_input
        self.time_training=time_training
        self.image = None
        self.X = tf.placeholder("float32", [None, nbp_input])
        n_h1 = 256 
        n_h2 = 128
        self.weights = {
            'encoder_h1' : tf.Variable(tf.random_normal([nbp_input, n_h1])),
            'encoder_h2' : tf.Variable(tf.random_normal([n_h1, n_h2])),
            'decoder_h1' : tf.Variable(tf.random_normal([n_h2, n_h1])),
            'decoder_h2' : tf.Variable(tf.random_normal([n_h1, nbp_input])),
        }
        self.biases = {
            'encoder_b1': tf.Variable(tf.random_normal([n_h1])),
            'encoder_b2': tf.Variable(tf.random_normal([n_h2])),
            'decoder_b1': tf.Variable(tf.random_normal([n_h1])),
            'decoder_b2': tf.Variable(tf.random_normal([nbp_input])),
        }

    def encode(self, x):
        self.el1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['encoder_h1']),self.biases['encoder_b1']))
        self.el2 = tf.nn.sigmoid(tf.add(tf.matmul(self.el1, self.weights['encoder_h2']) ,self.biases['encoder_b2']))
        return self.el2

    def decode(self, x):
        self.dl1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['decoder_h1']),self.biases['decoder_b1']))
        self.dl2 = tf.nn.sigmoid(tf.add(tf.matmul(self.dl1, self.weights['decoder_h2']),self.biases['decoder_b2']))
        return self.dl2

    def train(self, image, image_next):
        y_pred = self.predict(image)
        y_true = image_next

        cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)

        return [cost, optimizer]

    def predict(self, image):
        encode_op = self.encode(image)
        pred = self.decode(encode_op)
        return pred

    def accuracy(self, image, label):
        pred = self.predict(image)
        cost = tf.reduce_mean(tf.pow(label - pred, 2))
        return 1 - cost

    def act(self, observation, reward, done, info):
        next_image = info['cursor'].reshape(1,100).astype(np.float32)

        if self.image is None:
            self.image = next_image
            return (10, (0,1))

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            sess.run(self.train(self.image, next_image), feed_dict={X: self.image})
            print(sess.run(self.accuracy(self.image, next_image)))

        # print(self.accuracy(self.image, next_image))
        # print(next_image)
        self.image = next_image

        # Return action (don't try to predict digit, move in the same direction)
        return (10, (0,1))

