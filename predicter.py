import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

class Predicter:
    def __init__(self, learning_rate, nbp_input, move_distance):
        self.X = tf.placeholder("float32", [None, nbp_input + 1])
        self.Y = tf.placeholder("float32", [None, nbp_input])
        n_l1 = 256 
        n_l2 = 128
        weights = {
            'encoder_l1': tf.Variable(tf.random_normal([nbp_input + 1, n_l1])),
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
        
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.image = None

        self.direction = None # Direction of the current move
        self.move_distance = move_distance # Number of steps/pixels of a move
        self.move_steps = 0 # Number of steps done for the current move

    def train(self, image, direction, next_image):
        input_ = np.concatenate((image, [[direction]]), axis=1)
        cost, _ = self.sess.run([self.cost, self.optimizer], feed_dict={self.X: input_, self.Y: next_image})
        return cost

    def predict(self, image, direction):
        input_ = np.concatenate((image, [[direction]]), axis=1)
        return self.sess.run(self.decoder, feed_dict={self.X: input_})

    def accuracy(self, image, direction, next_image):
        input_ = np.concatenate((image, [[direction]]), axis=1)
        cost = self.sess.run(self.cost, feed_dict={self.X: input_, self.Y: next_image})
        return 1 - cost

    def act(self, observation, reward, done, info):
        if done:
            self.image = None
            return

        if self.move_steps % self.move_distance == 0:
            self.direction = random.randint(0, 1)
        self.move_steps += 1
        action = (10, self.direction)

        next_image = observation.reshape(1,-1).astype(np.float32) / 255
        if self.image is None:
            self.image = next_image
            return action

        self.train(self.image, action[1], next_image)
        self.image = next_image

        # Return action (don't try to predict digit, move in the same direction)
        return action

    def save_model(self, path):
        self.saver.save(self.sess, path)

    def load_model(self, path):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver.restore(self.sess, path)
