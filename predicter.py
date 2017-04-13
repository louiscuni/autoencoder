import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

class Predicter:
    """
    Class managing an autoencoder network predicting the cursor view
    at one pixel in a given direction in a NumGrid environment.
    """
    def __init__(self, cursor_size, learning_rate=0.001):
        nbp_input = np.prod(cursor_size)
        self.X = tf.placeholder("float32", [None, nbp_input + 1])
        self.Y = tf.placeholder("float32", [None, nbp_input])
        n_l1 = 130
        n_l2 = 70
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

    def train(self, images, directions, next_images):
        inputs = np.hstack((images, directions))
        cost, _ = self.sess.run([self.cost, self.optimizer], feed_dict={self.X: inputs, self.Y: next_images})
        return cost

    def predict(self, image, direction):
        input_ = np.hstack((image, [[direction]]))
        return self.sess.run(self.decoder, feed_dict={self.X: input_})

    def accuracy(self, image, direction, next_image):
        input_ = np.hstack((image, [[direction]]))
        cost = self.sess.run(self.cost, feed_dict={self.X: input_, self.Y: next_image})
        return 1 - cost

    def learn(self, numgrid, num_episodes, directions={0,1,2,3}, move_distance=10):
        for i in range(num_episodes):
            observation = numgrid.reset()
            done = False
            images = Predicter.normalize(observation)
            dirs = np.ndarray((0,1))
            t = 0
            while not done:
                if t % move_distance == 0:
                    direction = random.choice(tuple(directions))
                dirs = np.vstack((dirs, direction))
                action = (10, direction)
                observation, _, done, _ = numgrid.step(action)
                images = np.vstack((images, Predicter.normalize(observation)))
                t += 1
            self.train(images[:-1], dirs, images[1:])

    def save_model(self, path):
        return self.saver.save(self.sess, path)

    def load_model(self, path):
        self.sess.close()
        self.sess = tf.Session()
        self.saver.restore(self.sess, path)

    @staticmethod
    def normalize(observation):
        return observation.reshape(1,-1).astype(np.float32) / 255
