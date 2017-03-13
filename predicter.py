class Predict:
    def __init__(self, eta, batch_size...):
        self.eta = eta
        self.image = None

    def train(image):
        pass

    def predict(image):
        return pred

    def accuracy(self, image, label):
        pred = self.predict(image)
        return cost

    def act(self, observation, reward, done, info):
        if self.image is None:
            self.image = info['cursor']
            return (10, (0,1))

        print(self.accuracy(self.image, info['cursor']))
        self.image = info['cursor']

        # Return action (don't try to predict digit, move in the same direction)
        return (10, (0,1))
