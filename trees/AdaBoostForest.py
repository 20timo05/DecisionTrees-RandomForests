import numpy as np
from tqdm import tqdm
import random

from trees.Utilities import SharedUtility
from trees.DecisionTree import DecisionTree as DecisionTreeRoot


class AdaBoostForest(SharedUtility):
    def __init__(self, values, stump_count=10):
        self.values = values
        self.stump_count = stump_count

        self.generate_trees()

    def generate_trees(self):
        # assign weight to each row (in beginning: all equally important, after first stump: rows that were misclassified = more important)
        self.sample_weights = [1 / len(self.values)] * len(self.values)

        self.trees = []

        for i in tqdm(range(self.stump_count)):
            # create stump (decision tree with only two leaf nodes)
            stump = DecisionTreeRoot(self.values, use_entropy=False, max_depth=2)

            say, new_sample_weights = self.calc_say_and_new_sample_weight(stump, self.sample_weights)
            self.trees.append([stump, say])
            self.sample_weights = new_sample_weights

            self.values = self.create_new_dataset()

    def calc_say_and_new_sample_weight(self, stump, sample_weights):
        features, targets = self.values[:, :-1], self.values[:, -1]
        predictions = [int(stump.predict(feats)) for feats in features]
        correct_mask = targets == predictions

        # total error = sum of weights associated with incorrectly classified samples
        error = sum(
            [weight for weight, correct in zip(sample_weights, correct_mask) if correct]
        ) + 1e-5
        try:
            say = 1 / 2 * np.log((1 - error) / error)
        except:
            print("stoopppp")


        # update sample weightsI wan
        new_sample_weights = [weight * np.exp(say if not correct else -say) for weight, correct in zip(sample_weights, correct_mask)]
        # normalize new sample weights (divide by sum so that they add up to 1 again)
        new_sample_weights = new_sample_weights / sum(new_sample_weights)

        return say, new_sample_weights
    
    # sample random items out of original values based on their weight (the higher the sample weight, the more often it will occur in new dataset)
    def create_new_dataset(self):
        new_dataset = []
        for i in range(len(self.values)):
            # generate random number (0 - 1) and find the corresponding number in the weights list that the random number would fall into if weights were a distribution
            random_num = random.random()
            sample_idx = np.searchsorted(np.cumsum(self.sample_weights), random_num)
            new_dataset.append(self.values[sample_idx])
        
        return np.array(new_dataset)

    def predict(self, data_point):
        # [[prediction, say], ...]
        predictions = np.array([[stump.predict(data_point), say] for stump, say in self.trees])
        # group by predictions and sum say
        grouped_say = np.bincount(predictions[:, 0].astype(int), weights=predictions[:, 1])
        # find prediction with highest combined say
        final_prediction = np.argmax(grouped_say)

        return final_prediction
