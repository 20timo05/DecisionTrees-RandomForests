import numpy as np
import math

from trees.Utilities import SharedUtility

MIN_ENTROPY_GAIN = 0.1

class DecisionTree(SharedUtility):
    def __init__(self, values, feat_indices=None, use_entropy=False, max_depth=6, depth=1):
        self.values = values
        if feat_indices is not None:
            subset_cols = np.append(feat_indices, -1) # also include the target column
            self.values = self.values[:, subset_cols]
        
        self.feature_idx = None
        self.threshold = None
        self.children = []
        self.depth = depth

        self.feat_indices = feat_indices
        self.use_entropy = use_entropy

        self.max_depth = max_depth
    
    def generate_tree(self):
        if self.depth >= self.max_depth or len(self.values) <= 1: return
        # check if there is only 1 category within the values
        if DecisionTree.calc_entropy(self.values) == 0: return
        

        # [feature_idx, threshold_value, information_gain, portion1, portion2]
        best_divider = [None, None, 0]

        # loop through each possible feature which can be used for division of values (here: x coord, y coord)
        for feature_idx in range(self.values.shape[1] - 1):
            # loop through every possible threshold value (averages of adjacent values - ex.: [1, 3, 7, 10] => [2, 5, 8.5])
            feat_vals = sorted(self.values[:, feature_idx])
            thresholds = [(feat_vals[i] + feat_vals[i + 1]) / 2 for i in range(len(feat_vals) - 1)]

            for threshold in thresholds:
                # calculate information gain if this dividing criteria was used
                portion1 = np.array([val for val in self.values if val[feature_idx] <= threshold])
                portion2 = np.array([val for val in self.values if val[feature_idx] > threshold])
                
                if len(portion1) == 0 or len(portion2) == 0: continue

                information_gain = self.calc_information_gain(portion1, portion2)

                # check if this configuration is the best one yet
                if information_gain > best_divider[2]:
                    best_divider = [feature_idx, threshold, information_gain, portion1, portion2]
        
        # generate children based on the best dividing option if information_gain is good enough
        if best_divider[2] >= MIN_ENTROPY_GAIN:
            self.feature_idx = best_divider[0]           
            self.threshold = best_divider[1]
            self.children.append(DecisionTree(best_divider[3], use_entropy=self.use_entropy, depth=self.depth + 1))
            self.children.append(DecisionTree(best_divider[4], use_entropy=self.use_entropy, depth=self.depth + 1))

            self.children[0].generate_tree()
            self.children[1].generate_tree()

    def calc_information_gain(self, portion1, portion2):
        # calculate entropy/ gini impurity for values from category 0.0 and 1.0 and the one of the parent
        criterion_portion1 = DecisionTree.calc_entropy(portion1) if self.use_entropy else DecisionTree.calc_gini_impurity(portion1)
        criterion_portion2 = DecisionTree.calc_entropy(portion2) if self.use_entropy else DecisionTree.calc_gini_impurity(portion2)
        
        weighted_portion1 = criterion_portion1 * len(portion1) / len(self.values)
        weighted_portion2 = criterion_portion2 * len(portion2) / len(self.values)

        parent = DecisionTree.calc_entropy(self.values) if self.use_entropy else DecisionTree.calc_gini_impurity(self.values)

        information_gain = parent - weighted_portion1 - weighted_portion2

        return information_gain
        

    # return the target with the most datapoints in self.values
    def get_most_common_target(self):
        unique, counts = np.unique(self.values[:, -1], return_counts=True)
        return unique[counts.argmax()]


    def predict(self, data_point):
        if self.depth == 1 and self.feat_indices is not None:
            data_point = data_point[self.feat_indices]
        
        # check if leaf DecisionTree
        if len(self.children) == 0: return self.get_most_common_target()
        
        # check criteria
        if data_point[self.feature_idx] <= self.threshold: return self.children[0].predict(data_point)
        return self.children[1].predict(data_point)

    
    def getAllDecisions(self, depth):
        if len(self.children) == 0 or self.depth > depth: return []

        all_decisions = [[self.feature_idx, self.threshold]]

        all_decisions.extend(self.children[0].getAllDecisions(depth=depth))
        all_decisions.extend(self.children[1].getAllDecisions(depth=depth))

        return all_decisions
    
    @staticmethod
    def calc_gini_impurity(values):
        unique, counts = np.unique(values[:, -1], return_counts=True)
        category_portion = counts / len(values)
        return 1 - sum([cp**2 for cp in category_portion])

    @staticmethod
    def calc_entropy(values):
        unique, counts = np.unique(values[:, -1], return_counts=True)
        category_portion = counts / len(values)
        return sum([-cp * math.log2(cp) for cp in category_portion])