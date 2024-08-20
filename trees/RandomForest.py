import numpy as np
import math
from tqdm import tqdm

from trees.DecisionTree import DecisionTree as DecisionTreeRoot
from trees.Utilities import SharedUtility


class RandomForest(SharedUtility):
    def __init__(self, values, decision_tree_count=10, use_all_features=False):
        self.values = values
        self.tree_count = decision_tree_count
        
        self.num_feats = self.values.shape[1] - 1
        self.feat_count = round(math.sqrt(self.num_feats))

        self.use_all_features = use_all_features

        self.generate_trees()
    
    def generate_trees(self):
        self.decision_trees = []
        for i in tqdm(range(self.tree_count)):
            # Bootstrapping: Generate Random Subset (same size as values) with replacement
            subset = self.values[np.random.randint(self.values.shape[0], size=self.values.shape[0])]

            # Feature Selection: Only use some of the features (N = self.feature_count)
            feat_indices = np.random.choice(self.num_feats, size=self.feat_count, replace=False)
            
            # use all features in each decision tree
            if self.use_all_features: feat_indices = None
            
            decision_tree = DecisionTreeRoot(subset, feat_indices=feat_indices, use_entropy=False)
            decision_tree.generate_tree()

            self.decision_trees.append(decision_tree)
    
    def predict(self, values_point):
        # pass values point through every decision tree and return most common result
        predictions = [dec_tree.predict(values_point) for dec_tree in self.decision_trees]
        most_common_prediction = np.bincount(np.array(predictions).astype(int)).argmax()

        return most_common_prediction