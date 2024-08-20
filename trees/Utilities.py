from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
import numpy as np

class SharedUtility(ABC):
    def calc_accuracy(self, data):
        correct = 0
        for data_point in data:
            features, target = data_point[:-1], data_point[-1]
            if self.predict(features) == target:
                correct += 1
        
        return correct / len(data)
    
    @abstractmethod
    def predict(self):
        pass

    # ONLY WORKS IF THERE ARE ONLY 2 FEATURES & 2 TARGETS
    def draw(self, areas=False, dividing_lines=False):
        x = self.values[:, 0]
        y = self.values[:, 1]
        target = self.values[:, -1]

        # Plotting the data points
        plt.scatter(x[target == 0], y[target == 0], color='blue', label='Target 0')
        plt.scatter(x[target == 1], y[target == 1], color='orange', label='Target 1')

        # Adding labels and legend
        plt.xlabel('X')
        plt.ylabel('Y')

        if areas:
            # Creating a grid of points to evaluate the model on
            x_min, x_max = x.min() - 1, x.max() + 1
            y_min, y_max = y.min() - 1, y.max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                                    np.arange(y_min, y_max, 0.1))
            
            # Predicting the class for each point in the grid
            Z = np.array([self.predict([xi, yi]) for xi, yi in np.c_[xx.ravel(), yy.ravel()]])
            Z = Z.reshape(xx.shape)
            
            # Plotting the contour plot
            plt.contourf(xx, yy, Z, alpha=0.3, levels=np.linspace(0, 1, 3), colors=['blue', 'orange'])
            
            # Showing the legend
            plt.legend()

        if dividing_lines and hasattr(self, "getAllDecisions"):
            # add dividing lines
            all_decisions = self.getAllDecisions(depth=dividing_lines)
            for decision in all_decisions:
                feature_idx, threshold = decision
                LINE_LENGTH = 30
                if feature_idx == 0: plt.axline((threshold, -LINE_LENGTH/2), (threshold, LINE_LENGTH/2))
                else: plt.axline((-LINE_LENGTH/2, threshold), (LINE_LENGTH/2, threshold))

        plt.show()