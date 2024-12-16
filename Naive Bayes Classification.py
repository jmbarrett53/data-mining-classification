# Submit this file to Gradescope
import math
from typing import Dict, List, Tuple
# You may use any built-in standard Python libraries
# You may NOT use any non-standard Python libraries such as numpy, scikit-learn, etc.

num_C = 7 # Represents the total number of classes

class Solution:
  
  def prior(self, X_train: List[List[int]], Y_train: List[int]) -> List[float]:
    """Calculate the prior probabilities of each class
    Args:
      X_train: Row i represents the i-th training datapoint
      Y_train: The i-th integer represents the class label for the i-th training datapoint
    Returns:
      A list of length num_C where num_C is the number of classes in the dataset
    """
    # Something that is usefule to note is that the sum of all probabilities must equal 1

    # Prior probability I take to mean as the probability of a particular class occuring, regardless of any features or observations

    # Begin by counting number of instances of each class
    # Create a dictionary that holds class:num_instances as key:value pairs
    instances_dict = {}
    for label in Y_train:
      instances_dict[label] = 1 + instances_dict.get(label, 0)

    # print(instances_dict)

    class_priors = [0.1] * num_C
    for i in range(1, len(class_priors) + 1):
      class_priors[i - 1] += instances_dict.get(i, 0)
      class_priors[i - 1] = class_priors[i - 1] / (len(Y_train) + (0.1 * num_C))

    return class_priors

    
  def numerator_xi_prob(self, training_data, xi, f, c, class_labels):
    """
    Given training data, and a specified feature and class (both specified as an integer representing the index)
    Return the count of all instances where class = c and feature = f
    """
    count = 0
    for i, training_point in enumerate(training_data):
      # f is not what it's supposed to be
      if training_point[xi] == f and class_labels[i] == c:
        count += 1
    return count + 0.1


  def denomenator_xi_prob(self, training_data, xi, possible_feature_vals, c, class_labels):
    """
    Given training data, and a specified feature and class
    Return the count of all training instances where class = c * num unique vals for specified feature
    """
    count = 0
    for i, training_point in enumerate(training_data):
      if class_labels[i] == c:
        count += 1
    return count + 0.1 * len(possible_feature_vals[xi])

  def calculate_pX(self, probabilites):
    """
    Given a list of probabilities, combine and return them
    """
    ret_val = 1
    for p in probabilites:
      ret_val *= p

    return ret_val



  def label(self, X_train: List[List[int]], Y_train: List[int], X_test: List[List[int]]) -> List[int]:
    """Calculate the classification labels for each test datapoint
    Args:
      X_train: Row i represents the i-th training datapoint
      Y_train: The i-th integer represents the class label for the i-th training datapoint
      X_test: Row i represents the i-th testing datapoint
    Returns:
      A list of length M where M is the number of datapoints in the test set
    """
    # Calculate priors for each class
    priors = self.prior(X_train, Y_train)

    # Get possible values for each feature
    possible_feature_vals = [[0, 1] for _ in range(len(X_train[0]))]
    possible_feature_vals[-3] = [0, 2, 4, 5, 6, 8]  # For 'legs'

    # Predict labels for test points
    predictions = []
    for test_point in X_test:
      posteriors = []
      for c in range(1, num_C + 1):  # Iterate through classes
        likelihood = 1
        for xi, f in enumerate(test_point):
          if xi == 0:
            continue
          numerator = self.numerator_xi_prob(X_train, xi, f, c, Y_train)
          denominator = self.denomenator_xi_prob(X_train, xi, possible_feature_vals, c, Y_train)
          likelihood *= numerator / denominator
        posterior = priors[c - 1] * likelihood
        posteriors.append(posterior)
      predictions.append(posteriors.index(max(posteriors)) + 1)
    return predictions


    # classes = [1, 2, 3, 4, 5, 6, 7]

    # # Calculate posterior
    # posteriors = [0.1] * 7


    # priors = self.prior()
    # if len(priors) != len(posteriors):
    #   print("Class probabilites for priors and posteriors don't match")
    #   return -1

    # final_prob = []
    # for i in range(len(priors)):
    #   final_prob.append(priors[i] * posteriors[i])
    

    # labels = final_prob.index(max(final_prob))
    # return label





    # Is there a way to do this without a bunch of loops? We should probably try and do it like that
    # classes = [1, 2, 3, 4, 5, 6, 7]
    # possible_feature_vals = [
    #   [-1],
    #   [0, 1],
    #   [0, 1], 
    #   [0, 1],
    #   [0, 1], 
    #   [0, 1],
    #   [0, 1], 
    #   [0, 1],
    #   [0, 1], 
    #   [0, 1],
    #   [0, 1], 
    #   [0, 1],
    #   [0, 1], 
    #   [0, 2, 4, 5, 6, 8],
    #   [0, 1], 
    #   [0, 1], 
    #   [0, 1]
    # ]
    # prob_xi = []
    # prob_X = []

    


    # # for each data point, the predicted label y' is determined by the formula y' = argmax(y) [P(y)P(X|y)]
    # for test_point in X_test:
    #   # Calculate y' for each test point, y' = argmax(y) [P(y)P(X|y)]
      
    #   # create a method that calculates the number of training samples where class = c and xi = f
    #   for c in classes:
    #     for f in range(1, len(possible_feature_vals)):
    #       for real_val in possible_feature_vals[f]:
    #         for xi in range(1, 17):
    #           numerator = self.numerator_xi_prob(X_train, xi, real_val, c, Y_train)
    #           denomenator = self.denomenator_xi_prob(X_train, xi, possible_feature_vals, c, Y_train)
    #           prob_xi.append(numerator / denomenator)
    #       # Recombine P(xi|y) into P(X|y)
    #       pX = self.calculate_pX(prob_xi)
    #       prob_X.append(pX)

    #   prob_Y = self.prior(X_train, Y_train)
    #   pXY = []
    #   for i in range(len(prob_X)):
    #     pXY.append(prob_X[i] * prob_Y[i])
      
    #   max_p = -1
    #   for p in pXY:
    #     if p > max_p:
    #       max_p = p
    #   result = pXY.index(max_p)

    #   return result
          


    
  
# sample_X_train = [
#   ['aardvark',1,0,0,1,0,0,1,1,1,1,0,0,4,0,0,1],
#   ['worm',0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0], 
#   ['piranha',0,0,1,0,0,1,1,1,1,0,0,1,0,1,0,0],
#   ['gnat',0,0,1,0,1,0,0,0,0,1,0,0,6,0,0,0], 
#   ['oryx',1,0,0,1,0,0,0,1,1,1,0,0,4,1,0,1],
#   ['moth',1,0,1,0,1,0,0,0,0,1,0,0,6,0,0,0], 
#   ['skimmer',0,1,1,0,1,1,1,0,1,1,0,0,2,1,0,0],
#   ['crab',0,0,1,0,0,1,1,0,0,0,0,0,4,0,0,0], 
#   ['vampire',1,0,0,1,1,0,0,1,1,1,0,0,2,1,0,0],
#   ['slowworm',0,0,1,0,0,0,1,1,1,1,0,0,0,1,0,0]
# ]

# sample_X_test = [['bass',0,0,1,0,0,1,1,1,1,0,0,1,0,1,0,0]]

# sample_Y_train = [1,7,4,6,1,6,2,7,1,3]

# test = Solution()

# print(test.label(sample_X_train, sample_Y_train, sample_X_test))

