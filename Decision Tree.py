from typing import List
import math

class Node:

  """
  This class, Node, represents a single node in a decision tree. It is designed to store information about the tree
  structure and the specific split criteria at each node. It is important to note that this class should NOT be
  modified as it is part of the assignment and will be used by the autograder.

  The attributes of the Node class are:
  - split_dim: The dimension/feature along which the node splits the data (-1 by default, indicating uninitialized)
  - split_point: The value used for splitting the data at this node (-1 by default, indicating uninitialized)
  - label: The class label assigned to this node, which is the majority label of the data at this node. If there is a tie,
    the numerically smaller label is assigned (-1 by default, indicating uninitialized)
  - left: The left child node of this node (None by default). Either None or a Node object.
  - right: The right child node of this node (None by default) Either None or a Node object.
  """

  def __init__(self):
    self.split_dim = -1
    self.split_point = -1
    self.label = -1
    self.left = None
    self.right = None


class Solution:
  """
  Example usage of the Node class to build a decision tree using a custom method called split_node():

  # In the fit method, create the root node and call the split_node() method to build the decision tree
    self.root = Node()
    self.split_node(self.root, data, ..., depth=0)

  def split_node(self, node, data, ..., depth):
      # Your implementation to calculate split_dim, split_point, and label for the given node and data
      # ...

      # Assign the calculated values to the node
      node.split_dim = split_dim
      node.split_point = split_point
      node.label = label

      # Recursively call split_node() for the left and right child nodes if the current node is not a leaf node
      # Remember, a leaf node is one that either only has data from one class or one that is at the maximum depth
      if not is_leaf:
          left_child = Node()
          right_child = Node()

          split_node(left_child, left_data, ..., depth+1)
          split_node(right_child, right_data, ..., depth+1)
  """

  def get_max_label(self, label: List):
    """
    Returns the majority label given a list of labels
    """
    classes = set(label)
    counter = {}
    for item in label:
       counter[item] = 1 + counter.get(item, 0)

    return max(counter, key = counter.get)  
       
       

  def split_node(self, node, data, label, depth):
    """
    Recursively find and split on best point/dimension
    In essence, this function "builds" the tree
    """
    # Calculate split_dim, split_point, and label for the given node and data
    # For each feature, determine the split points
    if depth == 2 or len(set(label)) == 1:
       node.split_point = -1.0
       node.split_dim = -1
       node.label = self.get_max_label(label)
      #  print("Node complete: ")
      #  print(node.split_dim)
      #  print(node.split_point)
      #  print(node.label)

       return
    split_points = []
    dims = []
    for dim_id, feature in enumerate(zip(*data)):
        # *data unpacks the list
        # zip recombines lists in an element-position manner
        # print(type(feature))
        feature_list = list(feature)
        feature_list.sort()
        for i in range(1, len(feature_list)):
            split_points.append((feature_list[i-1] + feature_list[i]) / 2)
            dims.append(dim_id)
    # print(split_points)

        # For each split point, calculate information gain on splitting the dataset at that attribute and split point
    info_needed = []
    for j in range(len(split_points)):
        # if the following method calculates Info_A, then the point with the greatest info gain is the one with the least Info_A value
        info_needed.append((self.split_info(data, label, dims[j], split_points[j]), dims[j]))
            # Split the dataset on the best split point

  
    min_inf_needed = float('inf')
    min_inf_lab = []
    for tple in info_needed:
      if tple[0] < min_inf_needed:
        # tple takes the form of (info_needed, dim)
        min_inf_needed = tple[0]
        min_inf_lab.append(tple[1])

    for i in range(len(info_needed)):
      tpl = info_needed[i]
      element = tpl[0]
      if min_inf_needed == element:
        best_split = split_points[i]
        break
    
    # best_split = min_inf_needed
    if min_inf_lab:
      best_dim = min_inf_lab[-1] 
    else:
       return      
    # info_needed = sorted(info_needed, key=lambda x: x[0])
    # The best split is at the beginning
    # best_split, best_dim = info_needed[0]

    # Assign the calculated values to the node
    node.split_dim = best_dim
    node.split_point = best_split
    # print(label)
    node.label = self.get_max_label(label)

    # Make a function that determines if a node is a leaf node
    # Remember, a leaf node is one that either only has data from one class or one that is at the maximum depth
    if not depth == 2 and len(set(label)) != 1:
       left_child = Node()
       right_child = Node()
       left_data = []
       right_data = []
       left_label = []
       right_label = []

       for i in range(len(data)):
          if data[i][node.split_dim] <= node.split_point:   
            left_data.append(data[i])
            left_label.append(label[i])

          else:
            right_data.append(data[i])
            right_label.append(label[i])

        # Figure out a good way to remember which labels go where
       if left_data:
        node.left = left_child
        self.split_node(left_child, left_data, left_label, depth=depth+1)
       if right_data:
        node.right = right_child
        self.split_node(right_child, right_data, right_label, depth=depth+1)
        # Splitting can be thought of as creating two new nodes, assigning data points to each node, and then assigning 
                # Repeat procedure on two splits of data
      #  print("Node complete: ")
      #  print(node.split_dim)
      #  print(node.split_point)
      #  print(node.label)


  def split_info(self, data: List[List[float]], label: List[int], split_dim: int, split_point: float) -> float:
    """
    Compute the information needed to classify a dataset if it's split
    with the given splitting dimension and splitting point, i.e. Info_A in the slides.

    Parameters:
    data (List[List]): A nested list representing the dataset.
    label (List): A list containing the class labels for each data point.
    split_dim (int): The dimension/attribute index to split the data on.
    split_point (float): The value at which the data should be split along the given dimension.

    Returns:
    float: The calculated Info_A value for the given split. Do NOT round this value
    """
    left_data = []
    right_data = []
    left_label = []
    right_label = []

    for i in range(len(data)):
        if data[i][split_dim] <= split_point:   
            left_data.append(data[i])
            left_label.append(label[i])

        else:
            right_data.append(data[i])
            right_label.append(label[i])

    card_Dl = len(left_data)
    card_Dr = len(right_data)    
    
    left_dict = {}
    for ci in left_label:
       left_dict[ci] = 1 + left_dict.get(ci, 0)

    right_dict = {}
    for ci in right_label:
       right_dict[ci] = 1 + right_dict.get(ci, 0)

    pi_l = []
    for key, value in left_dict.items():
       pi_l.append(value / card_Dl)

    pi_r = []
    for key, value in right_dict.items():
       pi_r.append(value / card_Dr)

    info_Dl = 0
    info_Dr = 0
    for i in range(len(pi_l)):
        info_Dl -= (pi_l[i] * math.log2(pi_l[i]))
    for i in range(len(pi_r)):
       info_Dr -= (pi_r[i] * math.log2(pi_r[i]))

    Info_A = (card_Dl / len(data)) * info_Dl
    Info_A += (card_Dr / len(data)) * info_Dr
    return Info_A
      

  def fit(self, train_data: List[List[float]], train_label: List[int]) -> None:

    self.root = Node()

    """
    Fit the decision tree model using the provided training data and labels.

    Parameters:
    train_data (List[List[float]]): A nested list of floating point numbers representing the training data.
    train_label (List[int]): A list of integers representing the class labels for each data point in the training set.

    This method initializes the decision tree model by creating the root node. It then builds the decision tree starting 
    from the root node
    
    It is important to note that for tree structure evaluation, the autograder for this assignment
    first calls this method. It then performs tree traversals starting from the root node in order to check whether 
    the tree structure is correct. 
    
    So it is very important to ensure that self.root is assigned correctly to the root node
    
    It is best to use a different method (such as in the example above) to build the decision tree.
    """
    self.split_node(node=self.root, data=train_data, label=train_label, depth=0)
    # print(self.root)
    # print("Node complete: ")
    # print(self.root.split_dim)
    # print(self.root.split_point)
    # print(self.root.label)
    # print('test')
    


  def classify_datapoint(self, node, datapoint):
    # If at leaf node, return the label:
    if node.left is None and node.right is None:
      return node.label

    # Decide the direction to traverse based on the split dimension and split point
    if datapoint[node.split_dim] <= node.split_point:
      return self.classify_datapoint(node.left, datapoint)
    else:
      return self.classify_datapoint(node.right, datapoint)
    

  def classify(self, train_data: List[List[float]], train_label: List[int], test_data: List[List[float]]) -> List[int]:
    """
    Classify the test data using a decision tree model built from the provided training data and labels.
    This method first fits the decision tree model using the provided training data and labels by calling the
    'fit()' method.

    Parameters:
    train_data (List[List[float]]): A nested list of floating point numbers representing the training data.
    train_label (List[int]): A list of integers representing the class labels for each data point in the training set.
    test_data (List[List[float]]): A nested list of floating point numbers representing the test data.

    Returns:
    List[int]: A list of integer predictions, which are the label predictions for the test data after fitting
               the train data and labels to a decision tree.
    """
    # Build decision tree using fit() method
    self.fit(train_data=train_data, train_label=train_label)
    # Traverse through the decision tree to predict the labels
    predictions = []
    for datapoint in test_data:
      predictions.append(self.classify_datapoint(self.root, datapoint))

    return predictions


  """
  Students are encouraged to implement as many additional methods as they find helpful in completing
  the assignment. These methods can be implemented either as class methods of the Solution class or as
  global methods, depending on design preferences.

  For instance, one essential method that must be implemented is a method to build out the decision tree recursively.
  """


# testing_label = [1, 1, 1, 3, 1, 3, 3, 3]
# testing_data = [
#    [1.0, 1.0],
#    [1.0, 2.0],
#    [2.0, 1.0],
#    [2.0, 2.0],
#    [3.0, 1.0],
#    [3.0, 2.0],
#    [3.0, 3.0],
#    [4.5, 3.0]
#    ]

# testing_classification_needed = [
#   [1.0, 2.2],
#   [4.5, 1.0]
# ]
# test = Solution()
# print(test.split_info(data=testing_data, label=testing_label, split_dim=0, split_point=1.5))
# test.fit(train_data=testing_data, train_label=testing_label)
# print(test.classify(train_data=testing_data, train_label=testing_label, test_data=testing_classification_needed))

# testing2_labels = [1, 1, 1, 3, 1, 3, 3, 3]
# testing2_data = [
#   [1.0, 1.0],

# ]
