import pdb
import numpy as np
import code_for_hw3_part2 as hw3
from code_for_hw3_part2 import xval_learning_alg
from code_for_hw3_part2 import perceptron
from code_for_hw3_part2 import averaged_perceptron

#-------------------------------------------------------------------------------
# Auto Data
#-------------------------------------------------------------------------------

# Returns a list of dictionaries.  Keys are the column names, including mpg.
auto_data_all = hw3.load_auto_data('auto-mpg.tsv')

# The choice of feature processing for each feature, mpg is always raw and
# does not need to be specified.  Other choices are hw3.standard and hw3.one_hot.
# 'name' is not numeric and would need a different encoding.
features = [('cylinders', hw3.raw),
            ('displacement', hw3.raw),
            ('horsepower', hw3.raw),
            ('weight', hw3.raw),
            ('acceleration', hw3.raw),
            ## Drop model_year by default
            ## ('model_year', hw3.raw),
            ('origin', hw3.raw)]

# Construct the standard data and label arrays
auto_data, auto_labels = hw3.auto_data_and_labels(auto_data_all, features)
print('auto data and labels shape', auto_data.shape, auto_labels.shape)

if False:                               # set to True to see histograms
    import matplotlib.pyplot as plt
    for feat in range(auto_data.shape[0]):
        print('Feature', feat, features[feat][0])
        # Plot histograms in one window, different colors
        plt.hist(auto_data[feat,auto_labels[0,:] > 0])
        plt.hist(auto_data[feat,auto_labels[0,:] < 0])
        plt.show()
        # Plot histograms in two windows, different colors
        fig,(a1,a2) = plt.subplots(nrows=2)
        a1.hist(auto_data[feat,auto_labels[0,:] > 0])
        a2.hist(auto_data[feat,auto_labels[0,:] < 0])
        plt.show()

#-------------------------------------------------------------------------------
# Analyze auto data
#-------------------------------------------------------------------------------

# # Your code here to process the auto data
# params = {'T': 50}
#
# # Perform cross-validation and evaluate the perceptron classifier
# k = 10  # Number of folds for cross-validation
# accuracy = xval_learning_alg(perceptron, auto_data, auto_labels, k)
# # theta, theta_0 = perceptron(auto_data, auto_labels, params=params)
# print('Cross-validation accuracy:', accuracy)
# # print("Theta:", theta)
# # print("Theta_0:", theta_0)

#-------------------------------------------------------------------------------
# Review Data
#-------------------------------------------------------------------------------

# Returns lists of dictionaries.  Keys are the column names, 'sentiment' and 'text'.
# The train data has 10,000 examples
review_data = hw3.load_review_data('reviews.tsv')

# Lists texts of reviews and list of labels (1 or -1)
review_texts, review_label_list = zip(*((sample['text'], sample['sentiment']) for sample in review_data))

# The dictionary of all the words for "bag of words"
dictionary = hw3.bag_of_words(review_texts)

# The standard data arrays for the bag of words
review_bow_data = hw3.extract_bow_feature_vectors(review_texts, dictionary)
review_labels = hw3.rv(review_label_list)
print('review_bow_data and labels shape', review_bow_data.shape, review_labels.shape)

#-------------------------------------------------------------------------------
# Analyze review data
#-------------------------------------------------------------------------------

"""
# Your code here to process the review data
params = {'T': 10}  # You can adjust the value of T as needed
k=10  # Number of folds for cross-validation
accuracy = hw3.xval_learning_alg(hw3.perceptron, review_bow_data, review_labels, k)
print('Perceptron cross-validation accuracy:', accuracy)
theta, theta_0 = hw3.perceptron(review_bow_data, review_labels, params=params)
print("Perceptron Theta:", theta)
print("Perceptron Theta_0:", theta_0)
accuracy = hw3.xval_learning_alg(hw3.averaged_perceptron, review_bow_data, review_labels, k)
print('Averaged Perceptron cross-validation accuracy:', accuracy)
theta_avg, theta_0_avg = hw3.averaged_perceptron(review_bow_data, review_labels, params=params)
reverse_dictionary = hw3.reverse_dict(dictionary)

# Extract the weights corresponding to positive and negative predictions
positive_weights = theta_avg[:, 0]
negative_weights = -positive_weights

# Get the indices of the top 10 weights for positive and negative predictions
top_10_positive_indices = np.argsort(positive_weights)[-10:]
top_10_negative_indices = np.argsort(negative_weights)[-10:]

# Retrieve the corresponding words using the reverse dictionary
top_10_positive_words = [reverse_dictionary[idx] for idx in top_10_positive_indices]
top_10_negative_words = [reverse_dictionary[idx] for idx in top_10_negative_indices]

print("Top 10 positive words:", top_10_positive_words)
print("Top 10 negative words:", top_10_negative_words)
print("Averaged Perceptron Theta:", theta_avg)
print("Averaged Perceptron Theta_0:", theta_0_avg)
"""

#-------------------------------------------------------------------------------
# MNIST Data
#-------------------------------------------------------------------------------

"""
Returns a dictionary formatted as follows:
{
    0: {
        "images": [(m by n image), (m by n image), ...],
        "labels": [0, 0, ..., 0]
    },
    1: {...},
    ...
    9
}
Where labels range from 0 to 9 and (m, n) images are represented
by arrays of floats from 0 to 1
"""
mnist_data_all = hw3.load_mnist_data(range(10))

print('mnist_data_all loaded. shape of single images is', mnist_data_all[0]["images"][0].shape)

# HINT: change the [0] and [1] if you want to access different images
d0 = mnist_data_all[0]["images"]
d1 = mnist_data_all[1]["images"]
y0 = np.repeat(-1, len(d0)).reshape(1,-1)
y1 = np.repeat(1, len(d1)).reshape(1,-1)

# data goes into the feature computation functions
data = np.vstack((d0, d1))
# labels can directly go into the perceptron algorithm
labels = np.vstack((y0.T, y1.T)).T

def raw_mnist_features(x):
    """
    @param x (n_samples,m,n) array with values in (0,1)
    @return (m*n,n_samples) reshaped array where each entry is preserved
    """
    n_samples, m, n = x.shape
    return x.reshape((n_samples, m * n)).T
    # raise Exception("implement me!")

def row_average_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (m,n_samples) array where each entry is the average of a row
    """
    return np.mean(x, axis=1, keepdims=True)
    # raise Exception("modify me!")


def col_average_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (n,n_samples) array where each entry is the average of a column
    """
    return np.mean(x, axis=0, keepdims=True).T
    # raise Exception("modify me!")


def top_bottom_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (2,n_samples) array where the first entry of each column is the average of the
    top half of the image = rows 0 to floor(m/2) [exclusive]
    and the second entry is the average of the bottom half of the image
    = rows floor(m/2) [inclusive] to m
    """
    m = x.shape[0]
    return hw3.cv([np.mean(x[:m // 2, ]), np.mean(x[m // 2:, ])])
    # raise Exception("modify me!")

# use this function to evaluate accuracy
# acc = hw3.get_classification_accuracy(raw_mnist_features(data), labels)

#-------------------------------------------------------------------------------
# Analyze MNIST data
#-------------------------------------------------------------------------------

# Your code here to process the MNIST data
# Load MNIST data for labels 0 and 1
mnist_data_all = hw3.load_mnist_data([0, 1])

# HINT: change the [0] and [1] if you want to access different images
d0 = mnist_data_all[0]["images"]
d1 = mnist_data_all[1]["images"]
y0 = np.repeat(-1, len(d0)).reshape(1,-1)
y1 = np.repeat(1, len(d1)).reshape(1,-1)

# data goes into the feature computation functions
data = np.vstack((d0, d1))
# labels can directly go into the perceptron algorithm
labels = np.vstack((y0.T, y1.T)).T

# Evaluate accuracy using raw 0-1 features
acc_raw = hw3.get_classification_accuracycol_average_features((data), labels)
print("Accuracy using raw 0-1 features:", acc_raw)