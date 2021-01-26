import os
import datetime
import numpy as np
import test as ts

# config variables
# model_name = {   mobilenet}

# weights = {none | imagenet}
weights = "imagenet"
# include_top = {True | False}
include_top = False
# test_size = 0.1 ~ 0.9
test_size = 0.10
seed = 10
# classifier_name = {svm, lda, knn, reg}
classifier_name = "lda"
# already done = mobilenet | inceptionresnetv2 | xception | inceptionv3 | resnet50 | vgg16 | vgg19
model_name = "vgg19"

# cross-validation
# outer_shuffle = {True | False}
outer_shuffle = True
# outer_n_splits = {3, 5, 10}
outer_n_splits = 10

# inner_shuffle = {True | False}
inner_shuffle = True
# inner_n_splits = {3, 5, 10}
inner_n_splits = 3

# GridSearchCV
# refit = {True | False}
Grid_refit = True
# Grid_n_jobs = {1:4, -1}
Grid_n_jobs = 3


# define search spaces
knnspace = {
    "n_neighbors": np.arange(1, 2, 2),
    "metric": [
        "euclidean",
        "cityblock",
    ],  # “manhattan”, “chebyshev”, “minkowski”, “wminkowski”, “seuclidean”, “mahalanobis”],
    # "leaf_size": np.arange(1, 30, 2),
    # "p": [1, 2],
    # "weights": ["distance", "uniform"],
    # "algorithm":["auto", "ball_tree", "kd_tree", "brute"],
}

svmspace = {
    "probability": [True],
    "kernel": ["rbf", "linear", "poly", "sigmoid", "precomputed"],
    "decision_function_shape": ["ovr", "ovo"],
    "C": [0.1],  # , 1, 10, 100, 1000],
    "gamma": [1],  # , 0.1, 0.01, 0.001, 0.0001],
    "random_state": [seed],
}

ldaspace = {
    "solver": ["svd", "lsqr", "eigen"],
    "shrinkage": ["auto", "None"],
    # "n_components":
    # "priors":
    # "store_covariance":
    # "tol":
}

# dict_keys(['C', 'break_ties', 'cache_size', 'class_weight', 'coef0', 'decision_function_shape', 'degree', 'gamma', 'kernel', 'max_iter', 'probability', 'random_state', 'shrinking', 'tol', 'verbose'])
regspace = {
    "gamma": [0.01, 0.1],  # , 0.1, 0.01, 0.001, 0.0001],
}

# creating output paths
project_path = os.getcwd()

os.chdir(project_path)
os.system("mkdir output")

train_path = project_path + "/results"

os.chdir(project_path + "/output/")
os.system("mkdir " + model_name)
os.chdir(project_path)

features_path = project_path + "/output/" + model_name + "/features.pickle"
labels_path = project_path + "/output/" + model_name + "/labels.pickle"
regresults = project_path + "/output/" + model_name + "/regresults.txt"
svmresults = project_path + "/output/" + model_name + "/svmresults.txt"
knnresults = project_path + "/output/" + model_name + "/knnresults.txt"
ldaresults = project_path + "/output/" + model_name + "/ldaresults.txt"
results = project_path + "/output/" + model_name + "/results.txt"

model_path = project_path + "/output/" + model_name + "/model"
classifier_path = project_path + "/output/" + model_name + "/classifier.pickle"


datefile = (
    datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    + "-"
    + classifier_name
    + "-"
    + model_name
)

