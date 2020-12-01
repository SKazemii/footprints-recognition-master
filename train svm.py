# filter warnings
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


# organize imports

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# importing setting
import confiVariables as cfg


# import features and labels
with open(cfg.features_path, "rb") as handle:
    features = pickle.load(handle)

with open(cfg.labels_path, "rb") as handle:
    labels = pickle.load(handle)


# verify the shape of features and labels
print("[INFO] features shape: {}".format(features.shape))
print("[INFO] labels shape: {}".format(labels.shape))

print("[INFO] training started...")
# split the training and testing data
(trainData, testData, trainLabels, testLabels) = train_test_split(
    np.array(features), np.array(labels), test_size=cfg.test_size, random_state=cfg.seed
)


print("[INFO] splitted train and test data...")
print("[INFO] train data  : {}".format(trainData.shape))
print("[INFO] test data   : {}".format(testData.shape))
print("[INFO] train labels: {}".format(trainLabels.shape))
print("[INFO] test labels : {}".format(testLabels.shape))

# use logistic regression as the model
print("[INFO] creating model...")

model = svm.SVC(kernel="rbf", decision_function_shape="ovr", probability=True)
model.fit(trainData, trainLabels)


# use rank-1 and rank-5 predictions
print("[INFO] evaluating model...")
f = open(cfg.svmresults, "w")
rank_1 = 0
rank_2 = 0
rank_5 = 0

# loop over test data
for (label, features) in zip(testLabels, testData):
    # predict the probability of each class label and
    # take the top-5 class labels
    predictions = model.predict_proba(np.atleast_2d(features))[0]
    predictions = np.argsort(predictions)[::-1][:5]

    # rank-1 prediction increment
    if label == predictions[0]:
        rank_1 += 1

    # rank-2 prediction increment
    if label in predictions[:2]:
        rank_2 += 1

    # rank-5 prediction increment
    if label in predictions:
        rank_5 += 1

# convert accuracies to percentages
rank_1 = (rank_1 / float(len(testLabels))) * 100
rank_2 = (rank_2 / float(len(testLabels))) * 100
rank_5 = (rank_5 / float(len(testLabels))) * 100

# write the accuracies to file
f.write("Rank-1: {:.2f}%\n".format(rank_1))
f.write("Rank-2: {:.2f}%\n".format(rank_2))
f.write("Rank-5: {:.2f}%\n\n".format(rank_5))

# evaluate the model of test data
preds = model.predict(testData)

# write the cl˜assification report to file
f.write("{}\n".format(classification_report(testLabels, preds)))
f.close()

# dump classifier to file
print("[INFO] saving model...")
pickle.dump(model, open(cfg.classifier_path, "wb"))

# display the confusion matrix
print("[INFO] confusion matrix")

# get the list of training lables
labels = sorted(list(os.listdir(cfg.train_path)))

# plot the confusion matrix
cm = confusion_matrix(testLabels, preds)
sns.heatmap(cm, annot=True, cmap="Set2")
plt.show()
