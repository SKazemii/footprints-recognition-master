# filter warnings
import warnings

warnings.filterwarnings("ignore")


# organize imports

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

import numpy as np
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# importing setting
import confiVariables as cfg

# deleting the last results
if os.path.exists(cfg.results):
    os.remove(cfg.results)
elif os.path.exists(cfg.classifier_path):
    os.remove(cfg.classifier_path)
else:
    print("[INFO] The result file does not exist for deleting")


# import features and labels
with open(cfg.features_path, "rb") as handle:
    features = pickle.load(handle)

with open(cfg.labels_path, "rb") as handle:
    labels = pickle.load(handle)


# verify the shape of features and labels
print("[INFO] data shape: {}".format(features.shape))
print("[INFO] labels shape: {}".format(labels.shape))

print("[INFO] training started...")
# split the training and testing data
(trainData, testData, trainLabels, testLabels) = train_test_split(
    np.array(features), np.array(labels), test_size=cfg.test_size, random_state=cfg.seed
)

print("[INFO] splitted data into train and test set data...")
print("[INFO] test data   : {}".format(testData.shape))
print("[INFO] test labels : {}".format(testLabels.shape))

print(
    "[INFO] splitting the training dataset into {} folds\n\n".format(cfg.outer_n_splits)
)
cv_outer = StratifiedKFold(
    n_splits=cfg.outer_n_splits, shuffle=cfg.outer_shuffle, random_state=cfg.seed
)


outer_results_rank_1 = list()
outer_results_rank_2 = list()
outer_results_rank_5 = list()
cv = 1
for train_ix, test_ix in cv_outer.split(trainData, trainLabels):

    # split data
    trainingData, evaluationData = trainData[train_ix, :], trainData[test_ix, :]
    trainingLabels, evaluationLabels = trainLabels[train_ix], trainLabels[test_ix]

    print("[INFO] training data shape : {}".format(trainingData.shape))
    print("[INFO] training labels shape : {}\n\n".format(trainingLabels.shape))

    print("[INFO] evaluation data shape : {}".format(evaluationData.shape))
    print("[INFO] evaluation labels shape : {}\n\n".format(evaluationLabels.shape))

    # configure the inner cross-validation procedure
    cv_inner = StratifiedKFold(
        n_splits=cfg.inner_n_splits, shuffle=cfg.inner_shuffle, random_state=cfg.seed
    )

    if cfg.classifier_name == "knn":
        # use kNN as the model
        print("[INFO] creating model...")
        model = KNeighborsClassifier()

        # define search space
        space = cfg.knnspace

        # define search
        search = GridSearchCV(
            model,
            space,
            scoring="accuracy",
            n_jobs=cfg.Grid_n_jobs,
            cv=cv_inner,
            refit=cfg.Grid_refit,
        )
        # execute search
        result = search.fit(trainingData, trainingLabels)

        best_model = result.best_estimator_

    elif cfg.classifier_name == "svm":
        # use SVM as the model
        print("[INFO] creating model...")
        model = svm.SVC()

        # define search space
        space = cfg.svmspace

        # define search
        search = GridSearchCV(
            model,
            space,
            scoring="accuracy",
            n_jobs=cfg.Grid_n_jobs,
            cv=cv_inner,
            refit=cfg.Grid_refit,
        )
        # execute search
        result = search.fit(trainingData, trainingLabels)

        best_model = result.best_estimator_

    elif cfg.classifier_name == "lda":
        # use LDA as the model
        print("[INFO] creating model...")
        model = LinearDiscriminantAnalysis()

        # define search space
        space = cfg.ldaspace

        # define search
        search = GridSearchCV(
            model,
            space,
            scoring="accuracy",
            n_jobs=cfg.Grid_n_jobs,
            cv=cv_inner,
            refit=cfg.Grid_refit,
        )
        # execute search
        result = search.fit(trainingData, trainingLabels)

        best_model = result.best_estimator_

    elif cfg.classifier_name == "reg":
        # use logistic regression as the model
        print("[INFO] creating model...")
        model = LogisticRegression()

        # define search space
        space = cfg.regspace

        # define search
        search = GridSearchCV(
            model,
            space,
            scoring="accuracy",
            n_jobs=cfg.Grid_n_jobs,
            cv=cv_inner,
            refit=cfg.Grid_refit,
        )
        # execute search
        result = search.fit(trainingData, trainingLabels)

        best_model = result.best_estimator_

    else:
        print("[ERROR] could not find the classifier")

    print("[INFO] evaluating model...")
    if cv == 1:
        f = open(cfg.results, "a")

        f.write("##################################################\n")
        f.write("################## the settings ##################\n")
        f.write("##################################################\n\n")
        f.write("Grid_n_jobs:           {}\n".format(cfg.Grid_n_jobs))
        f.write("space:                 {}\n".format(space))
        f.write("inner_n_splits:        {}\n".format(cfg.inner_n_splits))
        f.write("outer_n_splits:        {}\n".format(cfg.outer_n_splits))
        f.write("features.shape:        {}\n".format(features.shape))
        f.write("trainingData.shape:    {}\n".format(trainingData.shape))
        f.write("evaluationData.shape:  {}\n".format(evaluationData.shape))
        f.write("testData.shape:        {}\n".format(testData.shape))
        f.write("test size:             {}\n\n\n".format(cfg.test_size))

    rank_1 = 0
    rank_2 = 0
    rank_5 = 0
    # loop over test data
    for (label, feature) in zip(evaluationLabels, evaluationData):
        # predict the probability of each class label and
        # take the top-5 class labels
        predictions = best_model.predict_proba(np.atleast_2d(feature))[0]
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
    rank_1 = (rank_1 / float(len(evaluationLabels))) * 100
    rank_2 = (rank_2 / float(len(evaluationLabels))) * 100
    rank_5 = (rank_5 / float(len(evaluationLabels))) * 100

    # write the accuracies to file
    f.write("##################################################\n")
    f.write("CV: {} ############################################\n".format(cv))
    f.write("##################################################\n\n")

    cv = cv + 1
    f.write("Rank-1: {:.2f}%\n".format(rank_1))
    f.write("Rank-2: {:.2f}%\n".format(rank_2))
    f.write("Rank-5: {:.2f}%\n\n".format(rank_5))

    # # store the result
    outer_results_rank_1.append(rank_1)
    outer_results_rank_2.append(rank_2)
    outer_results_rank_5.append(rank_5)


# dump classifier to file
#    print("[INFO] saving best_model...")
#    pickle.dump(best_model, open(cfg.classifier_path, "wb"))


# write the accuracies of training set to file
f.write("\n\n##################################################\n")
f.write("###### summarize the estimated performance #######\n")
f.write("##### of the best model on the training set ######\n")
f.write("##################################################\n\n")

f.write(
    "[mean,std] Accuracy Rank-1: [{:.2f}, {:.2f}]\n".format(
        np.mean(outer_results_rank_1), np.std(outer_results_rank_1)
    )
)
f.write(
    "[mean,std] Accuracy Rank-2: [{:.2f}, {:.2f}]\n".format(
        np.mean(outer_results_rank_2), np.std(outer_results_rank_2)
    )
)
f.write(
    "[mean,std] Accuracy Rank-5: [{:.2f}, {:.2f}]\n".format(
        np.mean(outer_results_rank_5), np.std(outer_results_rank_5)
    )
)


# write the accuracies of test set to file
f.write("\n\n##################################################\n")
f.write("###### summarize the estimated performance #######\n")
f.write("####### of the best model on the test set ########\n")
f.write("##################################################\n\n")


rank_1 = 0
rank_2 = 0
rank_5 = 0
for (label, feature) in zip(testLabels, testData):
    # predict the probability of each class label and
    # take the top-5 class labels
    predictions = best_model.predict_proba(np.atleast_2d(feature))[0]
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

f.write("Rank-1: {:.2f}%\n".format(rank_1))
f.write("Rank-2: {:.2f}%\n".format(rank_2))
f.write("Rank-5: {:.2f}%\n\n".format(rank_5))

f.write("best parameters are:\n {}\n\n".format(result.best_params_))


# evaluate the model of test data
preds = best_model.predict(testData)

# write the classification report to file
f.write("{}\n".format(classification_report(testLabels, preds)))
f.close()


# showing best results along with the best parameters
print("[INFO] Grid search best parameters: {}".format(result.best_params_))
print("-------------------------------------------------------------------\n\n")
print("[INFO] Accuracy Rank-1: {:.3f}%".format(rank_1))
print("[INFO] Accuracy Rank-2: {:.3f}%".format(rank_2))
print("[INFO] Accuracy Rank-5: {:.3f}%".format(rank_5))

# dump classifier to file
print("[INFO] saving best_model...")
pickle.dump(best_model, open(cfg.classifier_path, "wb"))

os.chdir(cfg.project_path + "/output/" + cfg.model_name)
if os.path.exists(cfg.project_path + "/output/" + cfg.model_name + "/classifiers/"):
    print("[INFO] The classifiers folder exists")
else:
    os.system("mkdir " + "classifiers")


os.system(
    "mv -f "
    + cfg.classifier_path
    + " "
    + cfg.project_path
    + "/output/"
    + cfg.model_name
    + "/classifiers/"
    + cfg.datefile
    + ".pickle"
)


os.chdir(cfg.project_path + "/output/" + cfg.model_name)
if os.path.exists(cfg.project_path + "/output/" + cfg.model_name + "/history/"):
    print("[INFO] The history folder exists")
else:
    os.system("mkdir " + "history")


os.system(
    "mv -f "
    + cfg.results
    + " "
    + cfg.project_path
    + "/output/"
    + cfg.model_name
    + "/history/"
    + cfg.datefile
    + ".txt"
)

# display the confusion matrix
# print("[INFO] confusion matrix")

# # get the list of training lables
# labels = sorted(list(os.listdir(cfg.train_path)))

# # plot the confusion matrix
# cm = confusion_matrix(testLabels, preds)
# sns.heatmap(cm, annot=True, cmap="Set2")
# plt.show()

