# global model_name
# global classifier_name
import confiVariables as cfg

for mdl in [
    "mobilenet",
    "inceptionresnetv2",
    "xception",
    "inceptionv3",
    "resnet50",
    "vgg16",
    "vgg19",
]:
    for clss in ["svm", "lda", "knn", "reg"]:
        cfg.model_name = mdl
        cfg.classifier_name = clss
        # exec(open("train.py").read())
