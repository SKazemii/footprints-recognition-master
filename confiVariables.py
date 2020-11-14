import os

# config variables
# model_name = {inceptionv3 | xception | inceptionresnetv2 | mobilenet}
# already done = resnet50 | vgg16 | vgg19
model_name = "test"
# weights = {none | imagenet}
weights = "imagenet"
# include_top = {True | False}
include_top = False
# test_size = 0.1 ~ 0.9
test_size = 0.10


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
results = project_path + "/output/" + model_name + "/results.txt"
model_path = project_path + "/output/" + model_name + "/model"
