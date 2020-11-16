# filter warnings
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

# keras imports
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.applications.xception import Xception, preprocess_input
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.models import model_from_json
from keras.layers import Input, Dense, GlobalAveragePooling2D

# other imports
from sklearn.preprocessing import LabelEncoder
import numpy as np
import glob
import cv2
import h5py
import os
import json
import datetime
import time
import pickle

# TODO: please cheak it
# for error "Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized."
# Answer might be found at: https://github.com/dmlc/xgboost/issues/1715
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# importing setting
import confiVariables as cfg


# create the pretrained models
# check for pretrained weight usage or not    output=base_model.get_layer("fc1").output
# check for top layers to be included or not  outputs=modelvvg16.layers[-2].output
if cfg.model_name == "vgg16":
    base_model = VGG16(weights=cfg.weights)
    model = Model(input=base_model.input, output=base_model.get_layer("fc1").output)
    image_size = (224, 224)
elif cfg.model_name == "test":
    base_model = VGG16(weights=cfg.weights)
    model = Model(input=base_model.input, output=base_model.get_layer("fc1").output)
    image_size = (224, 224)
elif cfg.model_name == "vgg19":
    base_model = VGG19(weights=cfg.weights)
    base_model.summary()
    model = Model(input=base_model.input, output=base_model.get_layer("fc1").output)
    image_size = (224, 224)
elif cfg.model_name == "resnet50":
    base_model = ResNet50(
        input_tensor=Input(shape=(224, 224, 3)),
        include_top=cfg.include_top,
        weights=cfg.weights,
    )
    base_model.summary()
    x = base_model.output
    predictions = GlobalAveragePooling2D()(x)

    model = Model(input=base_model.input, outputs=predictions)
    model.summary()

    image_size = (224, 224)
elif cfg.model_name == "inceptionv3":
    base_model = InceptionV3(
        include_top=cfg.include_top,
        weights=cfg.weights,
        input_tensor=Input(shape=(299, 299, 3)),
    )
    # base_model.summary()
    # add a global spatial average pooling layer
    x = base_model.output
    predictions = GlobalAveragePooling2D()(x)

    model = Model(input=base_model.input, outputs=predictions)
    image_size = (299, 299)
elif cfg.model_name == "inceptionresnetv2":
    base_model = InceptionResNetV2(
        include_top=cfg.include_top,
        weights=cfg.weights,
        input_tensor=Input(shape=(299, 299, 3)),
    )
    x = base_model.output
    predictions = GlobalAveragePooling2D()(x)
    model = Model(input=base_model.input, output=predictions)
    image_size = (299, 299)
elif cfg.model_name == "mobilenet":
    base_model = MobileNet(
        include_top=cfg.include_top,
        weights=cfg.weights,
        input_tensor=Input(shape=(224, 224, 3)),
        input_shape=(224, 224, 3),
    )
    x = base_model.output
    predictions = GlobalAveragePooling2D()(x)
    model = Model(input=base_model.input, output=predictions)
    image_size = (224, 224)

elif cfg.model_name == "xception":
    base_model = Xception(weights=cfg.weights)
    model = Model(
        input=base_model.input, output=base_model.get_layer("avg_pool").output
    )
    model.summary()
    image_size = (299, 299)
else:
    base_model = None

print("[INFO] successfully loaded base model and model...")

# path to training dataset
train_labels = os.listdir(cfg.train_path)

# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
le.fit([tl for tl in train_labels])

# variables to hold features and labels
features = []
labels = []

# just for test
if cfg.model_name == "test":
    train_labels = ["13"]


# loop over all the labels in the folder
for i, label in enumerate(train_labels):
    cur_path = cfg.train_path + "/" + label
    for image_path in glob.glob(cur_path + "/*.jpeg"):
        img = image.load_img(image_path, target_size=image_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature = model.predict(x)
        flat = feature.flatten()
        features.append(flat)
        labels.append(label)
    print("[INFO] completed label - " + label)

# encode the labels using LabelEncoder
le = LabelEncoder()
le_labels = le.fit_transform(labels)

# get the shape of training labels
print("[STATUS] extracted labels shape: {}".format(le_labels.shape))
print("[STATUS] extracted features shape: {}".format(features[1].shape))

# save features and labels
with open(cfg.features_path, "wb") as handle:
    pickle.dump(np.array(features), handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(cfg.labels_path, "wb") as handle:
    pickle.dump(np.array(le_labels), handle, protocol=pickle.HIGHEST_PROTOCOL)


# save model and weights
model_json = model.to_json()
with open(cfg.model_path + str(cfg.test_size * 100)[0:2] + ".json", "w") as json_file:
    json_file.write(model_json)

# save weights
model.save_weights(cfg.model_path + str(cfg.test_size * 100)[0:2] + ".h5")
print("[STATUS] saved model and weights to disk..")
print("[STATUS] features and labels saved..")

# end time
end = time.time()
print(
    "[STATUS] end time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
)
