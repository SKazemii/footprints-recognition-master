# organize imports
import os
import glob
import datetime
import pandas as pd


# get the input and output path
input_path = "/Users/saeedkazemi/Documents/Python/footprints-recognition-master/imgs"
output_path = (
    "/Users/saeedkazemi/Documents/Python/footprints-recognition-master/results"
)

csv_path = input_path + "/0output.csv"
print(csv_path)
csv_df = pd.read_csv(csv_path)
csv_df["fileName"] = list(range(1, 2659))

temp = list(range(1, 2659))
csv_df["fileName"] = [input_path + "/" + str(i) + ".jpeg" for i in temp]

# print the head of csv_df
print(csv_df.head())


# footprints class names
class_names = csv_df["subjects"].unique()
print("name of subjects = {}".format(class_names))


# get the class label limit
class_limit = class_names.__len__()
print("Number of subjects = {}".format(class_limit))

# change the current working directory
os.chdir(output_path)

# loop over the class labels
for x in range(1, class_limit):
    # create a folder for each subjects
    os.system("mkdir " + str(class_names[x]))


# loop over the images in the imgs folder
for ind in csv_df.index:
    original_path = csv_df["fileName"][ind]
    image = original_path.split("/")
    image = image[len(image) - 1]
    cur_path = (
        output_path
        + "/"
        + str(csv_df["subjects"][ind])
        + "/"
        + str(csv_df["group"][ind])
        + "-"
    )
    os.system("cp " + original_path + " " + cur_path + image)
