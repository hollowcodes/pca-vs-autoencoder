
from sklearn.decomposition import PCA
import json
import numpy as np
import matplotlib.pyplot as plt



def save_to_json(file_: str="", data: list=[]) -> None:
    """ save json-object to json-file
    :param str file_: path to json-file
    :param list data: json-object
    """

    with open(file_, "w+") as f:
        json.dump(data, f, indent=4)
    

def load_from_json(file_: str="") -> list:
    """ load dataset from json-file
    :param str file_: path to json-file
    :return list: dataset
    """
    
    with open(file_, "r") as f:
        data = json.load(f)

    return data


def create_projections(dataset: list=[]) -> list:
    # remove labels
    data = []
    for sample in dataset:
        data.append(sample[0])

    # create PCA 2D projection of the dataset
    pca = PCA(2)
    projected_data = pca.fit_transform(data)

    # add labels to the projections
    final_projections = []
    for i in range(len(dataset)):
        label = dataset[i][1]
        final_projections.append([projected_data[i].tolist(), label])

    return final_projections


# load datasets
train_set = load_from_json(file_="../datasets/iris-dataset/iris_train.json")
test_set = load_from_json(file_="../datasets/iris-dataset/iris_test.json")

# apply PCA to create 2d projections
train_projections = create_projections(dataset=train_set)
test_projections = create_projections(dataset=test_set)

# save projections
save_to_json(file_="iris_train_reduced.json", data=train_projections)
save_to_json(file_="iris_test_reduced.json", data=test_projections)
