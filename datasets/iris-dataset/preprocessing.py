
import pandas as pd
import numpy as np
import json
import random

random.seed(34)

classes = {"Setosa": [1, 0, 0], "Versicolor": [0, 1, 0], "Virginica": [0, 0, 1]}
iris_df = pd.read_csv("iris_raw.csv")

max_features = [7.9, 4.4, 6.9, 2.5]

final_iris_dataset = []
for index, row in iris_df.iterrows():
    sample = row.tolist()
    label = classes[sample[-1]]

    sample = sample[:-1]
    sample = [(sample[i] / max_features[i]) for i in range(len(sample))]
    sample = [sample, label]
    final_iris_dataset.append(sample)

random.shuffle(final_iris_dataset)

train_size = int(0.8 * len(final_iris_dataset))

train_set = final_iris_dataset[:train_size]
test_set = final_iris_dataset[train_size+1:]

with open("iris_train.json", "w+") as f:
    json.dump(train_set, f, indent=4)

with open("iris_test.json", "w+") as f:
    json.dump(test_set, f, indent=4)