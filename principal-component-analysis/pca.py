
from sklearn.decomposition import PCA
import json
import numpy as np
import matplotlib.pyplot as plt


# load dataset
with open("../datasets/iris-dataset/iris_train.json", "r") as f:
    dataset = json.load(f)

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


# save projections
with open("iris_reduced.json", "w+") as f:
    json.dump(final_projections, f, indent=4)


"""cluster1, cluster2, cluster3 = [], [], []
for i in range(len(dataset)):
    if dataset[i][1] == [1, 0, 0]:
        cluster1.append(projected_data[i])

    elif dataset[i][1] == [0, 1, 0]:
        cluster2.append(projected_data[i])

    elif dataset[i][1] == [0, 0, 1]:
        cluster3.append(projected_data[i])

cluster1 = list(zip(*cluster1))
plt.scatter(cluster1[0], cluster1[1], c="b")

cluster2 = list(zip(*cluster2))
plt.scatter(cluster2[0], cluster2[1], c="g")

cluster3 = list(zip(*cluster3))
plt.scatter(cluster3[0], cluster3[1], c="orange")

plt.show()"""