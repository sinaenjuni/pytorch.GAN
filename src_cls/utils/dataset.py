import torch
from torch.utils.data import Subset, ConcatDataset
import matplotlib.pyplot as plt
import seaborn as sns


def sliceDataset(dataset, class_index:dict, labels:torch.tensor, lratio:list):
    transformed_dataset = []
    subdata_count = {'class':[], 'count': []}

    for i, (name, idx) in enumerate(class_index.items()):
        target_label_indeces = torch.where(labels == idx)[0].numpy()
        print(i, target_label_indeces)
        data_subset = Subset(dataset, target_label_indeces)

        # if c != 0:
        ratio = len(data_subset) * (1 * lratio[i])
        ratio = int(ratio)

        data_subset = Subset(data_subset, range(ratio))
        print(len(data_subset))
        transformed_dataset += [data_subset]
        subdata_count['class'] += [name]
        subdata_count['count'] += [len(data_subset)]

    transformed_dataset = ConcatDataset(transformed_dataset)

    fig = plt.figure()
    sns.barplot(
        data=subdata_count,
        x="class",
        y="count"
    )

    return transformed_dataset, fig