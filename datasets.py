from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR100
from tqdm import tqdm
import pandas as pd
import numpy as np
import os


class CifarDataset(CIFAR100):
    def __init__(self, root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 test_transform=None,
                 target_test_transform=None,
                 download=False):
        """Dataset class representing cifar dataset

        """

        super(CifarDataset, self).__init__(root,
                                           train=train,
                                           transform=transform,
                                           target_transform=target_transform,
                                           download=download)
        self.target_test_transform = target_test_transform
        self.test_transform = test_transform
        self.Data = []
        self.Labels = []
        self.train = train


    def data_intialization(self):
        self.df = pd.DataFrame(self.index_subset())
        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.assign(id=self.df.index.values)
        # Convert arbitrary class names of dataset to ordered 0-(num_speakers - 1) integers
        self.unique_labels = sorted(self.df['label'].unique())
        self.labels_to_id = {self.unique_labels[i]: i for i in range(self.num_classes())}
        self.df = self.df.assign(label_id=self.df['label'].apply(lambda c: self.labels_to_id[c]))
        # Create dicts
        self.datasetid_to_images = self.df.to_dict()['image']
        self.datasetid_to_labels_id = self.df.to_dict()['label_id']

    def __getitem__(self, index):
        img, target = Image.fromarray(self.datasetid_to_images[index]), self.datasetid_to_labels_id[index]

        if self.transform:
            img = self.transform(img)
        elif self.test_transform:
            img = self.test_transform(img)

        if self.target_transform:
            target = self.target_transform(target)
        elif self.target_test_transform:
            target = self.target_test_transform(target)

        return img, target

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['label'].unique())

    def index_subset(self):
        """Index a subset by looping through all of its files and recording relevant information.

        # Arguments
            subset: Name of the subset

        # Returns
            A list of dicts containing information about all the image files in a particular subset of the
            miniImageNet dataset
        """
        images = []
        print('Indexing {}...'.format('Train' if self.train else 'Test'))
        # Quick first pass to find total for tqdm bar
        subset_len = len(self.Data)

        progress_bar = tqdm(total=subset_len)
        for i in range(subset_len):
            progress_bar.update(1)
            images.append({
                'label': self.Labels[i],
                'image': self.Data[i]
            })
        progress_bar.close()
        return images

    def concatenate(self, datas, labels):
        con_data = datas[0]
        con_label = labels[0]
        for i in range(1, len(datas)):
            con_data = np.concatenate((con_data, datas[i]), axis=0)
            con_label = np.concatenate((con_label, labels[i]), axis=0)
        return con_data, con_label

    # def getTestData(self, classes, exemplar_set, n_shot=-1):
    #     datas, labels = [], []
    #     # if len(exemplar_set) != 0:
    #     #     datas.append([exemplar[:50] for exemplar in exemplar_set])
    #     #     length = len(datas[0])
    #     #     labels.append([np.full((length), label) for label in range(len(exemplar_set))])
    #
    #     for label in range(classes[0], classes[1]):
    #         data = self.data[np.array(self.targets) == label]
    #         if n_shot>0:
    #             data = data[:n_shot]
    #         datas.append(data)
    #         labels.append(np.full((data.shape[0]), label))
    #
    #     datas, labels = self.concatenate(datas, labels)
    #     self.Data = datas if self.Data == [] else np.concatenate((self.Data, datas), axis=0)
    #     self.Labels = labels if self.Labels == [] else np.concatenate((self.Labels, labels), axis=0)
    #     print("the size of test set is %s" % (str(self.Data.shape)))
    #     print("the size of test label is %s" % str(self.Labels.shape))
    #     self.data_intialization()

    def getTrainData(self, classes, exemplar_set, n_shot = -1):

        datas, labels = [], []
        for label in range(len(exemplar_set)):
            exemplar = exemplar_set[label]
            datas.append(exemplar)
            labels.append(np.full(len(exemplar), label))

        for label in range(classes[0], classes[1]):
            data = self.data[np.array(self.targets) == label]
            if n_shot>0:
                data = data[:n_shot]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        self.Data, self.Labels = self.concatenate(datas, labels)
        print("the size of set is %s" % (str(self.Data.shape)))
        print("the size of label is %s" % str(self.Labels.shape))
        self.data_intialization()

    def get_image_class(self, label):
        return self.data[np.array(self.targets) == label]
