from torch.optim import Optimizer
from torch.nn import Module
from typing import Dict, List, Callable, Union
import torch.nn as nn
import torch
import numpy as np
from torch.nn import functional as F
from PIL import Image
import torch.optim as optim

from core import NShotTaskSampler, prepare_nshot_task
from datasets import CifarDataset
from torch.utils.data import DataLoader

from train import fit

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def pairwise_distances(x: torch.Tensor,
                       y: torch.Tensor,
                       matching_fn: str) -> torch.Tensor:
    """Efficiently calculate pairwise distances (or other similarity scores) between
    two sets of samples.

    # Arguments
        x: Query samples. A tensor of shape (n_x, d) where d is the embedding dimension
        y: Class prototypes. A tensor of shape (n_y, d) where d is the embedding dimension
        matching_fn: Distance metric/similarity score to compute between samples
    """
    n_x = x.shape[0]
    n_y = y.shape[0]
    if matching_fn == 'l2':
        distances = (
                x.unsqueeze(1).expand(n_x, n_y, -1) -
                y.unsqueeze(1).expand(n_x, n_y, -1)
        ).pow(2).sum(dim=2)
        return distances
    elif matching_fn == 'cosine':
        normalised_x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-8)
        normalised_y = y / (y.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-8)

        expanded_x = normalised_x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = normalised_y.unsqueeze(0).expand(n_x, n_y, -1)

        cosine_similarities = (expanded_x * expanded_y).sum(dim=2)
        return 1 - cosine_similarities
    elif matching_fn == 'dot':
        expanded_x = x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = y.unsqueeze(0).expand(n_x, n_y, -1)

        return -(expanded_x * expanded_y).sum(dim=2)
    else:
        raise (ValueError('Unsupported similarity function'))


def proto_net_episode(model: Module,
                      optimiser: Optimizer,
                      loss_fn: Callable,
                      x: torch.Tensor,

                      y: torch.Tensor,
                      n_shot: int,
                      k_way: int,
                      q_queries: int,
                      distance: str,
                      train: bool):
    """Performs a single training episode for a Prototypical Network.

    # Arguments
        model: Prototypical Network to be trained.
        optimiser: Optimiser to calculate gradient step
        loss_fn: Loss function to calculate between predictions and outputs. Should be cross-entropy
        x: Input samples of few shot classification task
        y: Input labels of few shot classification task
        n_shot: Number of examples per class in the support set
        k_way: Number of classes in the few shot classification task
        q_queries: Number of examples per class in the query set
        distance: Distance metric to use when calculating distance between class prototypes and queries
        train: Whether (True) or not (False) to perform a parameter update

    # Returns
        loss: Loss of the Prototypical Network on this task
        y_pred: Predicted class probabilities for the query set on this task
    """
    if train:
        # Zero gradients
        model.train()
        optimiser.zero_grad()
    else:
        model.eval()

    # print(model)
    # Embed all samples
    embeddings = model(x)
    # Samples are ordered by the NShotWrapper class as follows:
    # k lots of n support samples from a particular class
    # k lots of q query samples from those classes
    support = embeddings[:n_shot * k_way]
    queries = embeddings[n_shot * k_way:]
    prototypes = compute_prototypes(support, k_way, n_shot)
    # Calculate squared distances between all queries and all prototypes
    # Output should have shape (q_queries * k_way, k_way) = (num_queries, k_way)
    distances = pairwise_distances(queries, prototypes, distance)
    # Calculate log p_{phi} (y = k | x)
    log_p_y = (-distances).log_softmax(dim=1)
    loss = loss_fn(log_p_y, y)

    # Prediction probabilities are softmax over distances
    y_pred = (-distances).softmax(dim=1)

    if train:
        # Take gradient step
        loss.backward()
        optimiser.step()
    else:
        pass

    return loss, y_pred


def compute_prototypes(support: torch.Tensor, k: int, n: int) -> torch.Tensor:
    """Compute class prototypes from support samples.

    # Arguments
        support: torch.Tensor. Tensor of shape (n * k, d) where d is the embedding
            dimension.
        k: int. "k-way" i.e. number of classes in the classification task
        n: int. "n-shot" of the classification task

    # Returns
        class_prototypes: Prototypes aka mean embeddings for each class
    """
    # Reshape so the first dimension indexes by class then take the mean
    # along that dimension to generate the "prototypes" for each class
    class_prototypes = support.reshape(k, n, -1).mean(dim=1)
    return class_prototypes


class PROTO:

    def __init__(self, num_class, feature_extractor, task_size, memory_size, epochs, learning_rate, train_dataset,
                 test_dataset, n=1, q=1, distance='l2'):

        super(PROTO, self).__init__()
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.n = n
        self.q = q
        self.distance = distance

        self.model = feature_extractor
        self.train_exemplar_set = []
        self.test_exemplar_set = []
        self.class_mean_set = []
        self.num_class = num_class
        self.memory_size = memory_size
        self.task_size = task_size
        self.k = task_size
        self.episodes_per_epoch = 100
        self.train_loader = None
        self.test_loader = None

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    # get incremental train data
    # incremental
    def beforeTrain(self):
        self.model.eval()
        self.k = len(self.train_exemplar_set) + self.task_size
        classes = [self.num_class - self.task_size, self.num_class]
        self.train_loader, self.test_loader = self._get_train_and_test_dataloader(classes)
        self.model.train()
        self.model.to(device)

    def _get_train_and_test_dataloader(self, classes):
        self.train_dataset.getTrainData(classes, self.train_exemplar_set, n_shot=self.n + self.q)
        self.test_dataset.getTrainData(classes, self.test_exemplar_set, n_shot=self.n + self.q)
        train_loader = DataLoader(
            self.train_dataset,
            batch_sampler=NShotTaskSampler(self.train_dataset, self.episodes_per_epoch, self.n, self.k, self.q)
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_sampler=NShotTaskSampler(self.test_dataset, self.episodes_per_epoch, self.n, self.k, self.q)
        )
        return train_loader, test_loader

    # train model
    # compute loss
    # evaluate model
    def train(self):

        opt = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=0.00001)

        [epochs_accuracies,epochs_losses] = fit(
            self.model,
            opt,
            loss_fn=nn.NLLLoss().to(device),
            epochs=self.epochs,
            dataloader=self.train_loader,
            prepare_batch=prepare_nshot_task(self.n, self.k, self.q),
            fit_function=proto_net_episode,
            metric='categorical_accuracy',
            fit_function_kwargs={'n_shot': self.n, 'k_way': self.k, 'q_queries': self.q,
                                 'train': True,
                                 'distance': self.distance}
        )
        return [epochs_accuracies, epochs_losses]

    def _test(self, testloader, mode):
        if mode == 0:
            print("compute NMS")
        self.model.eval()
        correct, total = 0, 0
        for step, (imgs, labels) in enumerate(testloader):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = self.model(imgs) if mode == 1 else self.classify(imgs)
            predicts = torch.max(outputs, dim=1)[1] if mode == 1 else outputs
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracy = 100 * correct / total
        self.model.train()
        return accuracy

    # change the size of examplar
    def afterTrain(self, accuracy, transform, classify_transform):
        self.model.eval()
        print('...Computing Train Exemplar set...')
        self.class_mean_set = self.update_exemplar_set(classify_transform, transform, self.train_exemplar_set,
                                                       self.train_dataset, self.n+self.q)
        print('...Computing Test Exemplar set...')
        self.update_exemplar_set(classify_transform, transform, self.test_exemplar_set, self.test_dataset, self.n+self.q)
        self.num_class += self.task_size
        self.model.train()
        KNN_accuracy = self._test(self.test_loader, 0)
        print("NMS accuracy：" + str(KNN_accuracy.item()))
        # filename = 'model/accuracy_%.3f_KNN_accuracy_%.3f_increment_net.pkl' % (accuracy, KNN_accuracy)
        # torch.save(self.model, filename)
        # self.old_model = torch.load(filename)
        # self.old_model.to(device)
        # self.old_model.eval()
        return KNN_accuracy

    def update_exemplar_set(self, classify_transform, transform, exemplar_set, dataset,n):
        m = int(self.memory_size / self.num_class)
        self._reduce_exemplar_sets(m, exemplar_set)
        for i in range(self.num_class - self.task_size, self.num_class):
            print('construct class %s examplar:' % (i), end='')
            images = dataset.get_image_class(i)
            self._construct_exemplar_set(images, n, transform, exemplar_set)
        return self.compute_exemplar_class_mean(transform, classify_transform, exemplar_set)

    def _construct_exemplar_set(self, images, m, transform, exemplar_set):
        class_mean, feature_extractor_output = self.compute_class_mean(images, transform)
        exemplar = []
        # now_class_mean = np.zeros((1, 512))
        now_class_mean = np.zeros((1, 256))

        for i in range(m):
            x = class_mean - (now_class_mean + feature_extractor_output) / (i + 1)
            x = np.linalg.norm(x, axis=1)
            index = np.argmin(x)
            now_class_mean += feature_extractor_output[index]
            exemplar.append(images[index])

        print("the size of exemplar :%s" % (str(len(exemplar))))
        exemplar_set.append(exemplar)
        # self.exemplar_set.append(images)

    def _reduce_exemplar_sets(self, m, exemplar_set):
        for index in range(len(exemplar_set)):
            exemplar_set[index] = exemplar_set[index][:m]
            print('Size of class %d examplar: %s' % (index, str(len(exemplar_set[index]))))

    def Image_transform(self, images, transform):
        data = transform(Image.fromarray(images[0])).unsqueeze(0)
        for index in range(1, len(images)):
            data = torch.cat((data, transform(Image.fromarray(images[index])).unsqueeze(0)), dim=0)
        return data

    def compute_class_mean(self, images, transform):
        x = self.Image_transform(images, transform).to(device)
        feature_extractor_output = F.normalize(self.model(x).detach()).cpu().numpy()
        # feature_extractor_output = F.normalize(self.model.feature_extractor(x).detach()).cpu().numpy()
        # feature_extractor_output = self.model.feature_extractor(x).detach().cpu().numpy()
        class_mean = np.mean(feature_extractor_output, axis=0)
        return class_mean, feature_extractor_output

    def compute_exemplar_class_mean(self, transform, classify_transform, exemplar_set):
        class_mean_set = []
        for index in range(len(exemplar_set)):
            print("compute the class mean of %s" % (str(index)))
            exemplar = exemplar_set[index]
            # exemplar=self.train_dataset.get_image_class(index)
            class_mean, _ = self.compute_class_mean(exemplar, transform)
            class_mean_, _ = self.compute_class_mean(exemplar, classify_transform)
            class_mean = (class_mean / np.linalg.norm(class_mean) + class_mean_ / np.linalg.norm(class_mean_)) / 2
            class_mean_set.append(class_mean)
        return class_mean_set

    def classify(self, test):
        result = []
        test = F.normalize(self.model(test).detach()).cpu().numpy()
        # test = F.normalize(self.model.feature_extractor(test).detach()).cpu().numpy()
        # test = self.model.feature_extractor(test).detach().cpu().numpy()
        class_mean_set = np.array(self.class_mean_set)
        for target in test:
            x = target - class_mean_set
            x = np.linalg.norm(x, ord=2, axis=1)
            x = np.argmin(x)
            result.append(x)
        return torch.tensor(result)
