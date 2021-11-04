import torch.nn as nn
import torch
import numpy as np
from torch.nn import functional as F
from PIL import Image
import torch.optim as optim

from myNetwork import network_with_linear
from torch.utils.data import DataLoader

from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_one_hot(target, num_class):
    one_hot = torch.zeros(target.shape[0], num_class).to(device)
    one_hot = one_hot.scatter(dim=1, index=target.long().view(-1, 1), value=1.)
    return one_hot


class iCaRLmodel:

    def __init__(self, num_class, feature_extractor, batch_size, task_size, memory_size, epochs, learning_rate,
                 train_dataset, test_dataset):

        super(iCaRLmodel, self).__init__()
        self.epochs = epochs
        self.learning_rate = learning_rate
        feature_extractor.fc = nn.Linear(256, 100)
        self.model = network_with_linear(num_class, feature_extractor)
        self.train_exemplar_set = []
        self.test_exemplar_set = []
        self.class_mean_set = []
        self.num_class = num_class
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.task_size = task_size
        self.old_model = None
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset


        self.train_loader = None
        self.test_loader = None

    # get incremental train data
    # incremental
    def beforeTrain(self):
        self.model.eval()
        classes = [self.num_class - self.task_size, self.num_class]
        self.train_loader, self.test_loader = self._get_train_and_test_dataloader(classes)
        if self.num_class > self.task_size:
            self.model.Incremental_learning(self.num_class)
        self.model.train()
        self.model.to(device)

    def _get_train_and_test_dataloader(self, classes):
        self.train_dataset.getTrainData(classes, self.train_exemplar_set)
        self.test_dataset.getTrainData(classes, self.test_exemplar_set)
        train_loader = DataLoader(dataset=self.train_dataset,
                                  shuffle=True,
                                  batch_size=self.batch_size)

        test_loader = DataLoader(dataset=self.test_dataset,
                                 shuffle=True,
                                 batch_size=self.batch_size)

        return train_loader, test_loader

    '''
    def _get_old_model_output(self, dataloader):
        x = {}
        for step, (indexs, imgs, labels) in enumerate(dataloader):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                old_model_output = torch.sigmoid(self.old_model(imgs))
            for i in range(len(indexs)):
                x[indexs[i].item()] = old_model_output[i].cpu().numpy()
        return x
    '''

    # train model
    # compute loss
    # evaluate model
    def train(self):
        accuracy = 0
        opt = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=0.00001)

        epochs_losses = []
        epochs_accuracies = []
        for epoch in range(1, self.epochs+1):
            loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader), leave=False)
            for step, (images, target) in loop:
                images, target = images.to(device), target.to(device)
                # output = self.model(images)
                loss_value = self._compute_loss(images, target)
                opt.zero_grad()
                loss_value.backward()
                opt.step()

                ### Update PRogress Bar
                loop.set_description(f"Epoch [{epoch}/{self.epochs}]")
                loop.set_postfix(loss = loss_value.item())
            accuracy = self._test(self.test_loader, 1)
            print('epoch:%d,accuracy:%.3f' % (epoch, accuracy))
            epochs_losses.append(loss_value.item())
            epochs_accuracies.append(accuracy)
        return [epochs_accuracies, epochs_losses]

    def _test(self, testloader, mode):
        if mode == 0:
            print("compute NMS")
        self.model.eval()
        correct, total = 0, 0
        for setp, (imgs, labels) in enumerate(testloader):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = self.model(imgs) if mode == 1 else self.classify(imgs)
            predicts = torch.max(outputs, dim=1)[1] if mode == 1 else outputs
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracy = 100 * correct / total
        self.model.train()
        return accuracy

    def _compute_loss(self, imgs, target):
        output = self.model(imgs)
        target = get_one_hot(target, self.num_class)
        output, target = output.to(device), target.to(device)
        if self.old_model == None:
            return F.binary_cross_entropy_with_logits(output, target)
        else:
            # old_target = torch.tensor(np.array([self.old_model_output[index.item()] for index in indexs]))
            old_target = torch.sigmoid(self.old_model(imgs))
            old_task_size = old_target.shape[1]
            target[..., :old_task_size] = old_target
            return F.binary_cross_entropy_with_logits(output, target)

    # change the size of examplar
    def afterTrain(self, accuracy, transform, classify_transform):
        self.model.eval()
        print('...Computing Train Exemplar set...')
        self.class_mean_set = self.update_exemplar_set(classify_transform, transform, self.train_exemplar_set,
                                                       self.train_dataset)
        print('...Computing Test Exemplar set...')
        self.update_exemplar_set(classify_transform, transform, self.test_exemplar_set, self.test_dataset)
        self.num_class += self.task_size
        self.model.train()
        KNN_accuracy = self._test(self.test_loader, 0)
        print("NMS accuracy：" + str(KNN_accuracy.item()))
        # filename = 'model/accuracy_%.3f_KNN_accuracy_%.3f_increment:%d_net.pkl' % (accuracy, KNN_accuracy, i + 10)
        # torch.save(self.model, filename)
        # self.old_model = torch.load(filename)
        # self.old_model.to(device)
        # self.old_model.eval()
        return KNN_accuracy

    def update_exemplar_set(self, classify_transform, transform, exemplar_set, dataset):
        m = int(self.memory_size / self.num_class)
        self._reduce_exemplar_sets(m, exemplar_set)
        for i in range(self.num_class - self.task_size, self.num_class):
            print('construct class %s examplar:' % (i), end='')
            images = dataset.get_image_class(i)
            self._construct_exemplar_set(images, m, transform, exemplar_set)

        return self.compute_exemplar_class_mean(transform, classify_transform, exemplar_set)

    def _construct_exemplar_set(self, images, m, transform, exemplar_set):
        class_mean, feature_extractor_output = self.compute_class_mean(images, transform)
        exemplar = []
        now_class_mean = np.zeros((1, 256))

        for i in range(m):
            # shape：batch_size*512
            x = class_mean - (now_class_mean + feature_extractor_output) / (i + 1)
            # shape：batch_size
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
        feature_extractor_output = F.normalize(self.feature_extractor(x)).cpu().numpy()
        # feature_extractor_output = self.model.feature_extractor(x).detach().cpu().numpy()
        class_mean = np.mean(feature_extractor_output, axis=0)
        return class_mean, feature_extractor_output

    def feature_extractor(self, x):
        feature_extractor_output = self.model.feature_extractor(x).detach()
        feature_extractor_output = feature_extractor_output.view(feature_extractor_output.size(0), -1)
        return feature_extractor_output

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
        test = F.normalize(self.feature_extractor(test)).cpu().numpy()
        # test = self.model.feature_extractor(test).detach().cpu().numpy()
        class_mean_set = np.array(self.class_mean_set)
        for target in test:
            x = target - class_mean_set
            x = np.linalg.norm(x, ord=2, axis=1)
            x = np.argmin(x)
            result.append(x)
        return torch.tensor(result)
