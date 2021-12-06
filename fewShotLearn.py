from torchvision import transforms

from ResNet import resnet18, resnet34
from datasets import CifarDataset
from models import FeatureExtractor
from proto import PROTO
from iCaRL import iCaRLmodel
from utils import copy_model_parameters
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from datetime import datetime
import gc
import torch




def few_shot(folder, full):
    print('################### Start ###################')
    start_time = datetime.now()
    print(start_time)
    print('###############################################')

    # feature_extractor = FeatureExtractor(num_input_channels=3)
    feature_extractor = resnet18()#resnet34()
    img_size = 32
    memory_size = 5000
    epochs = 120
    learning_rate = 1
    step_size = 30
    gamma = 0.1
    task_size=100 if full else 50
    n = 18 if full else 25
    q = 4
    distance = 'dot'  # ('dot', 'cosine', 'l2')
    transform = transforms.Compose([  # transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

    train_transform = transforms.Compose([  # transforms.Resize(img_size),
        transforms.RandomCrop((32, 32), padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.24705882352941178),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

    test_transform = transforms.Compose([  # transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

    classify_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=1.),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5071, 0.4867, 0.4408),
                                                                  (0.2675, 0.2565, 0.2761))])

    train_dataset = CifarDataset('data', transform=train_transform, download=True)
    test_dataset = CifarDataset('data', test_transform=test_transform, train=False, download=True)


    print('################### Model Training ###################')
    print(datetime.now())
    print('###############################################')
    proto_model = PROTO(0, feature_extractor, task_size, memory_size, epochs, learning_rate, train_dataset,
                        test_dataset, n, q, distance, step_size, gamma)

    proto_model.beforeTrain()
    [accuracies, losses] = proto_model.train()

    KNN_accuracy = proto_model.afterTrain(transform, classify_transform)


    print('################### End Time ###################')
    end_time = datetime.now()
    print(end_time)
    print('###############################################')

    diff = (end_time - start_time).total_seconds()
    print('################### Total Time ###################')
    print(f'{diff//3600}:{(diff-diff%3600)//60}:{(diff%3600)%60}')
    print('###############################################')
    fig, axs = plt.subplots(2)
    axs[0].plot(range(1, epochs+1), accuracies, label='KNN accuracies')
    axs[0].legend(loc="upper left")
    axs[1].plot(range(1, epochs+1), losses, label='Task losses')
    axs[1].legend(loc="upper left")

    plt.savefig(f'{folder}/fewShot.png')


