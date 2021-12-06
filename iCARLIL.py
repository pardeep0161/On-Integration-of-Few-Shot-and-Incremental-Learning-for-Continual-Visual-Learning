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




def iCarl_il(folder, full, max_models=False):
    start_time = datetime.now()
    print('################### Start ###################')
    print(start_time)
    print('###############################################')

    # feature_extractor = FeatureExtractor(num_input_channels=3)
    feature_extractor = resnet18()#resnet34()
    img_size = 32
    batch_size = 32

    memory_size = 5000
    epochs = 120
    learning_rate = 1
    step_size = 30
    gamma = 0.1

    if full and max_models:
        epochs_list = [50, 60, 70, 80, 90, 100]
    elif full:
        epochs_list = [40, 60, 80, 100]
    else:
        epochs_list = [20, 30, 40, 50]

    intital_task = epochs_list[0]
    task_size = epochs_list[1]-epochs_list[0]
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
    task_accuracies = []
    task_losses = []

    print('################### Model 1 ###################')
    print(datetime.now())
    print('###############################################')
    iCaRL_model = iCaRLmodel(0, feature_extractor, batch_size, intital_task, memory_size, epochs, learning_rate,
                               train_dataset, test_dataset, step_size,gamma)

    iCaRL_model.beforeTrain()
    epochs_losses = iCaRL_model.train()
    KNN_accuracy = iCaRL_model.afterTrain(transform, classify_transform)
    task_accuracies.append(KNN_accuracy)
    task_losses.append(epochs_losses[-1])

    print('################### Model 2 ###################')
    print(datetime.now())
    print('###############################################')
    iCaRL_model.task_size = task_size
    iCaRL_model.learning_rate=learning_rate
    iCaRL_model.beforeTrain()
    epochs_losses = iCaRL_model.train()
    KNN_accuracy = iCaRL_model.afterTrain(transform, classify_transform)
    task_accuracies.append(KNN_accuracy)
    task_losses.append(epochs_losses[-1])

    print('################### Model 3 ###################')
    print(datetime.now())
    print('###############################################')
    iCaRL_model.learning_rate=learning_rate
    iCaRL_model.beforeTrain()
    epochs_losses = iCaRL_model.train()
    KNN_accuracy = iCaRL_model.afterTrain(transform, classify_transform)
    task_accuracies.append(KNN_accuracy)
    task_losses.append(epochs_losses[-1])

    print('################### Model 4 ###################')
    print(datetime.now())
    print('###############################################')
    iCaRL_model.learning_rate=learning_rate
    iCaRL_model.beforeTrain()
    epochs_losses = iCaRL_model.train()
    KNN_accuracy = iCaRL_model.afterTrain(transform, classify_transform)
    task_accuracies.append(KNN_accuracy)
    task_losses.append(epochs_losses[-1])

    if max_models:
        print('################### Model 5 ###################')
        print(datetime.now())
        print('###############################################')

        iCaRL_model.learning_rate = learning_rate
        iCaRL_model.beforeTrain()
        epochs_losses = iCaRL_model.train()
        KNN_accuracy = iCaRL_model.afterTrain(transform, classify_transform)
        task_accuracies.append(KNN_accuracy)
        task_losses.append(epochs_losses[-1])

        print('################### Model 6 ###################')
        print(datetime.now())
        print('###############################################')

        iCaRL_model.learning_rate = learning_rate
        iCaRL_model.beforeTrain()
        epochs_losses = iCaRL_model.train()
        KNN_accuracy = iCaRL_model.afterTrain(transform, classify_transform)
        task_accuracies.append(KNN_accuracy)
        task_losses.append(epochs_losses[-1])

    print('################### End Time ###################')
    iCaRL_model.learning_rate=learning_rate
    end_time = datetime.now()
    print(end_time)
    print('###############################################')

    diff = (end_time - start_time).total_seconds()
    print('################### Total Time ###################')
    print(f'{diff//3600}:{(diff-diff%3600)//60}:{(diff%3600)%60}')
    print('###############################################')
    fig, axs = plt.subplots(2)
    # axs[0].plot(epochs_list, KNN_accuracies, label='KNN accuracies')
    # axs[0].legend(loc="upper left")
    axs[0].plot(epochs_list, task_accuracies, label='KNN accuracies')
    axs[0].legend(loc="upper left")
    axs[1].plot(epochs_list, task_losses, label='Task losses')
    axs[1].legend(loc="upper left")

    plt.savefig(f'{folder}/iCarIL.png')
