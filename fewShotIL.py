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





def few_shot_il(folder, full, max_models=False):
    # feature_extractor = FeatureExtractor(num_input_channels=3)
    feature_extractor = resnet18()#resnet34()#
    img_size = 32
    batch_size = 32

    memory_size = 5000
    epochs = 120
    learning_rate = 1
    step_size = 30
    gamma = 0.1

    n = 18 if full else 25
    q = 4
    distance = 'dot'  # ('dot', 'cosine', 'l2')
    if full and max_models:
        epochs_list = [50, 60, 70, 80, 90, 100]
    elif full:
        epochs_list = [40, 60, 80, 100]
    else:
        epochs_list = [20, 30, 40, 50]

    intital_task = epochs_list[0]
    task_size = epochs_list[1] - epochs_list[0]
    print('################### Start ###################')
    start_time = datetime.now()
    print(start_time)
    print('###############################################')

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
    iCaRL_model_1 = iCaRLmodel(0, feature_extractor, batch_size, intital_task, memory_size, epochs, learning_rate,
                               train_dataset, test_dataset, step_size,gamma)

    iCaRL_model_1.beforeTrain()
    epochs_losses_1 = iCaRL_model_1.train()
    KNN_accuracy = iCaRL_model_1.afterTrain(transform, classify_transform)
    task_accuracies.append(KNN_accuracy)
    task_losses.append(epochs_losses_1[-1])

    print('################### Model 2 ###################')
    print(datetime.now())
    print('###############################################')
    proto_model_2 = PROTO(epochs_list[0], feature_extractor, task_size, memory_size, epochs, learning_rate, train_dataset,
                          test_dataset, n, q, distance, step_size, gamma)
    copy_model_parameters(iCaRL_model_1, proto_model_2, is_proto_source=False, is_proto_target=True)
    del iCaRL_model_1
    gc.collect()
    torch.cuda.empty_cache()
    proto_model_2.beforeTrain()
    [task_accuracies_2, epochs_losses_2] = proto_model_2.train()
    KNN_accuracy = proto_model_2.afterTrain(transform, classify_transform)
    # print(f' Task accuracy:{task_accuracies_2[-1]}')
    task_accuracies.append(KNN_accuracy)
    task_losses.append(epochs_losses_2[-1])

    print('################### Model 3 ###################')
    print(datetime.now())
    print('###############################################')
    proto_model_3 = PROTO(epochs_list[1], feature_extractor, task_size, memory_size, epochs, learning_rate, train_dataset,
                          test_dataset, n, q, distance, step_size, gamma)
    copy_model_parameters(proto_model_2, proto_model_3, is_proto_source=True, is_proto_target=True)

    del proto_model_2
    gc.collect()
    torch.cuda.empty_cache()
    proto_model_3.beforeTrain()
    [task_accuracies_3, epochs_losses_3] = proto_model_3.train()
    KNN_accuracy = proto_model_3.afterTrain(transform, classify_transform)
    # print(f' Task accuracy:{task_accuracies_3[-1]}')
    task_accuracies.append(KNN_accuracy)
    task_losses.append(epochs_losses_3[-1])

    print('################### Model 4 ###################')
    print(datetime.now())
    print('###############################################')
    iCaRL_model_4 = iCaRLmodel(epochs_list[2], feature_extractor, batch_size, task_size, memory_size, epochs, learning_rate,
                               train_dataset, test_dataset, step_size, gamma)
    copy_model_parameters(proto_model_3, iCaRL_model_4, is_proto_source=True, is_proto_target=False)

    del proto_model_3
    gc.collect()
    torch.cuda.empty_cache()
    iCaRL_model_4.beforeTrain()
    epochs_losses_4 = iCaRL_model_4.train()
    KNN_accuracy = iCaRL_model_4.afterTrain(transform, classify_transform)
    task_accuracies.append(KNN_accuracy)
    task_losses.append(epochs_losses_4[-1])


    if max_models:
        print('################### Model 5 ###################')
        print(datetime.now())
        print('###############################################')

        proto_model_5 = PROTO(epochs_list[3], feature_extractor, task_size, memory_size, epochs, learning_rate,
                              train_dataset,
                              test_dataset, n, q, distance, step_size)
        copy_model_parameters(iCaRL_model_4, proto_model_5, is_proto_source=False, is_proto_target=True)
        del iCaRL_model_4
        gc.collect()
        torch.cuda.empty_cache()
        proto_model_5.beforeTrain()
        [task_accuracies_5, epochs_losses_5] = proto_model_5.train()
        proto_model_5.afterTrain(transform, classify_transform)
        print(f' Task accuracy:{task_accuracies_5[-1]}')
        # KNN_accuracies.append(KNN_accuracy)
        task_accuracies.append(task_accuracies_5[-1])
        task_losses.append(epochs_losses_5[-1])

        print('################### Model 6 ###################')
        print(datetime.now())
        print('###############################################')
        proto_model_6 = PROTO(epochs_list[4], feature_extractor, task_size, memory_size, epochs, learning_rate,
                              train_dataset,
                              test_dataset, n, q, distance, step_size)
        copy_model_parameters(proto_model_5, proto_model_6, is_proto_source=True, is_proto_target=True)
        del proto_model_5
        gc.collect()
        torch.cuda.empty_cache()
        proto_model_6.learning_rate = learning_rate
        proto_model_6.beforeTrain()
        [task_accuracies_6, epochs_losses_6] = proto_model_6.train()
        proto_model_6.afterTrain(transform, classify_transform)
        print(f' Task accuracy:{task_accuracies_6[-1]}')
        # KNN_accuracies.append(KNN_accuracy)
        task_accuracies.append(task_accuracies_6[-1])
        task_losses.append(epochs_losses_6[-1])


    print('################### End Time ###################')
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

    plt.savefig(f'{folder}/fewShotIL.png')





