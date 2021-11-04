from torchvision import transforms

from datasets import CifarDataset
from models import FeatureExtractor
from proto import PROTO
from iCaRL import iCaRLmodel
from utils import copy_model_parameters
import matplotlib.pyplot as plt


numclass = 10
feature_extractor = FeatureExtractor(num_input_channels=3)
img_size = 32
batch_size = 128
task_size = 1
memory_size = 2000
epochs = 10
learning_rate = 0.002

transform = transforms.Compose([  # transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
old_model = None

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

iCaRL_model = iCaRLmodel(numclass, feature_extractor, batch_size, 5, memory_size, epochs, learning_rate, train_dataset, test_dataset)

iCaRL_model.beforeTrain()
[epochs_accuracies_1, epochs_losses_1] = iCaRL_model.train()
KNN_accuracy_1 = iCaRL_model.afterTrain(epochs_accuracies_1[-1], transform, classify_transform)

n=20
q=10
distance='dot' #'cosine'#'l2'

proto_model = PROTO(numclass, feature_extractor, task_size, memory_size, epochs, learning_rate, train_dataset,
                    test_dataset, n, q, distance)
copy_model_parameters(iCaRL_model, proto_model)
proto_model.beforeTrain()
[epochs_accuracies_2, epochs_losses_2] = proto_model.train()
KNN_accuracy_2 = proto_model.afterTrain(epochs_accuracies_2[-1], transform, classify_transform)
#
proto_model.beforeTrain()
[epochs_accuracies_3, epochs_losses_3] = proto_model.train()
KNN_accuracy_3 = proto_model.afterTrain(epochs_accuracies_3[-1], transform, classify_transform)

iCaRL_model = iCaRLmodel(numclass, feature_extractor, batch_size, task_size, memory_size, epochs, learning_rate, train_dataset, test_dataset)
copy_model_parameters(proto_model, iCaRL_model, False)
iCaRL_model.beforeTrain()
[epochs_accuracies_4, epochs_losses_4] = iCaRL_model.train()
KNN_accuracy_4 = iCaRL_model.afterTrain(epochs_accuracies_4[-1], transform, classify_transform)

proto_model = PROTO(numclass, feature_extractor, task_size, memory_size, epochs, learning_rate, train_dataset,
                    test_dataset, n, q, distance)
copy_model_parameters(iCaRL_model, proto_model)
proto_model.beforeTrain()
[epochs_accuracies_5, epochs_losses_5] = proto_model.train()
KNN_accuracy_5 = proto_model.afterTrain(epochs_accuracies_5[-1], transform, classify_transform)
#
proto_model.beforeTrain()
[epochs_accuracies_6, epochs_losses_6] = proto_model.train()
KNN_accuracy_6 = proto_model.afterTrain(epochs_accuracies_6[-1], transform, classify_transform)


KNN_accuracies = [KNN_accuracy_1, KNN_accuracy_2, KNN_accuracy_3, KNN_accuracy_4, KNN_accuracy_5, KNN_accuracy_6]

plt.figure(figsize=(10, 5))
plt.title("KNN_accuracies")
plt.plot([5,6,7,8,9,10], KNN_accuracies, label="KNN_accuracies")
plt.xlabel("No. of Classes")
plt.ylabel("Accuracy")
plt.legend()
plt.show()