from torch import nn
from myNetwork import network_with_linear


def copy_model_parameters(source, target, is_proto_target=True):
    target.epochs = source.epochs
    target.learning_rate = source.learning_rate
    if is_proto_target:
        feature_extractor = source.model.feature
        target.model = feature_extractor
    else:
        feature_extractor = source.model
        feature_extractor.fc = nn.Linear(256, 100)
        target.model = network_with_linear(source.num_class, feature_extractor)
    target.train_exemplar_set = source.train_exemplar_set
    target.test_exemplar_set = source.test_exemplar_set
    target.class_mean_set = source.class_mean_set
    target.num_class = source.num_class

    # target.memory_size = source.memory_size
    # target.task_size = source.task_size

    # target.train_loader = source.train_loader
    # target.test_loader = source.test_loader