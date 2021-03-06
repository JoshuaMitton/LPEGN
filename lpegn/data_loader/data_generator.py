import numpy as np
import data_loader.data_helper as helper
import utils.config
import torch
import random
from torch_geometric.data import DataLoader

def datagenerator(config):
    graphs = helper.load_dataset(config.dataset_name)
    random.shuffle(graphs)
    
    # if no fold specify creates random split to train and validation
    if config.num_fold is None:
        idx = len(graphs) // 10
        train_graphs, val_graphs = graphs[idx:], graphs[:idx]
    elif config.num_fold == 0:
        train_idx, test_idx = helper.get_parameter_split(config.dataset_name)
        train_graphs, val_graphs = graphs[train_idx], graphs[test_idx]
    else:
        train_idx, test_idx = helper.get_train_val_indexes(config.num_fold, config.dataset_name)
#         print(train_idx)
#         print(graphs)
        train_graphs = [graphs[idx] for idx in train_idx]
        val_graphs = [graphs[idx] for idx in test_idx]
#         train_graphs, val_graphs = graphs[train_idx], graphs[test_idx]
    
    train_loader = DataLoader(train_graphs, batch_size=config.batch_size , shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=config.batch_size , shuffle=False)
    
#     print(f'{graphs[0].x.shape}')
    
    return train_loader, val_loader, graphs[0].T2.shape[-1]
#     return train_loader, val_loader, graphs[0].x2.shape[-1]

# class DataGenerator:
#     def __init__(self, config):
#         self.config = config
#         # load data here
#         self.batch_size = self.config.batch_size
#         self.load_data()

#     # load the specified dataset in the config to the data_generator instance
#     def load_data(self):
#         graphs = helper.load_dataset(self.config.dataset_name)

#         # if no fold specify creates random split to train and validation
#         if self.config.num_fold is None:
#             graphs, labels = helper.shuffle(graphs, labels)
#             idx = len(graphs) // 10
#             self.train_graphs, self.train_labels, self.val_graphs, self.val_labels = graphs[idx:], labels[idx:], graphs[:idx], labels[:idx]
#         elif self.config.num_fold == 0:
#             train_idx, test_idx = helper.get_parameter_split(self.config.dataset_name)
#             self.train_graphs, self.train_labels, self.val_graphs, self.val_labels = graphs[train_idx], labels[
#                 train_idx], graphs[test_idx], labels[test_idx]
#         else:
#             train_idx, test_idx = helper.get_train_val_indexes(self.config.num_fold, self.config.dataset_name)
#             self.train_graphs, self.train_labels, self.val_graphs, self.val_labels = graphs[train_idx], labels[train_idx], graphs[test_idx], labels[
#                 test_idx]
#         # change validation graphs to the right shape
#         self.val_graphs = [np.expand_dims(g, 0) for g in self.val_graphs]
#         self.train_size = len(self.train_graphs)
#         self.val_size = len(self.val_graphs)


#     def next_batch(self):
#         return next(self.iter)

#     # initialize an iterator from the data for one training epoch
#     def initialize(self, is_train):
#         if is_train:
#             self.reshuffle_data()
#         else:
#             self.iter = zip(self.val_graphs, self.val_labels)

#     # resuffle data iterator between epochs
#     def reshuffle_data(self):
#         graphs, labels = helper.group_same_size(self.train_graphs, self.train_labels)
#         graphs, labels = helper.shuffle_same_size(graphs, labels)
#         graphs, labels = helper.split_to_batches(graphs, labels, self.batch_size)
#         self.num_iterations_train = len(graphs)
#         graphs, labels = helper.shuffle(graphs, labels)
#         self.iter = zip(graphs, labels)



# if __name__ == '__main__':
#     config = utils.config.process_config('../configs/example.json')
#     data = DataGenerator(config)
#     data.initialize(True)


