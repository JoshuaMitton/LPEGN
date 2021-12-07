from trainers.base_train import BaseTrain
# import tensorflow as tf
import torch
from tqdm import tqdm
import numpy as np
from utils import doc_utils
import matplotlib.pyplot as plt

class Trainer(BaseTrain):
    def __init__(self, model, train_loader, val_loader, config):
#     def __init__(self, model, data, config):
#         super(Trainer, self).__init__(model, config, data)
        super(Trainer, self).__init__(model, config)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # define loss function
        self.loss = torch.nn.CrossEntropyLoss(reduction='sum')

        # choose optimizer
#         if self.config.optimizer == 'momentum':
#             self.optimizer = torch.optim.SGD(list(self.net_eq.parameters())+list(self.net_fc.parameters()), lr=self.config.learning_rate, momentum=self.config.momentum)
#         elif self.config.optimizer == 'adam':
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=0.0)
#             self.optimizer = torch.optim.AdamW(list(self.net_eq.parameters())+list(self.net_fc.parameters()), lr=self.config.learning_rate, weight_decay=1.6724283607433983)
            
        # get learning rate with decay every 20 epochs
        self.learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            30,
            0.1,
            )

    def correct_predictions(self, output, labels):
        output = torch.argmax(output, dim=-1)
#         print(output)
#         print(labels)
#         print(torch.sum(output==labels, dtype=torch.float32))
        return torch.sum(output==labels, dtype=torch.float32) / len(output)

        # load the model from the latest checkpoint if exist
        # self.model.load(self.sess)

    def train(self):
#         for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs, 1):
#         self.model.module.net_eq.train()
#         self.model.module.net_fc.train()
        self.model.train()
#         self.model.net_fc.train()
        train_loss_save = []
        train_acc_save = []
        test_loss_save = []
        test_acc_save = []
        for cur_epoch in range(0, self.config.num_epochs, 1):
#             self.model.optimizer.zero_grad()
            # train epoch
            train_acc, train_loss = self.train_epoch(cur_epoch)
            train_loss_save.append(train_loss)
            train_acc_save.append(train_acc)
#             train_loss.backward()
#             self.model.optimizer.step()
#             self.model.module.learning_rate_scheduler.step()
            self.learning_rate_scheduler.step()
#             self.model.optimizer.param_groups[0]["lr"] = max(self.model.optimizer.param_groups[0]["lr"], 0.00001)
#             print(f'lr value : {self.model.optimizer}')
#             print(f'lr value : {self.model.optimizer.param_groups[0]["lr"]}')
#             print(f'lr value : {self.model.learning_rate_scheduler.get_last_lr()}')
#             self.sess.run(self.model.increment_cur_epoch_tensor)
            # validation step
            if self.config.val_exist:
                test_acc, test_loss = self.test(cur_epoch)
                # document results
                doc_utils.write_to_file_doc(train_acc, train_loss, test_acc, test_loss, cur_epoch, self.config)
                test_loss_save.append(test_loss)
                test_acc_save.append(test_acc)
                
            if cur_epoch%50 == 0 and cur_epoch!=0:
                torch.save({
                            'epoch':cur_epoch,
                            'model_state_dict':self.model.state_dict(),
                            'optimizer_state_dict':self.optimizer.state_dict(),
                            'loss':train_loss,
                            'accuracy':train_acc}, f'{self.config.checkpoint_dir}model_{cur_epoch}.pt')
        if self.config.val_exist:
            # creates plots for accuracy and loss during training
            doc_utils.create_experiment_results_plot(self.config.exp_name, "accuracy", self.config.summary_dir)
            doc_utils.create_experiment_results_plot(self.config.exp_name, "loss", self.config.summary_dir, log=True)

        return np.mean(np.asarray(test_acc_save[-10:])), np.mean(np.asarray(test_loss_save[-10:]))

    def train_epoch(self, num_epoch=None):
        """
       implement the logic of epoch:
       -loop on the number of iterations in the config and call the train step

        Train one epoch
        :param epoch: cur epoch number
        :return accuracy and loss on train set
        """
#         # initialize dataset
#         self.data_loader.initialize(is_train=True)

#         # initialize tqdm
#         tt = tqdm(range(self.data_loader.num_iterations_train), total=self.data_loader.num_iterations_train,
#                   desc="epoch-{}-".format(num_epoch))
        

        total_loss = []
        total_correct = []

        # Iterate over batches
#         for cur_it in tt:
        for data in tqdm(self.train_loader):
#             self.model.module.optimizer.zero_grad()
            self.optimizer.zero_grad()
            # One Train step on the current batch
#             loss, correct = self.train_step()
            loss, correct = self.train_step(data)
            
            loss.backward()
#             self.model.module.optimizer.step()
            self.optimizer.step()
#             print(f'loss device : {loss.device}')
#             print(f'correct device : {correct.device}')
            # update results from train_step func
            total_loss.append(loss.item())
            total_correct.append(correct.item())

#         # save model
#         if num_epoch % self.config.save_rate == 0:
#             self.model.save(self.sess)

#         loss_per_epoch = total_loss/self.data_loader.train_size
#         acc_per_epoch = total_correct/self.data_loader.train_size
#         print(total_correct)
#         print(len(self.train_loader))
#         loss_per_epoch = total_loss/len(self.train_loader)
#         acc_per_epoch = total_correct/len(self.train_loader)
        loss_per_epoch = np.mean(total_loss)#/len(self.train_loader)
        acc_per_epoch = np.mean(total_correct)#/len(self.train_loader)
        print("""
        Epoch-{}  loss:{:.4f} -- acc:{:.4f}
                """.format(num_epoch, loss_per_epoch, acc_per_epoch))

#         tt.close()
        return acc_per_epoch, loss_per_epoch

    def train_step(self, data):
#     def train_step(self):
        """
       implement the logic of the train step
       - run the tensorflow session
       - :return any accuracy and loss on current batch
       """
#         print(f'device memory cached : {torch.cuda.memory_cached(0)}')
#         print(f'device memory allocated : {torch.cuda.memory_allocated(0)}')
#         graphs, labels = self.data_loader.next_batch()
#         graphs = torch.from_numpy(graphs)
#         labels = torch.from_numpy(labels)
        if self.config.cuda:
            data = data.cuda()
#             graphs = graphs.cuda()
#             labels = labels.cuda()
#         print(f'graphs device : {graphs.device}')
#         print(f'data in : {data}')
        output = self.model(data)
#         output = self.model(graphs)
#         output = output.cpu().data
#         loss = self.model.module.loss(output, labels)
#         print(f'output : {output.shape}')
#         print(f'data.y : {data.y.shape}')
        loss = self.loss(output, data.y)
#         correct = self.model.module.correct_predictions(output, labels)
        correct = self.correct_predictions(output, data.y)
#         _, loss, correct = self.sess.run([self.model.train_op, self.model.loss, self.model.correct_predictions],
#                                      feed_dict={self.model.graphs: graphs, self.model.labels: labels,
#                                                 self.model.is_training: True})
        return loss, correct


    def test(self, epoch):
#         # initialize dataset
#         self.data_loader.initialize(is_train=False)

#         # initialize tqdm
#         tt = tqdm(range(self.data_loader.val_size), total=self.data_loader.val_size,
#                   desc="Val-{}-".format(epoch))

        total_loss = []
        total_correct = []

        # Iterate over batches
#         for cur_it in tt:
        for data in tqdm(self.val_loader):
            # One Train step on the current batch
#             graph, label = self.data_loader.next_batch()
#             label = np.expand_dims(label, 0)
# #             loss, correct = self.sess.run([self.model.loss, self.model.correct_predictions],
# #                                       feed_dict={self.model.graphs: graph, self.model.labels: label, self.model.is_training: False})
#             graph = torch.from_numpy(graph)
#             label = torch.from_numpy(label)
            if self.config.cuda:
                data = data.cuda()
#                 graph = graph.cuda()
#                 label = label.cuda()
            output = self.model(data)
#             output = self.model(graph)
            loss = self.loss(output, data.y)
            correct = self.correct_predictions(output, data.y)
            # update metrics returned from train_step func
            total_loss.append(loss.item())
            total_correct.append(correct.item())

#         test_loss = total_loss/self.data_loader.val_size
#         test_acc = total_correct/self.data_loader.val_size
#         test_loss = total_loss/len(self.val_loader)
#         test_acc = total_correct/len(self.val_loader)
        test_loss = np.mean(total_loss)#/len(self.train_loader)
        test_acc = np.mean(total_correct)#/len(self.train_loader)
        
        print("""
        Val-{}  loss:{:.4f} -- acc:{:.4f}
        """.format(epoch, test_loss, test_acc))

#         tt.close()
        return test_acc, test_loss
