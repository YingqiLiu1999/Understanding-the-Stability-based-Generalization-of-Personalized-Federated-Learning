import copy
import logging
import time
import pdb
import numpy as np
import torch
from torch import nn
from fedml_api.model.cv.cnn_meta import Meta_net
from fedml_core.trainer.model_trainer import ModelTrainer
try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class MyModelTrainer(ModelTrainer):
    def __init__(self, model, args=None, logger=None):
        super().__init__(model, args)
        self.args=args
        self.logger = logger
        self.body_params = [p  for name, p in model.named_parameters() if 'linear' not in name]
        self.head_params = [p  for name, p in model.named_parameters() if 'linear' in name]

      

    def set_masks(self, masks):
        self.masks=masks
        # self.model.set_masks(masks)

    def get_model_params(self):
        return copy.deepcopy(self.model.cpu().state_dict())

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters,strict=False) 

    def get_trainable_params(self):
        dict= {}
        for name, param in self.model.named_parameters():
            dict[name] = param
        return dict

    def train(self, train_data,test_data,  device,  args, round):
        # torch.manual_seed(0)
        # test_local_metrics = self.test(test_data,device,args)
        # p_test_acc = np.array(test_local_metrics['test_correct']) / np.array(test_local_metrics['test_total'])
        # self.logger.info('acc_before: {:.5f}'.format(p_test_acc))
        
        model = self.model
        model.to(device)
        model.train()
        
        if self.args.dataset == "emnist":
            body_params = [p for name, p in model.named_parameters() if 'output_layer' not in name]
            head_params = [p for name, p in model.named_parameters() if 'output_layer' in name]
        elif self.args.model == "resnet18" :
            body_params = [p for name, p in model.named_parameters() if 'linear' not in name]
            head_params = [p for name, p in model.named_parameters() if 'linear' in name]
        elif self.args.model =="vgg11":
            body_params = [p for name, p in model.named_parameters() if 'classifier' not in name]
            head_params = [p for name, p in model.named_parameters() if 'classifier' in name]
            
        criterion = nn.CrossEntropyLoss().to(device)
        for param in body_params:
            param.requires_grad = False #是否包含了私有部分
        for param in head_params:
            param.requires_grad = True
        head_optimizer = torch.optim.SGD([{'params': body_params, 'lr': 0.0, 'name': "body"},
                                    {'params': head_params, 'lr': args.lr_head*(args.lr_decay**round), "name": "head"}],
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)
        for epoch in range(args.head_epochs):
            epoch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model.forward(x)
                loss = criterion(log_probs, labels.long())
                loss.backward()
                # to avoid nan loss
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                head_optimizer.step()
                epoch_loss.append(loss.item())
                
            self.logger.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
                self.id, epoch, sum(epoch_loss) / len(epoch_loss)))
           
        for param in body_params:
            param.requires_grad = True 
        for param in head_params:
            param.requires_grad = False
        body_optimizer = torch.optim.SGD([{'params': body_params, 'lr':args.lr_body*(args.lr_decay**round), 'name': "body"},
                                    {'params': head_params, 'lr':0.0 , "name": "head"}],
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)
                
        for epoch in range(args.body_epochs):   
            epoch_loss = []     
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model.forward(x)
                loss = criterion(log_probs, labels.long())
                loss.backward()
                # to avoid nan loss
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                body_optimizer.step()
                epoch_loss.append(loss.item())

            self.logger.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
                self.id, epoch, sum(epoch_loss) / len(epoch_loss)))

        test_local_metrics1 = self.test(test_data,device,args)
        p_test_acc = np.array(test_local_metrics1['test_correct']) / np.array(test_local_metrics1['test_total'])
        self.logger.info('acc_after: {:.5f}'.format(p_test_acc))
        
        # train after de test
        train_local_metrics1 = self.test(train_data,device,args) #
        p_test_acc = np.array(train_local_metrics1['test_correct']) / np.array(train_local_metrics1['test_total'])
        self.logger.info('acc_train_after: {:.5f}'.format(p_test_acc))
        
        return test_local_metrics1,train_local_metrics1
    

    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0
        }

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target.long())

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)
        return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False

