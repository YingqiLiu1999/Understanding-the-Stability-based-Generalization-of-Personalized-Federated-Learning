import copy
import logging
import time
import pdb
import numpy as np
import torch
from torch import nn
from fedml_api.model.cv.cnn_meta import Meta_net
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class MyModelTrainer(ModelTrainer):
    def __init__(self, model, args=None, logger=None):
        super().__init__(model, args)
        self.args=args
        self.logger = logger
        self.body_params = [p  for name, p in model.named_parameters() if 'linear' and "classifier" not in name]
        self.head_params = [p  for name, p in model.named_parameters() if 'linear' and "classifier" in name]

    def set_masks(self, masks):
        self.masks=masks
        # self.model.set_masks(masks)

    def get_model_params(self):
        return copy.deepcopy(self.model.cpu().state_dict())

    def set_model_params(self, model_parameters):
        # self.model.load_state_dict(model_parameters)
        self.model.load_state_dict(model_parameters,strict=False) 

    def get_trainable_params(self):
        dict= {}
        for name, param in self.model.named_parameters():
            dict[name] = param
        return dict

    def train(self, train_data,  device,  args, round):
        # torch.manual_seed(0)
        model = self.model
        model.to(device)
        model.train()
        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        # for  name, p in model.named_parameters():
        #     print(name)
        body_params = [p for name, p in model.named_parameters() if 'linear' and "classifier" not in name]
        head_params = [p for name, p in model.named_parameters() if 'linear' and "classifier" in name]
        
        # if args.client_optimizer == "sgd":
            # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr* (args.lr_decay**round), momentum=args.momentum,weight_decay=args.wd)
        for epoch in range(args.head_epochs):
            epoch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                optimizer = torch.optim.SGD([{'params': body_params, 'lr': 0.0, 'name': "body"},
                                    {'params': head_params, 'lr': args.lr_head*(args.lr_decay**round), "name": "head"}],
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)
                model.zero_grad()

                log_probs = model.forward(x)
                loss = criterion(log_probs, labels.long())
                loss.backward()
                # to avoid nan loss
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                optimizer.step()
                epoch_loss.append(loss.item())
                
            self.logger.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
                self.id, epoch, sum(epoch_loss) / len(epoch_loss)))
        
        for epoch in range(args.body_epochs):       
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device) 
                optimizer = torch.optim.SGD([{'params': body_params, 'lr':args.lr_body*(args.lr_decay**round), 'name': "body"},
                                    {'params': head_params, 'lr':0.0 , "name": "head"}],
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)
                model.zero_grad()
                log_probs = model.forward(x)
                loss = criterion(log_probs, labels.long())
                loss.backward()
                # to avoid nan loss
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                optimizer.step()
                epoch_loss.append(loss.item())

            self.logger.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
                self.id, epoch, sum(epoch_loss) / len(epoch_loss)))

  
    def test(self, test_data, device, args):
        model = self.model
        model.to(device)
        # model.to(device)
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
