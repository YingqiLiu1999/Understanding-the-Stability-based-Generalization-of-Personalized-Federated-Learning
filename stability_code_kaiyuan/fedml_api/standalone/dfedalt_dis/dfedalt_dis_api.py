import copy
import logging
import pickle
import random
import pdb
import numpy as np
import torch
from fedml_api.standalone.dfedalt_dis.client import Client
import time
import gc

class DFedAltAPI(object):
    def __init__(self, dataset1,dataset2,dataset3, device, args, model_trainer, logger):
        self.logger = logger
        self.device = device
        self.args = args
        [train_data_num1, test_data_num1, train_data_global1, test_data_global1,
         train_data_local_num_dict1, train_data_local_dict1, test_data_local_dict1, class_num] = dataset1
        [train_data_num2, test_data_num2, train_data_global2, test_data_global2,
         train_data_local_num_dict2, train_data_local_dict2, test_data_local_dict2, class_num] = dataset2
        [train_data_num3, test_data_num3, train_data_global3, test_data_global3,
         train_data_local_num_dict3, train_data_local_dict3, test_data_local_dict3, class_num] = dataset3
        self.train_global = train_data_global1
        self.test_global = test_data_global1
        self.val_global = None
        self.train_data_num_in_total = train_data_num1
        self.test_data_num_in_total = test_data_num1
        self.client_list1 = []
        self.client_list2 = []
        self.client_list3 = []
        self.Budget = []
        self.train_data_local_num_dict1 = train_data_local_num_dict1
        self.train_data_local_dict1 = train_data_local_dict1
        self.test_data_local_dict1 = test_data_local_dict1
        self.model_trainer = model_trainer
        self._setup_clients3(train_data_local_num_dict1, train_data_local_dict1, test_data_local_dict1, model_trainer,\
                                train_data_local_num_dict2, train_data_local_dict2, test_data_local_dict2,\
                                    train_data_local_num_dict3, train_data_local_dict3, test_data_local_dict3)
        self.init_stat_info()

    def _setup_clients3(self, train_data_local_num_dict1, train_data_local_dict1, test_data_local_dict1, model_trainer,\
                                train_data_local_num_dict2, train_data_local_dict2, test_data_local_dict2,\
                                    train_data_local_num_dict3, train_data_local_dict3, test_data_local_dict3):
        self.logger.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_in_total):
            c = Client(client_idx, train_data_local_dict1[client_idx], test_data_local_dict1[client_idx],
                       train_data_local_num_dict1[client_idx], self.args, self.device, model_trainer, self.logger)
            self.client_list1.append(c)
            c = Client(client_idx, train_data_local_dict2[client_idx], test_data_local_dict2[client_idx],
                       train_data_local_num_dict2[client_idx], self.args, self.device, model_trainer, self.logger)
            self.client_list2.append(c)
            c = Client(client_idx, train_data_local_dict3[client_idx], test_data_local_dict3[client_idx],
                       train_data_local_num_dict3[client_idx], self.args, self.device, model_trainer, self.logger)
            self.client_list3.append(c)
        self.logger.info("############setup_clients (END)#############")

    def train(self):
        w_global1 = self.model_trainer.get_model_params()
        w_global2 = copy.deepcopy(w_global1)
        w_global3 = copy.deepcopy(w_global2)
        w_per_mdls1 = []
        w_per_mdls2 = []
        w_per_mdls3 = []
        # W_per_finetune = []
        # 初始化
        for clnt in range(self.args.client_num_in_total):
            w_per_mdls1.append(copy.deepcopy(w_global1))
            w_per_mdls2.append(copy.deepcopy(w_global2))
            w_per_mdls3.append(copy.deepcopy(w_global3))
        # device = {device} cuda:0apply mask to init weights
        for round_idx in range(self.args.comm_round):
            s_t = time.time()
            print("################Communication round : {}".format(round_idx))
            self.logger.info("################Communication round : {}".format(round_idx))
            w_locals = []

            ###11#############################################################################################################
            w_per_mdls_lstrd = copy.deepcopy(w_per_mdls1) 
            p_test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []}
            p_train_metrics= {
            'num_samples': [],
            'num_correct': [],
            'losses': []}

            for clnt_idx in self.client_list1:
                self.logger.info('@@@@@@@@@@@@@@@@ Training Client CM({}): {}'.format(round_idx, clnt_idx.client_idx))
                # update dataset
                """
                for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
                Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
                """
                nei_indexs = self._benefit_choose(round_idx, clnt_idx.client_idx, self.args.client_num_in_total,\
                                                  self.args.client_num_per_round, self.args.cs)
                if self.args.client_num_in_total != self.args.client_num_per_round:
                    nei_indexs = np.append(nei_indexs, clnt_idx.client_idx)
                nei_indexs = np.sort(nei_indexs) 
                w_local_mdl = self._aggregate_func(clnt_idx.client_idx, self.args.client_num_in_total,
                                                            self.args.client_num_per_round, nei_indexs,
                                                            w_per_mdls_lstrd)
                client = clnt_idx #self.client_list1[clnt_idx.client_idx]
                w_local_mdl,p_test_local_metrics,p_train_local_metrics = \
                    client.train(copy.deepcopy(w_local_mdl),w_per_mdls1[clnt_idx.client_idx],round_idx)
                
                p_test_metrics['num_samples'].append(copy.deepcopy(p_test_local_metrics['test_total']))
                p_test_metrics['num_correct'].append(copy.deepcopy(p_test_local_metrics['test_correct']))
                p_test_metrics['losses'].append(copy.deepcopy(p_test_local_metrics['test_loss']))
                p_train_metrics['num_samples'].append(copy.deepcopy(p_train_local_metrics['test_total']))
                p_train_metrics['num_correct'].append(copy.deepcopy(p_train_local_metrics['test_correct']))
                p_train_metrics['losses'].append(copy.deepcopy(p_train_local_metrics['test_loss']))
                w_per_mdls1[clnt_idx.client_idx] = copy.deepcopy(w_local_mdl)
                
            p_train_acc1 = sum(
            [np.array(p_train_metrics['num_correct'][i]) / np.array(p_train_metrics['num_samples'][i]) for i in
             range(self.args.client_num_in_total)]) / self.args.client_num_in_total
            p_train_loss1 = sum([np.array(p_train_metrics['losses'][i]) / np.array(p_train_metrics['num_samples'][i]) for i in
                            range(self.args.client_num_in_total)]) / self.args.client_num_in_total
            stats = {'Local model person_train_acc': p_train_acc1, 'person_train_loss': p_train_loss1}
            # self.stat_info["person_train_loss_after"].append(p_train_loss1)
            self.logger.info(stats)
            
            p_test_acc1 = sum(
            [np.array(p_test_metrics['num_correct'][i]) / np.array(p_test_metrics['num_samples'][i]) for i in
             range(self.args.client_num_in_total)]) / self.args.client_num_in_total
            p_test_loss1 = sum([np.array(p_test_metrics['losses'][i]) / np.array(p_test_metrics['num_samples'][i]) for i in
                            range(self.args.client_num_in_total)]) / self.args.client_num_in_total
            stats = {'Local model person_test_acc': p_test_acc1, 'person_test_loss': p_test_loss1}
            # self.stat_info["person_test_acc"].append(p_test_acc)
            self.logger.info(stats)
            
            print("person_test_acc_after:%.3f" %(p_test_acc1*100))
            del w_local_mdl,w_per_mdls_lstrd
            gc.collect()
            
            ####22############################################################################################################
            w_per_mdls_lstrd = copy.deepcopy(w_per_mdls2) 
            p_test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []}
            p_train_metrics= {
            'num_samples': [],
            'num_correct': [],
            'losses': []}
            for clnt_idx in self.client_list2:
                self.logger.info('@@@@@@@@@@@@@@@@ Training Client CM({}): {}'.format(round_idx, clnt_idx.client_idx))
                # update dataset
                """
                for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
                Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
                """
                nei_indexs = self._benefit_choose(round_idx, clnt_idx.client_idx, self.args.client_num_in_total,\
                                                  self.args.client_num_per_round, self.args.cs)
                if self.args.client_num_in_total != self.args.client_num_per_round:
                    nei_indexs = np.append(nei_indexs, clnt_idx.client_idx)
                nei_indexs = np.sort(nei_indexs) 
                w_local_mdl = self._aggregate_func(clnt_idx.client_idx, self.args.client_num_in_total,
                                                            self.args.client_num_per_round, nei_indexs,
                                                            w_per_mdls_lstrd)
                client = clnt_idx
                w_local_mdl,p_test_local_metrics,p_train_local_metrics = \
                    client.train(copy.deepcopy(w_local_mdl),w_per_mdls1[clnt_idx.client_idx],round_idx)
                
                p_test_metrics['num_samples'].append(copy.deepcopy(p_test_local_metrics['test_total']))
                p_test_metrics['num_correct'].append(copy.deepcopy(p_test_local_metrics['test_correct']))
                p_test_metrics['losses'].append(copy.deepcopy(p_test_local_metrics['test_loss']))
                p_train_metrics['num_samples'].append(copy.deepcopy(p_train_local_metrics['test_total']))
                p_train_metrics['num_correct'].append(copy.deepcopy(p_train_local_metrics['test_correct']))
                p_train_metrics['losses'].append(copy.deepcopy(p_train_local_metrics['test_loss']))
                w_per_mdls2[clnt_idx.client_idx] = copy.deepcopy(w_local_mdl)
                
            p_train_acc2 = sum(
            [np.array(p_train_metrics['num_correct'][i]) / np.array(p_train_metrics['num_samples'][i]) for i in
             range(self.args.client_num_in_total)]) / self.args.client_num_in_total
            p_train_loss2 = sum([np.array(p_train_metrics['losses'][i]) / np.array(p_train_metrics['num_samples'][i]) for i in
                            range(self.args.client_num_in_total)]) / self.args.client_num_in_total
            stats = {'Local model person_train_acc': p_train_acc2, 'person_train_loss': p_train_loss2}
            # self.stat_info["person_train_loss_after"].append(p_train_loss1)
            self.logger.info(stats)
            
            p_test_acc2 = sum(
            [np.array(p_test_metrics['num_correct'][i]) / np.array(p_test_metrics['num_samples'][i]) for i in
             range(self.args.client_num_in_total)]) / self.args.client_num_in_total
            p_test_loss2 = sum([np.array(p_test_metrics['losses'][i]) / np.array(p_test_metrics['num_samples'][i]) for i in
                            range(self.args.client_num_in_total)]) / self.args.client_num_in_total
            stats = {'Local model person_test_acc': p_test_acc2, 'person_test_loss': p_test_loss2}
            # self.stat_info["person_test_acc"].append(p_test_acc)
            self.logger.info(stats)
            
            print("person_test_acc_after:%.3f" %(p_test_acc2*100))
            del w_local_mdl,w_per_mdls_lstrd
            gc.collect()
            ####33#############################################################################################################
            w_per_mdls_lstrd = copy.deepcopy(w_per_mdls3) 
            p_test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []}
            p_train_metrics= {
            'num_samples': [],
            'num_correct': [],
            'losses': []}
            for clnt_idx in self.client_list3:
                self.logger.info('@@@@@@@@@@@@@@@@ Training Client CM({}): {}'.format(round_idx, clnt_idx.client_idx))
                # update dataset
                """
                for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
                Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
                """
                nei_indexs = self._benefit_choose(round_idx, clnt_idx.client_idx, self.args.client_num_in_total,\
                                                  self.args.client_num_per_round, self.args.cs)
                if self.args.client_num_in_total != self.args.client_num_per_round:
                    nei_indexs = np.append(nei_indexs, clnt_idx.client_idx)
                nei_indexs = np.sort(nei_indexs) 
                w_local_mdl = self._aggregate_func(clnt_idx.client_idx, self.args.client_num_in_total,
                                                            self.args.client_num_per_round, nei_indexs,
                                                            w_per_mdls_lstrd)
                client = clnt_idx
                w_local_mdl,p_test_local_metrics,p_train_local_metrics = \
                    client.train(copy.deepcopy(w_local_mdl),w_per_mdls1[clnt_idx.client_idx],round_idx)
                
                p_test_metrics['num_samples'].append(copy.deepcopy(p_test_local_metrics['test_total']))
                p_test_metrics['num_correct'].append(copy.deepcopy(p_test_local_metrics['test_correct']))
                p_test_metrics['losses'].append(copy.deepcopy(p_test_local_metrics['test_loss']))
                p_train_metrics['num_samples'].append(copy.deepcopy(p_train_local_metrics['test_total']))
                p_train_metrics['num_correct'].append(copy.deepcopy(p_train_local_metrics['test_correct']))
                p_train_metrics['losses'].append(copy.deepcopy(p_train_local_metrics['test_loss']))
                w_per_mdls3[clnt_idx.client_idx] = copy.deepcopy(w_local_mdl)
                
            p_train_acc3 = sum(
            [np.array(p_train_metrics['num_correct'][i]) / np.array(p_train_metrics['num_samples'][i]) for i in
             range(self.args.client_num_in_total)]) / self.args.client_num_in_total
            p_train_loss3 = sum([np.array(p_train_metrics['losses'][i]) / np.array(p_train_metrics['num_samples'][i]) for i in
                            range(self.args.client_num_in_total)]) / self.args.client_num_in_total
            stats = {'Local model person_train_acc': p_train_acc3, 'person_train_loss': p_train_loss3}
            # self.stat_info["person_train_loss_after"].append(p_train_loss1)
            self.logger.info(stats)
            
            p_test_acc3 = sum(
            [np.array(p_test_metrics['num_correct'][i]) / np.array(p_test_metrics['num_samples'][i]) for i in
             range(self.args.client_num_in_total)]) / self.args.client_num_in_total
            p_test_loss3 = sum([np.array(p_test_metrics['losses'][i]) / np.array(p_test_metrics['num_samples'][i]) for i in
                            range(self.args.client_num_in_total)]) / self.args.client_num_in_total
            stats = {'Local model person_test_acc': p_test_acc3, 'person_test_loss': p_test_loss3}
            # self.stat_info["person_test_acc"].append(p_test_acc)
            self.logger.info(stats)
            
            print("person_test_acc_after:%.3f" %(p_test_acc3*100))
            del w_local_mdl,w_per_mdls_lstrd
            gc.collect()
            ############################################################################################################
            p_train_acc_after = (p_train_acc1+p_train_acc2+p_train_acc3)/3
            p_train_loss_after = (p_train_loss1+p_train_loss2+p_train_loss3)/3
            p_test_acc_after = (p_test_acc1+p_test_acc2+p_test_acc3)/3
            p_test_loss_after = (p_test_loss1+p_test_loss2+p_test_loss3)/3
            
            # self.logger.info('person_train_loss={}'.format(p_train_loss))  
            # self.logger.info('person_test_acc={}'.format(p_test_acc))  
            # self.logger.info('person_train_acc={}'.format(p_train_acc))  
            # self.logger.info('person_test_loss={}'.format(p_test_loss))  
            stats = {'person_train_acc': p_train_acc_after, 'person_train_loss': p_train_loss_after}
            self.stat_info["person_train_acc"].append(p_train_acc_after)
            self.stat_info["train_loss_result"].append(p_train_loss_after) 
            self.logger.info(stats)
            
            stats = {'person_test_acc': p_test_acc_after, 'person_test_loss': p_test_loss_after}
            self.stat_info["person_test_acc"].append(p_test_acc_after)
            self.stat_info["test_loss_result"].append(p_test_loss_after) 
            self.logger.info(stats)
            
            #######################################cal dis#######################################
            globle_distance1 = 0
            globle_distance2 = 0
            globle_distance3 = 0 #w_per_mdls1[cur_clnt]
            for w1,w2,w3 in zip(w_per_mdls1, w_per_mdls2, w_per_mdls3):
                globle_distance1 += self.cal_distance(w1,w2)
                globle_distance2 += self.cal_distance(w2,w3)
                globle_distance3 += self.cal_distance(w3,w1)
                
            globle_distance1 = globle_distance1.cpu().detach().numpy()
            globle_distance2 = globle_distance2.cpu().detach().numpy()
            globle_distance3 = globle_distance3.cpu().detach().numpy()
            globle_distance = (globle_distance1+globle_distance2+globle_distance3)/3                
            stats = {'global_distance': globle_distance}
            self.stat_info["weight_distance"].append(globle_distance) 
            self.logger.info(stats)
            print("globle_distance:%.3f" %(globle_distance))
            
            if (round_idx+1) % 50 ==0:
                self.logger.info('person_test_acc50={}'.format(self.stat_info["person_test_acc"]))  
                self.logger.info('person_train_loss50={}'.format(self.stat_info["train_loss_result"]))  
                self.logger.info('weight_distance50={}'.format(self.stat_info["weight_distance"]))  

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])
            
        self.logger.info('test_acc_result499={}'.format(self.stat_info["person_test_acc"]))   
        self.logger.info('train_loss_result499={}'.format(self.stat_info["train_loss_result"]))  
        self.logger.info('weight_distance_result499={}'.format(self.stat_info["weight_distance"]))  
        self.logger.info('train_acc_result499={}'.format(self.stat_info["person_train_acc"]))  
        self.logger.info('test_loss_result499={}'.format(self.stat_info["test_loss_result"]))  
        
      
        return self.stat_info["train_loss_result"],self.stat_info["test_loss_result"],self.stat_info["weight_distance"],self.stat_info["person_test_acc"]
 

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        self.logger.info("client_indexes = %s" % str(client_indexes))
        return client_indexes
    
    def cal_distance(self,w1,w2):
        values1 = list(w1.values())
        values2 = list(w2.values())
        params1 = torch.cat([p.view(-1) for p in values1])
        params2 = torch.cat([p.view(-1) for p in values2])
        distance = torch.dist(params1, params2, p=2)
        
        return distance
    
    def _aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, _) = w_locals[idx]
            training_num += sample_num
        w_global ={}
        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    w_global[k] = local_model_params[k] * w
                else:
                    w_global[k] += local_model_params[k] * w
        return w_global

    def _test_on_all_clients3(self, w_global, w_per_mdls, round_idx,client_list):

        self.logger.info("################global_test_on_all_clients : {}".format(round_idx))

        g_test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        p_test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }
        
        p_train_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        for client_idx in range(self.args.client_num_in_total):
            # test data
            client = client_list[client_idx]
            # g_test_local_metrics = client.local_test(w_global, True)
            # g_test_metrics['num_samples'].append(copy.deepcopy(g_test_local_metrics['test_total']))
            # g_test_metrics['num_correct'].append(copy.deepcopy(g_test_local_metrics['test_correct']))
            # g_test_metrics['losses'].append(copy.deepcopy(g_test_local_metrics['test_loss']))

            p_test_local_metrics = client.local_test(w_per_mdls[client_idx], True)
            p_test_metrics['num_samples'].append(copy.deepcopy(p_test_local_metrics['test_total']))
            p_test_metrics['num_correct'].append(copy.deepcopy(p_test_local_metrics['test_correct']))
            p_test_metrics['losses'].append(copy.deepcopy(p_test_local_metrics['test_loss']))
            
            p_train_local_metrics = client.local_test(w_per_mdls[client_idx], False)
            p_train_metrics['num_samples'].append(copy.deepcopy(p_train_local_metrics['test_total']))
            p_train_metrics['num_correct'].append(copy.deepcopy(p_train_local_metrics['test_correct']))
            p_train_metrics['losses'].append(copy.deepcopy(p_train_local_metrics['test_loss']))

            """
            Note: CI environment is CPU-based computing. 
            The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
            """
            if self.args.ci == 1:
                break
        p_test_acc = sum(
            [np.array(p_test_metrics['num_correct'][i]) / np.array(p_test_metrics['num_samples'][i]) for i in
             range(self.args.client_num_in_total)]) / self.args.client_num_in_total
        p_test_loss = sum([np.array(p_test_metrics['losses'][i]) / np.array(p_test_metrics['num_samples'][i]) for i in
                           range(self.args.client_num_in_total)]) / self.args.client_num_in_total
        
        p_train_acc = sum(
            [np.array(p_train_metrics['num_correct'][i]) / np.array(p_train_metrics['num_samples'][i]) for i in
             range(self.args.client_num_in_total)]) / self.args.client_num_in_total
        p_train_loss = sum([np.array(p_train_metrics['losses'][i]) / np.array(p_train_metrics['num_samples'][i]) for i in
                           range(self.args.client_num_in_total)]) / self.args.client_num_in_total


     
        return p_train_loss, p_train_acc,p_test_acc,p_test_loss

    def record_avg_inference_flops(self, w_global, mask_pers=None):
        inference_flops = []
        for client_idx in range(self.args.client_num_in_total):

            if mask_pers == None:
                inference_flops += [self.model_trainer.count_inference_flops(w_global)]
            else:
                w_per = {}
                for name in mask_pers[client_idx]:
                    w_per[name] = w_global[name] * mask_pers[client_idx][name]
                inference_flops += [self.model_trainer.count_inference_flops(w_per)]
        avg_inference_flops = sum(inference_flops) / len(inference_flops)
        self.stat_info["avg_inference_flops"] = avg_inference_flops

    def init_stat_info(self):
        self.stat_info = {}
        self.stat_info["sum_comm_params"] = 0
        self.stat_info["sum_training_flops"] = 0
        self.stat_info["avg_inference_flops"] = 0
        self.stat_info["global_test_acc"] = []
        self.stat_info["global_train_acc"] = []
        self.stat_info["person_train_acc"] = []
        self.stat_info["final_masks"] = []
        
        self.stat_info["person_test_acc"] = []
        self.stat_info["train_loss_result"] =[]
        self.stat_info["test_loss_result"] =[]
        self.stat_info["distance_loss_result"] =[]
        self.stat_info["weight_distance"] = []

        

    def _benefit_choose(self, round_idx, cur_clnt, client_num_in_total, client_num_per_round, cs = False):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
            return client_indexes
        if cs == "random": 
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx + cur_clnt)  
            # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
            while cur_clnt in client_indexes:
                client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)

        elif cs == "ring":
            left = (cur_clnt - 1 + client_num_in_total) % client_num_in_total
            right = (cur_clnt + 1) % client_num_in_total
            client_indexes = np.asarray([left, right])
        elif cs == "grid":
            left = (cur_clnt - 1 + client_num_in_total) % client_num_in_total
            right = (cur_clnt + 1) % client_num_in_total
            top = (cur_clnt - 9 + client_num_in_total) % client_num_in_total
            down = (cur_clnt + 9 + client_num_in_total) % client_num_in_total
            client_indexes = np.asarray([left, right, top, down])

        elif cs =="exp": # (2^6<100<2^7)
            n1 = (cur_clnt + 1 + client_num_in_total) % client_num_in_total
            n2 = (cur_clnt + 2 + client_num_in_total) % client_num_in_total
            n3 = (cur_clnt + 4 + client_num_in_total) % client_num_in_total
            n4 = (cur_clnt + 8 + client_num_in_total) % client_num_in_total
            n5 = (cur_clnt + 16 + client_num_in_total) % client_num_in_total
            n6 = (cur_clnt + 32 + client_num_in_total) % client_num_in_total
            n7 = (cur_clnt + 64 + client_num_in_total) % client_num_in_total
            client_indexes = np.asarray([n1,n2,n3,n4,n5,n6,n7])

        elif cs == "full":
            client_indexes = np.arange(client_num_in_total)
            client_indexes = np.delete(client_indexes, cur_clnt)


        # self.logger.info("client_indexes = %s" % str(client_indexes))
        return client_indexes
        # For fedslim, we update the meta network for training purpose

    def _aggregate_func(self, cur_clnt, client_num_in_total, client_num_per_round, nei_indexs, w_per_mdls_lstrd):
        # self.logger.info('Doing local aggregation!')
        w_tmp = copy.deepcopy(w_per_mdls_lstrd[cur_clnt])
        w = 1 / len(nei_indexs)
        for k in w_tmp.keys():
            w_tmp[k] = w_tmp[k] - w_tmp[k]
            for clnt in nei_indexs:
                w_tmp[k] += w_per_mdls_lstrd[clnt][k] * w

        return w_tmp