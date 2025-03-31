import argparse
import logging
import os
import random
import sys
import pdb
import numpy as np
import torch
import time

torch.set_num_threads(1)
import os
sys.path.insert(0, os.path.abspath("/data/users/lyq/dfedalt_sysu240820/")) #
from fedml_api.model.cv.vgg import vgg11
from fedml_api.data_preprocessing.cifar10_dis.data_loader import load_partition_data_cifar10
from fedml_api.data_preprocessing.cifar100_dis.data_loader import load_partition_data_cifar100
from fedml_api.data_preprocessing.tiny_imagenet.data_loader import load_partition_data_tiny
from fedml_api.model.cv.resnet import  customized_resnet18, tiny_resnet18
from fedml_api.model.cv.cnn_cifar10 import cnn_cifar10, cnn_cifar100,cnn_emnist

from fedml_api.standalone.dfedalt_dis.dfedalt_dis_api import DFedAltAPI
from fedml_api.standalone.dfedalt_dis.my_model_trainer import MyModelTrainer

def logger_config(log_path, logging_name):
    logger = logging.getLogger(logging_name)
    logger.setLevel(level=logging.DEBUG)
    handler = logging.FileHandler(log_path, mode='w',encoding='UTF-8')
    handler.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--model', type=str, default='vgg11', metavar='N',
                        help="network architecture, supporting 'cnn_cifar10', 'cnn_cifar100', 'resnet18', 'vgg11'")

    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--momentum', type=float, default=0, metavar='N',
                        help='momentum')

    parser.add_argument('--data_dir', type=str, default='data/',
                        help='data directory, please feel free to change the directory to the right place')

    parser.add_argument('--partition_method', type=str, default='dir', metavar='N',
                        help="current supporting three types of data partition, one called 'dir' short for Dirichlet"
                             "one called 'n_cls' short for how many classes allocated for each client"
                             "and one called 'my_part' for partitioning all clients into PA shards with default latent Dir=0.3 distribution")

    parser.add_argument('--partition_alpha', type=float, default=0.1, metavar='PA',
                        help='available parameters for data partition method')

    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='local batch size for training')

    parser.add_argument('--client_optimizer', type=str, default='sgd',
                        help='SGD with momentum; adam')

    parser.add_argument('--lr_body', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    
    parser.add_argument('--lr_head', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.1)')

    parser.add_argument('--lr_decay', type=float, default=1, metavar='LR_decay',
                        help='learning rate decay (default: 0.998)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0)

    parser.add_argument('--head_epochs', type=int, default=1, metavar='EP',
                        help='local training epochs for each client')
    
    parser.add_argument('--body_epochs', type=int, default=5, metavar='EP',
                        help='local training epochs for each client')

    parser.add_argument('--client_num_in_total', type=int, default=20, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--frac', type=float, default=0.2, metavar='NN',
                        help='selection fraction each round')

    parser.add_argument('--comm_round', type=int, default=300,
                        help='how many round of communications we shoud use')

    parser.add_argument('--frequency_of_the_test', type=int, default=50,
                        help='the frequency of the algorithms')

    parser.add_argument('--gpu', type=int, default=5,help='gpu')

    parser.add_argument('--ci', type=int, default=0,help='CI')
    parser.add_argument('--cs', type=str, default='random',help='CI')
    
    parser.add_argument("--tag", type=str, default="test")

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument('--num_experiments', type=int, default=1,help='the number of experiments')
    
    return parser

def load_data(args, dataset_name):
    if dataset_name == "cifar10":
        args.data_dir += "cifar10"
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_cifar10(args.data_dir, args.partition_method,
                                args.partition_alpha, args.client_num_in_total, args.batch_size, logger)
        
    else:
        if dataset_name == "cifar100":
            args.data_dir += "cifar100"
            train_data_num, test_data_num, train_data_global, test_data_global, \
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
            class_num = load_partition_data_cifar100(args.data_dir, args.partition_method,
                                                     args.partition_alpha, args.client_num_in_total,
                                                     args.batch_size, logger)
        elif dataset_name == "tiny":
            args.data_dir += "tiny_imagenet"
            train_data_num, test_data_num, train_data_global, test_data_global, \
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
            class_num = load_partition_data_tiny(args.data_dir, args.partition_method,
                                                     args.partition_alpha, args.client_num_in_total,
                                                     args.batch_size, logger)
    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]
    return dataset

def load_data3(args, dataset_name):
    if dataset_name == "cifar10":
        args.data_dir += "cifar10"
        train_data_num1, test_data_num1, train_data_global1, test_data_global1, \
        train_data_local_num_dict1, train_data_local_dict1, test_data_local_dict1, class_num  = \
        load_partition_data_cifar10(args.data_dir, args.partition_method,
                                args.partition_alpha, args.client_num_in_total, args.batch_size, logger,idex=1)
        train_data_num2, test_data_num2, train_data_global2, test_data_global2, \
        train_data_local_num_dict2, train_data_local_dict2, test_data_local_dict2, class_num  = \
        load_partition_data_cifar10(args.data_dir, args.partition_method,
                                args.partition_alpha, args.client_num_in_total, args.batch_size, logger,idex=2)
        train_data_num3, test_data_num3, train_data_global3, test_data_global3, \
        train_data_local_num_dict3, train_data_local_dict3, test_data_local_dict3, class_num  = \
        load_partition_data_cifar10(args.data_dir, args.partition_method,
                                args.partition_alpha, args.client_num_in_total, args.batch_size, logger,idex=3)

    else:
        if dataset_name == "cifar100":
            args.data_dir += "cifar100"
            train_data_num1, test_data_num1, train_data_global1, test_data_global1, \
            train_data_local_num_dict1, train_data_local_dict1, test_data_local_dict1, class_num  = \
            load_partition_data_cifar100(args.data_dir, args.partition_method,
                                args.partition_alpha, args.client_num_in_total, args.batch_size, logger,idex=1)
            train_data_num2, test_data_num2, train_data_global2, test_data_global2, \
            train_data_local_num_dict2, train_data_local_dict2, test_data_local_dict2, class_num  = \
            load_partition_data_cifar100(args.data_dir, args.partition_method,
                                args.partition_alpha, args.client_num_in_total, args.batch_size, logger,idex=2)
            train_data_num3, test_data_num3, train_data_global3, test_data_global3, \
            train_data_local_num_dict3, train_data_local_dict3, test_data_local_dict3, class_num  = \
            load_partition_data_cifar100(args.data_dir, args.partition_method,
                                args.partition_alpha, args.client_num_in_total, args.batch_size, logger,idex=3)
    
        elif dataset_name == "tiny":
            args.data_dir += "tiny_imagenet"
            train_data_num, test_data_num, train_data_global, test_data_global, \
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
            class_num = load_partition_data_tiny(args.data_dir, args.partition_method,
                                                     args.partition_alpha, args.client_num_in_total,
                                                     args.batch_size, logger)
    
    dataset1 = [train_data_num1, test_data_num1, train_data_global1, test_data_global1,
               train_data_local_num_dict1, train_data_local_dict1, test_data_local_dict1, class_num]
    dataset2 = [train_data_num2, test_data_num2, train_data_global2, test_data_global2,
               train_data_local_num_dict2, train_data_local_dict2, test_data_local_dict2, class_num]
    dataset3 = [train_data_num3, test_data_num3, train_data_global3, test_data_global3,
               train_data_local_num_dict3, train_data_local_dict3, test_data_local_dict3, class_num]
    return dataset1,dataset2,dataset3


def create_model(args, model_name,class_num,logger):
    logger.info("create_model. model_name = %s" % (model_name))
    model = None

    if model_name == "cnn_cifar10":
        model = cnn_cifar10()
    elif model_name == "cnn_cifar100":
        model = cnn_cifar100()
        
    elif model_name == "cnn_emnist":   
        model = cnn_emnist(class_num)
        
    elif model_name =="resnet18" and args.dataset != 'tiny':
        model = customized_resnet18(class_num=class_num)
  
    elif model_name == "resnet18" and args.dataset == 'tiny':
        model = tiny_resnet18(class_num=class_num)
    elif model_name == "vgg11":
        model = vgg11(class_num)
        
        
    return model


def custom_model_trainer(args, model, logger):
    return MyModelTrainer(model, args, logger)


if __name__ == "__main__":

    parser = add_args(argparse.ArgumentParser(description='DFedAlt-standalone'))
    args = parser.parse_args()


    train_loss_result_list = []
    test_loss_result_list = []
    distance_weight_result_list = []
    test_acc_result_list = []
    
    for exper_index in range(args.num_experiments):
        random.seed(args.seed+exper_index)
        np.random.seed(args.seed+exper_index)
        torch.manual_seed(args.seed+exper_index)
        torch.cuda.manual_seed_all(args.seed+exper_index)
        torch.backends.cudnn.deterministic = True
        args. client_num_per_round = int(args.client_num_in_total* args.frac)
        
        data_partition = args.partition_method
        if data_partition != "homo":
            data_partition += str(args.partition_alpha)
        args.identity = "dfedalt_dis"  + "-"+data_partition
        args.identity +=  args.model
        
        args.identity  += "total-clnt" + str(args.client_num_in_total)
        args.identity += "-frac" + str(args.frac) 
        args.identity += "-cm" + str(args.comm_round) 
        args.identity +="-epochs"+ str(args.body_epochs) 
        args.identity +="-lr"+ str(args.lr_body) 
        args.identity += '-seed' + str(args.seed+exper_index)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        args.identity  += "-"+timestr

        cur_dir = os.path.abspath(__file__).rsplit("/", 1)[0]
        log_path = os.path.join(cur_dir, 'LOG/' + args.dataset + '/' + args.identity + '.log')
        logger = logger_config(log_path='LOG/' + args.dataset + '/' + args.identity + '.log', logging_name=args.identity)

        logger.info(args)
        device = torch.device("cuda:" + str(args.gpu) )
        logger.info(device)
        logger.info("running at device{}".format(device))

        # load data
        dataset1,dataset2,dataset3= load_data3(args, args.dataset)

        # create model.
        if args.dataset =="emnist":
            model = create_model(args, model_name=args.model, class_num= 62, logger = logger)
        elif args.dataset =="cifar10" :
            model = create_model(args, model_name=args.model, class_num= 10, logger = logger)
        elif args.dataset =="cifar100":
            model = create_model(args, model_name=args.model, class_num= 100, logger = logger)
        elif args.dataset =="tiny":
            model = create_model(args, model_name=args.model, class_num= 200, logger = logger)
        # print(model)
        model_trainer = custom_model_trainer(args, model, logger)
        logger.info(model)

        dfedAltAPI = DFedAltAPI(dataset1,dataset2,dataset3, device, args, model_trainer, logger)

        train_loss_result, test_loss_result, distance_weight_result , test_acc_result = dfedAltAPI.train()
        
        train_loss_result_list.append(train_loss_result)
        test_loss_result_list.append(test_loss_result)
        distance_weight_result_list.append(distance_weight_result)
        test_acc_result_list.append(test_acc_result)
        
        
    # ## list average
    train_loss_average = np.mean(train_loss_result_list,axis=0).tolist()
    test_loss_average =  np.mean(test_loss_result_list,axis=0).tolist()
    distance_weight_average =  np.mean(distance_weight_result_list,axis=0).tolist()
    test_acc_average =  np.mean(test_acc_result_list,axis=0).tolist()
    logger.info('test_acc_average={}'.format(test_acc_average)) 
    logger.info('train_loss_average={}'.format(train_loss_average))  
    logger.info('distance_weight_average={}'.format(distance_weight_average)) 
    logger.info('test_loss_average={}'.format(test_loss_average))  
    
    print("record over!")    
        
        
