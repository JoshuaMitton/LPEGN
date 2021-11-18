import os
os.umask(0o002)
import sys
sys.path.append(os.path.dirname(__file__)+'/../')

from data_loader.data_generator import datagenerator
from models.equivariant_gnn_v2 import equivariant_gnn
from trainers.trainer import Trainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils import doc_utils
from utils.utils import get_args
import torch
import numpy as np
import random

def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)

    except Exception as e:
        print("missing or invalid arguments %s" % e)
        exit(0)

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    
    if config.cuda:
        print(f'Using GPU : {torch.cuda.get_device_name(int(config.gpu))}')
    else:
        print(f'Using CPU')

    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    
    print("lr = {0}".format(config.learning_rate))
    print("decay = {0}".format(config.decay_rate))
    print(config.architecture)
    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    for exp in range(1, config.num_exp+1):
        for fold in range(1, 11):
            print("Experiment num = {0}\nFold num = {1}".format(exp, fold))
            # create your data generator
            config.num_fold = fold
            train_loader, val_loader, num_features_in = datagenerator(config)
            config.num_features_in = num_features_in
#             gpuconfig = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
#             gpuconfig.gpu_options.visible_device_list = config.gpus_list
#             gpuconfig.gpu_options.allow_growth = True
#             sess = tf.Session(config=gpuconfig)
            # create an instance of the model you want
#             model = invariant_basic(config, data)
            model = equivariant_gnn(config, config.weight_degrees, config.subgraph_degrees, config.degree_mapping)
            if config.cuda:
                model = model.cuda()
#                 model = torch.nn.DataParallel(model)
            # create trainer and pass all the previous components to it
            trainer = Trainer(model, train_loader, val_loader, config)
            # here you train your model
#             if fold == 1:
#                 pass
#             else:
            acc, loss = trainer.train()
            doc_utils.doc_results(acc, loss, exp, fold, config.summary_dir, ' ')
#             sess.close()
#             tf.reset_default_graph()

    doc_utils.summary_10fold_results(config.summary_dir)

if __name__ == '__main__':
    main()
