import torch
import argparse
import os
from multiprocessing import cpu_count

# Global parameters
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

cpu_num = cpu_count()
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)

def args_parser():
    parser = argparse.ArgumentParser()
    path_dir = os.path.dirname(__file__)
    # dl_training
    parser.add_argument('--seed', type=int, default=429023)
    parser.add_argument('--data_path', type=str, default="data/pair_data/pair_data.csv")
    parser.add_argument('--model_path', type=str, default="models/google-t5-t5-base")
    parser.add_argument('--objectives', nargs='+', default=["reflection", "coherence", "fluency"])
    parser.add_argument('--train_batch_size', type=int, default=10)
    parser.add_argument('--val_batch_size', type=int, default=10)
    parser.add_argument('--test_batch_size', type=int, default=10)    
    parser.add_argument('--test_history_size', type=int, default=10)
    parser.add_argument('--num_runs', type=int, default=10)
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--lora_path', type=str, default="results/")

    # dl_aggregation


    
    parser.add_argument('--test_name', type=str, default='Test01_ResNet18_LLT_Seed90_Lr0.01_BS256')
    parser.add_argument('--describe', type=str, help='you can write something about this test.')
    # parser.add_argument('--device', type=str, default='cuda', help='cuda/cpu')
    # parser.add_argument('--seed', type=int, default=90)

    parser.add_argument('--rule', type=str, default='LLT', \
                        help='LLT/AVG')
    parser.add_argument('--dataset_name', type=str, default='CIFAR', \
                        help='CIFAR/ImageNet')
    parser.add_argument('--network_name', type=str, default='ResNet18', \
                        help='LeNet/ResNet18/MobileNetV1/MobileNetV2/MobileNetV3L/MobileNetV3S')
    parser.add_argument('--num_classes', type=int, default=100, \
                        help='100/300/500/1000')
    
    parser.add_argument('--path_data', type=str, default=os.path.join(path_dir, 'Folder'))
    parser.add_argument('--path_cifar', type=str, default=os.path.join(path_dir, 'Folder/Data/Raw'))
    parser.add_argument('--path_imgnet', type=str, default=os.path.join(path_dir, 'Folder/Data/Raw/imagenet'))
    parser.add_argument('--path_tinyimagenet', type=str, default=os.path.join(path_dir, 'Folder/Data/Raw/tiny-imagenet-200'))
    parser.add_argument('--path_save_keypoints', type=str, default=os.path.join(path_dir, 'Results/'))

    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--num_candidate_clients', type=int, default=10)

    parser.add_argument('--batch_size_local_training', type=int, default=256)
    parser.add_argument('--batch_size_global_test', type=int, default=500, help='batch size in test eval')

    parser.add_argument('--num_rounds_global', type=int, default=100, help='global rounds')
    parser.add_argument('--num_epochs_local_training', type=int, default=4)
    parser.add_argument('--num_epochs_retrain', type=int, default=20)

    parser.add_argument('--lr_local_training', type=float, default=1e-2)
    parser.add_argument('--momentum', type=float, default=0.9)
    
    # Datasets
    parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor(IF)')
    parser.add_argument('--num_workers', default=8, type=int, help='dataloader num_workers')
    parser.add_argument('--prefetch_factor', default=8, type=int, help='dataloader prefetch_factor')
    parser.add_argument('--pin_memory', action='store_true', default=True, help='dataloader pin_memory')

    parser.add_argument('--save_log', action='store_true', default=False, help='save log when training')
    parser.add_argument('--save_options', action='store_true', default=False, help='save option when training')
    
    # FD_DC
    parser.add_argument('--lr_global_SCAFFOLD', type=float, default=0.90)
    parser.add_argument('--alpha_coef', type=float, default=1e-2, help='FedDC/FedDyn params')
    parser.add_argument('--mu', type=float, default=1e-4, help='prox params')

    # CReFF
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    args = parser.parse_args()

    return args

def save_options(args):
    args_dict = args.__dict__
    pth = '{}{}'.format(args.path_save_keypoints, args.test_name)
    if not os.path.exists(pth):
        os.mkdir(pth)
        
    with open('{}{}/log_options.txt'.format(args.path_save_keypoints, args.test_name), 'a') as f: 
        for eachArg, value in args_dict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')   
        f.close()