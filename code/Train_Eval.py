import os
import sys
sys.path.append("/home/qwe/data/liuhaitao/graphnn")
from tqdm import tqdm
import torch
#-----------------
# print(f"{torch.__version__=}")
# print(f"{torch.__file__=}")
# print(f"{torch.cuda.device_count()=}")
# print(f"{torch.cuda.is_available()=}")
# print(f"{torch.version.cuda=}")
#-----------------已停
# import torch
# from torch_geometric.data import Data
# from torch_geometric.loader import DataLoader
# from torch.utils.data import Subset, TensorDataset


import deepadr

from deepadr.dataset import *
from deepadr.utilities import *
from deepadr.chemfeatures import *
from deepadr.train_functions import *
from deepadr.model_gnn_ogb import GNN, DeepAdr_SiameseTrf, ExpressionNN

from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
import torch.multiprocessing as mp



def spawn_q_process(q_process):
    print(">>> spawning hyperparam search process")
    q_process.start()


def join_q_process(q_process):
    q_process.join()
    print("<<< joined hyperparam search process")


def create_q_process(queue, used_dataset, gpu_num, tphp, exp_dir, partition):
    #     fold_gpu_map = {0:gpu_num}
    return mp.Process(target=deepadr.train_functions.run_exp,
                      args=(queue, used_dataset, gpu_num, tphp, exp_dir, partition))



def main():
    rawdata_dir = '../data/raw/'
    processed_dir = '../data/processed/'
    up_dir = '..'

    report_available_cuda_devices()

    n_gpu = torch.cuda.device_count()

    device_cpu = get_device(to_gpu=False)
    # device_gpu = get_device(True, index=0)
#-----------------
    # print("torch:", torch.__version__)
    # print("CUDA:", torch.version.cuda)
#-------------已停
    # Preparing dataset
    # options:
    # 'total_thresh' + 4,3,2
    # 'loewe_thresh', 'hsa_thresh', 'bliss_thresh', 'zip_thresh' + 1

    score = 'total_thresh'
    score_val = 4
    # desired_cell_line = 'IST-MEL1'
    DSdataset_name = f'DrugComb_{score}_{score_val}'

    data_fname = 'data_v1'  # v2 for baseline models, v3 for additive samples

    targetdata_dir = create_directory(os.path.join(processed_dir, DSdataset_name, data_fname))
    targetdata_dir_raw = create_directory(os.path.join(targetdata_dir, "raw"))
    targetdata_dir_processed = create_directory(os.path.join(targetdata_dir, "processed"))
    targetdata_dir_exp = create_directory(os.path.join(targetdata_dir, "experiments"))
    # ReaderWriter.dump_data(dpartitions, os.path.join(targetdata_dir, 'data_partitions.pkl'))

    # Make sure to first run the "DDoS_Dataset_Generation" notebook first

    dataset = MoleculeDataset(root=targetdata_dir)


    used_dataset = dataset

    # If you want to use a smaller subset of the dataset for testing
    # smaller_dataset_len = int(len(dataset)/1)
    # used_dataset = dataset[:smaller_dataset_len]

    fold_partitions = get_stratified_partitions(dataset.data.y,
                                                num_folds=5,
                                                valid_set_portion=0.1,
                                                random_state=42)

    print("Number of training graphs: " + str(len(fold_partitions[0]['train'])))
    print("Number of validation graphs: " + str(len(fold_partitions[0]['validation'])))
    print("Number of testing graphs: " + str(len(fold_partitions[0]['test'])))

    # training parameters

    tp = {
        "batch_size": 300,
        "num_epochs": 100,

        "emb_dim": 100,
        "gnn_type": "gatv2",
        "num_layer": 3,
        "graph_pooling": "mean",  # attention

        "input_embed_dim": None,
        "gene_embed_dim": 1,
        "num_attn_heads": 2,
        "num_transformer_units": 1,
        "p_dropout": 0.3,
        #     "nonlin_func" : nn.ReLU(),
        "mlp_embed_factor": 2,
        "pooling_mode": 'attn',
        "dist_opt": 'cosine',

        "base_lr": 3e-4,  # 3e-4
        "max_lr_mul": 10,
        "l2_reg": 1e-7,
        "loss_w": 1.,
        "margin_v": 1.,

        "expression_dim": 64,
        "expression_input_size": 496,
        "exp_H1": 4096,
        "exp_H2": 1024
    }

    mp.set_start_method("spawn", force=True)

    queue = mp.Queue()
    q_processes = []

    time_stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    print("Start: " + time_stamp)

    for q_i in range(min(n_gpu, len(fold_partitions))):
        partition = fold_partitions[q_i]
        exp_dir = create_directory(os.path.join(targetdata_dir_exp, "fold_" + str(q_i) + "_" + time_stamp))
        create_directory(os.path.join(exp_dir, "predictions"))
        create_directory(os.path.join(exp_dir, "modelstates"))

        q_process = create_q_process(queue, dataset, q_i, tp, exp_dir, partition)
        q_processes.append(q_process)
        spawn_q_process(q_process)

    spawned_processes = n_gpu

    for q_i in range(min(n_gpu, len(fold_partitions))):
        join_q_process(q_processes[q_i])
        released_gpu_num = queue.get()
        print("released_gpu_num:", released_gpu_num)

    print("End: " + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

if __name__ == '__main__':
    main()






