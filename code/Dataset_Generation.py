import os
import sys
import numpy as np
import pandas as pd
import datetime
import seaborn as sns
import ogb
import ast

pd.set_option('display.max_rows', 100)

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader

import deepadr
from deepadr.dataset import *
from deepadr.utilities import *
from deepadr.chemfeatures import *
from deepadr.train_functions_flat import *
from deepadr.model_gnn_ogb import GNN, DeepAdr_SiameseTrf, ExpressionNN
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from sklearn.preprocessing import StandardScaler

rawdata_dir = '../data/raw/'
processed_dir = '../data/processed/'
up_dir = '..'

report_available_cuda_devices()

n_gpu = torch.cuda.device_count()

fdtype = torch.float32
print("torch:", torch.__version__)
print("CUDA:", torch.version.cuda)

# Preparing dataset
# options:
# 'total_thresh' + 4,3,2
# 'loewe_thresh', 'hsa_thresh', 'bliss_thresh', 'zip_thresh' + 1
#'A-673','SF-268','SNB-75','T98G','BT-549',

score = 'total_thresh'
score_val = 4
# desired_cell_line = 'IST-MEL1'
DSdataset_name = f'DrugComb_{score}_{score_val}'
data_fname = 'data_v1'  # v2 for baseline models, v3 for additive samples, v4 for additive baseline

df_drugcomb = pd.read_csv(f'../data/preprocessing/drugcomb_{score}_{score_val}.csv')
df_rma_landm = pd.read_csv('../data/preprocessing/df_rma_landm.tsv', sep="\t")

targetdata_dir = create_directory(os.path.join(processed_dir, DSdataset_name, data_fname))
targetdata_dir_raw = create_directory(os.path.join(targetdata_dir, "raw"))
targetdata_dir_processed = create_directory(os.path.join(targetdata_dir, "processed"))

data = df_drugcomb

data = data.drop(index=data[(data.Drug1.str.contains("Antibody")) | (data.Drug2.str.contains("Antibody"))].index)
data.index = range(len(data))
uniq_data = ddi_dataframe_to_unique_drugs(data)

uniq_data.Drug = [d.split("; ")[1] if ("; " in d) else d for d in uniq_data.Drug]
uniq_data.Drug = [d.split(";")[1] if (";" in d) else d for d in uniq_data.Drug]
uniq_data['Mol'] = [smiles_to_mol(smiles) for smiles in uniq_data.Drug]
uniq_mol = uniq_data[~uniq_data.Mol.isnull()]
uniq_mol['DataOGB'] = [smiles_to_graph_data_obj_ogb(smiles) for smiles in uniq_mol.Drug]
uniq_mol = uniq_mol.set_index("Drug_ID")
Draw.MolsToGridImage(uniq_mol.Mol.head(24), molsPerRow=6)

if (data_fname == 'data_v2' or data_fname == 'data_v4'):  # baseline model
    print("generating xFlat for", data_fname)
    uniq_mol['xFlat'] = [torch.mean(torch.clone(data.x).type(torch.float32), dim=0) for data in uniq_mol['DataOGB']]

y = data.Y.copy()
ReaderWriter.dump_data(y.values, os.path.join(targetdata_dir_raw, 'y.pkl'))
expression = np.array([df_rma_landm['DATA.' + str(c)].values for c in data["Cosmic_ID"]])
ReaderWriter.dump_data(expression, os.path.join(targetdata_dir_raw, 'expression.pkl'))
pairs = {i: (row.Drug1_ID, row.Drug2_ID) for i, row in data.iterrows()}
ReaderWriter.dump_data(pairs, os.path.join(targetdata_dir_raw, 'pairs.pkl'))
ReaderWriter.dump_data(data, os.path.join(targetdata_dir_raw, 'data_pairs.pkl'))

if (data_fname == 'data_v1' or data_fname == 'data_v3'):  # gnn model
    X = ReaderWriter.read_or_dump_data(file_name=norm_join_paths(targetdata_dir_raw, 'X.pkl'),
                                       data_gen_fun=get_X_all_pairdata_synergy,
                                       data_gen_params=(uniq_mol, pairs, "DataOGB"))

if (data_fname == 'data_v2' or data_fname == 'data_v4'):  # baseline model
    X = ReaderWriter.read_or_dump_data(file_name=norm_join_paths(targetdata_dir_raw, 'X_flat.pkl'),
                                       data_gen_fun=get_X_all_pairdata_synergy_flat,
                                       data_gen_params=(uniq_mol, pairs, "xFlat"))

dataset = MoleculeDataset(root=targetdata_dir,dataset='tdcSynergy')

print()
print(f'Dataset: {dataset}:')
print('====================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')
