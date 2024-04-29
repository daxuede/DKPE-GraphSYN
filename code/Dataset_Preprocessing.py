import os
import sys
import numpy as np
import pandas as pd
import datetime
import seaborn as sns
import ogb
import ast

pd.set_option('display.max_rows', 200)
import torch
from torch_geometric.data import Data, DataLoader
import deepadr
from deepadr.dataset import *
from deepadr.utilities import *
from deepadr.chemfeatures import *
from deepadr.train_functions_flat import *
from deepadr.model_gnn_ogb import GNN, DeepAdr_SiameseTrf, ExpressionNN
from ogb.graphproppred import Evaluator
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw

rawdata_dir = '../data/raw/'
processed_dir = '../data/processed/'
up_dir = '..'

report_available_cuda_devices()

n_gpu = torch.cuda.device_count()

fdtype = torch.float32

print("torch:", torch.__version__)
print("CUDA:", torch.version.cuda)

# DrugComb - v1.5
#目的是从一个CSV文件中读取数据并创建一个数据框（DataFrame），.dropna(): 这是一个pandas数据框方法，用于删除包含缺失值（NaN）的行
df_cellosaurus = pd.read_csv('../data/preprocessing/cellosaurus_cosmic_ids.txt', sep=',', header=None).dropna()

#将数据框中的两列转换成字典，第一列的元素作为键，第二列的元素作为相应的值。
dict_cellosaurus = dict(zip(df_cellosaurus[0], df_cellosaurus[1]))

df_drugcomb_drugs = pd.read_json('../data/preprocessing/drugs.json')

dict_smiles = dict(zip(df_drugcomb_drugs.dname, df_drugcomb_drugs.smiles))

df_drugcomb = pd.read_csv('../data/preprocessing/summary_v_1_5.csv')

#这一行代码创建了一个新的列"cosmicId"，其中的值是根据字典dict_cellosaurus映射的细胞系名称对应的Cosmic ID。如果细胞系名称在字典中找不到对应的Cosmic ID，则该值被设置为nan。
df_drugcomb["cosmicId"] = [dict_cellosaurus[cell] if cell in dict_cellosaurus.keys() else float('nan') for cell in
                           df_drugcomb['cell_line_name']]

df_drugcomb = df_drugcomb.replace({'\\N': float('nan')}).astype({"synergy_loewe": float}).dropna(subset=[
    'drug_row', 'drug_col', 'cell_line_name', 'synergy_zip', 'synergy_loewe', 'synergy_hsa', 'synergy_bliss', 'cosmicId'
])

df_drugcomb = df_drugcomb.astype({"cosmicId": int})


def synergy_threshold(val):
    res = 0
    if (val >= 10.0):
        res = 1
    if (val <= -10.0):
        res = -1
    return res


df_drugcomb["drug_row_smiles"] = [dict_smiles[drug] for drug in df_drugcomb.drug_row]
df_drugcomb["drug_col_smiles"] = [dict_smiles[drug] for drug in df_drugcomb.drug_col]
null_smiles = df_drugcomb[(df_drugcomb.drug_row_smiles == "NULL") | (df_drugcomb.drug_col_smiles == "NULL")].index
df_drugcomb = df_drugcomb.drop(index=null_smiles)

df_drugcomb["loewe_thresh"] = [synergy_threshold(val) for val in df_drugcomb.synergy_loewe]
df_drugcomb["zip_thresh"] = [synergy_threshold(val) for val in df_drugcomb.synergy_zip]
df_drugcomb["hsa_thresh"] = [synergy_threshold(val) for val in df_drugcomb.synergy_hsa]
df_drugcomb["bliss_thresh"] = [synergy_threshold(val) for val in df_drugcomb.synergy_bliss]



df_drugcomb["total_thresh"] = df_drugcomb[["loewe_thresh", "zip_thresh", "hsa_thresh", "bliss_thresh"]].sum(axis=1)

# Chose score

score = 'total_thresh'
score_val = 4
df_drugcomb_filter = df_drugcomb[df_drugcomb[score].abs() >= score_val].copy()
df_drugcomb_filter['Y'] = [1 if val >= score_val else 0 for val in df_drugcomb_filter[score]]
# df_drugcomb_filter['Y'] = df_drugcomb_filter[score]


# Drop duplicates删除重复项
dup_to_drop = []
df_drugcomb_filter_dedup = df_drugcomb_filter.copy()
cols = ['drug_row', 'drug_col', "cell_line_name"]
df_drugcomb_filter_dedup[cols] = np.sort(df_drugcomb_filter_dedup[cols].values, axis=1)
dup = df_drugcomb_filter_dedup.duplicated(subset=cols, keep=False)
dup_score = df_drugcomb_filter_dedup[dup][cols + ['Y']]
dup_val = dup_score.duplicated(keep=False)
print(dup_val.value_counts())
dup_val_true = df_drugcomb_filter_dedup[dup][cols + ['Y']][dup_val]  # same triplets and class
dup_val_false = df_drugcomb_filter_dedup[dup][cols + ['Y']][~dup_val]  # same triplets, other class
dup_val_true.duplicated(keep="first").value_counts()
dup_to_drop += list(dup_val_true[dup_val_true.duplicated(keep="first")].index)
dup2 = pd.concat([dup_val_false, dup_val_true[~dup_val_true.duplicated(keep="first")]], axis=0)
dup2_val = dup2.duplicated(subset=(cols), keep=False)  # .value_counts()
dup_to_drop += list(dup2[dup2_val].sort_values(cols).index)
df_drugcomb_filter = df_drugcomb_filter.drop(index=dup_to_drop)

# RMA
df_l1000 = pd.read_csv('../data/preprocessing/L1000genes.txt', sep='\t')
df_l1000_lm = df_l1000[df_l1000.Type == "landmark"]
lm_genes = list(df_l1000_lm.Symbol)
df_rma = pd.read_csv('../data/preprocessing/Cell_line_RMA_proc_basalExp.txt', sep='\t')
cosmic_found = set(df_drugcomb_filter.cosmicId)
cosmic_intersect = list(set(["DATA." + str(c) for c in cosmic_found]).intersection(set(df_rma.columns)))
df_drugcomb_filter = df_drugcomb_filter[
    df_drugcomb_filter.cosmicId.isin([int(c[len("DATA."):]) for c in cosmic_intersect])]
df_drugcomb_filter = df_drugcomb_filter.rename(columns={"drug_row": "Drug1_ID",
                                                        "drug_col": "Drug2_ID",
                                                        "cosmicId": "Cosmic_ID",
                                                        "cell_line_name": "Cell_Line_ID",
                                                        "drug_row_smiles": "Drug1",
                                                        "drug_col_smiles": "Drug2"})
# df_drugcomb_filter['Y'].value_counts()
# len(set(list(df_drugcomb_filter['Drug1_ID']) + list(df_drugcomb_filter['Drug2_ID'])))
# len(set(df_drugcomb_filter['Cell_Line_ID']))

# Gene Expression
df_rma_landm = df_rma[df_rma.GENE_SYMBOLS.isin(lm_genes)]
gene_gex = pd.DataFrame(df_rma_landm["GENE_SYMBOLS"].copy())
gene_gex["GEX"] = ["gex" + str(i) for i in range(len(gene_gex))]
# df_drugcomb_filter.Cell_Line_ID.value_counts()
gene_gex.to_csv('../data/preprocessing/gene_gex.tsv', sep='\t', index=False)

df_rma_landm.to_csv('../data/preprocessing/df_rma_landm.tsv', sep='\t')
col_sel = ['Drug1_ID', 'Drug2_ID', 'Cell_Line_ID', 'Cosmic_ID', 'Drug1', 'Drug2', 'Y']
df_drugcomb_filter[col_sel].to_csv(f'../data/preprocessing/drugcomb_{score}_{score_val}.csv', index=False)


