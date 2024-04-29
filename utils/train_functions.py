import os
import sys
import numpy
import pandas as pd
import datetime
import seaborn as sns
from tqdm import tqdm
from copy import deepcopy
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
import deepadr
from deepadr.GAN_MLP import DrugMLP
from deepadr.Transformer import Transformer
from deepadr.dataset import *
from deepadr.utilities import *
from deepadr.chemfeatures import *
from deepadr.model_gnn_ogb import GNN, DeepAdr_SiameseTrf, ExpressionNN, DeepSynergy
from deepadr.model_attn_siamese import GeneEmbAttention, GeneEmbProjAttention, DDCL_CNN, SimpleMLP
from ogb.graphproppred import Evaluator
import json
import functools
from torchvision.transforms import ToPILImage
from scipy.stats import gaussian_kde, stats
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

fdtype = torch.float32

torch.set_printoptions(precision=6)


def compose(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)


def F_score(a, b):
    return (2 * a * b) / (a + b)


def generate_tp_hp(tp, hp, hp_names):
    tphp = deepcopy(tp)
    for i, n in enumerate(hp_names):
        tphp[n] = hp[i]
    return tphp


def build_predictions_df(ids, true_class, pred_class, prob_scores):
    prob_scores_dict = {}
    for i in range(prob_scores.shape[-1]):
        prob_scores_dict[f'prob_score_class{i}'] = prob_scores[:, i]

    df_dict = {
        'id': ids,
        'true_class': true_class,
        'pred_class': pred_class
    }
    df_dict.update(prob_scores_dict)
    predictions_df = pd.DataFrame(df_dict)
    predictions_df.set_index('id', inplace=True)
    return predictions_df


def run_test(queue, used_dataset, gpu_num, tp, exp_dir, partition):
    print("gpu_num", gpu_num)


def run_exp(queue, used_dataset, gpu_num, tp, exp_dir, partition):  #

    num_classes = 2

    targetdata_dir_raw = os.path.abspath(exp_dir + "/../../raw")
    targetdata_dir_processed = os.path.abspath(exp_dir + "/../../processed")

    state_dict_dir = create_directory(os.path.join(exp_dir, 'modelstates'))

    device_gpu = get_device(True, index=gpu_num)
    print("gpu:", device_gpu)

    # Serialize data into file:
    json.dump(tp, open(exp_dir + "/hyperparameters.json", 'w'))

    tp['nonlin_func'] = nn.ReLU()

    expression_scaler = TorchStandardScaler()
    expression_scaler.fit(used_dataset.data.expression[partition['train']])

    train_dataset = Subset(used_dataset, partition['train'])
    val_dataset = Subset(used_dataset, partition['validation'])
    test_dataset = Subset(used_dataset, partition['test'])

    train_loader = DataLoader(train_dataset, batch_size=tp["batch_size"], shuffle=True, follow_batch=['x_a', 'x_b'])
    # print(train_loader.shape)
    valid_loader = DataLoader(val_dataset, batch_size=tp["batch_size"], shuffle=False, follow_batch=['x_a', 'x_b'])
    test_loader = DataLoader(test_dataset, batch_size=tp["batch_size"], shuffle=False, follow_batch=['x_a', 'x_b'])

    loaders = {"train": train_loader, "valid": valid_loader, "test": test_loader}

    gnn_model = GNN(gnn_type=tp["gnn_type"],
                    num_layer=tp["num_layer"],
                    emb_dim=tp["emb_dim"],
                    drop_ratio=0.5,
                    JK="multilayer",  # last
                    graph_pooling=tp["graph_pooling"],
                    virtual_node=False,
                    with_edge_attr=False).to(device=device_gpu, dtype=fdtype)

    expression_model = DeepSynergy(D_in=(2*tp["emb_dim"])+tp["expression_input_size"],
                                   H1=tp['exp_H1'], H2=tp['exp_H2'], drop=tp['p_dropout']).to(device=device_gpu,dtype=fdtype)

    gene_attn_model = SimpleMLP().to(device=device_gpu, dtype=fdtype)

    models_param = list(gnn_model.parameters()) + list(expression_model.parameters()) + list(
        gene_attn_model.parameters())

    model_name = "ogb"
    models = [(gnn_model, f'{model_name}_GNN'),
              (expression_model, f'{model_name}_Expression'),
              (gene_attn_model, f'{model_name}_GeneAttn'),
              ]

    y_weights = compute_class_weights(used_dataset.data.y[partition['train']])
    class_weights = torch.tensor(y_weights).type(fdtype).to(device_gpu)


    num_iter = len(train_loader)  # num_train_samples/batch_size
    c_step_size = int(np.ceil(5 * num_iter))  # this should be 2-10 times num_iter

    base_lr = tp['base_lr']
    max_lr = tp['max_lr_mul'] * base_lr  # 3-5 times base_lr
    optimizer = torch.optim.Adam(models_param, weight_decay=tp["l2_reg"], lr=base_lr)
    cyc_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr, max_lr, step_size_up=c_step_size,
                                                      mode='triangular', cycle_momentum=False)

    loss_nlll = torch.nn.NLLLoss(weight=class_weights, reduction='mean')  # negative log likelihood loss
    loss_contrastive = ContrastiveLoss(0.5, reduction='mean')

    valid_curve_aupr = []
    test_curve_aupr = []
    train_curve_aupr = []

    valid_curve_auc = []
    test_curve_auc = []
    train_curve_auc = []

    best_fscore = 0
    best_epoch = 0

    for epoch in range(tp["num_epochs"]):
        print("=====Epoch {}=====".format(epoch))
        print('Training...')

        for m, m_name in models:
            m.train()

        for i_batch, batch in enumerate(tqdm(train_loader, desc="Iteration")):
            batch = batch.to(device_gpu)

            h_a = gnn_model(batch.x_a, batch.edge_index_a, batch.edge_attr_a, batch.x_a_batch)
            h_b = gnn_model(batch.x_b, batch.edge_index_b, batch.edge_attr_b, batch.x_b_batch)


            expression_norm = expression_scaler.transform_ondevice(batch.expression, device=device_gpu)

            # image_tensors = []
            # for i in range(expression_norm.shape[0]):
            #     cell_line = expression_norm[i]
            #     left = cell_line[:int(len(cell_line) / 2)].reshape(-1, 1)
            #     right = cell_line[int(len(cell_line) / 2):].reshape(-1, 1)
            #     gene = torch.cat([left, right], axis=1).cpu()
            #     kde = gaussian_kde(gene.T)
            #     x = np.linspace(gene[0].min(), gene[0].max(), num=64)
            #     y = np.linspace(gene[1].min(), gene[1].max(), num=64)
            #     X, Y = np.meshgrid(x, y)
            #     positions = np.vstack([X.ravel(), Y.ravel()])
            #     # 计算估计的密度值
            #     density = np.reshape(kde(positions).T, X.shape)
            #     image_tensors.append(torch.tensor(density))
            #
            # # 将张量列表转换成一个张量
            # final_tensor = torch.stack(image_tensors)


            h_e = gene_attn_model(expression_norm.to("cuda").type(fdtype))


            triplet = torch.cat([h_a, h_b, h_e], axis=-1)

            logsoftmax_scores = expression_model(triplet)
            loss = loss_nlll(logsoftmax_scores, batch.y.type(torch.long))
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            cyc_scheduler.step()  # after each batch step the scheduler
            optimizer.zero_grad()  # Clear gradients.

        print('Evaluating...')

        perfs = {}

        for dsettype in ["train", "valid"]:
            for m, m_name in models:
                m.eval()

            pred_class = []
            ref_class = []
            prob_scores = []
            y_scores = []

            l_ids = []


            for i_batch, batch in enumerate(tqdm(loaders[dsettype], desc="Iteration")):
                batch = batch.to(device_gpu)

                h_a = gnn_model(batch.x_a, batch.edge_index_a, batch.edge_attr_a, batch.x_a_batch)
                h_b = gnn_model(batch.x_b, batch.edge_index_b, batch.edge_attr_b, batch.x_b_batch)

                expression_norm = expression_scaler.transform_ondevice(batch.expression, device=device_gpu)

                # image_tensors = []
                # for i in range(expression_norm.shape[0]):
                #     cell_line = expression_norm[i]
                #     left = cell_line[:int(len(cell_line) / 2)].reshape(-1, 1)
                #     right = cell_line[int(len(cell_line) / 2):].reshape(-1, 1)
                #     gene = torch.cat([left, right], axis=1).cpu()
                #     kde = gaussian_kde(gene.T)
                #     x = np.linspace(gene[0].min(), gene[0].max(), num=64)
                #     y = np.linspace(gene[1].min(), gene[1].max(), num=64)
                #     X, Y = np.meshgrid(x, y)
                #     positions = np.vstack([X.ravel(), Y.ravel()])
                #
                #     # 计算估计的密度值
                #     density = np.reshape(kde(positions).T, X.shape)
                #
                #     image_tensors.append(torch.tensor(density))
                #
                # # 将张量列表转换成一个张量
                # final_tensor = torch.stack(image_tensors)

                # 显示最终张量的形状
                # print(final_tensor)

                h_e = gene_attn_model(expression_norm.to("cuda").type(fdtype))

                triplet = torch.cat([h_a, h_b, h_e], axis=-1)

                logsoftmax_scores = expression_model(triplet)

                __, y_pred_clss = torch.max(logsoftmax_scores, -1)

                y_pred_prob = torch.exp(logsoftmax_scores.detach().cpu()).numpy()


                pred_class.extend(y_pred_clss.view(-1).tolist())
                ref_class.extend(batch.y.view(-1).tolist())
                prob_scores.append(y_pred_prob)
                l_ids.extend(batch.id.view(-1).tolist())



            prob_scores_arr = np.concatenate(prob_scores, axis=0)
            dset_perf = perfmetric_report(pred_class, ref_class, prob_scores_arr[:, 1], epoch,
                                          outlog=os.path.join(exp_dir, dsettype + ".log"))

            perfs[dsettype] = dset_perf

            if (dsettype == "valid"):

                fscore = F_score(perfs['valid'].s_aupr, perfs['valid'].s_auc)
                if (fscore > best_fscore):
                    best_fscore = fscore
                    best_epoch = epoch

                    for m, m_name in models:
                        torch.save(m.state_dict(), os.path.join(state_dict_dir, '{}.pkl'.format(m_name)))

        print({'Train': perfs['train'], 'Validation': perfs['valid']})

        train_curve_aupr.append(perfs['train'].s_aupr)
        valid_curve_aupr.append(perfs['valid'].s_aupr)
        test_curve_aupr.append(0.0)

        train_curve_auc.append(perfs['train'].s_auc)
        valid_curve_auc.append(perfs['valid'].s_auc)
        test_curve_auc.append(0.0)

    print('Finished training and validating!')

    for dsettype in ["test"]:

        if (len(os.listdir(state_dict_dir)) > 0):  # load state dictionary of saved models
            for m, m_name in models:
                m.load_state_dict(torch.load(os.path.join(state_dict_dir, '{}.pkl'.format(m_name)), map_location=device_gpu))

        for m, m_name in models:
            m.eval()

        pred_class = []
        ref_class = []
        prob_scores = []
        y_score = []

        l_ids = []


        for i_batch, batch in enumerate(tqdm(loaders[dsettype], desc="Iteration")):
            batch = batch.to(device_gpu)

            h_a = gnn_model(batch.x_a, batch.edge_index_a, batch.edge_attr_a, batch.x_a_batch)
            h_b = gnn_model(batch.x_b, batch.edge_index_b, batch.edge_attr_b, batch.x_b_batch)

            expression_norm = expression_scaler.transform_ondevice(batch.expression, device=device_gpu)

            # image_tensors = []
            # for i in range(expression_norm.shape[0]):
            #     cell_line = expression_norm[i]
            #     left = cell_line[:int(len(cell_line) / 2)].reshape(-1, 1)
            #     right = cell_line[int(len(cell_line) / 2):].reshape(-1, 1)
            #     gene = torch.cat([left, right], axis=1).cpu()
            #     kde = gaussian_kde(gene.T)
            #     x = np.linspace(gene[0].min(), gene[0].max(), num=64)
            #     y = np.linspace(gene[1].min(), gene[1].max(), num=64)
            #     X, Y = np.meshgrid(x, y)
            #     positions = np.vstack([X.ravel(), Y.ravel()])
            #
            #     # 计算估计的密度值
            #     density = np.reshape(kde(positions).T, X.shape)
            #     image_tensors.append(torch.tensor(density))
            #
            # # 将张量列表转换成一个张量
            # final_tensor = torch.stack(image_tensors)


            h_e = gene_attn_model(expression_norm.to("cuda").type(fdtype))


            triplet = torch.cat([h_a, h_b, h_e], axis=-1)

            logsoftmax_scores = expression_model(triplet)

            probabilities = torch.exp(logsoftmax_scores).detach()
            y_score1 = probabilities[:,1].cpu().numpy()

            __, y_pred_clss = torch.max(logsoftmax_scores, -1)

            y_pred_prob = torch.exp(logsoftmax_scores.detach().cpu()).numpy()

            y_score.extend(y_score1.tolist())
            pred_class.extend(y_pred_clss.view(-1).tolist())
            ref_class.extend(batch.y.view(-1).tolist())
            prob_scores.append(y_pred_prob)
            l_ids.extend(batch.id.view(-1).tolist())

        prob_scores_arr = np.concatenate(prob_scores, axis=0)
        #
        # # 假设 y_test 是测试集的真实标签，y_score 是模型预测的概率
        # y_test = ref_class
        # y_scoreyong = y_score
        #
        # # 创建一个DataFrame来存储这两个数组
        # df = pd.DataFrame({'y_test': y_test, 'y_scoreyong': y_scoreyong})
        #
        # # 将DataFrame保存为Excel文件，您可以指定路径和文件名
        # df.to_excel("../savedata/outputnew.xlsx", index=False)
        #
        # fpr, tpr, _ = roc_curve(y_test, y_scoreyong)
        # print('开始打印')
        # for threshold in _:
        #     print(threshold)
        # print('打印结束')
        # roc_auc = auc(fpr, tpr)
        # plt.figure()
        # plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver Operating Characteristic')
        # plt.legend(loc="lower right")
        # # plt.savefig('../code/newauc2.png')
        # plt.show()
        #
        # # 假设 y_test 是测试集的真实标签，y_scores 是模型预测的概率
        # precision, recall, _ = precision_recall_curve(y_test, y_scoreyong)
        # plt.figure()
        # plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.title('Precision-Recall Curve')
        # plt.legend(loc="lower left")
        # # plt.savefig('../code/newpr2.png')
        # plt.show()
        dset_perf = perfmetric_report(pred_class, ref_class, prob_scores_arr[:, 1], epoch,
                                      outlog=os.path.join(exp_dir, dsettype + ".log"))

        perfs[dsettype] = dset_perf

        if (dsettype == "test"):
            predictions_df = build_predictions_df(l_ids, ref_class, pred_class, prob_scores_arr)
            predictions_df.to_csv(os.path.join(exp_dir, 'predictions', f'epoch_{epoch}_predictions_{dsettype}.csv'))

        print({'Test': perfs['test']})

        test_curve_aupr.pop()
        test_curve_aupr.append(perfs['test'].s_aupr)

        test_curve_auc.pop()
        test_curve_auc.append(perfs['test'].s_auc)

    print('Finished testing!')

    queue.put(gpu_num)
