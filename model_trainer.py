import os
import numpy as np
import scipy.sparse as sp
import torch
import time
import json
import pathlib
from model import VGAE
# from PID import PIDControl
from evaluate import Evaluator
from sklearn.metrics import recall_score,precision_score,f1_score,accuracy_score
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
import pickle
from tqdm import tqdm

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)

class TrainerBase():
    def __init__(self):
        self.name = "TrainerBase"

    def train(self):
        raise NotImplementedError(self.name)

class ControlVGAETrainer(TrainerBase):
    def __init__(self, adj_matrix, features, args, dataset):
        super(ControlVGAETrainer).__init__()
        self.name = "ControlVGAETrainer"
        self.adj_matrix = adj_matrix
        self.features = features
        self.args = args
        self.dataset = dataset

        self.model = None
        self.result_embedding = None

    def train(self):
        print("Training using {}".format(self.name))

        # Store original adjacency matrix (without diagonal entries) for later
        adj_orig = self.adj_matrix
        adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
        adj_orig.eliminate_zeros()

        # adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj_matrix)
        self.adj_matrix[self.adj_matrix > 1] = 1

        adj_train = self.adj_matrix

        """"""
        # build adj_train matrix
        # adj_matrix_dense = self.adj_matrix.todense()
        #
        # if os.path.exists("test_mask.pkl"):
        #     print("USING EXISTING TEST MASK!")
        #     with open("test_mask.pkl", "rb") as fin:
        #         test_mask = pickle.load(fin)
        # else:
        #     test_mask = np.random.random(adj_matrix_dense.shape)
        #     test_mask[adj_matrix_dense != 1] = 0
        #     test_mask[self.args.num_user:, self.args.num_user:] = 0 # TODO now we do not evaluate tweet tweet
        #     test_mask[(0 < test_mask) & (test_mask < 0.1)] = 1
        #     test_mask[test_mask != 1] = 0
        #     test_mask[self.args.num_user:, :self.args.num_user] = test_mask[:self.args.num_user, self.args.num_user:].T
        #
        # print("Test samples: {}, One samples: {}".format(np.sum(test_mask), np.sum(adj_matrix_dense)))
        # adj_matrix_dense[test_mask == 1] = 0
        # print("After modification one samples: {}".format(np.sum(adj_matrix_dense)))
        # adj_train = sp.csr_matrix(adj_matrix_dense)
        # adj_train.eliminate_zeros()
        """"""

        # Some preprocessing
        adj_norm = preprocess_graph(adj_train)
        features = sparse_to_tuple(sp.coo_matrix(self.features))
        pos_weight = float(adj_train.shape[0] * adj_train.shape[0] - adj_train.sum()) / adj_train.sum() * self.args.pos_weight_lambda
        print("Pos weight: {}".format(pos_weight))
        norm = adj_train.shape[0] * adj_train.shape[0] / float((adj_train.shape[0] * adj_train.shape[0] - adj_train.sum()) * 2)

        adj_label = adj_train + sp.eye(adj_train.shape[0])

        adj_label = sparse_to_tuple(adj_label)

        adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T),
                                            torch.FloatTensor(adj_norm[1]),
                                            torch.Size(adj_norm[2])).to(self.args.device)
        adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T),
                                             torch.FloatTensor(adj_label[1]),
                                             torch.Size(adj_label[2])).to(self.args.device)
        features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T),
                                            torch.FloatTensor(features[1]),
                                            torch.Size(features[2])).to(self.args.device)

        weight_mask = adj_label.to_dense().view(-1) == 1
        weight_tensor = torch.ones(weight_mask.size(0)).to(self.args.device)
        weight_tensor[weight_mask] = pos_weight

        """"""
        # print(torch.sum((1 - adj_label_test.view(-1))))
        # weight_tensor *= (1 - adj_label_test.view(-1))
        """"""

        # init model and optimizer
        if self.args.model == "VGAE" or self.args.model == "HVGAE":
            self.model = VGAE(self.args, adj_norm)
            self.model.to(self.args.device)
        else:
            raise NotImplementedError(self.name)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=1e-4)

        for epoch in range(self.args.epochs):
            optimizer.zero_grad()

            embed = self.model.encode(features)

            if self.dataset.freeze_dict is not None:
                z_freezed = torch.zeros_like(embed)
                z_freezed[~self.dataset.freeze_mask] = embed[~self.dataset.freeze_mask]
                if self.dataset.freeze_tensor is not None:
                    z_freezed[self.dataset.freeze_mask] = self.dataset.freeze_tensor.to(self.args.device)
                embed = z_freezed

            A_pred = self.model.decode(embed)

            reconstruct_loss = norm * F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight=weight_tensor)
            loss = reconstruct_loss

            if self.args.axis_guidance and self.dataset.axis_guidance_N > 0:
                axis_weight = float(
                    self.dataset.axis_guidance_N * self.dataset.axis_guidance_N
                    - self.dataset.axis_guidance_adj_matrix.sum()) \
                              / self.dataset.axis_guidance_adj_matrix.sum() * self.args.pos_weight_lambda
                axis_norm = self.dataset.axis_guidance_N * self.dataset.axis_guidance_N / float(
                    (self.dataset.axis_guidance_N * self.dataset.axis_guidance_N - self.dataset.axis_guidance_adj_matrix.sum()) * 2)
                axis_weight_mask = self.dataset.axis_guidance_adj_matrix.view(-1) == 1
                axis_weight_tensor = torch.ones(axis_weight_mask.size(0)).to(self.args.device)
                axis_weight_tensor[axis_weight_mask] = axis_weight
                self.dataset.axis_guidance_units = self.dataset.axis_guidance_units.to(self.args.device)
                self.dataset.axis_guidance_adj_matrix = self.dataset.axis_guidance_adj_matrix.to(self.args.device)
                axis_weight_tensor = axis_weight_tensor.to(self.args.device)
                pred = torch.matmul(embed[self.dataset.axis_guidance_indexes], self.dataset.axis_guidance_units.t())
                pred = torch.sigmoid(pred)
                axis_guidance_loss = axis_norm * F.binary_cross_entropy(pred.view(-1), self.dataset.axis_guidance_adj_matrix.view(-1), weight=axis_weight_tensor)
                loss += axis_guidance_loss

            train_acc = self.get_acc(A_pred, adj_label)

            if epoch % 20 == 0:
                if self.args.axis_guidance and self.dataset.axis_guidance_N > 0:
                    print(f"Epoch: {epoch}, Rec_loss: {reconstruct_loss.item():.4f}, Axis_loss: {axis_guidance_loss.item():.4f}, Link_acc: {train_acc.item():.4f}")
                else:
                    print(f"Epoch: {epoch}, Rec_loss: {reconstruct_loss.item():.4f}, Link_acc: {train_acc.item():.4f}")

            loss.backward()
            optimizer.step()

        self.result_embedding = self.model.encode(features).cpu().detach().numpy()
        if self.dataset.freeze_dict is not None and self.dataset.freeze_tensor is not None:
            self.result_embedding[self.dataset.freeze_mask] = self.dataset.freeze_tensor.cpu().numpy()

    def save(self, path=None):
        path = self.args.output_path if path is None else path
        # Save result embedding of nodes
        with open(path / "args.json", 'w') as fout:
            json.dump({k: str(v) for k, v in vars(self.args).items()}, fout)
        with open(path / "embedding.bin", 'wb') as fout:
            pickle.dump(self.result_embedding, fout)
            print("Embedding and dependencies are saved in {}".format(path))
        if self.args.artifact_dir is not None:
            self.args.artifact_dir.mkdir(exist_ok=True, parents=True)
            artifact_file = self.args.artifact_dir / f"infovgae_artifact_{self.args.hidden2_dim}.pkl"
            with open(artifact_file, 'wb') as fout:
                output_dict = {}
                for i, asser in enumerate(self.dataset.asserlist):
                    output_dict[asser] = self.result_embedding[self.dataset.num_user + i]
                pickle.dump(output_dict, fout)
                print("freeze_dict artifact saved at {}".format(str(artifact_file)))

    def get_scores(self, adj_orig, edges_pos, edges_neg, adj_rec):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # Predict on test set of edges
        preds = []
        pos = []
        for e in edges_pos:
            # print(e)
            # print(adj_rec[e[0], e[1]])
            preds.append(sigmoid(adj_rec[e[0], e[1]].item()))
            pos.append(adj_orig[e[0], e[1]])

        preds_neg = []
        neg = []
        for e in edges_neg:
            preds_neg.append(sigmoid(adj_rec[e[0], e[1]].data))
            neg.append(adj_orig[e[0], e[1]])

        preds_all = np.hstack([preds, preds_neg])
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)

        return roc_score, ap_score

    def get_acc(self, adj_rec, adj_label):
        labels_all = adj_label.to_dense().view(-1).long()
        preds_all = (adj_rec > 0.5).view(-1).long()
        accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
        return accuracy
