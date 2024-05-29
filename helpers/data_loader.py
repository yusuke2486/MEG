import numpy as np
import scipy.sparse as sp
import torch

# Helper functions
def encode_onehot(labels):
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def accuracy(output, labels):
    # print(f'output.shape: {output.shape}')
    # print(f'output: {output}')
    preds = output.max(1)[1].type_as(labels)
    print(f'preds.shape: {preds.shape}')
    print(f'labels.shape: {labels.shape}')
    print(f'preds: {preds}')
    print(f'labels: {labels}')
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

# Data loader function
def load_data(path="data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

# Check the data loader
if __name__ == "__main__":
    adj, features, labels, idx_train, idx_val, idx_test = load_data()

    print("Adjacency Matrix:")
    print(adj)
    print("Features:")
    print(features)
    print("Labels:")
    print(labels)
    print("Training Indices:")
    print(idx_train)
    print("Validation Indices:")
    print(idx_val)
    print("Test Indices:")
    print(idx_test)

    print("Feature Matrix Shape:", features.shape)
    print("Adjacency Matrix Shape:", adj.shape)
    print("Number of Classes:", labels.max().item() + 1)
    print("Number of Training Samples:", len(idx_train))
    print("Number of Validation Samples:", len(idx_val))
    print("Number of Test Samples:", len(idx_test))
