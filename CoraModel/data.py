import os
import numpy as np
import scipy.sparse as sp
import torch

from utils import normalize_spadj, normalize_spfeatures


def default_data_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _find_file_recursive(root, filename_lower):
    filename_lower = str(filename_lower).lower()
    for r, _, files in os.walk(root):
        for f in files:
            if f.lower() == filename_lower:
                return os.path.join(r, f)
    return None


def load_cora(data_dir=None):
    data_dir = data_dir or default_data_dir()
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Cora data directory not found: {data_dir}")

    content_file = _find_file_recursive(data_dir, "cora.content")
    cites_file = _find_file_recursive(data_dir, "cora.cites")
    if content_file is None or cites_file is None:
        raise FileNotFoundError(f"Missing cora.content / cora.cites under {data_dir}")

    idx_features_labels = np.genfromtxt(content_file, dtype=np.dtype(str))
    features_unorm = sp.csr_matrix(idx_features_labels[:, 1:-1].astype(np.float32), dtype=np.float32)
    labels_raw = idx_features_labels[:, -1]

    idx = np.array(idx_features_labels[:, 0], dtype=np.dtype(str))
    idx_map = {j: i for i, j in enumerate(idx)}

    edges_unordered = np.genfromtxt(cites_file, dtype=np.dtype(str))
    edges_list = []
    for u, v in edges_unordered:
        if u in idx_map and v in idx_map:
            edges_list.append([idx_map[u], idx_map[v]])
    edges = np.array(edges_list, dtype=np.int32)
    adj = sp.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(features_unorm.shape[0], features_unorm.shape[0]),
        dtype=np.float32,
    )
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_noeye = adj.tocsr()
    adj_noeye.setdiag(0)
    adj_noeye.eliminate_zeros()

    adj_norm = adj_noeye + sp.eye(adj_noeye.shape[0], dtype=np.float32, format="csr")
    adj_norm = normalize_spadj(adj_norm)
    features = normalize_spfeatures(features_unorm)

    classes = sorted(list(set(labels_raw)))
    class_map = {c: i for i, c in enumerate(classes)}
    labels = np.array(list(map(class_map.get, labels_raw)), dtype=np.int64)

    labels = torch.LongTensor(labels)
    features = torch.FloatTensor(np.array(features.todense()))
    adj = torch.FloatTensor(np.array(adj_norm.todense()))
    adj_label = torch.FloatTensor(np.array(adj_noeye.todense()) != 0)
    return labels, adj, features, adj_label
