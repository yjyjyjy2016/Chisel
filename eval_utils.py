import numpy as np
import torch


# ==============================================================================
# Domain object and pIoU scoring
# ==============================================================================

MAXRN = 3000  # maximum number of residues in one target


class _DomainSegment:
    def __init__(self, d_name=''):
        self.d_name = d_name
        self.bgn = 0
        self.end = 0


class _Target:
    def __init__(self):
        self.d_num = 0
        self.ds_num = 0
        self.res_num = 0
        self.def_ = [-1] * MAXRN
        self.dm = [_DomainSegment() for _ in range(150)]


class _Prediction:
    def __init__(self):
        self.d_num = 0
        self.ds_num = 0
        self.res_num = 0
        self.pre = [-1] * MAXRN
        self.dm = [_DomainSegment() for _ in range(150)]


def adjacency_matrix_to_domain_object(matrix, object_type):
    """Convert binary adjacency matrix to domain object for pIoU scoring.

    Scans rows top-to-bottom. For each unassigned residue on the diagonal,
    assigns all residues connected to it (in the same row) to the same domain.

    Args:
        matrix: (N, N) binary adjacency matrix (numpy)
        object_type: 'target' or 'prediction'

    Returns:
        Domain object with .d_num, .ds_num, .res_num, .dm[], .def_/.pre[]
    """
    if object_type == 'target':
        obj = _Target()
        domain_flag = obj.def_
    elif object_type == 'prediction':
        obj = _Prediction()
        domain_flag = obj.pre
    else:
        raise ValueError(f"Invalid object type: {object_type}")

    obj.res_num = len(matrix)
    domain_counter = 0

    for i in range(obj.res_num):
        if domain_flag[i] == -1 and matrix[i][i] == 1:
            domain_flag[i] = domain_counter
            for j in range(i, obj.res_num):
                if matrix[i][j] == 1:
                    domain_flag[j] = domain_counter
            domain_counter += 1

    obj.d_num = domain_counter
    obj.ds_num = domain_counter

    for d in range(domain_counter):
        domain_indices = [i for i, x in enumerate(domain_flag) if x == d]
        if domain_indices:
            obj.dm[d].d_name = f"D{d + 1}"
            obj.dm[d].bgn = min(domain_indices)
            obj.dm[d].end = max(domain_indices)

    return obj


def calculate_piou(pred_matrix, target_matrix):
    """pIoU: for each target domain, find the predicted
    domain with highest intersection, accumulate intersection/union globally.

    Args:
        pred_matrix: (N, N) binary predicted adjacency matrix (numpy)
        target_matrix: (N, N) binary ground-truth adjacency matrix (numpy)

    Returns:
        float: pIoU score in [0, 1]
    """
    pred_obj = adjacency_matrix_to_domain_object(pred_matrix, 'prediction')
    target_obj = adjacency_matrix_to_domain_object(target_matrix, 'target')

    total_intersection = 0
    total_union = 0

    for j in range(target_obj.d_num):
        target_indices = [idx for idx, val in enumerate(target_obj.def_)
                          if val == j]
        best_intersection = 0
        best_union = len(target_indices)

        for i in range(pred_obj.d_num):
            pred_indices = [idx for idx, val in enumerate(pred_obj.pre)
                            if val == i]
            if pred_indices:
                intersection = len(set(pred_indices) & set(target_indices))
                union = len(set(pred_indices) | set(target_indices))
                if intersection > best_intersection:
                    best_intersection = intersection
                    best_union = union

        total_intersection += best_intersection
        total_union += best_union

    return total_intersection / total_union if total_union > 0 else 0.0


# ==============================================================================
# Public evaluation API (interface unchanged)
# ==============================================================================

def adj_to_labels(adj_matrix, threshold=0.5):
    """Convert adjacency matrix to domain labels via connected components.

    Args:
        adj_matrix: (N, N) numpy array or tensor, values in [0, 1]
        threshold: binarization threshold

    Returns:
        (N,) numpy array of domain labels (1-indexed, 0=unassigned)
    """
    if isinstance(adj_matrix, torch.Tensor):
        adj_matrix = adj_matrix.cpu().numpy()

    N = adj_matrix.shape[0]
    binary = (adj_matrix > threshold).astype(int)

    labels = np.zeros(N, dtype=int)
    domain_id = 0
    visited = set()

    for i in range(N):
        if i in visited:
            continue
        domain_id += 1
        queue = [i]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            labels[node] = domain_id
            for j in range(N):
                if j not in visited and binary[node, j] == 1:
                    queue.append(j)

    return labels


def labels_to_adj_matrix(labels):
    """Convert per-residue domain labels to binary adjacency matrix.

    Args:
        labels: (N,) array of domain IDs (0=unassigned, 1..K=domain)

    Returns:
        (N, N) binary adjacency matrix
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    N = len(labels)
    adj = np.zeros((N, N), dtype=int)
    for i in range(N):
        if labels[i] == 0:
            continue
        for j in range(N):
            if labels[i] == labels[j] and labels[j] > 0:
                adj[i, j] = 1
    return adj


def evaluate_predictions(pred_adj, true_labels, threshold=0.5):
    """Evaluate adjacency matrix prediction against true domain labels.

    Uses pIoU scoring.

    Args:
        pred_adj: (N, N) predicted adjacency matrix (float, 0-1)
        true_labels: (N,) true domain labels (1-indexed, 0=unassigned)
        threshold: binarization threshold for pred_adj

    Returns:
        dict with metrics: iou, num_pred_domains, num_true_domains
    """
    if isinstance(pred_adj, torch.Tensor):
        pred_adj = pred_adj.cpu().numpy()
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.cpu().numpy()

    # Binarize predicted adjacency
    pred_binary = (pred_adj > threshold).astype(int)

    # Build ground truth adjacency from labels
    true_adj = labels_to_adj_matrix(true_labels)

    # Compute pIoU
    piou = calculate_piou(pred_binary, true_adj)

    # Count domains for reporting
    pred_obj = adjacency_matrix_to_domain_object(pred_binary, 'prediction')
    target_obj = adjacency_matrix_to_domain_object(true_adj, 'target')

    return {
        'iou': piou,
        'num_pred_domains': pred_obj.d_num,
        'num_true_domains': target_obj.d_num,
    }


def batch_evaluate(pred_adjs, true_labels_batch, threshold=0.5):
    """Evaluate a batch of predictions using pIoU.

    Args:
        pred_adjs: (B, N, N) predicted adjacency matrices
        true_labels_batch: (B, N) true domain labels

    Returns:
        dict with averaged metrics
    """
    if isinstance(pred_adjs, torch.Tensor):
        pred_adjs = pred_adjs.cpu().numpy()
    if isinstance(true_labels_batch, torch.Tensor):
        true_labels_batch = true_labels_batch.cpu().numpy()

    B = pred_adjs.shape[0]
    all_ious = []
    all_pred_domains = []
    all_true_domains = []

    for i in range(B):
        result = evaluate_predictions(
            pred_adjs[i], true_labels_batch[i], threshold)
        all_ious.append(result['iou'])
        all_pred_domains.append(result['num_pred_domains'])
        all_true_domains.append(result['num_true_domains'])

    return {
        'mean_iou': np.mean(all_ious) if all_ious else 0.0,
        'mean_pred_domains': np.mean(all_pred_domains),
        'mean_true_domains': np.mean(all_true_domains),
        'per_sample_iou': all_ious,
    }


def compute_adjacency_metrics(pred_adj, true_adj, threshold=0.5):
    """Compute pixel-level metrics on adjacency matrices.

    Args:
        pred_adj: (N, N) predicted adjacency
        true_adj: (N, N) true adjacency

    Returns:
        dict with precision, recall, f1, accuracy
    """
    if isinstance(pred_adj, torch.Tensor):
        pred_adj = pred_adj.cpu().numpy()
    if isinstance(true_adj, torch.Tensor):
        true_adj = true_adj.cpu().numpy()

    pred_bin = (pred_adj > threshold).astype(int)
    true_bin = (true_adj > threshold).astype(int)

    tp = ((pred_bin == 1) & (true_bin == 1)).sum()
    fp = ((pred_bin == 1) & (true_bin == 0)).sum()
    fn = ((pred_bin == 0) & (true_bin == 1)).sum()
    tn = ((pred_bin == 0) & (true_bin == 0)).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)

    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'accuracy': float(accuracy),
    }

