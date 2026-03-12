import torch
import torch.nn.functional as F
import torchmetrics.functional as MF
import torchmetrics.retrieval as MR
from dbinfer_bench import DBBTaskType, DBBTaskMeta

negated = lambda f: lambda *args, **kwargs: -f(*args, **kwargs)


def root_mean_squared_error(logits, target):
    return MF.mean_squared_error(logits, target, squared=False)

def retrival_metric_wrapper(fn):
    def _wrapped(preds, labels, indexes):
        return fn(preds, labels.long(), task='binary')
    return _wrapped
    
def multiclass_log_loss(probabilities, target, **kwargs):
    target = target.long()
    log_probs = torch.log(probabilities) 
    return F.nll_loss(log_probs, target)  

METRIC_FN = {
    "classification": {
        "accuracy": MF.accuracy,
        "ap": MF.average_precision,
        "auroc": MF.auroc,
        "f1": MF.f1_score,
        "hinge": negated(MF.hinge_loss),
        "recall": MF.recall,
        "logloss": negated(multiclass_log_loss),
    },
    "regression": {
        "mae": negated(MF.mean_absolute_error),
        "mse": negated(MF.mean_squared_error),
        "msle": negated(MF.mean_squared_log_error),
        "pearson": MF.pearson_corrcoef,
        "rmse": negated(root_mean_squared_error),
        "r2": MF.r2_score,
    },
    "retrieval": {
        "hr": MR.RetrievalHitRate(),
        "mrr": MR.RetrievalMRR(),
        "ndcg": MR.RetrievalNormalizedDCG(),
        "auroc": retrival_metric_wrapper(MF.auroc),
    },
}

def get_metric_fn(meta : DBBTaskMeta):
    fn = METRIC_FN[meta.task_type][meta.evaluation_metric]
    if meta.task_type == DBBTaskType.classification:
        def _classification_wrapper(seeds, logits, labels):
            # Shape:
            #   - logits: (N, C) or (N,) if C == 1
            #   - labels : (N,)
            with torch.no_grad():
                preds = F.softmax(logits, dim=1)
                if hasattr(meta, 'num_classes'):
                    num_classes = meta.num_classes
                else:
                    num_classes = preds.shape[1]
                labels = labels.long()
                return fn(preds, labels,
                          num_classes=num_classes,
                          task='multiclass')
        return _classification_wrapper
    elif meta.task_type == DBBTaskType.regression:
        def _regression_wrapper(seeds, logits, targets):
            # Shape:
            #   - logits: (N,)
            #   - targets : (N,)
            with torch.no_grad():
                return fn(logits, targets.float())
        return _regression_wrapper
    elif meta.task_type == DBBTaskType.retrieval:
        def _retrieval_wrapper(query_idx, logits, labels):
            # Shape:
            #   - query_idx: (N,)
            #   - logits: (N,)
            #   - labels : (N,)
            with torch.no_grad():
                preds = torch.sigmoid(logits)
                return fn(preds, labels, indexes=query_idx)
        return _retrieval_wrapper
    else:
        raise ValueError(f"Unsupported task type {meta.task_type}")

def infer_task_type(metric_name: str):
    for task_type, metrics in METRIC_FN.items():
        if metric_name in metrics:
            return task_type
    raise ValueError(f"Invalid metric name {metric_name}.")

def retrieval_loss(logits, labels):
    return F.binary_cross_entropy(torch.sigmoid(logits), labels.float())

LOSS_FN = {
    'classification': F.cross_entropy,
    'regression': lambda logits, targets : F.mse_loss(logits, targets.float()),
    'retrieval': retrieval_loss,
}

def get_loss_fn(meta : DBBTaskMeta):
    fn = LOSS_FN[meta.task_type]
    return fn
