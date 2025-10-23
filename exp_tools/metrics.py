from sklearn.metrics import precision_score, recall_score

from exp_tools.data_utils import HAM10000_LABEL_MAP


def ham10000_precision(y_true, y_pred):
    """Returns class-wise precision for HAM10000 classification dataset."""
    precision = precision_score(y_true, y_pred, average=None, zero_division=0.0)
    return {
        label_str: precision[label_id]
        for label_str, label_id in HAM10000_LABEL_MAP.items()
    }


def ham10000_recall(y_true, y_pred):
    """Returns class-wise precision for HAM10000 classification dataset."""
    recall = recall_score(y_true, y_pred, average=None, zero_division=0.0)
    return {
        label_str: recall[label_id]
        for label_str, label_id in HAM10000_LABEL_MAP.items()
    }
