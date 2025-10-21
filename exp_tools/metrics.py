from sklearn.metrics import precision_score, recall_score

def average_precision(y_true, y_pred):
    return precision_score(y_true, y_pred, average="macro", zero_division=0.0)

def average_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, average="macro", zero_division=0.0)
