def get_mse(preds, labels):
    return ((preds - labels) ** 2).mean().item()

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.squeeze()
    
    return {
        "mse": get_mse(preds, labels)
    }