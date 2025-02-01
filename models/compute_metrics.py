# Metric computation
def compute_metrics(pred):
    logits, labels = pred.predictions, pred.label_ids
    predictions = np.argmax(logits, axis=-1)
    
    if len(predictions.shape) > 1:
        predictions = predictions.flatten()
    if len(labels.shape) > 1:
        labels = labels.flatten()

    print(f"Logits shape: {logits.shape}")
    print(f"Labels shape: {labels.shape}")

    roc_auc = roc_auc_score(labels, logits[:, 1]) if logits.shape[1] > 1 else 0.0

    print(f"ROC AUC: {roc_auc}")
    return {
        "roc_auc": roc_auc,
    }
