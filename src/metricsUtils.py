import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
from sklearn.preprocessing import label_binarize
import tensorflow as tf

class Evaluator:
    def __init__(self, class_names=['1_Pronacio', '2_Neutralis', '3_Szupinacio']):
      self.class_names = class_names

    def evaluate(self, y_true, y_pred_proba, history=None):
        y_pred = np.argmax(y_pred_proba, axis=1)
        n_classes = y_pred_proba.shape[1]
    
        metrics = {}
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision_macro"] = precision_score(y_true, y_pred, average="macro",zero_division=0)
        metrics["recall_macro"] = recall_score(y_true, y_pred, average="macro",zero_division=0)
        metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro")

        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)

        metrics["classification_report"] = classification_report(
            y_true, y_pred, target_names=self.class_names, output_dict=True
        )
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        roc_auc["macro"] = np.mean([roc_auc[i] for i in range(n_classes)])

        metrics["roc"] = {
            "fpr": fpr,
            "tpr": tpr,
            "auc": roc_auc
        }

        # -----------------------------
        # Learning curve (ha van)
        # -----------------------------
        if history is not None:
            metrics["history"] = history

        # -----------------------------
        # Error analysis
        # -----------------------------
        errors = np.where(y_pred != y_true)[0]
        metrics["errors"] = errors
        metrics["num_errors"] = len(errors)
        metrics["num_samples"] = len(y_true)

        return metrics


class OrdinalDistanceLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        # y_true: one-hot (batch, 3)
        # y_pred: softmax (batch, 3)

        class_indices = tf.constant([0., 1., 2.])

        y_true_idx = tf.argmax(y_true, axis=-1)
        y_true_idx = tf.cast(y_true_idx, tf.float32)

        # várható érték
        y_pred_idx = tf.reduce_sum(y_pred * class_indices, axis=-1)

        loss = tf.abs(y_true_idx - y_pred_idx)
        return tf.reduce_mean(loss)