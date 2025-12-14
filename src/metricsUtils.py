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

        # -----------------------------
        # Klasszikus metrikák
        # -----------------------------
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
        metrics["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
        metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro")

        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)
        metrics["classification_report"] = classification_report(
            y_true, y_pred, target_names=self.class_names, output_dict=True
        )

        # -----------------------------
        # ROC / AUC
        # -----------------------------
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        roc_auc["macro"] = np.mean([roc_auc[i] for i in range(n_classes)])
        metrics["roc"] = {"fpr": fpr, "tpr": tpr, "auc": roc_auc}

        # -----------------------------
        # Ordinal MAE (fő metrika)
        # -----------------------------
        ordinal_mae = np.mean(np.abs(y_true - y_pred))
        metrics["ordinal_mae"] = ordinal_mae

        # -----------------------------
        # Learning curve
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

        # Extra: hibák távolság szerinti eloszlása
        error_distances = np.abs(y_true[errors] - y_pred[errors])
        metrics["error_distances"] = error_distances

        return metrics

class OrdinalDistanceLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.7, beta=0.3, reduction="auto", name="ordinal_distanced_loss"):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.cce = tf.keras.losses.CategoricalCrossentropy()

    def call(self, y_true, y_pred):
        # y_true: one-hot (batch, 3)
        # y_pred: softmax (batch, 3)

        class_idx = tf.constant([0., 1., 2.])

        y_true_idx = tf.argmax(y_true, axis=-1)
        y_true_idx = tf.cast(y_true_idx, tf.float32)

        y_pred_expectation = tf.reduce_sum(y_pred * class_idx, axis=-1)

        ordinal_loss = tf.abs(y_true_idx - y_pred_expectation)

        ce_loss = self.cce(y_true, y_pred)
        
        neutral_idx = 1
        neutral_prob = y_pred[:, neutral_idx]

        neutral_penalty = tf.where(
            tf.equal(y_true_idx, 1),
            1.0 - neutral_prob,
            0.0
        )
        
        return self.alpha * ordinal_loss + self.beta * ce_loss
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "alpha": self.alpha,
            "beta": self.beta
        })
        return config
        
        
class OrdinalMAE(tf.keras.metrics.Metric):
    def __init__(self, name="ordinal_mae", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_idx = tf.argmax(y_true, axis=-1)
        y_pred_idx = tf.argmax(y_pred, axis=-1)
        error = tf.abs(tf.cast(y_true_idx, tf.float32) - tf.cast(y_pred_idx, tf.float32))
        self.total.assign_add(tf.reduce_sum(error))
        self.count.assign_add(tf.cast(tf.size(error), tf.float32))

    def result(self):
        return self.total / self.count
