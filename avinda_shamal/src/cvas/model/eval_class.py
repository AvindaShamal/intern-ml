import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
)
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import numpy as np


class ModelEvaluator:
    def __init__(
        self,
        network: nn.Module,
        test_loader: DataLoader,
    ):
        """test_loader : The DataLoader object for the test dataset."""
        self.network = network
        self.test_loader = test_loader
        self.y_true = []
        self.y_pred = []
        self.y_probs = []

    def test_model(self) -> dict:
        """
        Evaluates the model on the test dataset and calculates various metrics.

        Returns:
            dict: A dictionary containing evaluation metrics such as accuracy, precision, recall, F1 score,
                  confusion matrix, and optional ROC-AUC score.
        """
        metrics = {}
        self.network.eval()
        with torch.no_grad():
            for test_data in self.test_loader:
                test_images, test_labels = test_data
                test_outputs = self.network(test_images)
                _, predicted = torch.max(test_outputs, 1)
                # choose the class which has highest energy is the predicted class
                # _ gives the energy value for predicted class which is not used here
                self.y_true.extend(test_labels)
                self.y_pred.extend(predicted.numpy())
                # Calculate probabilities (for AUC)
                probabilities = F.softmax(
                    test_outputs, dim=1
                )  # Multiclass softmax
                self.y_probs.extend(
                    probabilities.numpy()
                )  # Append probabilities
        metrics["Accuracy"] = accuracy_score(self.y_true, self.y_pred)
        metrics["precision"] = precision_score(
            self.y_true, self.y_pred, average="weighted", zero_division=0
        )
        metrics["recall"] = recall_score(
            self.y_true, self.y_pred, average="weighted"
        )
        metrics["f1_score"] = f1_score(
            self.y_true, self.y_pred, average="weighted"
        )
        metrics["confusion_matrix"] = confusion_matrix(
            self.y_true, self.y_pred
        )
        # AUC
        if self.y_probs is not None:
            metrics["auc_score"] = roc_auc_score(
                self.y_true, self.y_probs, multi_class="ovr"
            )
            metrics["roc_auc"] = self.roc_auc()["micro"]
        # Print the metrics
        for key, value in metrics.items():
            if key == "confusion_matrix":
                print(f"{key}:\n{value}")
            else:
                print(f"{key}: {value:.4f}")
        return metrics

    def roc_auc(self):
        """
        calculate the auc score and Plots the ROC curve for the model per class.
        """
        # Binarize the true labels
        classes = datasets.CIFAR10(root="./data", train=False).classes
        n_classes = len(classes)
        y_true_bin = label_binarize(self.y_true, classes=range(n_classes))
        # Initialize dictionaries to store results
        fpr = {}
        tpr = {}
        roc_auc = {}
        # Compute ROC and AUC for each class
        for i in range(n_classes):
            y_probs_array = np.array(self.y_probs)
            fpr[i], tpr[i], _ = roc_curve(
                y_true_bin[:, i], y_probs_array[:, i]
            )
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and AUC
        fpr["micro"], tpr["micro"], _ = roc_curve(
            y_true_bin.ravel(), y_probs_array.ravel()
        )
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        if self.y_probs is not None:
            plt.figure(figsize=(10, 8))
            # Plot ROC curve for each class
            for i in range(n_classes):
                plt.plot(
                    fpr[i],
                    tpr[i],
                    label=f"{classes[i]} (AUC = {roc_auc[i]:.2f})",
                )
            # Plot micro-average ROC curve
            plt.plot(
                fpr["micro"],
                tpr["micro"],
                label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})',
                color="navy",
                linestyle="--",
            )
            # Add labels, legend, and title
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve for Multiclass Classification (CIFAR-10)")
            plt.legend(loc="lower right")
            plt.grid()
            plt.show()
        else:
            print(
                "ROC curve can only be plotted if probabilities are available."
            )
        return roc_auc
