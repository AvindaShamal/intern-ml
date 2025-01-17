import torch
import torch.optim.optimizer
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class ModelTrainer:
    """A class to train and test a PyTorch model."""

    def __init__(
        self,
        network: nn.Module,
        train_loader: DataLoader,
        device: torch.device,
        patience: int = 3,
    ) -> None:
        """Initializes the ModelTrainer.
        Args:
            network : The neural network model to train.
            train_loader : The DataLoader object for the training dataset.
            device : The device to use for training and testing (cpu or cuda).
            patience : The number of epochs to wait for early stopping.
        """
        self.network = network
        self.train_loader = train_loader
        self.device = device
        self.best_loss: float = float("inf")
        self.patience_counter: int = 0
        self.patience = patience

    def train_model(
        self,
        epochs: int,
        criterian: nn.Module,
        optimizer: optim.Optimizer,
        split: float = 0.8,
    ) -> tuple:
        """Trains the model on the training dataset.
        Args:
            epochs : The number of epochs to train the model.
            criterian : The loss function to use.
            optimizer : The optimizer to use for training.
            split : The percentage of the training data to use for validation.
        Returns:
            tuple: The trained model and a dictionary of metrics.
        """
        self.network.to(self.device)
        metrics = {
            "train_loss": [],
            "train_accuracy": [],
            "eval_loss": [],
            "eval_accuracy": [],
        }
        train_set, val_set = self.split_dataset(split)
        for epoch in range(epochs):  # loop over the dataset multiple times
            # Run training phase
            train_loss, train_accuracy = self.run_epoch(
                "train", criterian, optimizer, train_set
            )
            metrics["train_loss"].append(train_loss)
            metrics["train_accuracy"].append(train_accuracy)
            print(
                f"Epoch [{epoch + 1}/{epochs}], "
                f"Train Loss: {metrics['train_loss'][-1]:.4f}, Train Accuracy: {metrics['train_accuracy'][-1]:.2f}%"
            )
            # Run validation phase
            eval_loss, eval_accuracy = self.run_epoch(
                "eval", criterian, optimizer, val_set
            )
            metrics["eval_loss"].append(eval_loss)
            metrics["eval_accuracy"].append(eval_accuracy)
            print(
                f"Val Loss: {metrics['eval_loss'][-1]:.4f}, Val Accuracy: {metrics['eval_accuracy'][-1]:.2f}%"
            )
            # Check early stopping
            should_stop = self.early_stopping(val_loss=eval_loss)
            if should_stop:
                return self.network, metrics
        print("Finished Training")
        return self.network, metrics

    def split_dataset(self, split: float) -> tuple:
        """Splits the dataset into training and validation sets.
        Args:
            split : The percentage of the training data to use for validation.
        Returns:
            tuple: The training and validation DataLoader objects.
        """
        train_size = int(len(self.train_loader.dataset) * split)
        val_size = len(self.train_loader.dataset) - train_size
        train_data, val_data = random_split(
            self.train_loader.dataset, [train_size, val_size]
        )
        train_set = DataLoader(
            train_data, batch_size=self.train_loader.batch_size, shuffle=True
        )
        val_set = DataLoader(
            val_data, batch_size=self.train_loader.batch_size, shuffle=False
        )
        return train_set, val_set

    def run_epoch(
        self,
        phase: str,
        criterian: nn.Module,
        optimizer: optim.Optimizer,
        loader: DataLoader,
    ) -> tuple:
        """Runs a single epoch of training or evaluation.
        Args:
            phase : The phase of the epoch (train or eval).
            criterian : The loss function to use.
            optimizer : The optimizer to use for training.
            loader : The DataLoader object for the dataset.
        Returns:
            tuple: The loss and accuracy for the epoch.
        """
        if phase == "train":
            self.network.train()
        else:
            self.network.eval()
        running_loss = 0.0
        correct, total = 0, 0
        for data in loader:
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            # zero the parameter gradients
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == "train"):
                outputs = self.network(inputs)  # forward pass
                loss = criterian(outputs, labels)  # calculate the loss
                if phase == "train":
                    loss.backward()  # backward pass
                    optimizer.step()  # update the weights
            running_loss += loss.item() * inputs.size(0)  # Accumulate loss
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (
                (predicted == labels).sum().item()
            )  # count correct predictions
        epoch_loss = running_loss / len(loader.dataset)
        epoch_accuracy = correct / total * 100
        return epoch_loss, epoch_accuracy

    def early_stopping(
        self,
        val_loss: float,
    ) -> bool:
        """Checks if early stopping should be triggered.
        Args:
            val_loss (float): Current epoch's validation loss.
        Returns:
            tuple: where should stop and updated_best_loss
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                print("Early stopping triggered!")
                return True
        return False

    def plot_learning_curves(self, metrics: dict) -> None:
        """Plots the learning curves for the model.
        Args:
            metrics : A dictionary containing the training loss and accuracy.
        Returns:
            None
        """
        plt.figure(figsize=(10, 5))
        plt.plot(metrics["train_loss"], label="Training Loss")
        plt.plot(metrics["eval_loss"], label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(metrics["train_accuracy"], label="Training Accuracy")
        plt.plot(metrics["eval_accuracy"], label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Training Accuracy")
        plt.legend()
        plt.show()
