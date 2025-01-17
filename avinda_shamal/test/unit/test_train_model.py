import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from cvas.model import ModelTrainer

dummy_dataset = TensorDataset(
    torch.randn(100, 3, 32, 32), torch.randint(0, 10, (100,))
)
dummy_loader = DataLoader(dummy_dataset, batch_size=12, shuffle=True)


def test_split_dataset():
    trainer = ModelTrainer(
        nn.Linear(3, 10), dummy_loader, device=torch.device("cpu")
    )
    train_set, val_set = trainer.split_dataset(split=0.8)
    train_set1, val_set1 = trainer.split_dataset(split=0.7)

    assert len(train_set.dataset) == 80
    assert len(val_set1.dataset) == 30
    assert train_set1.batch_size == 12
    assert val_set.batch_size == 12


def test_run_epoch():
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
    trainer = ModelTrainer(model, dummy_loader, device=torch.device("cpu"))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # Test training phase
    train_loss, train_accuracy = trainer.run_epoch(
        "train", loss_fn, optimizer, dummy_loader
    )
    assert train_loss > 0
    assert 0 <= train_accuracy <= 100
    # Test validation phase
    eval_loss, eval_accuracy = trainer.run_epoch(
        "eval", loss_fn, optimizer, dummy_loader
    )
    assert eval_loss > 0
    assert 0 <= eval_accuracy <= 100


def test_early_stopping():
    trainer = ModelTrainer(nn.Linear(3, 10), None, torch.device("cpu"))
    # Test improvement in validation loss
    assert not trainer.early_stopping(val_loss=0.5)
    assert trainer.best_loss == 0.5
    assert trainer.patience_counter == 0
    # Test no improvement in validation loss
    assert not trainer.early_stopping(val_loss=0.55)
    assert trainer.patience_counter == 1
    assert not trainer.early_stopping(val_loss=0.6)
    assert trainer.patience_counter == 2
    # Trigger early stopping
    assert trainer.early_stopping(val_loss=0.6)
    assert trainer.patience_counter == 3
