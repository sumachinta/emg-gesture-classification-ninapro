import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")        # Apple GPU
    elif torch.cuda.is_available():
        return torch.device("cuda")       # NVIDIA GPUs (not on Macs)
    else:
        return torch.device("cpu")
    

class Standardize1D:
    def __init__(self, mean, std):
        self.mean, self.std = mean, std
    def __call__(self, x):
        return (x-self.mean)/self.std
    

class EMGDataset(Dataset):
    def __init__(self,X,y,transform=None):
        self.X=X # keeping as np arrays to save RAM copies
        self.y=y
        self.transform = transform
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self,idx):
        x = torch.from_numpy(self.X[idx]).float() #(C,L)
        y = torch.as_tensor(self.y[idx]).long()
        return x,y

class CNN_model(nn.Module):
    """One layer CNN model for EMG:  (B, 12, 400) -> (B, numClasses)
    Conv1d -> BatchNor -> ReLU -> Dropout -> global avg pool -> Linear
    """
    def __init__(self, inChannels:int = 12, numClasses:int = 17, conv1out: int = 64, kernel_size: int = 31, p_dropout: float = 0.2,
                 conv2out: int = 128, kernel_size2: int = 15, p_dropout2: float = 0.2, pool1_ks: int | None= 2):
        """Arguments:
        inChannels: number of input channels (EMG electrodes)
        numClasses: number of output classes (gestures)
        conv1out: number of filters identified. lower number to reduce overfitting or increase for underfitting
        kernel_size: kernel size or temporal receptive field in samples
        p_dropout: dropout probability"""
        super().__init__()

        # First conv layer
        self.conv1 = nn.Conv1d(in_channels=inChannels, out_channels=conv1out, kernel_size=kernel_size, padding=kernel_size//2)
        self.batchnorm1 = nn.BatchNorm1d(conv1out)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=p_dropout)
        self.pool1 = nn.MaxPool1d(pool1_ks) if pool1_ks else nn.Identity()

        # Second conv layer
        self.conv2 = nn.Conv1d(in_channels=conv1out, out_channels=conv2out, kernel_size=kernel_size2, padding=kernel_size2//2)
        self.batchnorm2 = nn.BatchNorm1d(conv2out)
        self.dropout2 = nn.Dropout(p=p_dropout2)

        # Third conv layer
        conv3out = conv2out  # keeping same number of channels
        self.conv3 = nn.Conv1d(in_channels=conv2out, out_channels=conv3out, kernel_size=kernel_size2, padding=kernel_size2//2)
        self.batchnorm3 = nn.BatchNorm1d(conv3out)
        self.dropout3 = nn.Dropout(p=p_dropout2)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1) # output size (B, conv1out, 1) so for each filter we have 1 value
        self.fc = nn.Linear(conv3out, numClasses) # final classification layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.activation(x)
        x = self.dropout3(x)

        x = self.global_avg_pool(x) # (B, conv1out, 1)
        x = x.squeeze(-1)           # (B, conv1out)
        x = self.fc(x)              # (B, numClasses)
        return x

def accuracy(logits, y):
    return (logits.argmax(dim=1) == y).float().mean().item()

# One loop of training
def train_one_epoch(model, train_loader, loss_fn, optimizer, device):
    model.train() # Set model to training mode
    running_loss, running_acc, n = 0.0, 0.0, 0
    for batch, (X, y) in enumerate(train_loader): # Loop over batches
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True) # clears old gradients from the last batch

        # Forward pass
        logits = model(X)
        loss = loss_fn(logits, y)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        batch_size = X.size(0)
        running_loss += loss.item() * batch_size
        running_acc += accuracy(logits, y) * batch_size
        n += batch_size

        return running_loss / n, running_acc / n


# One loop of validation
def validate_one_epoch(model, val_loader, loss_fn, device):
    model.eval() # Set model to evaluation mode
    running_loss, running_acc, n = 0.0, 0.0, 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = loss_fn(logits, y)

            batch_size = X.size(0)
            running_loss += loss.item() * batch_size
            running_acc += accuracy(logits, y) * batch_size
            n += batch_size

    return running_loss / n, running_acc / n

def get_lrs(optimizer):
    return [pg["lr"] for pg in optimizer.param_groups]