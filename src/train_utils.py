"""Training utilities for EEG models"""
import torch
import torch.optim as optim
from tqdm import tqdm
import os


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train model for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, targets) in enumerate(tqdm(dataloader, desc='Training')):
        data, targets = data.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if hasattr(criterion, '__class__') and 'CrossEntropy' in criterion.__class__.__name__:
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total if total > 0 else 0.0

    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, targets in tqdm(dataloader, desc='Validation'):
            data, targets = data.to(device), targets.to(device)

            outputs = model(data)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            all_predictions.append(outputs)
            all_targets.append(targets)

            if hasattr(criterion, '__class__') and 'CrossEntropy' in criterion.__class__.__name__:
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total if total > 0 else 0.0

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    return avg_loss, accuracy, all_predictions, all_targets


def save_checkpoint(model, optimizer, epoch, loss, accuracy, checkpoint_dir):
    """Save model checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }, checkpoint_path)

    return checkpoint_path


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    accuracy = checkpoint.get('accuracy', 0.0)
    return epoch, loss, accuracy


def get_optimizer(model, lr, weight_decay=0.0):
    """Get optimizer"""
    return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def get_scheduler(optimizer, num_epochs, warmup_epochs=5):
    """Get learning rate scheduler with warmup"""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
