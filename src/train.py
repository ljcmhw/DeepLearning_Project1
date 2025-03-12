import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

from dataset import load_cifar10_train, load_cifar10_val, convert_data_shape, CIFARDataset
from model import ResNet18

SEED = 2025
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

DATA_PATH = "./data/cifar-10-batches-py"  # Adjust path as needed
NUM_CLASSES = 10
BATCH_SIZE = 128
EPOCHS = 150
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

save_path = "best_model.pth"

# Load data
train_data_np, train_labels_np = load_cifar10_train(DATA_PATH)
val_data_np, val_labels_np = load_cifar10_val(DATA_PATH)

train_data_np = convert_data_shape(train_data_np)
val_data_np   = convert_data_shape(val_data_np)

CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD  = [0.2470, 0.2435, 0.2616]

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
])

train_dataset = CIFARDataset(train_data_np, train_labels_np, transform=train_transform)
val_dataset   = CIFARDataset(val_data_np,   val_labels_np,   transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Train set size: {len(train_dataset)}")
print(f"Val set size:   {len(val_dataset)}")

model = ResNet18(num_classes=NUM_CLASSES).to(DEVICE)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {num_params}")
assert num_params <= 5_000_000, "Model has more than 5 million parameters!"

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

best_val_acc = 0.0
best_epoch = 0

for epoch in range(1, EPOCHS+1):
    # Training
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = 100. * correct / total

    # Validation
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct_val += predicted.eq(labels).sum().item()
            total_val += labels.size(0)

    val_loss = val_loss / total_val
    val_acc = 100. * correct_val / total_val

    scheduler.step()

    print(f"Epoch [{epoch}/{EPOCHS}] "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% || "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch
        torch.save(model.state_dict(), save_path)
        print(f"--> Best model updated at epoch {epoch}, val_acc={val_acc:.2f}%")

print(f"Training complete. Best val_acc = {best_val_acc:.2f}% at epoch {best_epoch}")
