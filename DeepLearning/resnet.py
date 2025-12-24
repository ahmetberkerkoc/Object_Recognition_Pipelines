import os
import time
import copy
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import accuracy_score, f1_score, classification_report

from data import CustomDataset


# Custom Dataset to handle the numpy arrays from data.py
class NumpyDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # images are (H, W) grayscale from data.py
        img = self.images[idx]
        label = self.labels[idx]

        # Ensure image is uint8 for ToPILImage if it's not already
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        # Convert to 3 channels for ResNet (H, W) -> (H, W, 3)
        img = np.stack((img,) * 3, axis=-1)

        if self.transform:
            img = self.transform(img)

        if isinstance(img, torch.Tensor):
            img = img.float()

        # IMPORTANT: CrossEntropyLoss expects class indices as torch.long
        label = torch.tensor(label, dtype=torch.long)

        return img, label


def main():
    # Configuration
    data_dir = 'dataset2'
    num_epochs = 5
    batch_size = 32
    learning_rate = 0.001
    input_size = 224  # ResNet standard

    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "resnet50_best.pth")

    # Device configuration
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # 1. Load Data using data.py's CustomDataset
    print(f"Loading data from {data_dir} using CustomDataset...")
    if not os.path.exists(data_dir):
        if os.path.exists(os.path.join("Spm", data_dir)):
            data_dir = os.path.join("Spm", data_dir)

    data_dir = os.path.abspath(data_dir)
    dataset = CustomDataset(data_dir, resize_h=input_size, resize_w=input_size)

    print("Splitting dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test = dataset.xy(
        using_clss=-1,
        split=(0.6, 0.2, 0.2),
        shuffle_images_in_class=False,
        shuffle_indices=False,
        seed=0
    )

    print(f"Train shape: {X_train.shape}, Labels: {y_train.shape}")
    print(f"Val shape: {X_val.shape}, Labels: {y_val.shape}")
    print(f"Test shape: {X_test.shape}, Labels: {y_test.shape}")

    num_classes = dataset.num_clss
    print(f"Number of classes: {num_classes}")

    # 2. Define Transforms
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    data_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'test': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }

    # Ensure data is uint8 for ToPILImage
    if X_train.dtype != np.uint8: X_train = X_train.astype(np.uint8)
    if X_val.dtype != np.uint8:   X_val = X_val.astype(np.uint8)
    if X_test.dtype != np.uint8:  X_test = X_test.astype(np.uint8)

    train_dataset = NumpyDataset(X_train, y_train, transform=data_transforms['train'])
    val_dataset = NumpyDataset(X_val, y_val, transform=data_transforms['val'])
    test_dataset = NumpyDataset(X_test, y_test, transform=data_transforms['test'])

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0),
        'val':   DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0),
        'test':  DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0)
    }
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset), 'test': len(test_dataset)}

    # 4. Setup Model (ResNet50)
    print("Initializing ResNet50...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Freeze feature layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=learning_rate, momentum=0.9)

    # 5. Training Loop
    print("Starting training...")
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = -1.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase} Epoch {epoch+1}/{num_epochs}"):
                inputs = inputs.to(device)
                labels = labels.to(device)  # already long

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels).item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Save best on validation
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

                torch.save(
                    {
                        "model_state_dict": best_model_wts,
                        "num_classes": num_classes,
                        "input_size": input_size,
                        "best_val_acc": float(best_acc),
                        "epoch": epoch + 1,
                    },
                    ckpt_path
                )
                print(f"✅ Saved best model to: {ckpt_path}")

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # 6. Testing
    print("\nEvaluating on Test set...")
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloaders['test'], desc="Testing"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    test_acc = accuracy_score(all_labels, all_preds)
    test_f1_macro = f1_score(all_labels, all_preds, average='macro')
    test_f1_weighted = f1_score(all_labels, all_preds, average='weighted')

    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1 Score (Macro): {test_f1_macro:.4f}")
    print(f"Test F1 Score (Weighted): {test_f1_weighted:.4f}")

    # "test 1 score": per-class F1 (one-vs-rest / class-wise F1)
    # classification_report prints precision/recall/f1 for each class
    print("\nPer-class scores (Precision / Recall / F1):")
    print(classification_report(all_labels, all_preds, digits=4))

    # Also save final evaluated model weights (optional)
    final_path = os.path.join(save_dir, "resnet50_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Saved final model state_dict to: {final_path}")


if __name__ == '__main__':
    main()
