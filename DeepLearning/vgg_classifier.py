import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
import os
import time
from tqdm import tqdm
from data import CustomDataset

import matplotlib.pyplot as plt

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

        # Convert to 3 channels for VGG (H, W) -> (H, W, 3)
        img = np.stack((img,)*3, axis=-1)
        
        # To PIL Image or Tensor. transforms.ToTensor() expects (H, W, C) in [0, 255]
        # It converts to (C, H, W) in [0.0, 1.0]
        
        if self.transform:
            img = self.transform(img)
        
        # Explicitly cast to float32 to avoid any float64 issues
        if isinstance(img, torch.Tensor):
            img = img.float()
            
        # CrossEntropyLoss expects integer class indices (torch.long)
        label = torch.tensor(int(label), dtype=torch.long)
        return img, label


def _current_time_slug() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _save_confusion_matrix(cm: np.ndarray, out_path: str, title: str = "Confusion Matrix") -> None:
    """Save a readable confusion matrix with counts and per-class supports."""
    n = cm.shape[0]
    fig_w = min(18, max(8, 0.35 * n))
    fig_h = min(18, max(6, 0.35 * n))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ticks = np.arange(n)
    ax.set_xticks(ticks)

    supports = cm.sum(axis=1).astype(int)
    ytick_labels = [f"{i} (n={supports[i]})" for i in range(n)]
    ax.set_yticks(ticks)
    ax.set_yticklabels(ytick_labels)

    # annotate each cell
    thresh = cm.max() * 0.6 if cm.size else 0
    for i in range(n):
        for j in range(n):
            val = int(cm[i, j])
            ax.text(
                j,
                i,
                str(val),
                ha="center",
                va="center",
                fontsize=6,
                color="white" if val > thresh else "black",
            )

    ax.set_xticklabels([str(i) for i in range(n)], rotation=90)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def main():
    # Configuration
    data_dir = 'dataset1/train'
    num_epochs = 5
    batch_size = 32
    learning_rate = 0.001
    input_size = 224 # VGG standard

    # Output organization
    runs_root = "checkpoints"
    run_name = f"vgg16_{os.path.basename(os.path.normpath(data_dir))}_{_current_time_slug()}"
    run_dir = os.path.join(runs_root, run_name)
    os.makedirs(run_dir, exist_ok=True)
    best_ckpt_path = os.path.join(run_dir, "vgg16_best.pth")
    final_ckpt_path = os.path.join(run_dir, "vgg16_final.pth")
    cm_path = os.path.join(run_dir, "confusion_matrix_test.png")

    # Device configuration
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"Using device: {device}")
    print(f"Run dir: {os.path.abspath(run_dir)}")

    # 1. Load Data using data.py's CustomDataset
    print(f"Loading data from {data_dir} using CustomDataset...")
    # Ensure we point to the correct path
    if not os.path.exists(data_dir):
        if os.path.exists(os.path.join("Spm", data_dir)):
             data_dir = os.path.join("Spm", data_dir)
    
    # Initialize CustomDataset
    # Note: CustomDataset loads as grayscale. We resize to 224x224 for VGG.
    # Ensure data_dir is absolute path to avoid issues in CustomDataset
    data_dir = os.path.abspath(data_dir)
    dataset = CustomDataset(data_dir, resize_h=input_size, resize_w=input_size)
    
    # Get splits (Train 60%, Val 20%, Test 20%)
    # User requested "not shuffle" in previous turns, so we keep that consistency
    print("Splitting dataset...")
    # Note: data.py's xy method signature might have changed. 
    # Checking data.py content from previous turns, it accepts:
    # using_clss=-1, split=(0.6, 0.2, 0.2), shuffle_images_in_class=False, shuffle_indices=False, seed=None, save_split_dir=None
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
    # VGG normalization
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

    # 3. Create PyTorch Datasets and DataLoaders
    # Ensure data is uint8 for ToPILImage
    if X_train.dtype != np.uint8:
        X_train = X_train.astype(np.uint8)
    if X_val.dtype != np.uint8:
        X_val = X_val.astype(np.uint8)
    if X_test.dtype != np.uint8:
        X_test = X_test.astype(np.uint8)

    train_dataset = NumpyDataset(X_train, y_train, transform=data_transforms['train'])
    val_dataset = NumpyDataset(X_val, y_val, transform=data_transforms['val'])
    test_dataset = NumpyDataset(X_test, y_test, transform=data_transforms['test'])

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    }
    
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset), 'test': len(test_dataset)}

    # 4. Setup Model (VGG16)
    print("Initializing VGG16...")
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    
    # Freeze features layers
    for param in model.features.parameters():
        param.requires_grad = False
        
    # Modify Classifier
    # VGG16 classifier: (0): Linear(in_features=25088, out_features=4096) ... (6): Linear(in_features=4096, out_features=1000)
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, num_classes)
    
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr=learning_rate, momentum=0.9)
    # optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # 5. Training Loop
    print("Starting training...")
    since = time.time()
    
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase} Epoch {epoch+1}/{num_epochs}"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

                # Save best checkpoint
                torch.save(
                    {
                        "model_state_dict": best_model_wts,
                        "num_classes": int(num_classes),
                        "input_size": int(input_size),
                        "best_val_acc": float(best_acc),
                        "epoch": int(epoch + 1),
                        "data_dir": str(data_dir),
                    },
                    best_ckpt_path,
                )
                print(f"✅ Saved best checkpoint: {best_ckpt_path}")

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Save final weights (best-on-val) in a simple state_dict form
    torch.save(model.state_dict(), final_ckpt_path)
    print(f"Saved final model state_dict to: {final_ckpt_path}")

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

    # Confusion matrix on test
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    _save_confusion_matrix(cm, cm_path, title="VGG16 • Test Confusion Matrix")
    print(f"Saved test confusion matrix to: {cm_path}")

if __name__ == '__main__':
    main()
