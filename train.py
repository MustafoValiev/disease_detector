import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import numpy as np
# from tqdm import tqdm  <-- Commented out to prevent progress bar clutter
import sys
import copy 

# --- HYPERPARAMETERS & CONFIG ---
# SILENCED CONFIGURATION CHECKS FOR REPORT
# print("="*50)
# print("SYSTEM CONFIGURATION CHECK")
# print("="*50)
# print(f"PyTorch Version: {torch.__version__}")

if torch.cuda.is_available():
    # print(f"GPU Name:        {torch.cuda.get_device_name(0)}")
    DEVICE = torch.device('cuda')
else:
    # print("!! WARNING: CUDA NOT DETECTED. !!")
    DEVICE = torch.device('cpu')
# print(f"Active Device:   {DEVICE}")
# print("="*50 + "\n")

DATASET_PATH = './data' 
IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 16 
MAX_EPOCHS = 50     
LEARNING_RATE = 0.0001
NUM_CLASSES = 8
EARLY_STOPPING_PATIENCE = 10 

# --- DATA LOADING AND AUGMENTATION ---
def get_data_loaders(data_dir):
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        full_dataset = datasets.ImageFolder(root=data_dir)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {data_dir}. Check DATASET_PATH.")
        sys.exit(1)

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_dataset.dataset.transform = train_transforms
    test_dataset.dataset.transform = test_transforms 

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # SILENCED DATASET INFO
    # print(f"Total Images: {len(full_dataset)}")
    # print(f"Training Set: {train_size} images")
    # print(f"Test Set (for Validation/Evaluation): {test_size} images")
    # print(f"Classes: {full_dataset.classes}")
    
    return train_loader, test_loader, full_dataset.classes

# --- MODEL ARCHITECTURE ---
class SuperbCustomCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(SuperbCustomCNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, 300), 
            nn.BatchNorm1d(300), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(300, 150), 
            nn.BatchNorm1d(150), nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(150, num_classes)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.fc(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --- VALIDATION HELPER FUNCTION ---
def validate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_test_loss = test_loss / len(test_loader)
    test_acc = 100 * correct / total
    model.train()
    return avg_test_loss, test_acc

# --- TRAINING FUNCTION (CLEAN OUTPUT) ---
def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, max_epochs, patience):
    model.train()
    
    history = {'train_accuracy': [], 'train_loss': [], 'test_accuracy': [], 'test_loss': []}
    best_test_loss = float('inf')
    best_model_weights = None
    epochs_no_improve = 0
    
    # Removed initial start message to keep log clean
    # print("\nStarting Training with Early Stopping...")
    
    for epoch in range(max_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        # --- CLEAN LOOP (Removed tqdm progress bar) ---
        # We iterate directly over train_loader to avoid the progress bar output
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Calculate metrics
        epoch_train_acc = 100 * correct / total
        epoch_train_loss = running_loss / len(train_loader)
        epoch_test_loss, epoch_test_acc = validate_model(model, test_loader, criterion)
        
        history['train_accuracy'].append(epoch_train_acc)
        history['train_loss'].append(epoch_train_loss)
        history['test_accuracy'].append(epoch_test_acc)
        history['test_loss'].append(epoch_test_loss)
        
        scheduler.step(epoch_test_loss)
        
        # --- THE ONLY OUTPUT LINE ---
        print(f"Epoch [{epoch+1}/{max_epochs}] | Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | Test Loss: {epoch_test_loss:.4f}, Test Acc: {epoch_test_acc:.2f}%")

        # Early Stopping
        if epoch_test_loss < best_test_loss:
            best_test_loss = epoch_test_loss
            epochs_no_improve = 0
            best_model_weights = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs (Patience: {patience}).")
                break

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        # print("Loaded best model weights.") # Silenced

    # Plotting (Silent save)
    epochs_ran = len(history['train_accuracy'])
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs_ran + 1), history['train_accuracy'], label='Train Accuracy', marker='o')
    plt.plot(range(1, epochs_ran + 1), history['test_accuracy'], label='Test Accuracy', marker='o')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs_ran + 1), history['train_loss'], label='Train Loss', color='red', marker='o')
    plt.plot(range(1, epochs_ran + 1), history['test_loss'], label='Test Loss', color='orange', marker='o')
    plt.title('Training and Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves_optimized.png')
    # print("\nTraining curves saved as 'training_curves_optimized.png'") # Silenced
    
    return history

# --- FINAL EVALUATION ---
def evaluate_model(model, test_loader, classes):
    model.eval()
    y_true = []
    y_pred = []
    
    # Optional: Clean indicator that evaluation is happening
    print("\n" + "="*30)
    print("FINAL MODEL EVALUATION")
    print("="*30)
    
    with torch.no_grad():
        # We keep the loop clean (no tqdm) to match your style
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    # Calculate F1 Score
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # --- OUTPUTS FOR REPORT ---
    print(f"\n>>> FINAL WEIGHTED F1 SCORE: {f1:.4f} <<<")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=classes))
    
    # Generate Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (F1 Score: {f1:.4f})')
    plt.savefig('confusion_matrix.png')
    print("Confusion Matrix saved as 'confusion_matrix.png'")
    print("="*30 + "\n")

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    train_loader, test_loader, classes = get_data_loaders(DATASET_PATH)
    model = SuperbCustomCNN(num_classes=NUM_CLASSES).to(DEVICE)
    
    # SILENCED ARCHITECTURE SUMMARY
    # print("\n" + "="*30)
    # print("MODEL ARCHITECTURE SUMMARY")
    # print("="*30)
    # print(model)
    # num_params = count_parameters(model)
    # print(f"\nTotal Trainable Parameters: {num_params:,}")
    # print("="*30 + "\n")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    # Train
    train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, MAX_EPOCHS, EARLY_STOPPING_PATIENCE)
    
    # Evaluate
    evaluate_model(model, test_loader, classes)