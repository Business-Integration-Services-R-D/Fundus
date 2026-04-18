# %%
!unzip -qq "/kaggle/input/funduske/training_final.zip adlı dosyanın kopyası" -d /kaggle/working/data
!pip install -q comet_ml

# %%
import comet_ml
import os
import numpy as np
from tqdm import tqdm
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models import ResNet50_Weights
from sklearn.metrics import accuracy_score, f1_score
from PIL import Image

DATA_DIR = '/kaggle/working/data'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
TEST_DIR = os.path.join(DATA_DIR, 'test')
MODEL_SAVE_PATH = 'resnet50_classifier.pth'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 70
RANDOM_SEED = 42

COMET_API_KEY = "u4B1ofMbFXdIO9nq8zMrpgV1S"
COMET_PROJECT_NAME = "fundus"
COMET_WORKSPACE = "traick-classification"
COMET_EXPERIMENT_NAME = "DeepLabv3ResNet50"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNetClassifier, self).__init__()
        pretrained_model = deeplabv3_resnet50(weights_backbone=ResNet50_Weights.DEFAULT)
        self.backbone = pretrained_model.backbone
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        features = self.backbone(x)['out']
        x = self.avgpool(features)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class NoduleClassificationDataset(Dataset):
    def __init__(self, root_dir, transform=None, class_mapping=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        if not os.path.isdir(self.root_dir):
            print(f"Error: Directory not found at {self.root_dir}")
            self.classes = []
            self.class_to_idx = {}
            return

        if class_mapping:
            self.class_to_idx = class_mapping
            self.classes = sorted(class_mapping, key=class_mapping.get)
        else:
            class_names = sorted([d.name for d in os.scandir(root_dir) if d.is_dir()])
            self.classes = class_names
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        print(f"Loading data from: {os.path.basename(root_dir)}")
        print(f"Found classes: {self.classes}")
        
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            for fname in tqdm(sorted(os.listdir(class_dir)), desc=f"Scanning class {class_name}"):
                path = os.path.join(class_dir, fname)
                if path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    self.samples.append((path, class_idx))

        if not self.samples:
            print(f"Warning: No images found in {self.root_dir}")
        else:
            print(f"Found {len(self.samples)} samples for {len(self.classes)} classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            return torch.zeros(3, *IMAGE_SIZE), -1
            
        if self.transform:
            image = self.transform(image)
            
        return image, label

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

val_test_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

if __name__ == '__main__':
    print("Loading Datasets...")
    train_dataset = NoduleClassificationDataset(TRAIN_DIR, transform=train_transform)
    val_dataset = NoduleClassificationDataset(VAL_DIR, transform=val_test_transform, class_mapping=train_dataset.class_to_idx)
    
    if not train_dataset.classes or not val_dataset.classes:
        print("Exiting: Training or validation dataset is empty.")
        exit()

    assert train_dataset.class_to_idx == val_dataset.class_to_idx, "Train and Val datasets must have the same class mapping!"
    NUM_CLASSES = len(train_dataset.classes)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")

    model = ResNetClassifier(num_classes=NUM_CLASSES).to(DEVICE)

    # Class-weighted cross-entropy loss to handle class imbalance
    # Weights computed as inverse frequency: total / (num_classes * class_count)
    from collections import Counter
    label_counts = Counter([label for _, label in train_dataset.samples])
    total_samples = len(train_dataset.samples)
    class_weights = torch.tensor(
        [total_samples / (NUM_CLASSES * label_counts.get(i, 1)) for i in range(NUM_CLASSES)],
        dtype=torch.float32
    ).to(DEVICE)
    print(f"Class weights: {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1, verbose=True)

    experiment = None
    if COMET_API_KEY:
        try:
            from comet_ml import Experiment
            experiment = Experiment(
                api_key=COMET_API_KEY,
                project_name=COMET_PROJECT_NAME,
                workspace=COMET_WORKSPACE,
            )
            experiment.set_name(COMET_EXPERIMENT_NAME)
            hyper_params = {
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "num_epochs": NUM_EPOCHS,
                "epochs": NUM_EPOCHS,
                "image_size": IMAGE_SIZE[0],
                "model_architecture": "ResNet50_Backbone_Classifier",
                "optimizer": "Adam",
                "loss_function": "CrossEntropyLoss",
                "scheduler": "ReduceLROnPlateau",
                "num_classes": NUM_CLASSES,
                "device": device_name,
                "early_stopping_patience": "", 
                "model_name": "DeepLabv3_ResNet50_Classifier",
                "num_gradcam_images": "",
                "num_workers": 2,
                "use_cbam": False,
                "weight_decay": 0
            }
            experiment.log_parameters(hyper_params)
            experiment.log_others({"class_mapping": train_dataset.class_to_idx})
            print(f"Comet ML experiment initialized: {experiment.url}")
        except ImportError:
            print("Comet ML library not found. Please install it: pip install comet_ml")
            experiment = None
        except Exception as e:
            print(f"Failed to initialize Comet ML: {e}")
            experiment = None

    best_val_accuracy = 0.0
    print(f"\n--- Starting Training for {NUM_EPOCHS} Epochs ---")

    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        
        model.train()
        running_train_loss = 0.0
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", leave=False)
        for images, labels in train_progress:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            train_progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = running_train_loss / len(train_loader)

        model.eval()
        running_val_loss = 0.0
        all_preds = []
        all_labels = []
        val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]", leave=False)
        with torch.no_grad():
            for images, labels in val_progress:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = running_val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_accuracy = accuracy_score(all_labels, all_preds) if all_labels else 0
        val_f1_weighted = f1_score(all_labels, all_preds, average='weighted') if all_labels else 0
        
        scheduler.step(avg_val_loss)
        epoch_duration = time.time() - epoch_start_time

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"Val F1: {val_f1_weighted:.4f} | "
            f"Duration: {epoch_duration:.2f}s"
        )

        if experiment:
            experiment.log_metric("train_loss", avg_train_loss, step=epoch + 1)
            experiment.log_metric("val_loss", avg_val_loss, step=epoch + 1)
            experiment.log_metric("val_accuracy", val_accuracy, step=epoch + 1)
            experiment.log_metric("val_f1_weighted", val_f1_weighted, step=epoch + 1)
            experiment.log_metric("learning_rate", optimizer.param_groups[0]['lr'], step=epoch + 1)

            if all_labels:
                experiment.log_confusion_matrix(
                    y_true=all_labels,
                    y_predicted=all_preds,
                    labels=list(train_dataset.classes),
                    title=f"Confusion Matrix, Epoch {epoch+1}",
                    file_name=f"confusion-matrix-epoch-{epoch+1}.json"
                )

        if val_accuracy > best_val_accuracy:
            print(f"Validation accuracy improved ({best_val_accuracy:.4f} --> {val_accuracy:.4f}). Saving model...")
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            if experiment:
                experiment.log_model("best_model", MODEL_SAVE_PATH, overwrite=True)

    print("\n--- Training Finished ---")
    print(f"Best validation accuracy during training: {best_val_accuracy:.4f}")

    if os.path.exists(MODEL_SAVE_PATH):
        print(f"\n--- Starting Final Evaluation on Test Set ---")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        model.to(DEVICE)
        model.eval()

        test_dataset = NoduleClassificationDataset(TEST_DIR, transform=val_test_transform, class_mapping=train_dataset.class_to_idx)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

        if len(test_dataset) > 0:
            running_test_loss = 0.0
            all_test_preds = []
            all_test_labels = []
            test_progress = tqdm(test_loader, desc="Evaluating on Test Set", leave=False)
            
            with torch.no_grad():
                for images, labels in test_progress:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    running_test_loss += loss.item()
                    
                    _, preds = torch.max(outputs, 1)
                    all_test_preds.extend(preds.cpu().numpy())
                    all_test_labels.extend(labels.cpu().numpy())

            avg_test_loss = running_test_loss / len(test_loader) if len(test_loader) > 0 else 0
            test_accuracy = accuracy_score(all_test_labels, all_test_preds) if all_test_labels else 0
            test_f1_weighted = f1_score(all_test_labels, all_test_preds, average='weighted') if all_test_labels else 0

            print("\n--- Test Set Results ---")
            print(f"Test Loss: {avg_test_loss:.4f}")
            print(f"Test Accuracy: {test_accuracy:.4f}")
            print(f"Test F1 (Weighted): {test_f1_weighted:.4f}")

            if experiment:
                print("Logging test metrics to Comet ML...")
                experiment.log_metric("test_loss", avg_test_loss)
                experiment.log_metric("test_accuracy", test_accuracy)
                experiment.log_metric("test_f1_weighted", test_f1_weighted)
                
                if all_test_labels:
                    experiment.log_confusion_matrix(
                        y_true=all_test_labels,
                        y_predicted=all_test_preds,
                        labels=list(train_dataset.classes),
                        title="Final Test Confusion Matrix",
                        file_name="final-test-confusion-matrix.json"
                    )
        else:
            print("Test dataset is empty or contains no valid images. Skipping test evaluation.")

    else:
        print("\nModel file not found. Skipping final evaluation on the test set.")

    if experiment:
        experiment.end()
        print("Comet ML experiment finished.")


# %%
!rm -rf /kaggle/working/data


