# %%
!unzip -qq "/kaggle/input/funduske/training_final.zip adlı dosyanın kopyası" -d /kaggle/working/data

# %%
!pip install -q torch_geometric comet_ml
!pip install -q pyg_lib torch_scatter torch_sparse -f [https://data.pyg.org/whl/torch-2.3](https://data.pyg.org/whl/torch-2.3)
     

# %%
import os
import json
import time
import datetime
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from timm.layers import trunc_normal_, DropPath
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
import albumentations as A
from albumentations.pytorch import ToTensorV2
from glob import glob

# Attempt to import comet_ml
try:
    import comet_ml
except ImportError:
    print("comet_ml not found. Please install it: pip install comet_ml")
    # Create a dummy Experiment class to avoid errors if comet_ml is not installed

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='best_model.pth', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = -float('inf')
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_acc, model):
        score = val_acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
            self.counter = 0

    def save_checkpoint(self, val_acc, model):
        if self.verbose:
            self.trace_func(f'Validation accuracy increased ({self.val_acc_max:.2f}% --> {val_acc:.2f}%). Saving model to {self.path}')
        torch.save(model.state_dict(), self.path)
        self.val_acc_max = val_acc


# --- CBAM (Attention Module) ---
# Kept as it's an optional part of the new classification model

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                  nn.ReLU(),
                                  nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_map = self.conv1(x_cat)
        return self.sigmoid(x_map)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


# --- New Attention U-Net Model ---

class ConvBlock(nn.Module):
    """
    Standard building block for U-Net: (Conv2d -> BN -> ReLU) * 2
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class AttentionBlock(nn.Module):
    """
    Attention Gate (AG) for U-Net.
    Takes gating signal 'g' from lower layer and skip connection 'x'.
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUNet(nn.Module):
    """
    The core Attention U-Net model.
    Acts as the feature extractor backbone.
    """
    def __init__(self, in_channels=3, n_filters=64):
        super(AttentionUNet, self).__init__()
        
        # Encoder
        self.enc1 = ConvBlock(in_channels, n_filters)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = ConvBlock(n_filters, n_filters * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc3 = ConvBlock(n_filters * 2, n_filters * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc4 = ConvBlock(n_filters * 4, n_filters * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = ConvBlock(n_filters * 8, n_filters * 16)

        # Decoder
        self.up_conv4 = nn.ConvTranspose2d(n_filters * 16, n_filters * 8, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(F_g=n_filters * 8, F_l=n_filters * 8, F_int=n_filters * 4)
        self.dec_block4 = ConvBlock(n_filters * 16, n_filters * 8)

        self.up_conv3 = nn.ConvTranspose2d(n_filters * 8, n_filters * 4, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(F_g=n_filters * 4, F_l=n_filters * 4, F_int=n_filters * 2)
        self.dec_block3 = ConvBlock(n_filters * 8, n_filters * 4)

        self.up_conv2 = nn.ConvTranspose2d(n_filters * 4, n_filters * 2, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=n_filters * 2, F_l=n_filters * 2, F_int=n_filters)
        self.dec_block2 = ConvBlock(n_filters * 4, n_filters * 2)

        self.up_conv1 = nn.ConvTranspose2d(n_filters * 2, n_filters, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(F_g=n_filters, F_l=n_filters, F_int=n_filters // 2)
        self.dec_block1 = ConvBlock(n_filters * 2, n_filters)

        # Final output conv
        # This layer's output will be pooled for classification
        # This is also the target for Grad-CAM
        self.final_conv = nn.Conv2d(n_filters, n_filters, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)    # Skip connection 1
        
        e2 = self.pool1(e1)
        e2 = self.enc2(e2)  # Skip connection 2
        
        e3 = self.pool2(e2)
        e3 = self.enc3(e3)  # Skip connection 3
        
        e4 = self.pool3(e3)
        e4 = self.enc4(e4)  # Skip connection 4
        
        # Bottleneck
        b = self.pool4(e4)
        b = self.bottleneck(b)

        # Decoder
        d4 = self.up_conv4(b)
        e4_att = self.att4(g=d4, x=e4)
        d4 = torch.cat((e4_att, d4), dim=1)
        d4 = self.dec_block4(d4)

        d3 = self.up_conv3(d4)
        e3_att = self.att3(g=d3, x=e3)
        d3 = torch.cat((e3_att, d3), dim=1)
        d3 = self.dec_block3(d3)

        d2 = self.up_conv2(d3)
        e2_att = self.att2(g=d2, x=e2)
        d2 = torch.cat((e2_att, d2), dim=1)
        d2 = self.dec_block2(d2)

        d1 = self.up_conv1(d2)
        e1_att = self.att1(g=d1, x=e1)
        d1 = torch.cat((e1_att, d1), dim=1)
        d1 = self.dec_block1(d1)

        out = self.final_conv(d1)
        return out


class ClassificationAttentionUNet(nn.Module):
    """
    Wrapper model that uses AttentionUNet as a backbone for classification.
    """
    def __init__(self, in_channels, num_classes, n_filters=64, use_cbam=True):
        super().__init__()
        self.backbone = AttentionUNet(in_channels=in_channels, n_filters=n_filters)
        
        self.use_cbam = use_cbam
        if self.use_cbam:
            self.cbam = CBAM(n_filters) # CBAM on the final feature map
            
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(n_filters, num_classes)

    def forward(self, x):
        # Get the final feature map from the U-Net backbone
        feature_maps = self.backbone(x)
        
        # Optionally apply CBAM
        if self.use_cbam:
            feature_maps = self.cbam(feature_maps)
            
        # Global Average Pooling
        pooled_features = self.pool(feature_maps)
        
        # Flatten
        flattened_features = torch.flatten(pooled_features, 1)
        
        # Classification head
        output = self.head(flattened_features)
        return output


# --- Dataset Class ---
# (Unchanged)
class DRDataset(Dataset):
    def __init__(self, root_dir, label_map, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.label_map = label_map
        self.samples = []

        print(f"Scanning dataset folder: {root_dir}...")
        
        class_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

        for label_str in class_dirs:
            label_idx = self.label_map.get(label_str, -1)
            if label_idx == -1:
                print(f"Warning: Skipping directory '{label_str}' as it's not in the label map.")
                continue

            class_path = os.path.join(root_dir, label_str)
            
            image_files = glob(os.path.join(class_path, '*.[jJ][pP][gG]')) + \
                          glob(os.path.join(class_path, '*.[jJ][pP][eE][gG]')) + \
                          glob(os.path.join(class_path, '*.[pP][nN][gG]')) + \
                          glob(os.path.join(class_path, '*.[bB][mM][pP]'))

            for img_path in image_files:
                self.samples.append((img_path, label_idx))

        print(f"Found {len(self.samples)} images in {len(class_dirs)} classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]

        try:
            image = Image.open(img_path).convert("RGB")
            image_np = np.array(image)
        except Exception as e:
            print(f"Error opening or converting image: {img_path}. Error: {e}")
            return torch.zeros(3, 224, 224), -1, img_path

        if self.transform:
              try:
                  transformed = self.transform(image=image_np)
                  image_tensor = transformed['image']
              except Exception as e:
                  print(f"Error applying transform to image: {img_path}. Error: {e}")
                  return torch.zeros(3, 224, 224), label, img_path
        else:
              basic_transform = A.Compose([
                  A.Resize(224, 224),
                  A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                  ToTensorV2(),
              ])
              try:
                  transformed = basic_transform(image=image_np)
                  image_tensor = transformed['image']
              except Exception as e:
                  print(f"Error applying basic transform to image: {img_path}. Error: {e}")
                  return torch.zeros(3, 224, 224), label, img_path


        return image_tensor, label, img_path


# --- Helper Functions (Train, Eval, Label Map) ---
# (Unchanged)
def get_label_map_and_count_from_folder(train_dir):
    
    if not os.path.isdir(train_dir):
        raise ValueError(f"Training directory not found: {train_dir}")

    
    class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

    if not class_names:
        raise ValueError(f"No class subdirectories found in {train_dir}")

    label_map = {label: i for i, label in enumerate(class_names)}
    inverse_label_map = {i: label for label, i in label_map.items()}
    class_count = len(class_names)

    return label_map, class_count, inverse_label_map

def train_one_epoch(model, criterion, data_loader, optimizer, device, experiment, epoch):
    model.train()
    total_loss = 0
    correct_samples = 0
    total_samples = 0

    with experiment.train():
        for i, (images, targets, _) in enumerate(data_loader):
            
            if (targets == -1).any():
                print(f"Skipping batch {i} due to invalid target labels.")
                continue

            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)

            
            if targets.dtype != torch.long:
                 targets = targets.long()

            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_samples += (predicted == targets).sum().item()
            total_samples += targets.size(0)


    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
    accuracy = (correct_samples / total_samples) * 100 if total_samples > 0 else 0

    experiment.log_metric("loss", avg_loss, epoch=epoch)
    experiment.log_metric(f"accuracy", accuracy, epoch=epoch)

    return avg_loss, accuracy


@torch.no_grad()
def evaluate(data_loader, model, device, criterion, num_classes, inverse_label_map, experiment=None, epoch=None, context="val"):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    all_images = []
    all_img_paths = []

    for images, targets, img_paths in data_loader:
        
        valid_indices = targets != -1
        if not valid_indices.all():
            print(f"Skipping samples with invalid labels in evaluation batch.")
            images = images[valid_indices]
            targets = targets[valid_indices]
            img_paths = [p for i, p in enumerate(img_paths) if valid_indices[i]]
            if images.size(0) == 0:
                continue


        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        
        if targets.dtype != torch.long:
             targets = targets.long()

        outputs = model(images)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

        # Log test images
        if context == "test" and experiment:
            all_images.extend(images.cpu())
            all_img_paths.extend(img_paths)


    
    total_samples_processed = len(all_targets)
    avg_loss = total_loss / total_samples_processed if total_samples_processed > 0 else 0


    metrics = {}
    if all_targets:
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average='macro', zero_division=0, labels=range(num_classes)
        )

        cm = confusion_matrix(all_targets, all_preds, labels=range(num_classes))

        
        specificity_per_class = []
        for i in range(num_classes):
            tn = cm.sum() - (cm[i,:].sum() + cm[:,i].sum() - cm[i,i])
            fp = cm[:,i].sum() - cm[i,i]
            if tn + fp == 0:
                spec = np.nan
            else:
                spec = tn / (tn + fp)
            specificity_per_class.append(spec)

        
        specificity = np.nanmean(specificity_per_class) if not np.all(np.isnan(specificity_per_class)) else 0.0


        accuracy = accuracy_score(all_targets, all_preds) * 100

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1
        }

        if experiment:
            context_logger = experiment.validate() if context == "val" else experiment.test()
            with context_logger:
                experiment.log_metrics({
                    f"accuracy": metrics['accuracy'],
                    f"precision": metrics['precision'],
                    f"recall": metrics['recall'],
                    f"specificity": metrics['specificity'],
                    f"f1_score": metrics['f1_score'],
                    f"loss_overall": avg_loss,
                }, epoch=epoch)

                labels = [inverse_label_map.get(i, str(i)) for i in range(num_classes)]
                try:
                    experiment.log_confusion_matrix(
                        y_true=all_targets,
                        y_predicted=all_preds,
                        labels=labels,
                        title=f"{context.capitalize()} Confusion Matrix",
                        file_name=f"{context}_confusion-matrix-{epoch}.json",
                    )
                except Exception as cm_log_e:
                    print(f"Error logging confusion matrix to Comet ML: {cm_log_e}")
            
            # Log test images
            if context == "test":
                print(f"Logging {len(all_img_paths)} test images to Comet ML...")
                for img_tensor, true_label_idx, pred_label_idx, img_path in zip(all_images, all_targets, all_preds, all_img_paths):
                    true_label = inverse_label_map.get(true_label_idx, 'Unknown')
                    pred_label = inverse_label_map.get(pred_label_idx, 'Unknown')
                    
                    # De-normalize image for visualization
                    img_vis = img_tensor.numpy().transpose(1, 2, 0)
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img_vis = std * img_vis + mean
                    img_vis = np.clip(img_vis, 0, 1)

                    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                    ax.imshow(img_vis)
                    ax.set_title(f"True: {true_label} | Pred: {pred_label}")
                    ax.axis('off')
                    
                    try:
                        experiment.log_figure(
                            figure_name=f"Test_{os.path.basename(img_path)}",
                            figure=fig,
                            step=epoch
                        )
                    except Exception as log_e:
                        print(f"Failed to log test image figure to Comet ML: {log_e}")
                    plt.close(fig)


    elif experiment:
        context_logger = experiment.validate() if context == "val" else experiment.test()
        with context_logger:
            experiment.log_metric(f"loss_overall", avg_loss, epoch=epoch)

    return avg_loss, metrics


# --- Grad-CAM ---
# (Updated target_layer logic)

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.feature_maps = output.detach()

        def backward_hook(module, grad_in, grad_out):
            
            if grad_out and grad_out[0] is not None:
                self.gradients = grad_out[0].detach()
            else:
                self.gradients = None


        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        
        try:
             
             h = self.target_layer.register_full_backward_hook(backward_hook)
             self.hooks.append(h)
        except AttributeError:
             
             print("Using register_backward_hook (older PyTorch version detected)")
             h = self.target_layer.register_backward_hook(backward_hook)
             self.hooks.append(h)


    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        
        is_training = self.model.training
        self.model.eval()

        outputs = self.model(x)

        if class_idx is None:
            class_idx = torch.argmax(outputs, dim=1).item()

        target_score = outputs[:, class_idx]

        
        
        x.requires_grad_(True)
        if target_score.requires_grad:
            
            
            target_score.backward(retain_graph=False)
        else:
             print("Warning: Target score does not require gradients. Grad-CAM might not work correctly.")
             
             if is_training: self.model.train()
             return np.zeros(x.shape[2:], dtype=np.float32), class_idx


        if self.gradients is None or self.feature_maps is None:
             
             if is_training: self.model.train()
             raise ValueError("Gradients or feature maps are not available after backward. Check hook registration or backward call.")


        
        grads = self.gradients[0]
        fmaps = self.feature_maps[0]

        weights = torch.mean(grads, dim=[1, 2], keepdim=True)
        cam = torch.sum(weights * fmaps, dim=0, keepdim=True)

        cam = F.relu(cam)

        
        cam = F.interpolate(cam.unsqueeze(0), size=x.shape[2:], mode='bilinear', align_corners=False)

        
        cam = cam.squeeze()
        if cam.numel() > 0 and cam.max() > cam.min():
               cam -= cam.min()
               cam /= cam.max()
        else:
               cam = torch.zeros_like(cam)


        
        if is_training: self.model.train()

        
        cam_np = cam.detach().cpu().numpy()

        return cam_np, class_idx


    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.feature_maps = None
        self.gradients = None


def visualize_and_save_grad_cam(model, loader, device, num_classes, inverse_label_map, output_dir, experiment, num_images=4):
    print(f"Generating Grad-CAM visualizations for {num_images} images...")
    os.makedirs(output_dir, exist_ok=True)

    
    model.eval()

    
    try:
        
        target_layer = model.backbone.final_conv
        print(f"Targeting layer for Grad-CAM: {target_layer.__class__.__name__}")
    except AttributeError:
        print("Could not automatically find backbone.final_conv. Please verify target layer.")
        
        return

    
    grad_cam_extractor = GradCAM(model, target_layer)

    images_generated = 0
    data_iter = iter(loader)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    while images_generated < num_images:
        try:
            images, targets, img_paths = next(data_iter)
        except StopIteration:
            print("Reached end of dataloader while generating Grad-CAM images.")
            break
        except Exception as e:
            print(f"Error getting next batch from dataloader: {e}")
            break


        for i in range(images.size(0)):
            if images_generated >= num_images:
                break

            
            if targets[i].item() == -1:
                print(f"Skipping invalid sample detected in Grad-CAM loop (path: {img_paths[i]}).")
                continue


            img_tensor_batch = images[i:i+1].to(device)
            true_label_idx = targets[i].item()
            img_path = img_paths[i]

            
            if true_label_idx < 0 or true_label_idx >= num_classes:
                 print(f"Skipping image {os.path.basename(img_path)} due to invalid label index: {true_label_idx}")
                 continue

            true_label = inverse_label_map.get(true_label_idx, 'Unknown')

            
            img_vis = img_tensor_batch[0].cpu().numpy().transpose(1, 2, 0)
            img_vis = std * img_vis + mean
            img_vis = np.clip(img_vis, 0, 1)

            try:
                
                heatmap, pred_idx = grad_cam_extractor(img_tensor_batch)

                pred_label = inverse_label_map.get(pred_idx, 'Unknown')

                heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
                heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

                
                img_vis_uint8 = np.uint8(255 * img_vis)

                superimposed_img = heatmap_colored * 0.4 + img_vis_uint8
                superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)


                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                ax1.imshow(img_vis)
                ax1.set_title(f'Original Image\nFile: {os.path.basename(img_path)}')
                ax1.axis('off')

                ax2.imshow(superimposed_img)
                ax2.set_title(f'Grad-CAM\nTrue: {true_label} | Pred: {pred_label}')
                ax2.axis('off')

                plt.tight_layout()

                if experiment:
                    try:
                        experiment.log_figure(
                            figure_name=f"Grad-CAM_{os.path.basename(img_path)}",
                            figure=fig
                        )
                    except Exception as log_e:
                        print(f"Failed to log Grad-CAM figure to Comet ML: {log_e}")


                safe_filename = "".join([c if c.isalnum() else "_" for c in os.path.basename(img_path)])
                save_path = os.path.join(output_dir, f"gradcam_img{images_generated}_{safe_filename}.png")

                plt.savefig(save_path)
                plt.close(fig)
                images_generated += 1


            except Exception as e:
                print(f"Could not generate Grad-CAM for image {os.path.basename(img_path)}. Error: {e}")
                


    
    grad_cam_extractor.remove_hooks()
    print(f"Grad-CAM visualizations generation finished. Saved to {output_dir}")



def main(args):
    
    experiment = comet_ml.Experiment(
        api_key=args.comet_api_key,
        project_name=args.comet_project_name,
        workspace=args.comet_workspace,
    )
    experiment.set_name(args.comet_experiment_name)
    experiment.log_parameters(vars(args))

    device = torch.device(args.device)
    cudnn.benchmark = True

    print("Deriving label map from training folder structure...")
    train_dir = os.path.join(args.data_path, 'train')
    try:
        label_map, num_classes, inverse_label_map = get_label_map_and_count_from_folder(train_dir)
        print(f"Number of classes found: {num_classes}")
        print("Label map derived:", label_map)
    except Exception as e:
        print(f"Error deriving label map from folders: {e}")
        experiment.log_other("error", f"Failed to derive label map: {e}")
        experiment.end()
        return

    os.makedirs(args.output_dir, exist_ok=True)
    try:
        with open(os.path.join(args.output_dir, "label_map.json"), 'w') as f:
            json.dump(label_map, f, indent=4)
        with open(os.path.join(args.output_dir, "inverse_label_map.json"), 'w') as f:
             json.dump(inverse_label_map, f, indent=4)
    except Exception as e:
        print(f"Error saving label map files: {e}")

    
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.Rotate(limit=15, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.CLAHE(p=0.5),
        A.Blur(blur_limit=3, p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    print("Loading datasets from folders...")
    try:
        val_dir = os.path.join(args.data_path, 'val')
        test_dir = os.path.join(args.data_path, 'test')

        dataset_train = DRDataset(root_dir=train_dir, label_map=label_map, transform=train_transform)
        dataset_val = DRDataset(root_dir=val_dir, label_map=label_map, transform=val_transform)
        dataset_test = DRDataset(root_dir=test_dir, label_map=label_map, transform=val_transform)

        
        if len(dataset_train) == 0:
             print("Error: Training dataset is empty. Check data_path and folder structure.")
             experiment.log_other("error", "Training dataset empty.")
             experiment.end()
             return
        if len(dataset_val) == 0:
             print("Warning: Validation dataset is empty.")
             

    except Exception as e:
         print(f"Error loading datasets: {e}")
         experiment.log_other("error", f"Failed to load datasets: {e}")
         experiment.end()
         return

    data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    data_loader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    data_loader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)


    print(f"Creating model: ClassificationAttentionUNet...")
    model = ClassificationAttentionUNet(
        in_channels=3,
        num_classes=num_classes,
        n_filters=64, # Starting filters for the U-Net
        use_cbam=args.use_cbam
    )
    
    # Removed the ConvNeXt-specific finetuning block.
    # The Attention U-Net will be trained from scratch.

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Class-weighted cross-entropy loss to handle class imbalance
    # Weights computed as inverse frequency: total / (num_classes * class_count)
    from collections import Counter
    label_counts = Counter([label for _, label in dataset_train.samples])
    total_samples = len(dataset_train.samples)
    class_weights = torch.tensor(
        [total_samples / (num_classes * label_counts.get(i, 1)) for i in range(num_classes)],
        dtype=torch.float32
    ).to(device)
    print(f"Class weights: {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    early_stopper = EarlyStopping(patience=args.early_stopping_patience, verbose=True, path=os.path.join(args.output_dir, 'best_model.pth'))

    
    best_epoch = 0

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, criterion, data_loader_train, optimizer, device, experiment, epoch)

        
        if len(dataset_val) > 0:
            val_loss, val_metrics = evaluate(data_loader_val, model, device, criterion, num_classes, inverse_label_map, experiment, epoch, "val")
            val_acc = val_metrics.get('accuracy', 0)
        else:
            val_loss, val_metrics, val_acc = 0, {}, 0
            print("Skipping validation due to empty dataset.")


        print(f"--- Epoch {epoch+1}/{args.epochs} ---")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        if len(dataset_val) > 0:
            print(f"  Val Loss: {val_loss:.4f}     | Val Acc: {val_acc:.2f}%")
            if val_metrics:
                 print(f"  Precision: {val_metrics['precision']:.4f} | Recall (Sensitivity): {val_metrics['recall']:.4f}")
                 print(f"  Specificity: {val_metrics['specificity']:.4f} | F1-Score: {val_metrics['f1_score']:.4f}")
        else:
            print("  Validation skipped.")

        
        if val_acc > early_stopper.val_acc_max:
             best_epoch = epoch

        early_stopper(val_acc, model)
        if early_stopper.early_stop:
            print("Early stopping triggered")
            experiment.log_other("early_stopping_triggered_at_epoch", epoch)
            break

    
    experiment.log_other("best_epoch", best_epoch)

    if early_stopper.best_score is not None:
         experiment.log_metric("best_val_accuracy", early_stopper.best_score)
         try:
            experiment.log_model("best_model", os.path.join(args.output_dir, 'best_model.pth'))
         except Exception as e:
            print(f"Error logging best model to Comet ML: {e}")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time {total_time_str}')
    experiment.log_other("training_duration", total_time_str)


    
    print("\n--- Starting Testing Phase ---")
    
    if len(dataset_test) > 0:
        
        best_model_path = os.path.join(args.output_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            print("Loading best model for testing...")
            try:
                model.load_state_dict(torch.load(best_model_path, map_location=device))
            except Exception as e:
                 print(f"Error loading best model state_dict: {e}. Testing with last epoch model.")
                 experiment.log_other("warning", "Used last epoch model for testing due to load error.")
        else:
            print("Warning: Best model file not found. Testing with the last epoch model.")
            experiment.log_other("warning", "Best model file not found, testing with last epoch model.")


        test_loss, test_metrics = evaluate(data_loader_test, model, device, criterion, num_classes, inverse_label_map, experiment, best_epoch, "test")


        print(f"\n--- Test Results (using model from epoch {best_epoch+1 if os.path.exists(best_model_path) else args.epochs}) ---")
        print(f"  Test Loss: {test_loss:.4f}")
        if test_metrics:
             test_acc = test_metrics.get('accuracy', 0)
             print(f"  Test Accuracy: {test_acc:.2f}%")
             print(f"  Precision: {test_metrics['precision']:.4f} | Recall (Sensitivity): {test_metrics['recall']:.4f}")
             print(f"  Specificity: {test_metrics['specificity']:.4f} | F1-Score: {test_metrics['f1_score']:.4f}")
        else:
             print("  Test metrics could not be calculated (possibly empty test set or evaluation error).")


        
        print("\n--- Generating Grad-CAM visualizations ---")
        try:
            visualize_and_save_grad_cam(
                model=model,
                loader=data_loader_test,
                device=device,
                num_classes=num_classes,
                inverse_label_map=inverse_label_map,
                output_dir=os.path.join(args.output_dir, "grad_cam_visualizations"),
                experiment=experiment,
                num_images=args.num_gradcam_images
            )
        except Exception as e:
             print(f"Error during Grad-CAM generation: {e}")
             experiment.log_other("warning", f"Grad-CAM generation failed: {e}")

    else:
        print("Test dataset is empty. Skipping testing and Grad-CAM generation.")
        experiment.log_other("info", "Test dataset empty, skipping test phase.")


    print("Experiment finished.")
    experiment.end()


if __name__ == '__main__':
    class Args:
        # Model & Training Hyperparameters
        batch_size = 16
        epochs = 20
        lr = 1e-5 # Learning rate
        weight_decay = 0.05
        early_stopping_patience = 15
        use_cbam = True # Whether to use CBAM on the U-Net's output features

        # Data and Output Paths
        # IMPORTANT: Update this path to your dataset location
        data_path = '/kaggle/working/data' 
        output_dir = 'AttnUNet_Classification' # Changed from ConvNeXt_XL

        # System Config
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        num_workers = 2

        # Visualization
        num_gradcam_images = 4 # Number of Grad-CAM images to generate from test set

        # Comet ML Config
        # DO NOT REMOVE: API key provided by user
        comet_api_key = "u4B1ofMbFXdIO9nq8zMrpgV1S"
        comet_project_name = "fundus"
        comet_workspace = "traick-classification"
        comet_experiment_name = "AttnUNet-Classification-GradCAM" # Changed from ConvNeXt-XL

    args = Args()
    
    # Ensure the data_path exists
    if not os.path.isdir(args.data_path):
        print(f"Error: data_path not found at '{args.data_path}'")
        print("Please update the 'data_path' variable in the 'Args' class to point to your dataset.")
        print("The dataset directory should contain 'train', 'val', and 'test' subfolders.")
    else:
        main(args)


# %%
!rm -rf /kaggle/working/data


