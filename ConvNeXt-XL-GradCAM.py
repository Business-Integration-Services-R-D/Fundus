# %% In [1]:
get_ipython().system('unzip -qq "/kaggle/input/funduske/training_final.zip adlı dosyanın kopyası" -d /kaggle/working/data')

# %% In [2]:
get_ipython().system('pip install -q torch_geometric comet_ml')
get_ipython().system('pip install -q pyg_lib torch_scatter torch_sparse -f [https://data.pyg.org/whl/torch-2.3](https://data.pyg.org/whl/torch-2.3)')
     

# %% In [3]:
import comet_ml

# %% In [4]:
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


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                 requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_feature_maps(self, x):
          for i in range(4):
              x = self.downsample_layers[i](x)
              x = self.stages[i](x)
          return x

    def forward_features(self, x):
        x = self.forward_feature_maps(x)
        return self.norm(x.mean([-2, -1]))

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


MODEL_CONFIGS = {
    'convnext_tiny': {'depths': [3, 3, 9, 3], 'dims': [96, 192, 384, 768]},
    'convnext_xlarge': {'depths': [3, 3, 27, 3], 'dims': [256, 512, 1024, 2048]},
}

def create_convnext(model_name='convnext_tiny', num_classes=1000):
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model {model_name} not supported. Available models: {list(MODEL_CONFIGS.keys())}")
    config = MODEL_CONFIGS[model_name]
    model = ConvNeXt(depths=config['depths'], dims=config['dims'], num_classes=num_classes)
    return model


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


class SingleTaskConvNeXt(nn.Module):
    def __init__(self, model_name, num_classes, use_cbam=True):
        super().__init__()
        self.backbone = create_convnext(model_name, num_classes=1000)

        feature_dim = MODEL_CONFIGS[model_name]['dims'][-1]

        self.use_cbam = use_cbam
        if self.use_cbam:
              self.cbam = CBAM(feature_dim)

        self.head = nn.Linear(feature_dim, num_classes)

        self.backbone.head = nn.Identity()


    def forward(self, x):
          feature_maps = self.backbone.forward_feature_maps(x)

          if self.use_cbam:
              refined_maps = self.cbam(feature_maps)
              features = self.backbone.norm(refined_maps.mean([-2, -1]))
          else:
              features = self.backbone.norm(feature_maps.mean([-2, -1]))

          output = self.head(features)
          return output


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

    for images, targets, _ in data_loader:
        
        valid_indices = targets != -1
        if not valid_indices.all():
            print(f"Skipping samples with invalid labels in evaluation batch.")
            images = images[valid_indices]
            targets = targets[valid_indices]
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

    elif experiment:
        context_logger = experiment.validate() if context == "val" else experiment.test()
        with context_logger:
            experiment.log_metric(f"loss_overall", avg_loss, epoch=epoch)

    return avg_loss, metrics


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
        
        target_layer = model.backbone.stages[-1]
        print(f"Targeting layer for Grad-CAM: {target_layer.__class__.__name__}")
    except AttributeError:
        print("Could not automatically find backbone.stages[-1]. Please verify target layer.")
        
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


    print(f"Creating model: {args.model_name}...")
    model = SingleTaskConvNeXt(
        model_name=args.model_name,
        num_classes=num_classes,
        use_cbam=args.use_cbam
    )

    if args.finetune:
        
        print(f"Loading pretrained weights for backbone from {args.finetune}")
        try:
            if args.finetune.startswith('http://') or args.finetune.startswith('https://'):
                  checkpoint = torch.hub.load_state_dict_from_url(args.finetune, map_location='cpu', check_hash=True)
            else:
                  checkpoint = torch.load(args.finetune, map_location='cpu')

            checkpoint_model = checkpoint.get('model', checkpoint)


            
            state_dict = checkpoint_model.copy()
            keys_to_remove = [k for k in state_dict if k.startswith('head.')]
            if keys_to_remove:
                  print(f"Removing keys from pretrained checkpoint: {keys_to_remove}")
                  for k in keys_to_remove:
                      del state_dict[k]


            
            msg = model.backbone.load_state_dict(state_dict, strict=False)
            print("Pretrained weights loading message:", msg)
            if msg.missing_keys or msg.unexpected_keys:
                  print("Note: Some keys were missing or unexpected during weight loading, which is normal when replacing the head.")
        except Exception as e:
              print(f"Error loading pretrained weights: {e}")
              experiment.log_other("warning", f"Failed to load pretrained weights: {e}")


    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

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
            print(f"  Val Loss: {val_loss:.4f}    | Val Acc: {val_acc:.2f}%")
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
        model_name = 'convnext_xlarge'
        batch_size = 16
        epochs = 30
        lr = 1e-5
        weight_decay = 0.05
        early_stopping_patience = 15
        use_cbam = True

        data_path = '/kaggle/working//data'

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        output_dir = 'ConvNeXt_XL'
        num_workers = 2
        finetune = 'https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth'
        num_gradcam_images = 4

        comet_api_key = "u4B1ofMbFXdIO9nq8zMrpgV1S"
        comet_project_name = "fundus"
        comet_workspace = "traick-classification"
        comet_experiment_name = "ConvNeXt-XL-GradCAM"

    args = Args()
    main(args)

