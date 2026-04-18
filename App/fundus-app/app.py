import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101
from torchvision.models import ResNet50_Weights, ResNet101_Weights
from timm.layers import trunc_normal_, DropPath
import os

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATHS = {
    "ConvNeXt": os.path.join(BASE_DIR, "models", "convnext-xl.pth"),
    "AttentionUNet": os.path.join(BASE_DIR, "models", "attentionUnet.pth"),
    "ResNet50": os.path.join(BASE_DIR, "models", "resnet50.pth"),
    "ResNet101": os.path.join(BASE_DIR, "models", "resnet101.pth")
}

# Diabetic Retinopathy severity grades (5-class)
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
NUM_CLASSES = len(CLASS_NAMES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODEL DEFINITIONS ---

# 1. Attention Blocks & CBAM (Used by ConvNeXt and AttentionUNet)
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

# 2. ConvNeXt Components
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
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
                 layer_scale_init_value=1e-6, head_init_scale=1.):
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
    config = MODEL_CONFIGS[model_name]
    model = ConvNeXt(depths=config['depths'], dims=config['dims'], num_classes=num_classes)
    return model

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

# 3. Attention U-Net Components
class ConvBlock(nn.Module):
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
    def __init__(self, in_channels=3, n_filters=64):
        super(AttentionUNet, self).__init__()
        self.enc1 = ConvBlock(in_channels, n_filters)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = ConvBlock(n_filters, n_filters * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc3 = ConvBlock(n_filters * 2, n_filters * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc4 = ConvBlock(n_filters * 4, n_filters * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = ConvBlock(n_filters * 8, n_filters * 16)
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
        self.final_conv = nn.Conv2d(n_filters, n_filters, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.pool1(e1)
        e2 = self.enc2(e2)
        e3 = self.pool2(e2)
        e3 = self.enc3(e3)
        e4 = self.pool3(e3)
        e4 = self.enc4(e4)
        b = self.pool4(e4)
        b = self.bottleneck(b)
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
    def __init__(self, in_channels, num_classes, n_filters=64, use_cbam=True):
        super().__init__()
        self.backbone = AttentionUNet(in_channels=in_channels, n_filters=n_filters)
        self.use_cbam = use_cbam
        if self.use_cbam:
            self.cbam = CBAM(n_filters)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(n_filters, num_classes)

    def forward(self, x):
        feature_maps = self.backbone(x)
        if self.use_cbam:
            feature_maps = self.cbam(feature_maps)
        pooled_features = self.pool(feature_maps)
        flattened_features = torch.flatten(pooled_features, 1)
        output = self.head(flattened_features)
        return output

# 4. ResNet Wrappers (DeepLabV3 Backbone)
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes, backbone='resnet50'):
        super(ResNetClassifier, self).__init__()
        if backbone == 'resnet50':
            pretrained_model = deeplabv3_resnet50(weights_backbone=ResNet50_Weights.DEFAULT)
        elif backbone == 'resnet101':
            pretrained_model = deeplabv3_resnet101(weights_backbone=ResNet101_Weights.DEFAULT)
        else:
            raise ValueError("Invalid backbone")
            
        self.backbone = pretrained_model.backbone
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        features = self.backbone(x)['out']
        x = self.avgpool(features)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# --- INFERENCE UTILS ---

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def load_model(model_name):
    path = MODEL_PATHS[model_name]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")

    # Architecture Initialization
    if model_name == "ConvNeXt":
        model = SingleTaskConvNeXt(model_name='convnext_xlarge', num_classes=NUM_CLASSES, use_cbam=True)
    elif model_name == "AttentionUNet":
        model = ClassificationAttentionUNet(in_channels=3, num_classes=NUM_CLASSES, n_filters=64, use_cbam=True)
    elif model_name == "ResNet50":
        model = ResNetClassifier(num_classes=NUM_CLASSES, backbone='resnet50')
    elif model_name == "ResNet101":
        model = ResNetClassifier(num_classes=NUM_CLASSES, backbone='resnet101')
    
    # State Dict Loading with Error Handling
    state_dict = torch.load(path, map_location=DEVICE)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        # Provide specific advice if ResNet101 looks like AttentionUNet
        if model_name == "ResNet101" and "cbam" in str(e):
             raise RuntimeError("It looks like the file uploaded for ResNet101 actually contains an AttentionUNet or ConvNeXt model (found keys like 'cbam' or 'enc1'). Please check the file path.")
        raise e

    model.to(DEVICE)
    model.eval()
    return model

def predict(image, model_choice):
    if image is None:
        return None
    
    transform = get_transforms()
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    try:
        model = load_model(model_choice)
    except Exception as e:
        return {f"Error loading model: {str(e)}": 0}

    with torch.no_grad():
        output = model(img_tensor)
        probabilities = F.softmax(output, dim=1)[0]
    
    return {CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))}

# --- GRADIO INTERFACE ---

if __name__ == "__main__":
    iface = gr.Interface(
        fn=predict,
        inputs=[
            gr.Image(type="pil", label="Upload Fundus Image"),
            gr.Dropdown(choices=list(MODEL_PATHS.keys()), value="ConvNeXt", label="Select Model")
        ],
        outputs=gr.Label(num_top_classes=NUM_CLASSES, label="Predictions"),
        title="Fundus Image Classification",
        description=f"Select a trained model to classify the fundus image. (Configured for {NUM_CLASSES} classes)"
    )
    iface.launch(share=True, debug=True)