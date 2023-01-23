import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class ViTEmbeddings(nn.Module):
  def __init__(self, image_size=1775, patch_size=25, num_channels=1, em_dim=4608, drop=0.1) -> None:
    super().__init__()
    assert image_size % patch_size == 0, "image size must be an integer multiply of patch size"
    self.patch_embedding = nn.Conv2d(num_channels, em_dim, kernel_size=patch_size, stride=patch_size, bias=False)
    self.position_embedding = nn.Parameter(torch.zeros(1, (image_size // patch_size)**2 + 1, em_dim))
    self.cls_token = nn.Parameter(torch.zeros(1, 1, em_dim))
    self.dropout = nn.Dropout(drop)
    
  def forward(self, x):
      x = self.patch_embedding(x).flatten(2).transpose(1, 2)  # [B,C, H, W] -> [B, num_patches, em_dim]
      x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], dim=1)  # [B, num_patches+1, em_dim]
      x = x + self.position_embedding
      x = self.dropout(x)
      return x

class ViTEncoder(nn.Module):
  def __init__(self, num_layers, image_size, patch_size, num_channels, em_dim, drop, num_heads, ff_dim):
    super().__init__()
    self.embedding = ViTEmbeddings(image_size, patch_size, num_channels, em_dim, drop)
    encoder_layer = TransformerEncoderLayer(em_dim, nhead=num_heads, dim_feedforward=ff_dim, activation='gelu', dropout=drop)
    self.encoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
    self.layernorm = nn.LayerNorm(em_dim, eps=1e-6)

  def forward(self, x):
    x = self.embedding(x)
    x = x.transpose(0, 1)  # Switch batch and sequence length dimension for pytorch convention
    x = self.encoder(x)
    x = self.layernorm(x[0])
    return x

class Residual_Block(nn.Module):
  def __init__(self, channels, kernel_size = 3):
    super().__init__()
    self.conv2d = nn.Conv2d(channels, channels, kernel_size, stride=1, padding=1)
    self.relu_bn = nn.Sequential(
        nn.ReLU(),
        nn.BatchNorm2d(channels)
    )

  def forward(self, x):
    y = self.conv2d(x)
    y = self.relu_bn(y)
    y = self.conv2d(y)
    y+=x
    out = self.relu_bn(y)
    return out

def ConvTransBlock(in_channel, out_channel, kernel_size, stride, padding):
  return nn.Sequential(
      nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(out_channel),
      nn.LeakyReLU(negative_slope=0.01, inplace=True)
  )

class ViTGenerator(nn.Module):
  def __init__(self, num_layers=4, image_size=1775, patch_size=25, num_channels=1, em_dim=4608, drop=0.1, num_heads=12, ff_dim=3072):
    super().__init__()
    self.vit_encoder = ViTEncoder(num_layers=num_layers, image_size=image_size, patch_size=patch_size, 
                                  num_channels=num_channels, em_dim=em_dim, drop=drop, num_heads=num_heads, ff_dim=ff_dim)
    self.dense = nn.Linear(em_dim,4608)
    self.unflatten = nn.Unflatten(1,(128, 6, 6))
    self.ConvBlock1 = ConvTransBlock(128, 64, 5, 3, 1)
    self.ResidualBlock1 = Residual_Block(64)
    self.ConvBlock2 = ConvTransBlock(64, 32, 5, 3, 1)
    self.ResidualBlock2 = Residual_Block(32)
    self.ConvBlock3 = ConvTransBlock(32, 16, 5, 3, 1)
    self.ResidualBlock3 = Residual_Block(16)
    self.ConvBlock4 = ConvTransBlock(16, 1, 6, 4, 1)
    self.ResidualBlock4 = Residual_Block(1)
    self.Conv2d = nn.Conv2d(1, 1, 1, stride=1, padding=1)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = self.vit_encoder(x)
    x = self.dense(x)
    x = self.unflatten(x)
    
    x = self.ConvBlock1(x)
    x = self.ResidualBlock1(x)

    x = self.ConvBlock2(x)
    x = self.ResidualBlock2(x)
    
    x = self.ConvBlock3(x)
    x = self.ResidualBlock3(x)
    
    x = self.ConvBlock4(x)
    x = self.ResidualBlock4(x)
    
    x = self.Conv2d(x)
    x = self.sigmoid(x)
    return x