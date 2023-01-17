import torch.nn as nn

# Encoder Layers (7100x7100)
def conv(in_channels, out_channels):
  return nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=3, padding=1),
    # nn.LeakyReLU(negative_slope=0.01, inplace=True),
    nn.GELU(),
    nn.BatchNorm2d(out_channels),
  )
def conv_max(in_channels, out_channels):
  return nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=3, padding=1),
    nn.MaxPool2d(3, stride=2),
    # nn.LeakyReLU(negative_slope=0.01, inplace=True),
    nn.GELU(),
    nn.BatchNorm2d(out_channels),
  )

# Encoder Layers (1775x1775)
def conv2(in_channels, out_channels):
  return nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2),
    nn.GELU(),
    nn.BatchNorm2d(out_channels),
  )
def conv_max2(in_channels, out_channels):
  return nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=3),
    nn.MaxPool2d(3, stride=2),
    nn.GELU(),
    nn.BatchNorm2d(out_channels),
  )

# Decoder Layers
def convTrans(in_channels, out_channels):
  return nn.Sequential(
    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=2),
    nn.GELU()
  )
def up_conv(in_channels, out_channels, padding):
  return nn.Sequential(
    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=5, stride=2, padding=padding),
    nn.GELU(),
    nn.BatchNorm2d(out_channels),
    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
  )

class Encoder(nn.Module):
  def __init__(self):
    super().__init__()
    # self.block1 = conv(1, 16)
    # self.block2 = conv_max(16, 32)
    # self.block3 = conv_max(32, 64)
    # self.block4 = conv_max(64, 128)
    self.block1 = conv_max2(1, 16)
    self.block2 = conv2(16, 32)
    self.block3 = conv_max2(32, 64)
    self.block4 = conv2(64, 128)

  def forward(self, x):
    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = self.block4(x)
    return x

class Decoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.block1 = up_conv(128, 64, 1)
    self.block2 = convTrans(64, 32)
    self.block3 = up_conv(32, 16, 2)
    self.block4 = convTrans(16, 1)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = self.block4(x)
    return self.sigmoid(x)


class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.encoder = Encoder()
    self.phase_decoder = Decoder()
    self.amp_decoder = Decoder()

  def forward(self, diffraction):
    latent_z = self.encoder(diffraction)
    phase = self.phase_decoder(latent_z)
    amp = self.amp_decoder(latent_z)
    return phase, amp