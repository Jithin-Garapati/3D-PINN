import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.float32))
        self.weights2 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.float32))
        self.weights3 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.float32))
        self.weights4 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.float32))

    def forward(self, x):
        batchsize = x.shape[0]
        x = x.to(torch.float32)
        
        # Compute Fourier coefficients
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, 
                           device=x.device, dtype=torch.cfloat)
        
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            torch.einsum("bixyz,ioxyz->boxyz", 
                        x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], 
                        torch.view_as_complex(torch.stack([self.weights1, self.weights2], dim=-1)))

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

class FNO3D(nn.Module):
    def __init__(self, modes1, modes2, modes3, width=32, in_channels=4, out_channels=4):
        super(FNO3D, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.fc0 = nn.Linear(in_channels, self.width)
        
        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        
        self.bn0 = nn.BatchNorm3d(self.width)
        self.bn1 = nn.BatchNorm3d(self.width)
        self.bn2 = nn.BatchNorm3d(self.width)
        self.bn3 = nn.BatchNorm3d(self.width)
        
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        batch_size = x.shape[0]
        grid_size = x.shape[1:-1]  # (x, y, z)
        
        x = x.to(torch.float32)
        
        x = self.fc0(x)  # (batch, x, y, z, width)
        
        x = x.permute(0, 4, 1, 2, 3)  # (batch, width, x, y, z)
        
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(self.bn0(x))
        
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(self.bn1(x))
        
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(self.bn2(x))
        
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        x = F.gelu(self.bn3(x))
        
        x = x.permute(0, 2, 3, 4, 1)  # (batch, x, y, z, width)
        
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        
        return x 