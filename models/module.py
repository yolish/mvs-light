import torch
import torch.nn as nn
import torch.nn.functional as F

# Implementation of differentiable warping from PatchMatchNet 
def differentiable_warping(
    src_fea: torch.Tensor, src_proj: torch.Tensor, ref_proj: torch.Tensor, depth_samples: torch.Tensor
):
    """Differentiable homography-based warping, implemented in Pytorch.

    Args:
        src_fea: [B, C, H, W] source features, for each source view in batch
        src_proj: [B, 4, 4] source camera projection matrix, for each source view in batch
        ref_p  roj: [B, 4, 4] reference camera projection matrix, for each ref view in batch
        depth_samples: [B, Ndepth, H, W] virtual depth layers
    Returns:
        warped_src_fea: [B, C, Ndepth, H, W] features on depths after perspective transformation
    """

    batch, channels, height, width = src_fea.shape
    num_depth = depth_samples.shape[1]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid(
            [
                torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                torch.arange(0, width, dtype=torch.float32, device=src_fea.device),
            ]
        )
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]

        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_samples.view(
            batch, 1, num_depth, height * width
        )  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        # avoid negative depth
        negative_depth_mask = proj_xyz[:, 2:] <= 1e-3
        proj_xyz[:, 0:1][negative_depth_mask] = float(width)
        proj_xyz[:, 1:2][negative_depth_mask] = float(height)
        proj_xyz[:, 2:3][negative_depth_mask] = 1.0
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1  # [B, Ndepth, H*W]
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy

    # Set align_corners = True in higher torch versions
    warped_src_fea = F.grid_sample(
        src_fea,
        grid.view(batch, num_depth * height, width, 2),
        mode="bilinear",
        padding_mode="zeros"
    )

    return warped_src_fea.view(batch, channels, num_depth, height, width)

class ConvBnReLU(nn.Module):
    """Implements 2d Convolution + batch normalization + ReLU"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        pad: int = 1,
        dilation: int = 1,
    ) -> None:
        """initialization method for convolution2D + batch normalization + relu module
        Args:
            in_channels: input channel number of convolution layer
            out_channels: output channel number of convolution layer
            kernel_size: kernel size of convolution layer
            stride: stride of convolution layer
            pad: pad of convolution layer
            dilation: dilation of convolution layer
        """
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=pad, dilation=dilation, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method"""
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvBnReLU3D(nn.Module):
    """Implements of 3d convolution + batch normalization + ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        pad: int = 1,
        dilation: int = 1,
    ) -> None:
        """initialization method for convolution3D + batch normalization + relu module
        Args:
            in_channels: input channel number of convolution layer
            out_channels: output channel number of convolution layer
            kernel_size: kernel size of convolution layer
            stride: stride of convolution layer
            pad: pad of convolution layer
            dilation: dilation of convolution layer
        """
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=pad, dilation=dilation, bias=False
        )
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method"""
        return F.relu(self.bn(self.conv(x)), inplace=True)
