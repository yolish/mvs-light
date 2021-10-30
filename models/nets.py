import torch
from torch import nn
from models.module import ConvBnReLU3D, differentiable_warping, ConvBnReLU
import torch.nn.functional as F

# Basic UNet arch from: https://raw.githubusercontent.com/4uiiurz1/pytorch-nested-unet/master/archs.py
# Modified to return and take in levels 1 and 2 of the encoder
class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input, forward_decoder=True, mvs_features=None):
        x0_0 = self.conv0_0(input)

        if mvs_features is None:
            x1_0 = self.conv1_0(self.pool(x0_0))
            x2_0 = self.conv2_0(self.pool(x1_0))

        else:
            x1_0 = self.conv1_0(self.pool(x0_0)) + mvs_features[1]
            x2_0 = self.conv2_0(self.pool(x1_0)) + mvs_features[2]

        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
        output = None
        if forward_decoder:
            x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
            x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
            x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
            x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))
            output = self.final(x0_4)
        # Level 1: H/2, W/2  Level 2: H/4, W/4,
        return output, {1:x1_0, 2:x2_0}


class PixelwiseNet(nn.Module):
    """Pixelwise Net: A simple pixel-wise view weight network, composed of 1x1x1 convolution layers
    and sigmoid nonlinearities, takes the initial set of similarities to output a number between 0 and 1 per
    pixel as estimated pixel-wise view weight.

    1. The Pixelwise Net is used in adaptive evaluation step
    2. The similarity is calculated by ref_feature and other source_features warped by differentiable_warping
    3. The learned pixel-wise view weight is estimated in the first iteration of Patchmatch and kept fixed in the
    matching cost computation.
    """

    def __init__(self, G: int) -> None:
        """Initialize method

        Args:
            G: the feature channels of input will be divided evenly into G groups
        """
        super(PixelwiseNet, self).__init__()
        self.conv0 = ConvBnReLU3D(in_channels=G, out_channels=16, kernel_size=1, stride=1, pad=0)
        self.conv1 = ConvBnReLU3D(in_channels=16, out_channels=8, kernel_size=1, stride=1, pad=0)
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.output = nn.Sigmoid()

    def forward(self, x1: torch.Tensor) -> torch.Tensor:
        """Forward method for PixelwiseNet

        Args:
            x1: pixel-wise view weight, [B, G, Ndepth, H, W], where G is the number of groups
        """
        # [B,1,H,W]
        return torch.max(self.output(self.conv2(self.conv1(self.conv0(x1))).squeeze(1)), dim=1)[0].unsqueeze(1)


class RefinementNet(nn.Module):
    """Depth map refinement network from patchmatchnet modified to work on full res depth """

    def __init__(self):
        """Initialize"""

        super(RefinementNet, self).__init__()

        # img: [B,3,H,W]
        self.conv0 = ConvBnReLU(in_channels=3, out_channels=8)
        # depth map:[B,1,H,W]
        self.conv1 = ConvBnReLU(in_channels=1, out_channels=8)
        self.conv2 = ConvBnReLU(in_channels=8, out_channels=8)

        self.bn = nn.BatchNorm2d(8)
        self.conv3 = ConvBnReLU(in_channels=16, out_channels=8)
        self.res = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, padding=1, bias=False)

    def forward(
        self, img: torch.Tensor, depth_0: torch.Tensor, depth_min: torch.Tensor, depth_max: torch.Tensor
    ) -> torch.Tensor:
        """Forward method
        Args:
            img: input reference images (B, 3, H, W)
            depth_0: current depth map (B, 1, H, W)
            depth_min: pre-defined minimum depth (B, )
            depth_max: pre-defined maximum depth (B, )
        Returns:
            depth: refined depth map (B, 1, H, W)
        """

        batch_size = depth_min.size()[0]
        # pre-scale the depth map into [0,1]
        depth = (depth_0 - depth_min.view(batch_size, 1, 1, 1)) / (depth_max - depth_min).view(batch_size, 1, 1, 1)

        conv_img = self.conv0(img)
        conv_depth = F.relu(self.bn(self.conv2(self.conv1(depth))), inplace=True)
        # depth residual
        res = self.res(self.conv3(torch.cat((conv_depth, conv_img), dim=1)))
        del conv_img
        del conv_depth

        depth = depth + res
        # convert the normalized depth back
        return depth * (depth_max - depth_min).view(batch_size, 1, 1, 1) + depth_min.view(batch_size, 1, 1, 1)


class MVSLight(nn.Module):
    """ Implementation of MVSLight"""

    def __init__(self, return_intermidiate=False):
        """Initialize MVSLight
        """
        super(MVSLight, self).__init__()

        self.depth_net = UNet(num_classes=1)
        # number of groups for group-wise correlation
        self.G = 4
        self.pixel_wise_net = PixelwiseNet(self.G)
        self.refine_net = RefinementNet()
        self.levels = [1,2]
        self.return_intermidiate = return_intermidiate

    def forward(self, imgs, proj_matrices, depth_min, depth_max):
        """Forward method for MVSLight

        Args:
            images: different stages of images (B, 3, H, W) stored in the dictionary
            proj_matrices: different stages of camera projection matrices (B, 4, 4) stored in the dictionary
            depth_min: minimum virtual depth (B, )
            depth_max: maximum virtual depth (B, )

        Returns:
            output tuple of initial_depth, merged_depth and refined_depth
        """
        imgs = torch.unbind(imgs, 1)

        # Step 1: regress initial depth and extract features
        ref_img = imgs[0]
        init_depth, ref_features = self.depth_net(ref_img)
        depth_min = depth_min.to(ref_img.dtype)
        depth_max = depth_max.to(ref_img.dtype)


        # Step 2: extract feature maps in: images; out: ??-channel feature maps
        src_features = [self.depth_net(img, forward_decoder=False)[1] for img in imgs[1:]]
        del imgs

        # Step 3: warp features, calculate view weight and apply a weighted sum
        mvs_feat = {}
        for level in self.levels:
            # list of len = n_views-1, each element is of shape B x C X H_l X W_l
            level_src_features = [feat_dict.get(level) for feat_dict in src_features]
            # The features of the reference image at this level, shape:  B x C X H_l X W_l
            level_ref_feature = ref_features.get(level)
            batch, feature_channel, height, width = level_ref_feature.shape
            level_ref_feature = level_ref_feature.view(batch, self.G, feature_channel // self.G, height, width)
            # Projection matrices for the given level: B x n_views x 2 x 4 x 4
            level_proj_matrices = proj_matrices.get(level)
            level_proj_matrices = torch.unbind(level_proj_matrices, 1)
            level_ref_proj, level_src_proj = level_proj_matrices[0], level_proj_matrices[1:]

            for src_feature, src_proj in zip(level_src_features, level_src_proj):

                warped_feature = differentiable_warping(
					src_feature, src_proj[:,0], level_ref_proj[:,0], F.interpolate(init_depth,
                                                [height, width], mode='bilinear',
                                                align_corners=False)
				).view(batch, self.G, feature_channel // self.G, height, width)

                # group-wise correlation
                similarity = (warped_feature * level_ref_feature)
                # pixel-wise view weight
                view_weight = self.pixel_wise_net(similarity)
                if mvs_feat.get(level) is None:
                    mvs_feat[level] = view_weight*src_feature
                else:
                    mvs_feat[level] += view_weight*src_feature

        # Step 4 re-compute depth with aggregared mvs features
        mvs_depth, _ = self.depth_net(ref_img, mvs_features=mvs_feat)
        # Step 5: Refine depth
        refined_depth = self.refine_net(ref_img, mvs_depth, depth_min, depth_max).squeeze(1)

        outputs = {"refined_depth":refined_depth}
        if self.return_intermidiate:
            outputs = {"init_depth":init_depth.squeeze(1), "mvs_depth":mvs_depth.squeeze(1), "refined_depth":refined_depth}

        return outputs


class MVSLightLoss(nn.Module):
    def __init__(self, w_init=1.0, w_mvs=1.0, w_refined=1.0):
        super(MVSLightLoss, self).__init__()
        self.w_init = w_init
        self.w_mvs = w_mvs
        self.w_refined = w_refined
        self.l1_smooth_loss = nn.SmoothL1Loss(reduction='mean')

    def forward(self, inputs, depth_gt, mask):
        mask = mask.to(dtype=torch.bool)
        depth_loss_init = self.l1_smooth_loss(inputs["init_depth"][mask], depth_gt[mask])
        depth_loss_mvs = self.l1_smooth_loss(inputs["mvs_depth"][mask], depth_gt[mask])
        depth_loss_refined = self.l1_smooth_loss(inputs["refined_depth"][mask], depth_gt[mask])
        total_depth_loss = self.w_init*depth_loss_init + self.w_mvs*depth_loss_mvs + self.w_refined*depth_loss_refined
        return depth_loss_init, depth_loss_mvs, depth_loss_refined, total_depth_loss