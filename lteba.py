"""
Enhanced version supporting more classes and excellent performance on CT data.
"""

import torch
from torch import nn
from torch.nn import functional as F
from typing import Dict, Tuple, Type
import torchvision.models as models


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 7,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 7,
        iou_head_hidden_dim: int = 256,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

        self.seg_head = nn.Conv2d(transformer_dim // 8, num_classes, kernel_size=1)
        self.edge_head = nn.Conv2d(transformer_dim // 8, num_classes, kernel_size=1)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        if len(image_embeddings.shape) == 3:
            image_embeddings = image_embeddings.unsqueeze(0)
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            src = image_embeddings
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]

        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)

        masks = self.seg_head(upscaled_embedding)
        edges = self.edge_head(upscaled_embedding)
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred, upscaled_embedding, edges


class LTEBADetailAdapter(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        resnet = models.resnet34(pretrained=True)
        self.lte = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.layer1
        )

        self.conv_trans = nn.Sequential(
            nn.ConvTranspose2d(32 + 64, 32, kernel_size=3, padding=1,
                               output_padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.seg_head = nn.Conv2d(32, num_classes, kernel_size=3, padding=1)
        self.edge_head = nn.Conv2d(32, num_classes, kernel_size=3, padding=1)

    def forward(
        self,
        F_ue: torch.Tensor,
        current_slice: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        F_lt = self.lte(current_slice.repeat(1, 3, 1, 1))
        F_fr = self.conv_trans(torch.cat([F_ue, F_lt], dim=1))
        s1 = self.seg_head(F_fr)
        e1 = self.edge_head(F_fr)
        return s1, e1


class CDSAM(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder,
        prompt_encoder,
        mask_decoder,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.lteba = LTEBADetailAdapter(num_classes=num_classes)

    def forward(
        self,
        imgs: torch.Tensor,
        pt: Tuple[torch.Tensor, torch.Tensor],
        bbox: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        F_enc = self.image_encoder(imgs)

        se, de = self.prompt_encoder(points=pt, boxes=None, masks=None)

        s0, _, F_ue, e0 = self.mask_decoder(
            image_embeddings=F_enc,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=se,
            dense_prompt_embeddings=de,
            multimask_output=True,
        )

        current_slice = imgs[:, 1:2, :, :]
        s1, e1 = self.lteba(F_ue, current_slice)

        s0_up = F.interpolate(s0, (256, 256), mode="bilinear", align_corners=False)
        masks = torch.mul(s1, s0_up)

        return {
            "masks": masks,
            "s0": s0,
            "e0": e0,
            "e1": e1,
        }
