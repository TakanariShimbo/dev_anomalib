"""Feature Extractor.

This script extracts features from a CNN network
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import warnings

import timm
import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)


class TimmFeatureExtractor(nn.Module):
    """Extract features from a CNN.

    Args:
        backbone (nn.Module): The backbone to which the feature extraction hooks are attached.
        layers (Iterable[str]): List of layer names of the backbone to which the hooks are attached.
        pre_trained (bool): Whether to use a pre-trained backbone. Defaults to True.
        requires_grad (bool): Whether to require gradients for the backbone. Defaults to False.
            Models like ``stfpm`` use the feature extractor model as a trainable network. In such cases gradient
            computation is required.

    Example:
        >>> import torch
        >>> from anomalib.models.components.feature_extractors import TimmFeatureExtractor

        >>> model = TimmFeatureExtractor(model="resnet18", layers=['layer1', 'layer2', 'layer3'])
        >>> input = torch.rand((32, 3, 256, 256))
        >>> features = model(input)

        >>> [layer for layer in features.keys()]
            ['layer1', 'layer2', 'layer3']
        >>> [feature.shape for feature in features.values()]
            [torch.Size([32, 64, 64, 64]), torch.Size([32, 128, 32, 32]), torch.Size([32, 256, 16, 16])]
    """

    def __init__(self, backbone: str, layers: list[str], pre_trained: bool = True, requires_grad: bool = False):
        super().__init__()
        self.backbone = backbone
        self.layers = layers
        self.idx = self._map_layer_to_idx()
        self.requires_grad = requires_grad
        self.feature_extractor = timm.create_model(
            backbone,
            pretrained=pre_trained,
            features_only=True,
            exportable=True,
            out_indices=self.idx,
        )
        self.out_dims = self.feature_extractor.feature_info.channels()
        self._features = {layer: torch.empty(0) for layer in self.layers}

    def _map_layer_to_idx(self, offset: int = 3) -> list[int]:
        """Maps set of layer names to indices of model.

        Args:
            offset (int) `timm` ignores the first few layers when indexing please update offset based on need

        Returns:
            Feature map extracted from the CNN
        """
        idx = []
        features = timm.create_model(
            self.backbone,
            pretrained=False,
            features_only=False,
            exportable=True,
        )
        for i in self.layers:
            try:
                idx.append(list(dict(features.named_children()).keys()).index(i) - offset)
            except ValueError:
                warnings.warn(f"Layer {i} not found in model {self.backbone}")
                # Remove unfound key from layer dict
                self.layers.remove(i)

        return idx

    def forward(self, inputs: Tensor) -> dict[str, Tensor]:
        """Forward-pass input tensor into the CNN.

        Args:
            inputs (Tensor): Input tensor

        Returns:
            Feature map extracted from the CNN
        """
        if self.requires_grad:
            features = dict(zip(self.layers, self.feature_extractor(inputs)))
        else:
            self.feature_extractor.eval()
            with torch.no_grad():
                features = dict(zip(self.layers, self.feature_extractor(inputs)))
        return features


class OutputsHolder:
    def __init__(self):
        self._outputs = []

    def append(self, output):
        self._outputs.append(output)

    @property
    def outputs(self):
        return self._outputs


class FeatureExtractor(nn.Module):
    """Extract features from a CNN.

    Args:
        backbone (nn.Module): The backbone to which the feature extraction hooks are attached.
        layers (Iterable[str]): List of layer names of the backbone to which the hooks are attached.
        pre_trained (bool): Whether to use a pre-trained backbone. Defaults to True.
        requires_grad (bool): Whether to require gradients for the backbone. Defaults to False.
            Models like ``stfpm`` use the feature extractor model as a trainable network. In such cases gradient
            computation is required.

    Example:
        >>> import torch
        >>> from anomalib.models.components.feature_extractors import TimmFeatureExtractor

        >>> model = TimmFeatureExtractor(model="resnet18", layers=['layer1', 'layer2', 'layer3'])
        >>> input = torch.rand((32, 3, 256, 256))
        >>> features = model(input)

        >>> [layer for layer in features.keys()]
            ['layer1', 'layer2', 'layer3']
        >>> [feature.shape for feature in features.values()]
            [torch.Size([32, 64, 64, 64]), torch.Size([32, 128, 32, 32]), torch.Size([32, 256, 16, 16])]
    """

    def __init__(self, backbone: str, layers: list[str], pre_trained: bool = True, requires_grad: bool = False):
        super().__init__()
        self.backbone = backbone
        self.layers = layers
        self.requires_grad = requires_grad
        self.feature_extractor = timm.create_model(
            backbone,
            pretrained=pre_trained,
            features_only=True,
            exportable=True,
        )
        self._outputs_holder = OutputsHolder()
        self.out_dims = []
        self.out_sizes = []  
        self._register_layer_to_output()
        
    def _hook(self, module, input, output):
        self._outputs_holder.append(output)
        
    def _recursive_getattr(self, obj, layers):
        try:
            int(layers[0])
            layer_type = "index"
        except:
            layer_type = "name"

        try:
            if layer_type == "name":
                next_obj = getattr(obj, layers[0], None)
            else:
                next_obj = obj[int(layers[0])]
        except:
            next_obj = None
    
        if len(layers) == 1:
            return next_obj
        else:
            return self._recursive_getattr(next_obj, layers[1:])

    def _register_layer_to_output(self) -> None:
        for layer in self.layers:
            layers_list = layer.split('.')
            target_layer = self._recursive_getattr(self.feature_extractor, layers_list)
            
            if target_layer is None:
                warnings.warn(f"Layer {layer} not found in model {self.backbone}")
                self.layers.remove(layer)
            else:
                target_layer.register_forward_hook(self._hook)

    def forward(self, inputs: Tensor) -> dict[str, Tensor]:
        """Forward-pass input tensor into the CNN.

        Args:
            inputs (Tensor): Input tensor

        Returns:
            Feature map extracted from the CNN
        """
        if self.requires_grad:
            self._outputs_holder = OutputsHolder()
            self.feature_extractor(inputs)
            features = dict(zip(self.layers, self._outputs_holder.outputs))
            self.out_dims = [output.shape[1] for output in self._outputs_holder.outputs]
            self.out_sizes = [output.shape[2] for output in self._outputs_holder.outputs]
        else:
            self.feature_extractor.eval()
            with torch.no_grad():
                self._outputs_holder = OutputsHolder()
                self.feature_extractor(inputs)
                features = dict(zip(self.layers, self._outputs_holder.outputs))
                self.out_dims = [output.shape[1] for output in self._outputs_holder.outputs]
                self.out_sizes = [output.shape[2] for output in self._outputs_holder.outputs]
        return features
