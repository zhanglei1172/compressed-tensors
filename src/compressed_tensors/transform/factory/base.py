# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from typing import List, Optional

import torch
import torch.nn.utils.parametrize as P
import tqdm
from compressed_tensors.registry.registry import RegistryMixin, T
from compressed_tensors.transform import (
    TransformArgs,
    TransformLocation,
    TransformScheme,
)
from compressed_tensors.utils import (
    align_module_device,
    delete_offload_module,
    has_offloaded_params,
    match_modules_set,
    match_named_modules,
    patch_attr,
    register_offload_module,
    update_offload_parameter,
)
from compressed_tensors.utils.internal import InternalModule
from compressed_tensors.utils.offload import delete_from_weights_map
from torch import Tensor
from torch.nn import Module, Parameter

__all__ = ["TransformFactory", "TransformBase"]


class TransformFactory(RegistryMixin, ABC):
    """
    Abstract factory base used to create and apply transforms to a model

    :param name: name associated with transform scheme
    :param scheme: transform scheme which defines how transforms should be created
    :param seed: random seed used to transform weight randomization
    """

    transforms: List["TransformBase"]

    def __init__(self, name: str, scheme: TransformScheme, seed: Optional[int] = None):
        self.name = name
        self.scheme = scheme
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    @classmethod
    def from_scheme(cls: type[T], scheme: TransformScheme, **kwargs) -> T:
        """
        Create a transform factory from a scheme

        :param scheme: defines how transforms should be created
        :param kwargs: TransformFactory constructor arguments
        :return: subclass of `TransformFactory` corresponding to the scheme type
        """
        constructor = cls.get_value_from_registry(name=scheme.type)
        return constructor(scheme=scheme, **kwargs)

    @abstractmethod
    def create_transform(self, module: Module, args: TransformArgs) -> "TransformBase":
        """
        Abstract method which defines how a transform should be created. May utilize
        caching to maximize shared memory

        :param module: parent module that transform will be applied to
        :param args: defines how the transform will be applied to the module
        :return: instance of TransformBase
        """
        raise NotImplementedError()

    def apply_to_model(self, model: Module, use_tqdm=True):
        """
        Create transforms and apply them to the model

        :param model: module to apply transforms to
        """
        desc = f"Applying {self.name} transforms"
        if self.scheme.block_wise:
            targets, args = zip(
                *[(target, arg) for arg in self.scheme.apply for target in arg.targets]
            )
            for modules in tqdm.tqdm(
                match_modules_set(model, targets), desc=desc, disable=(not use_tqdm)
            ):
                for module, arg in zip(modules, args):
                    sequential_onload = not has_offloaded_params(module) and self.scheme.sequential_onload
                    ori_device = next(module.parameters()).device
                    if sequential_onload:
                        module.to("cuda", non_blocking=True)
                    self._apply_to_module(module, arg)
                    if sequential_onload:
                        module.to(ori_device, non_blocking=True)

                self._clear_weights_cache()

        else:
            modules_args = [
                (module, arg)
                for arg in self.scheme.apply
                for _, module in match_named_modules(model, arg.targets, arg.ignore)
            ]
            for module, arg in tqdm.tqdm(
                modules_args, desc=desc, disable=(not use_tqdm)
            ):
                sequential_onload = not has_offloaded_params(module) and self.scheme.sequential_onload
                ori_device = next(module.parameters()).device
                if sequential_onload:
                    module.to("cuda", non_blocking=True)
                self._apply_to_module(module, arg)
                if sequential_onload:
                    module.to(ori_device, non_blocking=True)

        self._clear_weights_cache()

    def _clear_weights_cache(self):
        """
        Clear any cached weights used to create new transforms
        """
        raise NotImplementedError()

    def _apply_to_module(self, module: Module, args: TransformArgs):
        """
        Create transforms and apply them to the module

        :param module: target module to apply transforms to
        :param args: defines how the transform will be applied to the target module
        """
        if has_offloaded_params(module):
            if module._hf_hook.place_submodules:
                raise NotImplementedError(
                    "Applying transforms to offloaded submodules with "
                    "`place_submodules=True` is not supported"
                )

        # create transform as submodule
        transform_name = f"{self.name}_{args.location}"
        transform = self.create_transform(module, args)
        register_offload_module(module, transform_name, transform)

        # register input transformation hook
        if args.location == TransformLocation.INPUT:

            def input_hook(_, args):
                input = args[0]
                return transform(input)

            module.register_forward_pre_hook(input_hook, prepend=True)

        # eagerly apply transformation to weight
        elif args.location in (
            TransformLocation.WEIGHT_INPUT,
            TransformLocation.WEIGHT_OUTPUT,
        ):
            # fuse transform into weight
            assert hasattr(module, "weight")
            bias = False
            with torch.no_grad(), align_module_device(module):
                update_offload_parameter(module, "weight", transform(module.weight))
                if (
                    args.location == TransformLocation.WEIGHT_OUTPUT
                    and hasattr(module, "bias")
                    and module.bias is not None
                    and not args.inverse
                ):
                    update_offload_parameter(module, "bias", transform(module.bias))
                    bias = True

            if self.scheme.requires_grad:
                # for training, the weight changes with every forward pass
                # so we can leverage parametrization to propagate the gradient
                # if has_offloaded_params(module):
                #     raise ValueError("Offloaded training is not supported")
                with align_module_device(module):
                    clear_offload = True
                    if "weight" not in module._parameters:
                        clear_offload = False
                    P.register_parametrization(module, "weight", transform)
                    if has_offloaded_params(module):
                        weights_map = module._hf_hook.weights_map
                        if clear_offload:
                            delete_from_weights_map(weights_map, "weight")
                            weights_map.dataset.all_keys.remove(
                                f"{weights_map.prefix}weight"
                            )
                            module._hf_hook.original_devices.pop("weight", None)
                    # update_offload_parameter(module, "weight", module.weight)
                    if bias:
                        clear_offload = True
                        if "bias" not in module._parameters:
                            clear_offload = False
                        P.register_parametrization(module, "bias", transform)
                        if has_offloaded_params(module) and clear_offload:
                            delete_from_weights_map(weights_map, "bias")
                            weights_map.dataset.all_keys.remove(
                                f"{weights_map.prefix}bias"
                            )
                            module._hf_hook.original_devices.pop("bias", None)
                        # update_offload_parameter(module, "bias", module.bias)

            else:
                # transform is no longer needed (unfusing is not supported)
                delete_offload_module(module, transform_name)

        # register output transformation hook
        elif args.location == TransformLocation.OUTPUT:

            def output_hook(_, _input, output):
                return transform(output)

            module.register_forward_hook(output_hook)

        # other locations such as q_attn and k_attn have not been implemented
        else:
            raise NotImplementedError()


class TransformBase(InternalModule, ABC):
    """
    Represents the application of a transform accord to TransformArgs
    """

    args: TransformArgs
    weight: Parameter
    _dynamic_tied_weights_keys: List[str] = ["weight"]

    @abstractmethod
    def forward(self, value: Tensor) -> Tensor:
        raise NotImplementedError()

    def right_inverse(self, value: Tensor) -> Tensor:
        with patch_attr(self.args, "inverse", not self.args.inverse):
            return self.forward(value)

    def __repr__(self):
        return f"{self.__class__.__name__}(inverse={self.args.inverse})"
