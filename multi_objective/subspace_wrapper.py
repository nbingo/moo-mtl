import torch
import torch.nn as nn
from typing import Type, Optional
from collections import OrderedDict, namedtuple
import numpy as np
import warnings


def to_subspace_class(model_class: 'Type[nn.Module]', num_vertices: Optional[int] = 2,
                      verbose: Optional[bool] = False) -> 'Type[nn.Module]':
    class SubspaceModelClass(model_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Saving the original keys to make exporting these same keys easy when we want the values of the parameters
            # of the underlying model at a specific alpha
            self.orig_state_dict_keys = self.state_dict().keys()
            self.verbose = verbose
            # These will be the vertices of our n-simplex
            self.register_buffer('num_base_parameters', torch.Tensor([len(list(self.parameters()))]))
            self.register_buffer('num_vertices', torch.Tensor([num_vertices]))
            # Have to store names of the original parameters that we will make copies of for proper state_dict loading
            self.orig_parameter_names = [name for name, _ in self.named_parameters()]
            pre_copy_state_dict = self.state_dict().keys()
            # Have to register all of these as parameters (not registers) so that they're copied to the correct device
            # and so that they get backpropagated automatically
            self.parametrization_points = nn.ParameterList(
                [nn.Parameter(p.clone()) for p in self.parameters() for _ in range(num_vertices)])
            # We need the key names of these added parameters to properly load state_dict later on
            self.parametrization_points_keys = list(self.state_dict().keys())[len(pre_copy_state_dict):]
            # Create a map from the names of our copied parameters to the parameters that they came from for easy
            # state_dict loading!
            self.param_point_keys_to_orig_state_keys = {
                param_point_key: self.orig_parameter_names[int(torch.floor(i / self.num_vertices))]
                for i, param_point_key in enumerate(self.parametrization_points_keys)
            }
            assert len(self.parametrization_points_keys) / len(
                self.orig_parameter_names) == num_vertices, f'The number of original keys is {len(self.orig_parameter_names)}, but the number of copied keys is {len(self.parametrization_points_keys)}, which is not {num_vertices} times the number of original keys. '

            # Just start w/ equal weights for everything (i.e., at center of simplex)
            self.register_buffer('alpha', torch.full((num_vertices,), np.sqrt(num_vertices) / num_vertices))
            # Boolean to keep track if we changed alpha and not the parameters of the underlying model yet. Every
            # time alpha is changed we have to update the model parameters to represent the parameters parametrized
            # by that alpha. We don't need to reset parameters if alpha does not change, however, since the changed
            # underlying parameters will still track the alpha that they had before.
            self.alpha_updated = True

        def set_alpha(self, alpha: torch.Tensor) -> None:
            if alpha.shape[0] != int(self.num_vertices):
                raise ValueError(f'Alpha must have size of self.num_vertices={self.num_vertices}')
            self.alpha.copy_(alpha)
            self.alpha_updated = True

        def load_state_dict(self, state_dict: 'OrderedDict[str, torch.Tensor]', strict: bool = False) \
                -> 'tuple[set, set]':
            incompatible_keys = super().load_state_dict(state_dict=state_dict, strict=strict)
            missing_keys = set(incompatible_keys.missing_keys)
            unexpected_keys = set(incompatible_keys.unexpected_keys)
            if len(unexpected_keys) > 0:
                warnings.warn(f'Unexpected keys found while loading: {unexpected_keys}', RuntimeWarning)
            if len(missing_keys) > 0:
                if verbose:
                    print(
                        f'Found {len(missing_keys)} missing keys, '
                        f'and assuming that they are copies for the parametrization so will fill up accordingly.')
                # Go through the parameters and copy values into them according to the original parameter that they
                # came from if they're a missing key
                for name, param in self.named_parameters():
                    if name in missing_keys:
                        missing_keys -= {name}
                        with torch.no_grad():
                            param.copy_(state_dict[self.param_point_keys_to_orig_state_keys[name]])
            return missing_keys, unexpected_keys

        def _set_params_at_alpha(self) -> None:
            if self.verbose:
                print('Setting parameters from subspace parametrization...')
            for i, name in enumerate(self.orig_parameter_names):
                if i >= int(self.num_base_parameters):
                    break
                to_stack = [
                    self.parametrization_points[
                        np.ravel_multi_index([i, j], (int(self.num_base_parameters), int(self.num_vertices)))] *
                    self.alpha[j]
                    for j in range(int(self.num_vertices))
                ]
                new_p = torch.mean(torch.stack(to_stack, dim=0), axis=0)
                # To set the value of the parameters of the underlying model, we first have to remove them and then
                # reset them since PyTorch thinks they're a leaf parameter, but in reality in this context they
                # aren't since we're computing them as a result of our parametrization. So resetting them allows us
                # to have them as non-leaf parameters.
                del_attr(self, name.split("."))
                set_attr(self, name.split("."), new_p)
            if self.verbose:
                print('Done setting parameters!')
            self.alpha_updated = False

        def state_dict_at_alpha(self, alpha: torch.Tensor) -> 'dict[str, torch.Tensor]':
            # Sets the state dict of the underlying model for the given alpha and then returns just the state dict of
            # the underlying model.
            orig_alpha = self.alpha.clone()
            self.set_alpha(alpha)
            self._set_params_at_alpha()
            # Get the state_dict of the underlying model
            # THIS RETURNS ALL PARAMETERS! BAD
            state_dict = self.state_dict()
            state_dict = OrderedDict([(name, state_dict[name]) for name in self.orig_state_dict_keys])
            # reset our params
            self.set_alpha(orig_alpha)
            self._set_params_at_alpha()

            return state_dict

        def forward(self, *args, alpha: torch.Tensor = None, **kwargs):
            if (self.alpha is None) and (alpha is None):
                raise RuntimeError('Alpha must be defined before a forward passs. '
                                   'Call model.set_alpha(<alpha>) or set alpha in the forward pass.')
            if self.alpha_updated and alpha is not None:
                warnings.warn('Passing in new alpha in forward pass along with setting new alpha, so it is unclear '
                              'which one to use. Please only use model.set_alpha(<alpha>) or set the alpha in the '
                              'forward pass via a keyword argument. Currently going with the alpha set in the '
                              'forward pass.', RuntimeWarning)
            if alpha is not None:
                self.set_alpha(alpha)
            # set the parameters according to alpha
            if self.alpha_updated:
                self._set_params_at_alpha()
            return super().forward(*args, **kwargs)

    return SubspaceModelClass


def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])


def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)