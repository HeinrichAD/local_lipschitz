from copy import deepcopy
import logging
import torch
import torch.nn as nn
from torchextractor import Extractor
from typing import Any, Dict, List, NamedTuple, Set, Union

from . import utils


SUPPORTED_LAYERS = (
    nn.AdaptiveAvgPool2d,
    nn.Conv2d,
    nn.Dropout,
    nn.Flatten,
    nn.Linear,
    nn.MaxPool2d,
    nn.ReLU,
    nn.Sequential,
)


class LayerInfo(NamedTuple):
    layer: nn.Module
    input: torch.Tensor
    output: torch.Tensor


def relu(x):
    return (x > 0) * x


def get_layer_info(
    net: nn.Module,
    x0: torch.Tensor,
    *,
    extractor_overrides: Dict[str, Any] = dict()
) -> List[LayerInfo]:
    '''
    get all supported layers and their inputs and outputs of the nominal input x0

    net: nn.Module, the network
    x0: torch.Tensor, nominal input
    extractor_overrides: Dict[str, Any], overrides for torchextractor.Extractor
    '''

    unsupported_layers: Set[str] = set()

    def capture_fn(module: nn.Module, input: Any, output: Any, module_name: str, feature_maps: Dict[str, Any]):
        feature_maps[module_name] = LayerInfo(module, input[0], output)

    def module_filter_fn(module: nn.Module, name: str) -> bool:
        if len(list(module.children())) > 0:
            # just skip branch modules and only take leaf modules
            return False
        if isinstance(module, SUPPORTED_LAYERS):
            return True
        unsupported_layers.add(module.__class__.__name__)
        return False

    kwargs = dict(module_filter_fn=module_filter_fn, capture_fn=capture_fn)
    kwargs.update(extractor_overrides)
    model = Extractor(deepcopy(net), **kwargs)
    _, features = model(x0)

    if len(unsupported_layers) > 0:
        logging.getLogger("local_lipschitz.global_bound").warning(
            'THESE TYPES OF MODULE/LAYER HAVE NOT BEEN SUPPORTED YET AND WILL BE SKIPPED: %s',
            ", ".join(sorted(unsupported_layers))
        )

    return [layer_info for _, layer_info in features.items()]


def global_bound(net: nn.Module, x0: torch.Tensor, *, extractor_overrides: Dict[str, Any] = dict()) -> List[float]:
    '''
    calculate the global Lipschitz bound of a feedforward neural network
    '''

    layer_infos = get_layer_info(net, x0, extractor_overrides=extractor_overrides)
    lip = [None] * len(layer_infos)

    for i, layer_info in enumerate(layer_infos):
        layer, x_in, _ = layer_info

        if isinstance(layer, nn.Sequential):
            affine_func = layer[0]  # should be nn.Conv2d or nn.Linear
            spec_norm, _ = utils.get_RAD(affine_func, x_in.shape, d=None, r_squared=None)
            lip[i] = spec_norm.item()

        elif isinstance(layer, (nn.Conv2d, nn.Linear)):
            spec_norm, _ = utils.get_RAD(layer, x_in.shape, d=None, r_squared=None)
            lip[i] = spec_norm.item()

        elif isinstance(layer, nn.MaxPool2d):
            # lipschitz constant
            lip[i] = utils.max_pool_lip(layer)

        elif isinstance(layer, nn.AdaptiveAvgPool2d):
            # this layer does nothing when the input is 3x224x224
            lip[i] = 1

        elif isinstance(layer, nn.Flatten):
            lip[i] = 1

        elif isinstance(layer, nn.Dropout):
            lip[i] = 1

        elif isinstance(layer, nn.ReLU):
            lip[i] = 1

        else:
            logging.getLogger("local_lipschitz.global_bound").error(
                'THIS TYPE OF LAYER HAS NOT BEEN SUPPORTED YET: %s',
                layer.__class__.__name__
            )
            # Note: in this case the return will contain None values

    return lip


def local_bound(
    net: nn.Module,
    x0: torch.Tensor,
    eps: Union[int, float, torch.Tensor],
    batch_size: int = 32,
    *,
    extractor_overrides: Dict[str, Any] = dict()
) -> float:
    '''
    calculate the local Lipschitz bound of a feedforward neural network
    '''

    layer_infos = get_layer_info(net, x0, extractor_overrides=extractor_overrides)
    n_layers = len(layer_infos)

    # a list for each layer where each list item is the diagonal elements of
    # the D matrix in other words
    d = [None]*(n_layers+1)

    # bound of network
    L_net = 1.  # Lipschitz bound of full network  # NOSONAR
    i = 0
    while i < n_layers:
        layer, x_in, x_out = layer_infos[i]

        # affine-ReLU
        if (
            i + 1 < n_layers and
            isinstance(layer, (nn.Conv2d, nn.Linear)) and
            isinstance(layer_infos[i+1].layer, nn.ReLU)
        ):
            # get l vector
            aiTD = utils.get_aiTD(  # NOSONAR
                layer, x_in.shape, x_out.shape,
                d=d[i], batch_size=batch_size)

            # get spectral norm
            with torch.no_grad():
                y0 = layer(x_in).flatten()  # A@x0 + b

            # ybar and R
            y0 = y0.double()
            ybar = eps*aiTD + y0

            # "flat" inds occur when a_i^T D equals zero, and all y_i equal y_{0,i}
            inds_flat = (ybar == y0)  # cspell: ignore inds

            r = (relu(ybar) - relu(y0))/(ybar - y0)
            # this should replace any nans from the previous operation with 0s
            r[inds_flat] = 0
            RAD_norm, V = utils.get_RAD(  # NOSONAR
                layer, x_in.shape, d=d[i], r_squared=r**2)
            L = RAD_norm.item()

            i += 1  # increment i so we skip the ReLU
            d[i+1] = (ybar > 0)

        # sole affine
        elif isinstance(layer, (nn.Conv2d, nn.Linear)):
            A_norm, V = utils.get_RAD(layer, x_in.shape, d=d[i])  # NOSONAR
            L = A_norm.item()
            d[i+1] = None

        # max pooling
        elif isinstance(layer, nn.MaxPool2d):
            di = d[i].view(x_in.shape)
            di = di.to(torch.float)
            d_i1 = layer(di).flatten().to(torch.bool)
            d[i+1] = d_i1
            L = utils.max_pool_lip(layer)

        # adaptive average pooling
        # (doesn't change the input when input is nominal sizes)
        elif isinstance(layer, nn.AdaptiveAvgPool2d):
            # if the adaptive avg pool function doesn't change the input
            if torch.equal(x_in, x_out):
                d[i+1] = d[i]
                L = 1
            else:
                logging.getLogger("local_lipschitz.local_bound").error(
                    'THE ADAPTIVE AVG POOL SECTION IS NOT IMPLEMENTED FOR INPUTS & OUTPUTS OF DIFFERENT SIZES'
                )
                # ignore this layer for now
                d[i+1] = d[i]
                L = 1

        # flatten
        elif isinstance(layer, nn.Flatten):
            d_i = layer(torch.unsqueeze(d[i], dim=0))
            d[i+1] = d_i.flatten()
            L = 1

        # dropout
        elif isinstance(layer, nn.Dropout):
            d[i+1] = d[i]
            L = 1

        # any other type of layer
        else:
            logging.getLogger("local_lipschitz.local_bound").error(
                'NETWORK BOUND IS NOT IMPLEMENTED FOR THIS TYPE OF LAYER: %s',
                layer.__class__.__name__
            )
            # ignore this layer for now
            d[i+1] = d[i]
            L = 1

        # update
        eps *= L
        L_net *= L
        i += 1

    return L_net
