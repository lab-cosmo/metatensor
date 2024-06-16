import importlib.metadata

__version__ = importlib.metadata.version("metatensor-operations")


from ._utils import NotEqualError  # noqa
from ._checks import checks_enabled, unsafe_disable_checks, unsafe_enable_checks  # noqa

from .abs import abs  # noqa
from .add import add  # noqa
from .allclose import (  # noqa
    allclose,
    allclose_block,
    allclose_block_raise,
    allclose_raise,
)
from .block_from_array import block_from_array  # noqa
from .detach import detach, detach_block  # noqa
from .divide import divide  # noqa
from .dot import dot  # noqa
from .drop_blocks import drop_blocks  # noqa
from .empty_like import empty_like, empty_like_block  # noqa
from .equal import equal, equal_block, equal_block_raise, equal_raise  # noqa
from .equal_metadata import (  # noqa
    equal_metadata,
    equal_metadata_block,
    equal_metadata_raise,
    equal_metadata_block_raise,
)
from .is_contiguous import (  # noqa: F401
    is_contiguous,
    is_contiguous_block,
)
from .join import join  # noqa
from .lstsq import lstsq  # noqa
from .make_contiguous import (  # noqa: F401
    make_contiguous,
    make_contiguous_block,
)
from .manipulate_dimension import (  # noqa
    append_dimension,
    insert_dimension,
    permute_dimensions,
    remove_dimension,
    rename_dimension,
)
from .multiply import multiply  # noqa
from .one_hot import one_hot  # noqa
from .ones_like import ones_like, ones_like_block  # noqa
from .random_like import random_uniform_like, random_uniform_like_block  # noqa
from .pow import pow  # noqa
from .reduce_over_samples import (  # noqa
    mean_over_samples,
    mean_over_samples_block,
    std_over_samples,
    std_over_samples_block,
    sum_over_samples,
    sum_over_samples_block,
    var_over_samples,
    var_over_samples_block,
)
from .remove_gradients import remove_gradients, remove_gradients_block  # noqa
from .requires_grad import requires_grad, requires_grad_block  # noqa
from .slice import slice, slice_block  # noqa
from .solve import solve  # noqa
from .sort import sort, sort_block  # noqa
from .split import split, split_block  # noqa
from .subtract import subtract  # noqa
from .unique_metadata import unique_metadata, unique_metadata_block  # noqa
from .zeros_like import zeros_like, zeros_like_block  # noqa
