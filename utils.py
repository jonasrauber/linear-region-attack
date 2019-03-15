import jax.numpy as np
import jax
from jax.interpreters import xla


def is_device_array(x):
    return isinstance(x, xla.DeviceArray)


def scatter(indices, values, length):
    assert indices.ndim == 1
    assert values.shape[1:] == indices.shape
    batch_size = values.shape[0]

    def f(a):
        return a[:, indices]

    base = np.zeros((batch_size, length), np.float32)
    _, grad = jax.vjp(f, base)
    (out,) = grad(values)
    return base + out


# def scatter(indices, values, length):
#     dnums = jax.lax.ScatterDimensionNumbers(
#         update_window_dims=(),
#         inserted_window_dims=(0,),
#         scatter_dims_to_operand_dims=(0,),
#         index_vector_dim=1)
#     indices = np.atleast_1d(np.asarray([indices]).squeeze())
#     values = np.atleast_1d(np.asarray([values]).squeeze())
#     assert indices.ndim == values.ndim == 1
#     assert indices.shape == values.shape
#     return jax.lax.scatter_add(np.zeros(length, values.dtype), indices, values, dnums)


def onehot(index, length, dtype=np.int32):
    assert isinstance(index, int)
    onehot = np.arange(length) == index
    onehot = onehot.astype(dtype)
    return onehot
