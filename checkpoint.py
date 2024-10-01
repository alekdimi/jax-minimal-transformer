from typing import Any, TypeVar

import jax.numpy as jnp
import tensorflow as tf

T = TypeVar("T")


def _get_nested_attr(object, attr_string) -> Any:
    attributes = attr_string.split(".")
    for attribute in attributes:
        object = getattr(object, attribute)
    return object


def _set_nested_attr(object, attr_string, value) -> None:
    attributes = attr_string.split(".")
    for attribute in attributes[:-1]:
        object = getattr(object, attribute)
    setattr(object, attributes[-1], value)


def load_weights(object: T, checkpoint_path: str, verbose: bool = False) -> T:
    """Load weights from a checkpoint with a dict structure."""
    checkpoint = tf.train.load_checkpoint(checkpoint_path)
    assert checkpoint is not None
    for name, shape in tf.train.list_variables(checkpoint_path):
        try:
            attr_name = name.replace("/", ".")
            attr = _get_nested_attr(object, attr_name)
            weights = jnp.array(checkpoint.get_tensor(name)).squeeze()  # type: ignore
            assert (
                weights.shape == attr.shape
            ), f"Shape mismatch: {weights.shape} != {attr.shape}"
            _set_nested_attr(object, attr_name, weights)
            if verbose:
                print(f"Loaded: {name} -> {attr_name}, Shape: {shape}")
        except AttributeError as attr_error:
            raise AttributeError(
                f"Expected: {name} to be an attribute of {object}"
            ) from attr_error
    return object
