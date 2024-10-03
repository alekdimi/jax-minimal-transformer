from calendar import c
import enum
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


class SamplingMethod(enum.StrEnum):
    GREEDY = "greedy"
    TOP_K = "top_k"
    NUCLEUS = "nucleus"


def _greedy(logits: Float[Array, "batch* vocab_size"]) -> Float[Array, "batch 1"]:
    return jnp.argmax(logits, axis=-1, keepdims=True)


def _top_k(
    logits: Float[Array, "batch* vocab_size"],
    k: int = 10,
    num_samples: int = 1,
    seed: int = 0,
) -> Float[Array, "batch num_samples"]:
    logits = jnp.moveaxis(logits, -1, 0)
    num_logits, batch_size = logits.shape[0], logits.shape[1:]
    assert (
        0 < k <= num_logits
    ), "k must be greater than 0 and less than or equal to the number of logits."
    top_k_indices = jnp.argsort(logits, axis=0, descending=True)[k : k + 1]
    top_k_logit = jnp.take_along_axis(logits, top_k_indices, axis=0)
    top_k_logits = jnp.where(logits >= top_k_logit, logits, 0)
    top_k_logits = jnp.moveaxis(top_k_logits, 0, axis)
    samples = jax.random.categorical(
        key=jax.random.PRNGKey(seed),
        logits=top_k_logits,
        axis=axis,
        shape=(num_samples, *batch_size),
    )
    return jnp.moveaxis(samples, 0, axis)


def _nucleus(
    logits: Float[Array, "batch* vocab_size"], p: float = 0.9
) -> Float[Array, "batch 1"]:
    """Nucleus (top p) sampling: https://arxiv.org/abs/1904.09751

    Args:
        logits: The logits to sample from across the last axis.
        p: The probability of sampling from the top (p * 100)% of the distribution.

    Returns:
        The sampled indices.
    """
    assert 0 < p < 1, "p must be between 0 and 1"
    sorted_logits = jnp.sort(logits, axis=-1, descending=True)
    sorted_probs = jax.nn.softmax(sorted_logits, axis=-1)
    cumulative_probs = jnp.cumsum(sorted_probs, axis=-1)
    del cumulative_probs
    return logits


def sample(
    logits: Float[Array, "batch* vocab_size"], method: str | SamplingMethod, **kwargs
) -> Float[Array, "batch 1"]:
    method = SamplingMethod(method) if isinstance(method, str) else method
    match method:
        case SamplingMethod.GREEDY:
            return _greedy(logits)
        case SamplingMethod.TOP_K:
            return _top_k(logits, **kwargs)
        case SamplingMethod.NUCLEUS:
            return _nucleus(logits, **kwargs)
        case _:
            raise ValueError(f"Unknown sampling method: {method}.")
