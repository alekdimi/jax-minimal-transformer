from typing import Callable

import jax.numpy as jnp


class LayerNorm:
    def __init__(self, features: int):
        self.features = features

        self.b = jnp.zeros(features)
        self.g = jnp.ones(features)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        mu = jnp.mean(x, axis=-1, keepdims=True)
        sigma = jnp.std(x, axis=-1, keepdims=True)
        return self.b + self.g * (x - mu) / sigma


class Dense:
    def __init__(self, in_features: int, out_features: int, bias: bool):
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias

        self.w = jnp.zeros((in_features, out_features))
        if self.has_bias:
            self.b = jnp.zeros(out_features)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        result = jnp.dot(x, self.w)
        if self.has_bias:
            result += self.b
        return result


class TokenEmbedder(Dense):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__(in_features=vocab_size, out_features=d_model, bias=False)

    def __call__(self, x: jnp.ndarray, reverse: bool) -> jnp.ndarray:
        if reverse:
            return jnp.squeeze(self.w @ jnp.expand_dims(x, axis=-1), axis=-1)
        else:
            return self.w[x]


class FeedForwardNetwork:
    def __init__(
        self, in_out_features: int, hidden_features: int, activation: Callable
    ):
        self.c_fc = Dense(in_out_features, hidden_features, bias=True)
        self.activation = activation
        self.c_proj = Dense(hidden_features, in_out_features, bias=True)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        return x
