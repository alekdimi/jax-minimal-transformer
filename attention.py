import jax
import jax.numpy as jnp

from . import layers


class CausalAttention:
    def __init__(self, num_heads: int, d_model: int, debug: bool):
        self.num_heads = num_heads
        self.d_model = d_model
        assert self.d_model % self.num_heads == 0
        self.d_head = self.d_model // self.num_heads
        self.norm_factor: int = 1 / (self.d_head**0.5)

        self.c_attn = layers.Dense(
            in_features=d_model, out_features=3 * d_model, bias=True
        )
        self.c_proj = layers.Dense(in_features=d_model, out_features=d_model, bias=True)

        self.kv_cache = None
        self.debug = debug

    def _reshape_heads(self, x: jnp.ndarray, transpose: bool):
        B, T, _ = x.shape
        x = x.reshape(B, T, self.num_heads, self.d_head)
        x = x.transpose(0, 2, 1, 3)
        if transpose:
            x = x.transpose(0, 1, 3, 2)
        return x

    def _append_to_kv_cache(
        self, KT: jnp.ndarray, V: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        if self.kv_cache is None:
            # Prefill phase
            self.kv_cache = (KT, V)
        else:
            # Decode phase
            KT = jnp.concatenate([self.kv_cache[0], KT], axis=-1)
            V = jnp.concatenate([self.kv_cache[1], V], axis=-2)
            self.kv_cache = (KT, V)
        return self.kv_cache

    def __call__(self, x: jnp.ndarray, kv_cache: bool) -> jnp.ndarray:
        QKV = self.c_attn(x)
        Q, K, V = jnp.split(QKV, 3, axis=-1)
        Q = self._reshape_heads(Q, transpose=False)
        KT = self._reshape_heads(K, transpose=True)
        V = self._reshape_heads(V, transpose=False)

        prefill = self.kv_cache is None
        if kv_cache:
            KT, V = self._append_to_kv_cache(KT, V)

        QKT = (self.norm_factor * Q) @ KT

        if prefill:
            mask = jnp.tril(jnp.ones((QKT.shape[-2], QKT.shape[-1])))
        else:
            mask = jnp.ones((QKT.shape[-2], QKT.shape[-1]))

        attention = jnp.where(mask == 0, -1e10, QKT)

        out = jax.nn.softmax(attention, axis=-1) @ V
        B, H, T, D = out.shape
        out = out.transpose(0, 2, 1, 3).reshape(B, T, H * D)
        return self.c_proj(out)
