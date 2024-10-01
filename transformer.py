import dataclasses

import jax
import jax.numpy as jnp

from . import model_types
from . import layers
from . import attention


@dataclasses.dataclass
class TransformerConfig(model_types.MakeableConfig["Transformer"]):
    vocab_size: int
    seq_len: int
    d_model: int
    num_heads: int
    num_layers: int
    debug: bool

    def make(self) -> "Transformer":
        return Transformer(self)


class TransformerBlock:
    def __init__(self, num_heads: int, d_model: int, debug: bool):
        self.num_heads = num_heads
        self.d_model = d_model

        self.ln_1 = layers.LayerNorm(features=d_model)
        self.ln_2 = layers.LayerNorm(features=d_model)
        self.attn = attention.CausalAttention(
            num_heads=num_heads, d_model=d_model, debug=debug
        )
        self.mlp = layers.FeedForwardNetwork(
            in_out_features=d_model, hidden_features=d_model * 4, activation=jax.nn.gelu
        )

    def __call__(self, x: jnp.ndarray, kv_cache: bool) -> jnp.ndarray:
        x = x + self.attn(self.ln_1(x), kv_cache=kv_cache)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer:
    def __init__(self, config: "TransformerConfig"):
        self.vocab_size = config.vocab_size
        self.seq_len = config.seq_len
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.num_layers = config.num_layers
        self.debug = config.debug

        self.positional_embeddings = jnp.zeros((self.seq_len, self.d_model))
        self.token_embedder = layers.TokenEmbedder(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
        )

        self.blocks = []
        for i in range(self.num_layers):
            block = TransformerBlock(
                num_heads=self.num_heads,
                d_model=self.d_model,
                debug=self.debug and not i,
            )
            self.blocks.append(block)
            setattr(self, f"h{i}", block)

        self.ln_f = layers.LayerNorm(features=self.d_model)

        self.position = None

    @property
    def wpe(self):
        return self.positional_embeddings

    @wpe.setter
    def wpe(self, value):
        self.positional_embeddings = value

    @property
    def wte(self):
        return self.token_embedder.w

    @wte.setter
    def wte(self, value):
        self.token_embedder.w = value

    def _update_start_end_position(self, kv_cache: bool, num_tokens: int) -> tuple[int, int]:
        if self.position is None:
            # Prefill phase, which is just 0 to num_tokens.
            if kv_cache:
                self.position = num_tokens
            return (0, num_tokens)
        else:
            # Decode phase.
            assert num_tokens == 1
            start = self.position
            self.position += num_tokens
            return (start, self.position)

    def __call__(self, x: jnp.ndarray, kv_cache: bool) -> jnp.ndarray:
        """Transformer forward pass, visualized at:
        https://lilianweng.github.io/posts/2019-01-31-lm/#transformer-decoder-as-language-model
        """
        _, T = x.shape
        start, end = self._update_start_end_position(kv_cache, num_tokens=T)

        token_embeddings = self.token_embedder(x, reverse=False)
        position_embeddings = jnp.expand_dims(
            self.positional_embeddings[start:end], axis=0
        )
        assert position_embeddings.shape == token_embeddings.shape, (
            position_embeddings.shape,
            token_embeddings.shape,
        )
        x = token_embeddings + position_embeddings

        for block in self.blocks:
            x = block(x, kv_cache=kv_cache)
        x = self.ln_f(x)
        logits = self.token_embedder(x, reverse=True)
        return logits
