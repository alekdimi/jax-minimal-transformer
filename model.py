from typing import Any

import json
import os

import dataclasses
import jax.numpy as jnp

from . import checkpoint
from . import model_types
from . import sampling
from . import tokenizer
from . import transformer


@dataclasses.dataclass
class GPTConfig(model_types.MakeableConfig["GPT"]):
    tokenizer_config: tokenizer.TokenizerConfig
    transformer_config: transformer.TransformerConfig

    def make(self) -> "GPT":
        return GPT(self)


class GPT:
    def __init__(self, config: GPTConfig):
        self.config = config
        self.tokenizer = config.tokenizer_config.make()
        self.model = config.transformer_config.make()

    @classmethod
    def from_pretrained(
        cls, checkpoint_path: str | None = None, verbose: bool = False
    ) -> "GPT":
        checkpoint_path = checkpoint_path or "/Users/alek/Checkpoints/GPT2/124M"
        with open(os.path.join(checkpoint_path, "hparams.json"), "r") as file:
            hparams = json.load(file)

        gpt2 = GPTConfig(
            tokenizer_config=tokenizer.TokenizerConfig(model="gpt2"),
            transformer_config=transformer.TransformerConfig(
                vocab_size=hparams["n_vocab"],
                seq_len=hparams["n_ctx"],
                d_model=hparams["n_embd"],
                num_heads=hparams["n_head"],
                num_layers=hparams["n_layer"],
                debug=False,
            ),
        ).make()
        return checkpoint.load_weights(
            gpt2, checkpoint_path=checkpoint_path, verbose=verbose
        )

    def get_logits(self, tokens: jnp.ndarray, kv_cache: bool) -> jnp.ndarray:
        if kv_cache and self.model.position is not None:
            tokens = tokens[:, -1:]
        logits = self.model(tokens, kv_cache=kv_cache)
        return logits[:, -1, :]

    def reset_kv_cache(self):
        self.model.position = None
        for block in self.model.blocks:
            block.attn.kv_cache = None

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        kv_cache: bool = False,
        streaming: bool = True,
        output_logits: bool = False,
        sampling_method: str = "greedy",
    ) -> model_types.ModelOutput:
        if streaming:
            print(prompt, end="", flush=True)
        self.reset_kv_cache()
        tokens = jnp.array([self.tokenizer.encode(prompt)])
        all_logits = jnp.empty((1, 0, self.model.vocab_size))

        while len(tokens[0]) < max_tokens:
            logits = self.get_logits(tokens=tokens, kv_cache=kv_cache)
            result = sampling.sample(logits, method=sampling_method)

            if output_logits:
                all_logits = jnp.concatenate(
                    (all_logits, jnp.expand_dims(logits, 1)), axis=1
                )

            tokens = jnp.concatenate([tokens, result], axis=-1)
            new_token = result[0, 0]
            if new_token == 50256:  # Break if <|endoftext|> is encountered
                break
            if streaming:
                print(self.tokenizer.decode([new_token]), end="", flush=True) # type: ignore
        if streaming:
            print()
        return model_types.ModelOutput(
            text=[self.tokenizer.decode(tokens[0].tolist())],
            tokens=tokens,
            logits=all_logits if output_logits else None,
        )
