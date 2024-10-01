from typing import Protocol, TypeVar

import dataclasses
import jax.numpy as jnp

T = TypeVar("T", covariant=True)


class MakeableConfig(Protocol[T]):
    def make(self) -> T: ...


@dataclasses.dataclass
class ModelOutput:
    text: list[str]  # [B]
    tokens: jnp.ndarray  # [B, T]
    logits: jnp.ndarray | None  # [B, T, V]
