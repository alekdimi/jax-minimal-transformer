import dataclasses

import tiktoken

from . import model_types

Tokenizer = tiktoken.Encoding


@dataclasses.dataclass
class TokenizerConfig(model_types.MakeableConfig[Tokenizer]):
    model: str

    def make(self) -> Tokenizer:
        return tiktoken.encoding_for_model(self.model)
