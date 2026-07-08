import tiktoken


class Tokenizer:
    def __init__(self, encoding_name: str = "gpt2"):
        self._enc = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str) -> list[int]:
        return self._enc.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, ids: list[int]) -> str:
        return self._enc.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self._enc.n_vocab
