import attrs

@attrs.define
class SpecialToken:
    token: int
    char: str
    
@attrs.define
class Vocab:

    def encode_text(self, text: str) -> list[int]:
        """Encode text based input as string to list of tokens"""
        raise NotImplementedError()

    def decode_tokens(self, tokens: list[int]) -> str:
        """Decode list of tokens to string"""
        raise NotImplementedError()

    def __len__(self):
        """Returns number of tokens"""
        raise NotImplementedError()
    
    @property
    def bos(self) -> SpecialToken:
        """Beginning of sequenc token"""
        raise NotImplementedError()
    
    @property
    def eos(self) -> SpecialToken:
        """End of sequence token"""
        raise NotImplementedError()
    
    @property
    def pad(self) -> SpecialToken:
        """Padding token"""
        raise NotImplementedError()

    @property
    def special_tokens(self):
        return [special.token for special in self.special]

    @property
    def special_characters(self):
        return [special.char for special in self.special]

    @property
    def special(self):
        return [
            self.bos,
            self.eos,
            self.pad
        ]

