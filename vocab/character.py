import attrs
from vocab import Vocab, SpecialToken

@attrs.define
class CharacterVocab(Vocab):
    """Vocab for character level models"""

    # List of all possible characters
    characters: list[str] = attrs.field()

    # Token : Character mappings
    char_to_token: dict[str, int] = attrs.field(init=False)
    token_to_char: dict[int, str] = attrs.field(init=False)

    def __attrs_post_init__(self):

        if not all([c not in self.special_characters for c in self.characters]):
            raise ValueError(f"Special characters {self.special_characters} are reserved by {self.__class__.__name__}.")
        
        # Sort characters and insert special characters at correct indices
        self.characters = sorted(self.characters)
        for special in self.special: self.characters.insert(special.token, special.char)

        # Create mappings reserving space for special characters
        self.char_to_token = {c: t for t, c in enumerate(self.characters)}
        self.token_to_char = {t: c for t, c in enumerate(self.characters)}

    def encode_text(self, text: str) -> list[int]:
        """Encode text based input as string to list of tokens"""
        return [self.char_to_token[c] for c in text]

    def decode_tokens(self, tokens: list[int]) -> str:
        """Decode list of tokens to string"""
        return ''.join([self.token_to_char[t] for t in tokens])

    def __len__(self):
        """Returns number of tokens"""
        return len(self.characters)

    @property
    def bos(self):
        return SpecialToken(1, "<")

    @property
    def eos(self):
        return SpecialToken(2, ">")
    
    @property
    def pad(self):
        return SpecialToken(0, " ")


