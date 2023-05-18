import numpy as np
from .cmu_dict import valid_symbols, CMUDict
from .number import normalize_numbers
import re
from g2p_en import G2p
_pad = "_"
_punctuation = '!\'(),.:;? '
_special = "-"
_letters = 'abcdefghijklmnopqrstuvwxyz'

ipa = [f"@{s}" for s in valid_symbols]

symbols = [_pad] + ipa + list(_special) + list(_punctuation) 

class Cleaner:
    def __init__(self) -> None:
        pass
    def clean(self, seq: str):
        seq = normalize_numbers(seq)
        seq = re.sub(r'\"', "", seq)
        seq = re.sub(r"\'", "", seq)
        seq = re.sub(r"\-", " - ", seq)
        seq = re.sub(r"\,", " , ", seq)
        seq = re.sub(r"\.", " . ", seq)
        seq = re.sub(r"[\"|\'|\(|\)|~`:!@#$^&*]", "", seq)
        seq = seq.strip()
        seq = seq.lower()
        seq = re.sub("\s\s+", " ", seq)
        return seq

class Dictionary:
    def __init__(self) -> None:
        self.cmu_dict = CMUDict("./preprocessing/lexicon/cmudict.txt")
        self.g2e = G2p()
    def get_pronunciation(self, word: str):
        return self.cmu_dict.lookup(word)
    def lookup(self, word: str):
        phonemes = self.cmu_dict.lookup(word)
        if phonemes is None:
            return None
        elif len(phonemes) == 1:
            phonemes = phonemes[0]
        else:
            phonemes = phonemes[1]
        result = []
        for phoneme in phonemes.split(" "):
            result.append(symbols.index(f"@{phoneme}"))
        return result

class TextProcessor:
    def __init__(self) -> None:
        self.dictionary = Dictionary()
        self.cleaner = Cleaner()
    def padding_sequence(self, sequence, padding: str, maxlen: int) -> np.ndarray:
        delta = maxlen - len(sequence)
        zeros = np.zeros(delta, dtype=np.int64)

        if padding.strip().lower() == 'post':
            return np.concatenate((sequence, zeros), axis=0)
        elif padding.strip().lower() == 'pre':
            return np.concatenate((zeros, sequence), axis=0)

    def truncating_sequence(self, sequence, truncating: str, maxlen: int) -> np.ndarray:
        if truncating.strip().lower() == 'post':
            return sequence[0:maxlen]
        elif truncating.strip().lower() == 'pre':
            delta = sequence.shape[0] - maxlen
            return sequence[delta: len(sequence)]

    def pad_sequences(self, sequences: list, maxlen: int, padding: str = 'post', truncating: str = 'post') -> np.ndarray:
        result = []
        for _, sequence in enumerate(sequences):
            delta = sequence.shape[0] - maxlen
            if delta < 0:
                sequence = self.padding_sequence(sequence, padding, maxlen)
            elif delta > 0:
                sequence = self.truncating_sequence(sequence, truncating, maxlen)
            result.append(sequence)
        
        return np.array(result)
    
    def handle_characters_phoneme(self, word: str):
        phonemes = []
        for character in word:
            phoneme = self.dictionary.lookup(character)
            if phoneme is None:
                phonemes.append(symbols.index(character))
            else:
                phonemes.append(phoneme)
        return phonemes
    
    def concat_phonemes(self, phonemes: list):
        result = []
        for item in phonemes:
            print(item)
            result += item
        return np.array(result)
    def process(self, data: list):
        digits = []
        max_len = 0
        for item in data:
            cleanned = self.cleaner.clean(item)
            word_phonemes  = []
            for word in cleanned.split(" "):
                phonemes = self.dictionary.g2e(word)
                temp = []
                for phoneme in phonemes:
                    if phoneme in valid_symbols:
                        temp.append(symbols.index(f"@{phoneme}"))
                    else:
                        temp.append(symbols.index(f"{phoneme}"))
                word_phonemes += temp
                word_phonemes.append(symbols.index(" "))
            if max_len < len(word_phonemes):
                max_len = len(word_phonemes)
            digits.append(np.array(word_phonemes))
        padded = self.pad_sequences(digits, maxlen=max_len)
        return padded
