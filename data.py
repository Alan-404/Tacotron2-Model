import numpy as np
import pandas as pd
import io

from preprocessing.text import TextProcessor
from preprocessing.audio import AudioProcessor


def program(data_path: str,
            sample_rate: int,
            duration: float,
            mono: bool,
            frame_size: int,
            hop_length: int,
            n_mels: int):
    audio_processor = AudioProcessor(sample_rate, duration, mono, frame_size, hop_length, n_mels)
    df = pd.read_csv(data_path)
    files = df['file'].to_list()
    contents = df['content'].to_list()

