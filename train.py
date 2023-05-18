import torch
from preprocessing.text import TextProcessor, symbols
from preprocessing.audio import AudioProcessor
from trainer import Tacotron2Trainer
import pickle
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--mini_batch", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=0.00003)
parser.add_argument("--device", type=str, default="cpu")

parser.add_argument("--data_folder", type=str)
parser.add_argument("--checkpoint", type=str)

def load_data(path: str):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def program(data_folder: str, checkpoint: str, epochs: int, batch_size: int, mini_batch: int, learning_rate: float, device: str):
    audio = load_data(data_folder + "/audio.pkl")
    text = load_data(data_folder + "/text.pkl")


    trainer = Tacotron2Trainer(token_size=len(symbols), n_mel_channels=audio.shape[1], checkpoint=checkpoint, device=device)

    audio = torch.tensor(audio)
    text = torch.tensor(text)
    
    trainer.fit(text, audio, epochs=epochs, batch_size=batch_size, mini_batch=mini_batch, learning_rate=learning_rate)

    print("Finished Training")

if __name__ == "__main__":
    args = parser.parse_args()

    if args.data_folder is None or args.checkpoint is None:
        print("Missing Information")
    else:
        program(
            data_folder=args.data_folder,
            checkpoint=args.checkpoint,
            epochs=args.epochs,
            batch_size=args.batch_size,
            mini_batch=args.mini_batch,
            learning_rate=args.learning_rate,
            device=args.device
        )
