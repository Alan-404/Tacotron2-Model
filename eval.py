#%%
import torch
from preprocessing.text import TextProcessor, symbols
from model.tacotron2 import Tacotron2
# %%
model = Tacotron2(token_size=len(symbols), n_mel_channels=80, checkpoint='./saved_models/tacotron2.pt')
# %%
seq = 'Hello world'
# %%
text_processor = TextProcessor()
# %%
out = text_processor.process([seq])
# %%
out
# %%
out = torch.tensor(out)
# %%
out.size()
#%%
device = torch.device('cuda')
# %%
out = out.to(device)
# %%
result = model.predict(out, max_decoder_steps=800, gate_threshold=0.5)
# %%
result[0].shape
# %%
import librosa.display as display
# %%

display.specshow(result[0][0].detach().cpu().numpy())
# %%
result[0][0]
# %%
result[1]
# %%

# %%
