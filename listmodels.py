import torch
from TTS.api import TTS
tts = TTS(model_name="", progress_bar=False).to('cuda')
models = tts.list_models()
for m in sorted(models):
  print(m)
