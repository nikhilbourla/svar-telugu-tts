import torch
import torch.nn as nn

class SpeakerEmbedding(nn.Module):
    def __init__(self,embed_dim,speakers_size):
        super().__init__()
        self.speaker_embd=nn.Embedding(speakers_size,embed_dim)

    def forward(self,speakers_ids):
        sp_embed=self.speaker_embd(speakers_ids)
        return sp_embed