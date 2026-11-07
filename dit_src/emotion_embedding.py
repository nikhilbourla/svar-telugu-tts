import torch
import torch.nn as nn

class EmotionEmbedding(nn.Module):
    def __init__(self,embed_dim,emotions_size):
        super().__init__()
        self.emo_embd=nn.Embedding(emotions_size,embed_dim)

    def forward(self,emotion_ids):
        emo_embds=self.emo_embd(emotion_ids)
        return emo_embds