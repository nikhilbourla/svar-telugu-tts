import torch.nn as nn

from config import Config
from text_embedding import TextEmbedding
from semantic_embedding import Semantic_embedding
from speaker_embedding import SpeakerEmbedding
from positonal_embedding import PositionalEmbedding
from encoder import Encoder
from decoder import Decoder


class TextToSemantic(nn.Module):

    def __init__(self):
        super().__init__()
        d = Config.d_model
        self.text_embedding = TextEmbedding(Config.TEXT_VOCAB, d, padding_idx=Config.TEXT_PAD)
        self.sem_embedding = Semantic_embedding(Config.SEM_VOCAB, d, padding_idx=Config.SEM_PAD)
        self.speaker_embedding = SpeakerEmbedding(d, Config.num_speakers)
        self.position = PositionalEmbedding(d)          
        self.drop = nn.Dropout(Config.dropout)

        self.encoder = Encoder(Config.enc_layers, d, Config.n_heads, Config.ffn_mult, Config.dropout)
        self.decoder = Decoder(Config.dec_layers, d, Config.n_heads, Config.ffn_mult, Config.dropout)
        self.out = nn.Linear(d, Config.SEM_VOCAB)

    def encode(self, text_ids, text_pad_mask):
        x = self.text_embedding(text_ids)                      
        x = self.drop(x + self.position(x))
        return self.encoder(x, text_pad_mask)                  

    def decode(self, dec_in, enc_out, text_pad_mask, speaker_id, dec_pad_mask=None):
        x = self.sem_embedding(dec_in)                          
        x = x + self.speaker_embedding(speaker_id).unsqueeze(1) 
        x = self.drop(x + self.position(x))
        h = self.decoder(x, enc_out, text_pad_mask, dec_pad_mask)
        return self.out(h)                                      
    
    def forward(self, text_ids, text_pad_mask, dec_in, dec_pad_mask, speaker_id):
        enc_out = self.encode(text_ids, text_pad_mask)
        return self.decode(dec_in, enc_out, text_pad_mask, speaker_id, dec_pad_mask)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
