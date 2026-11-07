from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import math
import os
import re
import sys
import time
import types
import unicodedata
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

EMOTION_ID = 0                      # only one emotion exists right now
NUM_SEM_UNITS_DEFAULT = 200



@dataclass
class SynthSettings:
    seed: int = 42
    temperature: float = 0.8        
    top_k: int = 5                  
    top_p: float = 1.0              
    t2s_attempts: int = 2           
    t2s_batch: int = 8              
    max_token_run: int = 100        
    # ---- DiT
    dit_steps: int = 65             
    dit_batch: int = 4              
    dit_fp16: bool = False

    # ---- chunking
    sem_per_piece: float = 4.5      
    budget_fraction: float = 0.85   
    use_context: bool = False       
    max_resplit_depth: int = 3

    # ---- stitching
    pause_ms: Dict[str, int] = field(
        default_factory=lambda: {".": 350, "?": 350, "!": 350, ",": 150, "": 0})
    crossfade_ms: int = 20          
    trim_db: float = -45.0          
    keep_ms: int = 20               
    lead_keep_ms: int = 80          
    tail_keep_ms: int = 250         
    edge_pad_ms: int = 50           
    tail_pad_ms: int = 150          
    debug_dir: Optional[str] = None  
    verbose: bool = False



# 1. normalization 
# Keep the cleaning steps identical to the training version.
KEEP_PUNCT = {",", ".", "?", "!"}
ZERO_WIDTH = {"\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"}
TO_PERIOD = {"\u0964", "\u0965", "\u2026"}
BRACE_RE = re.compile(r"\{[^{}]*\}\s*\{([^{}]*)\}")


def _is_telugu(ch: str) -> bool:
    return 0x0C00 <= ord(ch) <= 0x0C7F


def _is_latin(ch: str) -> bool:
    return "LATIN" in unicodedata.name(ch, "")


def _is_digit(ch: str) -> bool:
    return unicodedata.category(ch) == "Nd"


def normalize_lenient(raw: str) -> Tuple[str, List[str]]:
    """Return (clean_text, notes). Same cleaning as training, but nothing is rejected."""
    t = unicodedata.normalize("NFC", raw)
    t = "".join(c for c in t if c not in ZERO_WIDTH)
    if BRACE_RE.search(t):
        t = BRACE_RE.sub(lambda m: m.group(1), t)
    for c in TO_PERIOD:
        t = t.replace(c, ".")

    notes = []
    if any(_is_latin(c) for c in t):
        notes.append("Latin letters found")
    if any(_is_digit(c) for c in t):
        notes.append("digits found")
    if notes:
        notes = ["not supported (not seen in training, they will not be spoken correctly): "
                 + ", ".join(notes) + ". Write them out in Telugu script."]

    out = []
    for c in t:
        if c in KEEP_PUNCT or c == " " or _is_telugu(c) or _is_latin(c) or _is_digit(c):
            out.append(c)
        elif (unicodedata.category(c)[0] in ("P", "S", "C", "Z")
              or (unicodedata.category(c) == "Lm" and not _is_telugu(c))):
            out.append(" ")
        else:
            out.append(c)
    t = "".join(out)
    t = re.sub(r"\s+([,.?!])", r"\1", t)
    t = re.sub(r"([,.?!])(\s*[,.?!])+", r"\1", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t, notes



# 2. splitting
_SENT_RE = re.compile(r"(?<=[.?!])\s+")
_CLAUSE_RE = re.compile(r"(?<=,)\s+")


def boundary_of(text: str) -> str:
    """Punctuation that ends a chunk: '.', '?', '!', ',' or '' (split inside a sentence)."""
    last = text.rstrip()[-1:]
    return last if last in (".", "?", "!", ",") else ""


def bisect_text(text: str) -> Optional[Tuple[str, str]]:
    """Cut a chunk in two near the middle: sentence end, else comma, else space. Never inside a word."""
    cands = [m.end() for m in re.finditer(r"(?<=[.?!])\s+", text)]
    if not cands:
        cands = [m.end() for m in re.finditer(r"(?<=,)\s+", text)]
    if not cands:
        cands = [m.end() for m in re.finditer(r"\s+", text)]
    if not cands:
        return None
    mid = len(text) / 2
    pos = min(cands, key=lambda p: abs(p - mid))
    a, b = text[:pos].strip(), text[pos:].strip()
    return (a, b) if a and b else None


class ChunkSplitter:

    def __init__(self, sp, max_text_len: int, budget_tokens: float, sem_per_piece: float):
        self.sp = sp
        self.max_text_len = max_text_len
        self.budget = budget_tokens
        self.spp = sem_per_piece

    def _pieces(self, text: str) -> int:
        return len(self.sp.encode(text))

    def _fits(self, text: str, limit: Optional[float] = None) -> bool:
        limit = self.budget if limit is None else limit
        n = self._pieces(text)
        return n <= self.max_text_len and n * self.spp <= limit

    def split(self, text: str) -> List[str]:
        return self._split(text.strip(), 0)

    def _split(self, text: str, level: int) -> List[str]:
        if self._fits(text):
            return [text]
        if level >= 2:
            return self._split_words(text)
        units = _SENT_RE.split(text) if level == 0 else _CLAUSE_RE.split(text)
        units = [u for u in units if u]
        if len(units) <= 1:
            return self._split(text, level + 1)
        parts: List[str] = []
        for u in units:
            parts.extend([u] if self._fits(u) else self._split(u, level + 1))
        return self._pack(parts, self.budget)

    def _split_words(self, text: str) -> List[str]:
        words = text.split()
        if len(words) <= 1:
            return [text]                      
        est = self._pieces(text) * self.spp
        k = max(2, math.ceil(est / self.budget))
        limit = min(self.budget, est / k * 1.15)   
        return self._pack(words, limit)

    def _pack(self, parts: List[str], limit: float) -> List[str]:
        out, cur = [], ""
        for p in parts:
            cand = f"{cur} {p}" if cur else p
            if cur and not self._fits(cand, limit):
                out.append(cur)
                cur = p
            else:
                cur = cand
        if cur:
            out.append(cur)
        return out



# 3. t2s backend with a decoder cache
def longest_run(a: np.ndarray) -> int:
    if len(a) == 0:
        return 0
    change = np.flatnonzero(np.diff(a) != 0)
    edges = np.concatenate(([-1], change, [len(a) - 1]))
    return int(np.max(np.diff(edges)))


class T2SBackend:
    
    def __init__(self, model, cfg: dict, device: str):
        self.model = model
        self.device = device
        self.d_model = cfg["d_model"]
        self.text_pad = cfg["TEXT_PAD"]
        self.text_bos = cfg["TEXT_BOS"]
        self.text_eos = cfg["TEXT_EOS"]
        self.text_unk = cfg.get("TEXT_UNK", 1)
        self.sem_pad = cfg["SEM_PAD"]
        self.sem_bos = cfg["SEM_BOS"]
        self.sem_eos = cfg["SEM_EOS"]
        self.num_units = cfg.get("NUM_SEM_UNITS", NUM_SEM_UNITS_DEFAULT)
        self.min_len = cfg.get("min_sem_len", 25)
        self.max_sem_len = cfg.get("max_sem_len", 1150)
        self._pos = None

    def text_ids(self, sp, text: str) -> List[int]:
        return [self.text_bos] + sp.encode(text) + [self.text_eos]

    
    def _pos_table(self) -> torch.Tensor:
        if self._pos is None:
            n = self.max_sem_len + 2                      
            pe = self.model.position(torch.zeros(1, n, self.d_model, device=self.device))
            pe = pe.reshape(-1, self.d_model)
            if pe.shape[0] < n:
                raise RuntimeError("positional table shorter than expected")
            self._pos = pe[:n]
        return self._pos

    def _embed(self, tokens: torch.Tensor, spk_vec: torch.Tensor, start: int) -> torch.Tensor:
        x = self.model.sem_embedding(tokens)
        x = x + spk_vec.unsqueeze(1)
        n = tokens.shape[1]
        return x + self._pos_table()[start:start + n].unsqueeze(0)

    @staticmethod
    def _layer(layer, x, cache, enc, text_mask):
        h = layer.norm1(x)
        cache = h if cache is None else torch.cat([cache, h], dim=1)
        if x.shape[1] == 1:                               
            a = layer.self_attn(h, cache, key_pad_mask=None, causal=False)
        else:                                             
            a = layer.self_attn(h, h, key_pad_mask=None, causal=True)
        x = x + a
        x = x + layer.cross_attn(layer.norm2(x), enc, key_pad_mask=text_mask)
        x = x + layer.ffn(layer.norm3(x))
        return x, cache

    def _run(self, x, caches, enc, text_mask):
        for i, layer in enumerate(self.model.decoder.layers):
            x, caches[i] = self._layer(layer, x, caches[i], enc, text_mask)
        return x, caches

    def _head(self, x_last: torch.Tensor) -> torch.Tensor:
        return self.model.out(self.model.decoder.final_norm(x_last))

    def _pad_text(self, id_lists: List[List[int]]):
        B, L = len(id_lists), max(len(t) for t in id_lists)
        ids = torch.full((B, L), self.text_pad, dtype=torch.long, device=self.device)
        mask = torch.ones((B, L), dtype=torch.bool, device=self.device)      
        for i, t in enumerate(id_lists):
            ids[i, :len(t)] = torch.tensor(t, dtype=torch.long, device=self.device)
            mask[i, :len(t)] = False
        return ids, mask

    
    def _sample(self, logits, new_count: int, s: SynthSettings, gen) -> int:
        logits = logits.float().clone()
        logits[self.sem_pad] = float("-inf")
        logits[self.sem_bos] = float("-inf")
        if new_count < self.min_len:
            logits[self.sem_eos] = float("-inf")
        if s.temperature <= 0:
            return int(logits.argmax())
        logits = logits / s.temperature
        if s.top_k and s.top_k > 0:
            kth = torch.topk(logits, min(s.top_k, logits.numel())).values[-1]
            logits[logits < kth] = float("-inf")
        if s.top_p < 1.0:
            sl, si = torch.sort(logits, descending=True)
            probs = torch.softmax(sl, dim=-1)
            cum = torch.cumsum(probs, dim=-1)
            sl[cum - probs > s.top_p] = float("-inf")
            logits = torch.full_like(logits, float("-inf")).scatter(0, si, sl)
        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, 1, generator=gen))

    
    @torch.inference_mode()
    def generate(self, id_lists: List[List[int]], speaker_id: int, prefix: List[int],
                 max_new: int, s: SynthSettings, gens: list) -> List[Tuple[np.ndarray, bool]]:
        """Decode B chunks together. `prefix` (semantic tokens of a previous chunk) is only used with B == 1.
        Returns, per chunk, (new tokens without BOS/EOS, finished_with_EOS)."""
        dev = self.device
        B = len(id_lists)
        if prefix and B != 1:
            raise ValueError("a context prefix is only supported for a single chunk")
        text_t, text_mask = self._pad_text(id_lists)
        enc = self.model.encode(text_t, text_mask)                            
        spk_vec = self.model.speaker_embedding(
            torch.full((B,), speaker_id, dtype=torch.long, device=dev))

        caches = [None] * len(self.model.decoder.layers)
        start = torch.tensor([[self.sem_bos] + [int(t) for t in prefix]] * B,
                             dtype=torch.long, device=dev)
        x, caches = self._run(self._embed(start, spk_vec, 0), caches, enc, text_mask)
        logits = self._head(x[:, -1])                                         
        pos = start.shape[1]

        out: List[List[int]] = [[] for _ in range(B)]
        finished = [False] * B
        done = [False] * B
        for _ in range(max_new + 1):
            nxt = []
            for i in range(B):
                if done[i]:
                    nxt.append(0)
                    continue
                tok = self._sample(logits[i], len(out[i]), s, gens[i])
                if tok == self.sem_eos:
                    finished[i] = done[i] = True
                    nxt.append(0)
                elif len(out[i]) >= max_new:                                  
                    done[i] = True
                    nxt.append(0)
                else:
                    out[i].append(tok)
                    nxt.append(tok)
            if all(done):
                break
            step = torch.tensor(nxt, dtype=torch.long, device=dev).unsqueeze(1)
            x, caches = self._run(self._embed(step, spk_vec, pos), caches, enc, text_mask)
            logits = self._head(x[:, 0])
            pos += 1

        res = []
        for i in range(B):
            toks = np.array(out[i], dtype=np.int64)
            res.append((toks[toks < self.num_units], finished[i]))
        return res

    
    @torch.inference_mode()
    def self_check(self, sp, text: str, speaker_id: int, steps: int = 40, tol: float = 1e-3) -> bool:
        dev = self.device
        ids = self.text_ids(sp, text)
        text_t, text_mask = self._pad_text([ids])
        spk = torch.tensor([speaker_id], dtype=torch.long, device=dev)
        enc = self.model.encode(text_t, text_mask)
        spk_vec = self.model.speaker_embedding(spk)

        caches = [None] * len(self.model.decoder.layers)
        dec = [self.sem_bos]
        x, caches = self._run(self._embed(torch.tensor([dec], device=dev), spk_vec, 0),
                              caches, enc, text_mask)
        cached = self._head(x[:, -1])[0]
        worst = 0.0
        for _ in range(steps):
            dec_t = torch.tensor([dec], dtype=torch.long, device=dev)
            full = self.model(text_t, text_mask, dec_t, torch.zeros_like(dec_t, dtype=torch.bool), spk)[0, -1]
            worst = max(worst, float((full - cached).abs().max()))
            tok = int(cached[:self.num_units].argmax())
            dec.append(tok)
            step = torch.tensor([[tok]], dtype=torch.long, device=dev)
            x, caches = self._run(self._embed(step, spk_vec, len(dec) - 1), caches, enc, text_mask)
            cached = self._head(x[:, 0])[0]
        ok = worst < tol
        print(f"t2s cache check: max |logit difference| over {steps} steps = {worst:.2e}  ->  "
              f"{'PASS' if ok else 'FAIL (do not use the cache; send me attention.py)'}")
        return ok



@contextlib.contextmanager
def isolated_import(folder: Optional[str]):
    """Import your flat model files (model.py, positonal_embedding.py, ...) from `folder` without
    clashing with same-named files of another folder. Build the model INSIDE the `with` block."""
    if folder is None:
        yield
        return
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"source folder not found: {folder}")
    names = {os.path.splitext(f)[0] for f in os.listdir(folder) if f.endswith(".py")}
    saved = {n: sys.modules.pop(n) for n in list(names) if n in sys.modules}
    sys.path.insert(0, folder)
    importlib.invalidate_caches()
    try:
        yield
    finally:
        for n in names:
            sys.modules.pop(n, None)
        for n in [k for k, v in sys.modules.items() if getattr(v, "_svar_stub", False)]:
            del sys.modules[n]
        sys.modules.update(saved)
        if folder in sys.path:
            sys.path.remove(folder)


def _default_src(name: str) -> Optional[str]:
    """<folder of this file>/<name> if it exists (t2s_src / dit_src), else None = flat layout."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    return path if os.path.isdir(path) else None



# 4. DiT backend
_DIT_PARTS = ["semantic_embedding", "acoustic_projection", "position_embedding",
              "timestep_embedding", "speaker_embedding", "emotion_embedding", "dit"]


def _sub_state(sd: dict, name: str) -> dict:
    p = name + "."
    return {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}


def _find_stat(name: str, *sources: dict):
    for src in sources:
        keys = [k for k in src if name in k.lower()]
        keys.sort(key=lambda k: (not k.lower().startswith("latent"), len(k)))
        if keys:
            return src[keys[0]]
    raise KeyError(f"latent {name} not found in dit.safetensors or latent_stats.safetensors")


class DiTBackend:
    def __init__(self, dit_sd: dict, stats_sd: dict, cfg: dict, device: str,
                 frames_per_token: float, max_frames: int, src_dir: Optional[str] = None,
                 frame_offset: int = 0):
        E, L, H = cfg["embed_dim"], cfg["num_layers"], cfg["num_heads"]
        D, V = cfg["latent_dim"], cfg["vocab_size"]
        with isolated_import(src_dir):
            from semantic_embedding import Semantic_embedding
            from acoustic_projection import AcousticProjection
            from positional_embedding import PositionalEmbedding
            from timestep_embedding import TimestepEmbedding
            from speaker_embedding import SpeakerEmbedding
            from emotion_embedding import EmotionEmbedding
            from dit import Dit
            mods = {
                "semantic_embedding": Semantic_embedding(V, E),
                "acoustic_projection": AcousticProjection(D, E),
                "position_embedding": PositionalEmbedding(E),
                "timestep_embedding": TimestepEmbedding(E),
                "speaker_embedding": SpeakerEmbedding(E, cfg["speaker_size"]),
                "emotion_embedding": EmotionEmbedding(E, cfg["emotion_size"]),
                "dit": Dit(E, E, H, L, D),
            }
        for name in _DIT_PARTS:
            mods[name].load_state_dict(_sub_state(dit_sd, name))
            mods[name].to(device).eval()
        self.m = mods
        self.device = device
        self.latent_dim = D
        self.sem_pad = V - 1                      
        self.fpt = frames_per_token
        self.max_frames = max_frames
        self.offset = frame_offset      
        self.mean = _find_stat("mean", dit_sd, stats_sd).to(device)
        self.std = _find_stat("std", dit_sd, stats_sd).to(device)

    def n_frames(self, n_tokens: int) -> int:
        return min(math.ceil(n_tokens * self.fpt) + self.offset, self.max_frames)

    @torch.inference_mode()
    def sample(self, token_lists: List[np.ndarray], speaker_id: int,
               seeds: List[int], steps: int, fp16: bool) -> List[torch.Tensor]:
        dev, m = self.device, self.m
        B = len(token_lists)
        S = max(len(t) for t in token_lists)
        nf = [self.n_frames(len(t)) for t in token_lists]
        T = max(nf)

        sem = torch.full((B, S), self.sem_pad, dtype=torch.long, device=dev)
        sem_mask = torch.ones((B, S), dtype=torch.bool, device=dev)       
        ac_mask = torch.ones((B, T), dtype=torch.bool, device=dev)
        x = torch.zeros((B, T, self.latent_dim), device=dev)
        for i, toks in enumerate(token_lists):
            sem[i, :len(toks)] = torch.as_tensor(toks, dtype=torch.long, device=dev)
            sem_mask[i, :len(toks)] = False
            ac_mask[i, :nf[i]] = False
            g = torch.Generator(device=dev)
            g.manual_seed(int(seeds[i]))
            x[i, :nf[i]] = torch.randn((nf[i], self.latent_dim), generator=g, device=dev)

        spk = torch.full((B,), speaker_id, dtype=torch.long, device=dev)
        emo = torch.full((B,), EMOTION_ID, dtype=torch.long, device=dev)

        use_amp = fp16 and str(dev).startswith("cuda")
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            semantic = m["semantic_embedding"](sem)
            semantic = semantic + m["position_embedding"](semantic)       
            speaker = m["speaker_embedding"](spk)
            emotion = m["emotion_embedding"](emo)

            dt = 1.0 / steps
            t_val = 1.0                                                   
            for _ in range(steps):
                t_tensor = torch.full((B,), t_val, device=dev)
                x_proj = m["acoustic_projection"](x)
                dit_in = x_proj + m["position_embedding"](x_proj)
                condition = m["timestep_embedding"](t_tensor) + speaker + emotion
                velocity = m["dit"](dit_in, semantic, condition, sem_mask, ac_mask)
                x = x - velocity.float() * dt
                t_val -= dt

        x = x.float() * self.std + self.mean
        return [x[i, :nf[i]] for i in range(B)]



# 5. EnCodec decoder
class Codec:
    def __init__(self, path_or_id: str, device: str):
        from transformers import EncodecModel
        self.model = EncodecModel.from_pretrained(path_or_id).to(device).eval()
        self.device = device
        self.sample_rate = int(self.model.config.sampling_rate)

    @torch.inference_mode()
    def decode(self, latent: torch.Tensor) -> np.ndarray:
        z = latent.transpose(0, 1).unsqueeze(0).to(self.device)           
        audio = self.model.decoder(z)
        return audio.reshape(-1).float().cpu().numpy()



# 6. stitching
def _frame_rms(x: np.ndarray, hop: int) -> np.ndarray:
    n = len(x) // hop
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    f = x[:n * hop].reshape(n, hop)
    return np.sqrt((f ** 2).mean(axis=1) + 1e-12)


def edge_silence_ms(x: np.ndarray, sr: int, thr_db: float) -> Tuple[float, float]:
    hop = max(1, int(sr * 0.010))
    rms = _frame_rms(x, hop)
    if len(rms) == 0 or rms.max() < 1e-5:
        return 0.0, 0.0
    voiced = np.flatnonzero(rms > rms.max() * (10 ** (thr_db / 20)))
    head = voiced[0] * hop
    tail = len(x) - (voiced[-1] + 1) * hop
    return 1000.0 * head / sr, 1000.0 * tail / sr


def trim_silence(x: np.ndarray, sr: int, thr_db: float,
                 keep_head_ms: int, keep_tail_ms: int) -> Tuple[np.ndarray, float]:
    hop = max(1, int(sr * 0.010))
    rms = _frame_rms(x, hop)
    if len(rms) == 0 or rms.max() < 1e-5:
        return x, 0.0
    thr = rms.max() * (10 ** (thr_db / 20))
    voiced = np.flatnonzero(rms > thr)
    start = max(voiced[0] * hop - int(sr * keep_head_ms / 1000), 0)
    end = min((voiced[-1] + 1) * hop + int(sr * keep_tail_ms / 1000), len(x))
    return x[start:end], float(np.sqrt(np.mean(rms[voiced] ** 2)))


def _fade_edges(x: np.ndarray, sr: int, ms: int = 5) -> np.ndarray:
    n = min(int(sr * ms / 1000), len(x) // 2)
    if n > 0:
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
        x = x.copy()
        x[:n] *= ramp
        x[-n:] *= ramp[::-1]
    return x


def stitch(wavs: List[np.ndarray], boundaries: List[str], sr: int, s: SynthSettings) -> np.ndarray:
    """wavs[i] is followed by a join of type boundaries[i] ('.', '?', '!', ',' or '')."""
    trimmed, levels = [], []
    last = len(wavs) - 1
    for i, w in enumerate(wavs):
        head = s.lead_keep_ms if i == 0 else s.keep_ms          
        tail = s.tail_keep_ms if i == last else s.keep_ms        
        t, lvl = trim_silence(w.astype(np.float32), sr, s.trim_db, head, tail)
        trimmed.append(t)
        levels.append(lvl)

    
    ref = float(np.median([l for l in levels if l > 0])) if any(l > 0 for l in levels) else 0.0
    for i, (t, lvl) in enumerate(zip(trimmed, levels)):
        if ref > 0 and lvl > 0:
            trimmed[i] = t * float(np.clip(ref / lvl, 0.5, 2.0))
    trimmed = [_fade_edges(t, sr) for t in trimmed]

    out = trimmed[0]
    for i in range(1, len(trimmed)):
        nxt = trimmed[i]
        pause = s.pause_ms.get(boundaries[i - 1], 0)
        if pause > 0:
            out = np.concatenate([out, np.zeros(int(sr * pause / 1000), dtype=np.float32), nxt])
        else:
            xf = min(int(sr * s.crossfade_ms / 1000), len(out), len(nxt))
            if xf > 0:
                w = np.linspace(0.0, 1.0, xf, dtype=np.float32)
                mix = out[-xf:] * (1.0 - w) + nxt[:xf] * w
                out = np.concatenate([out[:-xf], mix, nxt[xf:]])
            else:
                out = np.concatenate([out, nxt])

    out = np.concatenate([np.zeros(int(sr * s.edge_pad_ms / 1000), dtype=np.float32), out,
                          np.zeros(int(sr * s.tail_pad_ms / 1000), dtype=np.float32)])
    peak = float(np.max(np.abs(out))) if len(out) else 0.0
    if peak > 0.99:
        out = out * (0.99 / peak)
    return out.astype(np.float32)


def _prepare_config(t2s_cfg: dict) -> None:
    try:
        import config as cfg_mod
        Cfg = cfg_mod.Config
    except ImportError:
        cfg_mod = types.ModuleType("config")
        Cfg = type("Config", (), {})
        cfg_mod.Config = Cfg
        cfg_mod._svar_stub = True              
        sys.modules["config"] = cfg_mod
    for k, v in t2s_cfg.items():
        setattr(Cfg, k, v)
    if not hasattr(Cfg, "dropout"):
        Cfg.dropout = 0.0



# 7. the public class
@dataclass
class _Item:
    text: str
    depth: int = 0
    attempt: int = 0
    use_ctx: bool = False
    tokens: Optional[np.ndarray] = None
    finished: bool = False
    good: bool = False
    resolved: bool = False


class Svar:
    def __init__(self, t2s: T2SBackend, dit: DiTBackend, codec: Codec, sp,
                 t2s_cfg: dict, dit_cfg: dict, device: str):
        self.t2s, self.dit, self.codec, self.sp, self.device = t2s, dit, codec, sp, device
        self.speaker_to_id = {k.lower(): v for k, v in t2s_cfg["speaker_to_id"].items()}
        self.max_text_len = t2s_cfg.get("max_text_len", 450)
        self.max_sem_len = t2s_cfg.get("max_sem_len", 1150)
        self.fpt = dit.fpt
        # hard cap on generated tokens: whatever the DiT can still take (small safety margin)
        self.hard_cap = int((dit.max_frames - dit.offset) / dit.fpt) - 10
        self.sample_rate = codec.sample_rate

    @classmethod
    def from_pretrained(cls, model_dir: str, device: Optional[str] = None,
                        encodec_path: str = "facebook/encodec_24khz",
                        t2s_src: Optional[str] = None, dit_src: Optional[str] = None) -> "Svar":
        t2s_src = t2s_src or _default_src("t2s_src")
        dit_src = dit_src or _default_src("dit_src")
        import sentencepiece as spm
        from safetensors.torch import load_file
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        j = lambda n: os.path.join(model_dir, n)

        with open(j("t2s_config.json"), encoding="utf-8") as f:
            t2s_cfg = json.load(f)
        with open(j("dit_config.json"), encoding="utf-8") as f:
            dit_cfg = json.load(f)

        sp = spm.SentencePieceProcessor(model_file=j("telugu_char.model"))

        with isolated_import(t2s_src):
            _prepare_config(t2s_cfg)
            from model import TextToSemantic
            t2s_model = TextToSemantic()
        t2s_model = t2s_model.to(device)
        t2s_model.load_state_dict(load_file(j("t2s.safetensors"), device="cpu"))
        t2s_model.eval()

        dit = DiTBackend(
            load_file(j("dit.safetensors"), device="cpu"),
            load_file(j("latent_stats.safetensors"), device="cpu"),
            dit_cfg, device,
            frames_per_token=float(dit_cfg.get("frames_per_token", 1.5)),
            max_frames=int(dit_cfg.get("max_latent_frames", 1100)),
            src_dir=dit_src,
            frame_offset=int(dit_cfg.get("frame_offset", 0)),
        )
        return cls(T2SBackend(t2s_model, t2s_cfg, device), dit, Codec(encodec_path, device),
                   sp, t2s_cfg, dit_cfg, device)

    def _speaker_id(self, speaker) -> int:
        if isinstance(speaker, int) and speaker in self.speaker_to_id.values():
            return speaker
        key = str(speaker).strip().lower()
        if key not in self.speaker_to_id:
            raise ValueError(f"speaker must be one of {sorted(self.speaker_to_id)}")
        return self.speaker_to_id[key]

    def _gen_rows(self, items: List["_Item"], spk: int, s: SynthSettings, state: dict,
                  ctx: Optional[Tuple[str, np.ndarray]]) -> None:
        texts = [it.text for it in items]
        prefix: List[int] = []
        if ctx is not None and len(items) == 1:           
            ctx_text, ctx_tokens = ctx
            joined = f"{ctx_text} {texts[0]}"
            if (len(self.sp.encode(joined)) <= self.max_text_len
                    and len(ctx_tokens) + self.hard_cap + 1 <= self.max_sem_len):
                texts[0], prefix = joined, [int(t) for t in ctx_tokens]
        id_lists = [self.t2s.text_ids(self.sp, t) for t in texts]
        for it, ids in zip(items, id_lists):
            n_unk = sum(1 for i in ids if i == self.t2s.text_unk)
            if n_unk and it.attempt == 0:
                warnings.warn(f"{n_unk} unknown character(s) in: {it.text[:40]!r}")
        gens = []
        for _ in items:
            g = torch.Generator(device=self.device)
            g.manual_seed(s.seed + 7919 * state["n"])
            state["n"] += 1
            gens.append(g)
        results = self.t2s.generate(id_lists, spk, prefix, self.hard_cap, s, gens)
        for it, (toks, fin) in zip(items, results):
            it.tokens, it.finished = toks, fin
            it.good = bool(fin and len(toks) >= self.t2s.min_len
                           and longest_run(toks) < s.max_token_run)

    def _resolve(self, texts: List[str], spk: int, s: SynthSettings, state: dict,
                 ctx=None) -> List[Tuple[str, np.ndarray]]:
        items = [_Item(t, use_ctx=(i == 0 and ctx is not None and s.use_context))
                 for i, t in enumerate(texts)]
        while True:
            todo = [it for it in items if not it.resolved]
            if not todo:
                break
            with_ctx = [it for it in todo if it.use_ctx]
            plain = [it for it in todo if not it.use_ctx]
            for it in with_ctx:
                self._gen_rows([it], spk, s, state, ctx)
            for b in range(0, len(plain), max(1, s.t2s_batch)):
                self._gen_rows(plain[b:b + s.t2s_batch], spk, s, state, None)

            new_items: List[_Item] = []
            for it in items:
                if it.resolved:
                    new_items.append(it)
                elif it.finished and it.good:
                    it.resolved = True
                    new_items.append(it)
                elif not it.finished and it.depth < s.max_resplit_depth and bisect_text(it.text):
                    a, b2 = bisect_text(it.text)
                    new_items += [_Item(a, it.depth + 1, 0, it.use_ctx), _Item(b2, it.depth + 1)]
                elif it.finished and it.attempt + 1 < s.t2s_attempts:
                    new_items.append(_Item(it.text, it.depth, it.attempt + 1, it.use_ctx))
                else:
                    warnings.warn(f"chunk may be imperfect (no clean end or repeated units): {it.text[:40]!r}")
                    it.resolved = True
                    new_items.append(it)
            items = new_items
        return [(it.text, it.tokens) for it in items]

    @torch.inference_mode()
    def synthesize(self, text: str, speaker="male",
                   settings: Optional[SynthSettings] = None) -> Tuple[np.ndarray, int]:
        s = settings or SynthSettings()
        t_start = time.time()
        spk = self._speaker_id(speaker)

        clean, notes = normalize_lenient(text)
        for n in notes:
            warnings.warn(n)
        if not any(_is_telugu(c) for c in clean):
            raise ValueError("no Telugu text to speak")

        splitter = ChunkSplitter(self.sp, self.max_text_len,
                                 s.budget_fraction * self.hard_cap, s.sem_per_piece)
        chunks = splitter.split(clean)
        if s.verbose:
            print(f"{len(chunks)} chunk(s)")

        state = {"n": 0}
        if s.use_context:
            pieces: List[Tuple[str, np.ndarray]] = []
            ctx = None
            for ch in chunks:
                res = self._resolve([ch], spk, s, state, ctx)
                pieces.extend(res)
                ctx = res[-1]
        else:
            pieces = self._resolve(chunks, spk, s, state)
        pieces = [(t, k) for t, k in pieces if len(k) > 0]
        if not pieces:
            raise RuntimeError("t2s produced no tokens")
        if s.verbose:
            for t, k in pieces:
                print(f"  t2s: {len(k)} tokens ({len(k) / 50:.1f}s)  {t[:30]!r}")

        order = sorted(range(len(pieces)), key=lambda i: len(pieces[i][1]))
        latents: Dict[int, torch.Tensor] = {}
        for b in range(0, len(order), s.dit_batch):
            idx = order[b:b + s.dit_batch]
            out = self.dit.sample([pieces[i][1] for i in idx], spk,
                                  [s.seed + 10007 + i for i in idx], s.dit_steps, s.dit_fp16)
            for i, lat in zip(idx, out):
                latents[i] = lat

        wavs = [self.codec.decode(latents[i]) for i in range(len(pieces))]
        if s.verbose or s.debug_dir:
            for i, w in enumerate(wavs):
                h, t = edge_silence_ms(w, self.sample_rate, s.trim_db)
                print(f"  chunk {i}: raw audio {len(w) / self.sample_rate:.2f}s | "
                      f"silence made by the model: start {h:.0f} ms, end {t:.0f} ms")
        if s.debug_dir:
            import soundfile as sf
            os.makedirs(s.debug_dir, exist_ok=True)
            for i, w in enumerate(wavs):
                sf.write(os.path.join(s.debug_dir, f"chunk_{i}_raw.wav"), w, self.sample_rate)
        boundaries = [boundary_of(t) for t, _ in pieces]
        wav = stitch(wavs, boundaries, self.sample_rate, s)
        if s.verbose:
            print(f"done: {len(wav) / self.sample_rate:.1f}s audio in {time.time() - t_start:.1f}s")
        return wav, self.sample_rate

    def synthesize_to_file(self, text: str, path: str, speaker="male",
                           settings: Optional[SynthSettings] = None) -> str:
        import soundfile as sf
        wav, sr = self.synthesize(text, speaker, settings)
        sf.write(path, wav, sr, subtype="PCM_16")
        return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--speaker", default="male", help="male or female")
    ap.add_argument("--text", default=None)
    ap.add_argument("--text_file", default=None, help="UTF-8 text file")
    ap.add_argument("--out", default="svar_out.wav")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=65)
    ap.add_argument("--context", action="store_true", help="experimental: previous chunk as context")
    ap.add_argument("--debug_dir", default=None, help="save raw untrimmed chunk wavs here and print their edge silence")
    ap.add_argument("--check", action="store_true", help="verify the t2s cache against the full model, then exit")
    ap.add_argument("--encodec_path", default="facebook/encodec_24khz")
    ap.add_argument("--t2s_src", default=None, help="folder with the t2s python files (default: t2s_src next to this file)")
    ap.add_argument("--dit_src", default=None, help="folder with the DiT python files (default: dit_src next to this file)")
    args = ap.parse_args()

    text = None
    if args.text_file:
        with open(args.text_file, encoding="utf-8") as f:
            text = f.read()
    elif args.text:
        text = args.text

    tts = Svar.from_pretrained(args.model_dir, encodec_path=args.encodec_path,
                               t2s_src=args.t2s_src, dit_src=args.dit_src)
    if args.check:
        sample = text 
        clean, _ = normalize_lenient(sample)
        ok = tts.t2s.self_check(tts.sp, clean, tts._speaker_id(args.speaker))
        raise SystemExit(0 if ok else 1)
    if text is None:
        raise SystemExit("give --text or --text_file")
    s = SynthSettings(seed=args.seed, dit_steps=args.steps, use_context=args.context, verbose=True,
                      debug_dir=args.debug_dir)
    path = tts.synthesize_to_file(text, args.out, args.speaker, s)
    print("saved", path)


if __name__ == "__main__":
    main()
