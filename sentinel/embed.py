"""The Sentinel - the embedding half of the search substrate.

multilingual-e5-small, 384-dim, ONNX, CPU. It does NOT touch VRAM, which is
the point: a 4B model at 60k context already needs ~10.7 GB and does not fit
in 6 GB, so nothing about search may compete for that memory.

Chosen over bge-small-en-v1.5 because that one is English-only and would have
failed SILENTLY on a bilingual vault. Same 384 dimensions, so the swap costs
nothing structurally. Changing the model again invalidates every vector and
forces a one-off full re-embed.
"""

from __future__ import annotations

import struct
from pathlib import Path

DIM = 384


def to_blob(vec) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def from_blob(blob: bytes):
    return struct.unpack(f"<{len(blob) // 4}f", blob)


class E5Encoder:
    """Mean pooling over the last hidden state, then L2 normalisation.

    NOT the pooler output: e5 is trained with mean pooling, and taking the
    pooler would be the kind of mistake that produces plausible-looking
    vectors and quietly worse retrieval.

    The prefixes are CONSTRUCTOR ARGUMENTS and default to empty. Measured on
    three query pairs, `query:` / `passage:` gave lower scores and narrower
    margins with identical ranking - but three pairs settle nothing, so the
    re-test has to be a config change rather than a rewrite.
    """

    def __init__(self, model_dir: str | Path, query_prefix: str = "",
                 passage_prefix: str = "", max_len: int = 512):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        d = Path(model_dir)
        onnx = d / "onnx" / "model.onnx" if (d / "onnx" / "model.onnx").exists() \
            else d / "model.onnx"
        if not onnx.exists():
            raise FileNotFoundError(f"no model.onnx under {d}")
        tok = d / "tokenizer.json"
        if not tok.exists():
            raise FileNotFoundError(f"no tokenizer.json under {d}")

        self.tok = Tokenizer.from_file(str(tok))
        self.tok.enable_truncation(max_length=max_len)
        self.tok.enable_padding()
        self.sess = ort.InferenceSession(
            str(onnx), providers=["CPUExecutionProvider"])
        # Read the input names rather than assuming them: some exports want
        # token_type_ids and some do not, and feeding an input the graph does
        # not declare is a hard failure at run time.
        self.inputs = {i.name for i in self.sess.get_inputs()}
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix

    def encode(self, texts: list[str], kind: str = "passage"):
        import numpy as np

        prefix = self.query_prefix if kind == "query" else self.passage_prefix
        enc = self.tok.encode_batch([prefix + t for t in texts])
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)

        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        feed = {k: v for k, v in feed.items() if k in self.inputs}

        last_hidden = self.sess.run(None, feed)[0]
        m = mask[..., None].astype(last_hidden.dtype)
        pooled = (last_hidden * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
        norm = np.linalg.norm(pooled, axis=1, keepdims=True)
        return pooled / np.clip(norm, 1e-9, None)


def embed_pending(index, encoder, batch: int = 32) -> int:
    """Fill the vector store. Returns how many vectors were computed.

    Deduplicated by stripped hash, so identical chunks and re-inserted chunks
    whose markup changed cost nothing.
    """
    pending = index.pending_embed()
    if not pending:
        return 0
    done = 0
    for i in range(0, len(pending), batch):
        rows = pending[i:i + batch]
        vecs = encoder.encode([r["content"] for r in rows], kind="passage")
        index.db.executemany(
            "INSERT OR REPLACE INTO vectors (stripped_hash, vec) VALUES (?,?)",
            [(r["stripped_hash"], to_blob(v)) for r, v in zip(rows, vecs)])
        done += len(rows)
    index.db.commit()
    return done
