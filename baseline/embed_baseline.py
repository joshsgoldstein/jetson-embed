#!/usr/bin/env python3
"""Reference embeddings from the Python stack, for verifying the Rust server.

Why this exists: a port is only correct if it produces the same numbers as the
original. This script runs any sentence-transformers model exactly the way its
authors intended (their pooling, their normalisation, their prompt prefixes) and
writes the vectors to JSON. `jetson-embed verify` then embeds the same texts and
reports cosine similarity. A correct port scores > 0.999 on every text.

Usage:
    python embed_baseline.py --model BAAI/bge-base-en-v1.5
    python embed_baseline.py --model nomic-ai/nomic-embed-text-v1.5 --prefix "search_document: "
    python embed_baseline.py --model perplexity-ai/pplx-embed-v1-0.6b --trust-remote-code
"""
import argparse, json, pathlib, sys, time

TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Jetson AGX Orin has 64 GB of unified memory shared by CPU and GPU.",
    "Attention cost grows with the square of sequence length.",
    "¿Dónde está la biblioteca?",
    "a",
    "",  # empty string: models must not crash on it
    " ".join(["padding"] * 300),  # long input: exercises truncation
]

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prefix", default="", help="Prompt prefix some models need (nomic: 'search_document: ')")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--device", default=None, help="cuda | cpu | mps (default: auto)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    model = SentenceTransformer(args.model, trust_remote_code=args.trust_remote_code, device=args.device)
    load_s = time.time() - t0
    texts = [args.prefix + t for t in TEXTS]
    t0 = time.time()
    vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    embed_s = time.time() - t0

    out = {
        "model": args.model,
        "prefix": args.prefix,
        "dim": int(vecs.shape[1]),
        "max_seq_length": int(getattr(model, "max_seq_length", 0) or 0),
        "device": str(model.device),
        "load_s": round(load_s, 2),
        "embed_s": round(embed_s, 3),
        "texts": TEXTS,
        "embeddings": [[float(x) for x in v] for v in vecs],
    }
    path = pathlib.Path(args.out or f"reference/{args.model.replace('/', '__')}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    print(f"{args.model}: dim={out['dim']} max_seq={out['max_seq_length']} device={out['device']} "
          f"load={load_s:.1f}s embed={embed_s:.3f}s -> {path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
