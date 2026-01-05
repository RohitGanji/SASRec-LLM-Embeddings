import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


def build_item_text(df: pd.DataFrame) -> list[str]:
    # expects columns like: movieId, title, genres (based on your items.parquet)
    titles = df.get("title", pd.Series([""] * len(df))).fillna("")
    genres = df.get("genres", pd.Series([""] * len(df))).fillna("")
    # simple, effective text format
    return [f"Title: {t}. Genres: {g}." for t, g in zip(titles, genres)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ml-1m")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--out", default=None, help="output .npy path")
    args = ap.parse_args()

    artifacts = Path("artifacts") / args.dataset
    items_path = artifacts / "items.parquet"
    id_maps_path = artifacts / "id_maps.json"

    if args.out is None:
        args.out = str(artifacts / "llm_item_emb.npy")

    # Load metadata + id mapping
    items = pd.read_parquet(items_path)
    with open(id_maps_path, "r") as f:
        maps = json.load(f)

    # raw/original item id (movieId) -> internal/dense item id
    raw_to_internal = maps["item_original_to_dense"]
    raw_to_internal = {int(k): int(v) for k, v in raw_to_internal.items()}


    # Determine itemnum
    itemnum = max(raw_to_internal.values())

    # Build text per internal item_id (1..itemnum)
    text_by_internal = [""] * (itemnum + 1)  # index 0 unused
    texts = build_item_text(items)

    # Map each row to internal id and store text
    if "movieId" not in items.columns:
        raise KeyError(f"items.parquet must contain movieId column. Found: {items.columns.tolist()}")

    missing = 0
    for movie_id, text in zip(items["movieId"].tolist(), texts):
        movie_id = int(movie_id)
        if movie_id not in raw_to_internal:
            missing += 1
            continue
        internal_id = raw_to_internal[movie_id]
        text_by_internal[internal_id] = text

    print(f"Loaded items: {len(items)} | mapped to internal ids: {itemnum} | missing rows: {missing}")
    empty = sum(1 for i in range(1, itemnum + 1) if text_by_internal[i].strip() == "")
    print(f"Internal items with empty text: {empty}/{itemnum}")

    # Embed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.model, device=device)

    # If any empty texts, give them a placeholder (prevents weird embeddings)
    for i in range(1, itemnum + 1):
        if text_by_internal[i].strip() == "":
            text_by_internal[i] = "Title: . Genres: ."

    emb = model.encode(
        text_by_internal[1:],
        batch_size=args.batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,  # good default for dot-product similarity
    )

    # Save with padding row at 0 so index aligns with internal item_id
    out = np.zeros((itemnum + 1, emb.shape[1]), dtype=np.float32)
    out[1:] = emb.astype(np.float32)

    np.save(args.out, out)
    print(f"Saved embeddings: {out.shape} -> {args.out}")


if __name__ == "__main__":
    main()
