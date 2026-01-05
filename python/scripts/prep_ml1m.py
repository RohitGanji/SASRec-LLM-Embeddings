from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def read_ml1m(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    ML-1M files are '::' separated and historically contain non-utf8 chars.
    Use latin-1 to avoid decode errors.
    """
    raw_dir = raw_dir.resolve()

    ratings_path = raw_dir / "ratings.dat"
    movies_path = raw_dir / "movies.dat"
    users_path = raw_dir / "users.dat"

    if not ratings_path.exists():
        raise FileNotFoundError(f"Missing: {ratings_path}")
    if not movies_path.exists():
        raise FileNotFoundError(f"Missing: {movies_path}")
    if not users_path.exists():
        raise FileNotFoundError(f"Missing: {users_path}")

    ratings = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        header=None,
        names=["userId", "movieId", "rating", "timestamp"],
        encoding="latin-1",
    )

    movies = pd.read_csv(
        movies_path,
        sep="::",
        engine="python",
        header=None,
        names=["movieId", "title", "genres"],
        encoding="latin-1",
    )

    users = pd.read_csv(
        users_path,
        sep="::",
        engine="python",
        header=None,
        names=["userId", "gender", "age", "occupation", "zip"],
        encoding="latin-1",
    )

    # Ensure dtypes
    ratings["userId"] = ratings["userId"].astype("int32")
    ratings["movieId"] = ratings["movieId"].astype("int32")
    ratings["timestamp"] = ratings["timestamp"].astype("int64")

    movies["movieId"] = movies["movieId"].astype("int32")
    users["userId"] = users["userId"].astype("int32")

    return ratings, movies, users


def make_dense_maps(values: pd.Series) -> dict[int, int]:
    """
    Deterministic mapping:
      - sort unique original IDs ascending
      - assign dense ids 1..N
    """
    uniq = sorted(values.unique().tolist())
    return {int(orig): int(i + 1) for i, orig in enumerate(uniq)}


def write_sasrec_txt(
    ratings: pd.DataFrame,
    user_map: dict[int, int],
    item_map: dict[int, int],
    out_txt: Path,
    out_txt_with_ts: Path,
) -> None:
    """
    SASRec expects: each line "user_id item_id"
    and interactions are appended to User[u] in file order, so file MUST be time-sorted per user.
    """
    df = ratings[["userId", "movieId", "timestamp"]].copy()
    df["user_id"] = df["userId"].map(user_map).astype("int32")
    df["item_id"] = df["movieId"].map(item_map).astype("int32")

    # Stable order: per-user chronological
    df = df.sort_values(["user_id", "timestamp", "item_id"], ascending=[True, True, True])

    out_txt.parent.mkdir(parents=True, exist_ok=True)

    # Write with timestamp for sanity checks
    with out_txt_with_ts.open("w") as f:
        for u, i, ts in zip(df["user_id"], df["item_id"], df["timestamp"]):
            f.write(f"{int(u)} {int(i)} {int(ts)}\n")

    # Write minimal SASRec format
    with out_txt.open("w") as f:
        for u, i in zip(df["user_id"], df["item_id"]):
            f.write(f"{int(u)} {int(i)}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Reading raw ML-1M from: {raw_dir.resolve()}")
    ratings, movies, users = read_ml1m(raw_dir)
    print(f"[INFO] ratings: {len(ratings):,} rows | movies: {len(movies):,} | users: {len(users):,}")

    # Only keep movies that appear in ratings (consistent universe)
    movies_in_ratings = movies[movies["movieId"].isin(ratings["movieId"].unique())].copy()

    # Dense mappings (deterministic)
    user_map = make_dense_maps(ratings["userId"])
    item_map = make_dense_maps(ratings["movieId"])

    # Inverse maps
    inv_user_map = {v: k for k, v in user_map.items()}
    inv_item_map = {v: k for k, v in item_map.items()}

    # Write mapping json (clean, explicit)
    id_maps = {
        "user_original_to_dense": user_map,
        "item_original_to_dense": item_map,
        "user_dense_to_original": inv_user_map,
        "item_dense_to_original": inv_item_map,
    }
    maps_path = out_dir / "id_maps.json"
    maps_path.write_text(json.dumps(id_maps, indent=2, sort_keys=True))
    print(f"[INFO] Wrote: {maps_path}")

    # Write SASRec txt
    out_txt = out_dir / "ml-1m.txt"
    out_txt_with_ts = out_dir / "ml-1m_with_ts.txt"
    write_sasrec_txt(ratings, user_map, item_map, out_txt, out_txt_with_ts)
    print(f"[INFO] Wrote: {out_txt}")
    print(f"[INFO] Wrote: {out_txt_with_ts}")

    # Artifacts for later (metadata joins / cold start)
    interactions = ratings.copy()
    interactions["user_id"] = interactions["userId"].map(user_map).astype("int32")
    interactions["item_id"] = interactions["movieId"].map(item_map).astype("int32")
    interactions = interactions.sort_values(["user_id", "timestamp", "item_id"], ascending=True)

    items = movies_in_ratings.copy()
    items["item_id"] = items["movieId"].map(item_map).astype("int32")
    items = items.sort_values(["item_id"])

    users_df = users.copy()
    users_df["user_id"] = users_df["userId"].map(user_map).astype("int32")
    users_df = users_df.dropna(subset=["user_id"]).copy()
    users_df["user_id"] = users_df["user_id"].astype("int32")
    users_df = users_df.sort_values(["user_id"])

    interactions.to_parquet(out_dir / "interactions.parquet", index=False)
    items.to_parquet(out_dir / "items.parquet", index=False)
    users_df.to_parquet(out_dir / "users.parquet", index=False)

    print("[DONE] ml-1m artifacts ready.")
    print("       Next: copy/symlink ml-1m.txt into python/data as data/ml-1m.txt")


if __name__ == "__main__":
    main()