# cold_stats.py
# Usage: python cold_stats.py --dataset ml-1m --k 5

import argparse
from collections import Counter

from utils import data_partition

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., ml-1m). Expects data/<dataset>.txt")
    parser.add_argument("--k", type=int, default=5, help="Cold threshold: item is cold if train_count <= k")
    args = parser.parse_args()

    user_train, user_valid, user_test, usernum, itemnum = data_partition(args.dataset)

    # 1) Count item occurrences in TRAIN ONLY
    train_item_counts = Counter()
    for u, items in user_train.items():
        train_item_counts.update(items)

    # Ensure all item ids 1..itemnum exist in dict (missing => 0)
    def train_count(i: int) -> int:
        return train_item_counts.get(i, 0)

    # 2) Cold items mask based on TRAIN counts
    k = args.k
    cold_items = [i for i in range(1, itemnum + 1) if train_count(i) <= k]

    # Distribution for counts 0..k, plus a catch-all bucket
    dist_0_to_k = {c: 0 for c in range(0, k + 1)}
    gt_k = 0
    for i in range(1, itemnum + 1):
        c = train_count(i)
        if c <= k:
            dist_0_to_k[c] += 1
        else:
            gt_k += 1

    # 3) Cold test cases (based on whether the TEST ground-truth item is cold in TRAIN)
    total_test_cases = 0
    cold_test_cases = 0
    # also useful: cold breakdown by train_count(test_item)
    cold_test_count_breakdown = {c: 0 for c in range(0, k + 1)}

    for u in range(1, usernum + 1):
        if len(user_test.get(u, [])) < 1:
            continue
        total_test_cases += 1
        test_item = user_test[u][0]
        c = train_count(test_item)
        if c <= k:
            cold_test_cases += 1
            cold_test_count_breakdown[c] += 1

    # 4) Print results
    total_items = itemnum
    num_cold_items = len(cold_items)

    print(f"\nDataset: {args.dataset}")
    print(f"Users: {usernum}, Items: {itemnum}")
    print(f"Cold threshold K: {k}  (cold if train_count <= K)\n")

    print("TRAIN item frequency distribution (over ALL items):")
    for c in range(0, k + 1):
        print(f"  items with train_count == {c}: {dist_0_to_k[c]}")
    print(f"  items with train_count  > {k}: {gt_k}")

    pct_cold_items = (num_cold_items / total_items * 100.0) if total_items else 0.0
    print(f"\nCold items: {num_cold_items}/{total_items}  ({pct_cold_items:.2f}%)")

    if total_test_cases > 0:
        pct_cold_tests = cold_test_cases / total_test_cases * 100.0
    else:
        pct_cold_tests = 0.0

    print(f"\nTest cases total (users with test): {total_test_cases}")
    print(f"Cold test cases (test_item cold by TRAIN count): {cold_test_cases}  ({pct_cold_tests:.2f}%)")

    print("\nCold test cases breakdown by train_count(test_item):")
    for c in range(0, k + 1):
        print(f"  train_count == {c}: {cold_test_count_breakdown[c]}")

    print()

if __name__ == "__main__":
    main()
