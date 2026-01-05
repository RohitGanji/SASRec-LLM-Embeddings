# fit_llm_proj.py
import argparse
import os
import numpy as np
import torch

from model import SASRec
from utils import data_partition

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ckpt", required=True, help="warm-only trained SASRec .pth (zero-shot split)")
    ap.add_argument("--maxlen", type=int, default=50)
    ap.add_argument("--hidden_units", type=int, default=50)
    ap.add_argument("--num_blocks", type=int, default=2)
    ap.add_argument("--num_heads", type=int, default=1)
    ap.add_argument("--dropout_rate", type=float, default=0.2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--llm_emb_path", required=True)
    ap.add_argument("--llm_dim", type=int, default=384)

    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    ap.add_argument("--out", default=None, help="output .pth (defaults to <ckpt>.llmproj.pth)")
    args = ap.parse_args()

    # Build dataset split (your current utils.py data_partition determines warm/cold)
    user_train, user_valid, user_test, usernum, itemnum = data_partition(args.dataset)

    # Load LLM embeddings
    llm_np = np.load(args.llm_emb_path)  # [itemnum+1, llm_dim]
    assert llm_np.shape == (itemnum + 1, args.llm_dim), f"got {llm_np.shape}, expected {(itemnum+1, args.llm_dim)}"
    llm_emb = torch.from_numpy(llm_np).float()

    # Build warm mask from TRAIN counts (warm = appears in train at least once)
    train_item_count = np.zeros(itemnum + 1, dtype=np.int32)
    for u in user_train:
        for it in user_train[u]:
            train_item_count[it] += 1
    warm_ids = np.where(train_item_count > 0)[0]
    warm_ids = warm_ids[warm_ids != 0]  # exclude padding id 0

    # Instantiate model (must match checkpoint architecture)
    class A: pass
    a = A()
    a.device = args.device
    a.norm_first = True  # you're using preLN; doesn't affect item_emb/llm_proj shapes
    a.hidden_units = args.hidden_units
    a.maxlen = args.maxlen
    a.num_blocks = args.num_blocks
    a.num_heads = args.num_heads
    a.dropout_rate = args.dropout_rate
    a.use_llm = True
    a.llm_dim = args.llm_dim

    model = SASRec(usernum, itemnum, a).to(args.device)

    # Load warm-only SASRec weights (strict=False ok because llm_proj not trained there)
    state = torch.load(args.ckpt, map_location=torch.device(args.device))
    model.load_state_dict(state, strict=False)

    # Freeze everything except llm_proj
    for p in model.parameters():
        p.requires_grad = False
    for p in model.llm_proj.parameters():
        p.requires_grad = True

    # Targets: learned collaborative embeddings for warm items
    with torch.no_grad():
        target_item_emb = model.item_emb.weight.detach().cpu()  # [itemnum+1, hidden]

    X = llm_emb[warm_ids]                      # [N, llm_dim]
    Y = target_item_emb[warm_ids]              # [N, hidden_units]

    # Train llm_proj to map X -> Y
    model.train()
    opt = torch.optim.AdamW(model.llm_proj.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.MSELoss()

    X = X.to(args.device)
    Y = Y.to(args.device)

    n = X.shape[0]
    bs = min(args.batch_size, n)

    for ep in range(1, args.epochs + 1):
        perm = torch.randperm(n, device=args.device)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            pred = model.llm_proj(X[idx])
            loss = loss_fn(pred, Y[idx])

            opt.zero_grad()
            loss.backward()
            opt.step()

            tot += loss.item() * idx.numel()

        if ep % 20 == 0 or ep == 1:
            print(f"epoch {ep:4d} | mse {tot / n:.6f} | warm_items {n}")

    # Save updated checkpoint (includes trained llm_proj + original SASRec weights)
    out = args.out or (args.ckpt + ".llmproj.pth")
    torch.save(model.state_dict(), out)
    print("saved:", out)

if __name__ == "__main__":
    main()
