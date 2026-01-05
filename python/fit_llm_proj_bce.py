# fit_llm_proj_bce.py
import argparse
import random
import numpy as np
import torch

from model import SASRec
from utils import data_partition


def sample_batch(user_train, usernum, itemnum, batch_size, maxlen):
    """
    Same sampling logic as utils.sample_function(), but returns numpy arrays.
    Negatives are sampled from [1..itemnum] excluding user's train items (same as baseline).
    """
    uids = np.random.randint(1, usernum + 1, size=batch_size)

    seq = np.zeros((batch_size, maxlen), dtype=np.int32)
    pos = np.zeros((batch_size, maxlen), dtype=np.int32)
    neg = np.zeros((batch_size, maxlen), dtype=np.int32)

    for b, uid in enumerate(uids):
        # ensure user has at least 2 train interactions
        while len(user_train[uid]) <= 1:
            uid = np.random.randint(1, usernum + 1)
        uids[b] = uid

        nxt = user_train[uid][-1]
        idx = maxlen - 1
        ts = set(user_train[uid])

        for it in reversed(user_train[uid][:-1]):
            seq[b, idx] = it
            pos[b, idx] = nxt

            # negative sampling (same style as baseline)
            t = np.random.randint(1, itemnum + 1)
            while t in ts:
                t = np.random.randint(1, itemnum + 1)
            neg[b, idx] = t

            nxt = it
            idx -= 1
            if idx == -1:
                break

    return uids, seq, pos, neg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ckpt", required=True, help="trained SASRec checkpoint (zero-shot split)")
    ap.add_argument("--out", default=None, help="output path (default: <ckpt>.llmproj_bce.pth)")

    ap.add_argument("--maxlen", type=int, default=50)
    ap.add_argument("--hidden_units", type=int, default=50)
    ap.add_argument("--num_blocks", type=int, default=2)
    ap.add_argument("--num_heads", type=int, default=1)
    ap.add_argument("--dropout_rate", type=float, default=0.2)
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--llm_emb_path", required=True)
    ap.add_argument("--llm_dim", type=int, default=384)

    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps_per_epoch", type=int, default=100)

    # Logging / stopping
    ap.add_argument("--log_every", type=int, default=10, help="print step loss every N steps")
    ap.add_argument("--patience", type=int, default=5, help="early stop patience on epoch avg loss")
    ap.add_argument("--min_delta", type=float, default=1e-4, help="min improvement in avg loss to reset patience")
    ap.add_argument("--max_steps", type=int, default=0, help="if >0, stop after this many total optimizer steps")
    ap.add_argument("--save_best", action="store_true", default=True, help="save best (lowest avg loss) checkpoint")

    args = ap.parse_args()

    # Repro
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # dataset split (uses your current utils.py data_partition)
    user_train, user_valid, user_test, usernum, itemnum = data_partition(args.dataset)

    # load LLM embeddings
    llm_np = np.load(args.llm_emb_path)
    assert llm_np.shape == (itemnum + 1, args.llm_dim), f"llm_np.shape={llm_np.shape}, expected {(itemnum+1, args.llm_dim)}"
    llm_emb_t = torch.from_numpy(llm_np).float()

    # dummy cold mask buffer (not used in this script's training path, but model expects it)
    cold_mask_np = np.zeros(itemnum + 1, dtype=np.bool_)
    cold_mask_t = torch.from_numpy(cold_mask_np)

    # build args object matching model signature
    class A:
        pass

    a = A()
    a.device = args.device
    a.norm_first = True  # keep consistent with your preLN runs
    a.hidden_units = args.hidden_units
    a.maxlen = args.maxlen
    a.num_blocks = args.num_blocks
    a.num_heads = args.num_heads
    a.dropout_rate = args.dropout_rate
    a.use_llm = True
    a.llm_dim = args.llm_dim

    model = SASRec(usernum, itemnum, a).to(args.device)

    # move buffers to same device
    model.set_llm_buffers(llm_emb_t.to(args.device), cold_mask_t.to(args.device))

    # load checkpoint
    state = torch.load(args.ckpt, map_location=torch.device(args.device))
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("load_state_dict strict=False", flush=True)
    print("  missing keys:", missing, flush=True)
    print("  unexpected keys:", unexpected, flush=True)

    # freeze everything except llm_proj
    for p in model.parameters():
        p.requires_grad = False
    for p in model.llm_proj.parameters():
        p.requires_grad = True

    model.train()

    bce = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(
        model.llm_proj.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        weight_decay=1e-4
    )

    num_batch = args.steps_per_epoch

    # Early stopping state
    best_avg = float("inf")
    bad_epochs = 0

    # Total step cap
    total_steps = 0
    max_steps = args.max_steps if args.max_steps and args.max_steps > 0 else None

    out = args.out or (args.ckpt + ".llmproj_bce.pth")
    best_out = out + ".best"

    def save_state(path: str):
        # save full state_dict (minus buffers) so main.py can load it directly
        state_out = model.state_dict()
        state_out.pop("llm_item_emb", None)
        state_out.pop("cold_item_mask", None)
        torch.save(state_out, path)

    for ep in range(1, args.epochs + 1):
        tot_loss = 0.0

        for step in range(num_batch):
            u, seq, pos, neg = sample_batch(user_train, usernum, itemnum, args.batch_size, args.maxlen)

            pos_t = torch.from_numpy(pos).to(args.device)
            neg_t = torch.from_numpy(neg).to(args.device)

            # log_feats from warm SASRec pipeline; pass numpy because model.log2feats() expects it
            log_feats = model.log2feats(seq)

            # pos/neg embeddings always via LLM->proj so llm_proj gets gradients
            pos_embs = model.llm_proj(model.llm_item_emb[pos_t.long()])
            neg_embs = model.llm_proj(model.llm_item_emb[neg_t.long()])

            pos_logits = (log_feats * pos_embs).sum(dim=-1)
            neg_logits = (log_feats * neg_embs).sum(dim=-1)

            pos_labels = torch.ones_like(pos_logits, device=args.device)
            neg_labels = torch.zeros_like(neg_logits, device=args.device)

            opt.zero_grad()
            idx = torch.where(pos_t != 0)
            loss = bce(pos_logits[idx], pos_labels[idx]) + bce(neg_logits[idx], neg_labels[idx])
            loss.backward()
            opt.step()

            tot_loss += loss.item()
            total_steps += 1

            if args.log_every > 0 and (step % args.log_every == 0):
                print(f"epoch {ep} step {step}/{num_batch} loss {loss.item():.4f} total_steps={total_steps}", flush=True)

            if max_steps is not None and total_steps >= max_steps:
                avg = tot_loss / max(1, step + 1)
                print(f"Reached max_steps={max_steps}. Stopping. (epoch={ep}, step={step}, avg_loss={avg:.6f})", flush=True)
                # Save final + best logic before exit
                avg_epoch = tot_loss / max(1, step + 1)
                if args.save_best and (best_avg - avg_epoch > args.min_delta):
                    best_avg = avg_epoch
                    save_state(best_out)
                    print(f"saved best so far: {best_out} (avg_loss={best_avg:.6f})", flush=True)
                save_state(out)
                print("saved:", out, flush=True)
                return

        avg_loss = tot_loss / max(1, num_batch)
        print(f"epoch {ep:3d} | avg_loss {avg_loss:.6f}", flush=True)

        # Save best on epoch boundary
        if args.save_best and (best_avg - avg_loss > args.min_delta):
            best_avg = avg_loss
            bad_epochs = 0
            save_state(best_out)
            print(f"saved best so far: {best_out} (avg_loss={best_avg:.6f})", flush=True)
        else:
            bad_epochs += 1
            if args.patience > 0 and bad_epochs >= args.patience:
                print(f"Early stop at epoch {ep} (best_avg={best_avg:.6f}, last_avg={avg_loss:.6f})", flush=True)
                break

    # Save final
    save_state(out)
    print("saved:", out, flush=True)
    if args.save_best:
        print("best checkpoint:", best_out, flush=True)


if __name__ == "__main__":
    main()