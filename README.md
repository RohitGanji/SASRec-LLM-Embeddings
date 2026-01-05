# SASRec + LLM Item Embeddings for Zero-Shot Cold-Start Recommendation

This project extends **SASRec** to handle **item cold-start** using **LLM-derived item embeddings**, evaluated under a **strict zero-shot setting** (cold items never appear in training).

---

## Motivation
Sequential recommenders like SASRec cannot recommend unseen items because item embeddings are learned only from training data.  
We ask:

> **Can LLM item embeddings enable effective zero-shot item recommendation when properly aligned to the collaborative embedding space?**

---

## Method (High-Level)
1. Generate **LLM item embeddings** from item metadata.
2. Add a **linear projection head** mapping LLM embeddings → SASRec latent space.
3. **Freeze SASRec**, train only the projection using a **ranking-aware BCE loss**.
4. At inference:
   - Warm items → SASRec item embeddings  
   - Cold items → projected LLM embeddings
5. Evaluate **cold items only** (no leakage).

---

## Dataset & Split
- **Dataset:** MovieLens-1M  
- **Model:** SASRec (Pre-LayerNorm)  
- **Sequence length:** 50  
- **Cold-start split:**  
  - Cold items never appear in TRAIN  
  - Cold items appear only as TEST targets  
  - **1705 cold-test users**

---

## Results (5 Seeds, Mean ± Std)

### Zero-Shot Cold Items
| Method | HR@10 | NDCG@10 |
|------|------|--------|
| SASRec (baseline) | 0.0000 | 0.0000 |
| **SASRec + LLM (BCE-trained proj)** | **0.3742 ± 0.0046** | **0.1720 ± 0.0022** |

### Overall Test (Warm + Cold)
| Method | HR@10 | NDCG@10 |
|------|------|--------|
| **SASRec + LLM (BCE-trained proj)** | **0.6954 ± 0.0007** | **0.4629 ± 0.0003** |

**Observations**
- Cold-start recall improves from **0 → 0.37 HR@10**
- Very low variance across seeds
- No degradation of warm-item performance

---

## Reproducibility

### Train LLM Projection
```bash
python fit_llm_proj_bce.py \
  --dataset ml-1m \
  --ckpt ml-1m_zeroshot_preLN/SASRec.epoch=860.lr=0.001.layer=2.head=1.hidden=50.maxlen=50.pth \
  --llm_emb_path artifacts/ml-1m/llm_item_emb.npy \
  --llm_dim 384 \
  --epochs 50 \
  --steps_per_epoch 100 \
  --batch_size 2048 \
  --lr 1e-3
