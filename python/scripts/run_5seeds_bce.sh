#!/usr/bin/env bash
set -euo pipefail

# ---- CONFIG ----
DATASET="ml-1m"
CKPT="ml-1m_zeroshot_preLN/SASRec.epoch=860.lr=0.001.layer=2.head=1.hidden=50.maxlen=50.pth"
LLM_EMB="artifacts/ml-1m/llm_item_emb.npy"
LLM_DIM=384
MAXLEN=50

# Projection training params
EPOCHS=50
STEPS_PER_EPOCH=100
LR=1e-3
BS=2048

# Early stopping / logging (fit_llm_proj_bce.py)
PATIENCE=5
MIN_DELTA=1e-4
LOG_EVERY=10
# If >0, caps TOTAL optimizer steps and can stop early regardless of epochs.
# Example: 5000 total steps ~= (EPOCHS * STEPS_PER_EPOCH) if those multiply to 5000.
MAX_STEPS=0

# Use best checkpoint (lowest avg loss) for eval
USE_BEST=1

TRAIN_DIR="eval_llmproj_bce"
OUTDIR="results/llmproj_bce_5seeds"
JSONL="${OUTDIR}/runs.jsonl"

SEEDS=(0 1 2 3 4)
# ---------------

mkdir -p "${OUTDIR}"
rm -f "${JSONL}"

echo "Writing results to: ${JSONL}"
echo "Outdir: ${OUTDIR}"
echo "Seeds: ${SEEDS[*]}"
echo "Config: epochs=${EPOCHS} steps_per_epoch=${STEPS_PER_EPOCH} bs=${BS} lr=${LR} patience=${PATIENCE} min_delta=${MIN_DELTA} log_every=${LOG_EVERY} max_steps=${MAX_STEPS} use_best=${USE_BEST}"

for SEED in "${SEEDS[@]}"; do
  echo "=============================="
  echo "SEED ${SEED}: TRAIN llm_proj"
  echo "=============================="

  OUT_PTH="${OUTDIR}/seed${SEED}.pth"
  TRAIN_LOG="${OUTDIR}/seed${SEED}.train.txt"

  TRAIN_ARGS=(
    --dataset "${DATASET}"
    --ckpt "${CKPT}"
    --llm_emb_path "${LLM_EMB}"
    --llm_dim "${LLM_DIM}"
    --maxlen "${MAXLEN}"
    --epochs "${EPOCHS}"
    --steps_per_epoch "${STEPS_PER_EPOCH}"
    --lr "${LR}"
    --batch_size "${BS}"
    --seed "${SEED}"
    --out "${OUT_PTH}"
    --patience "${PATIENCE}"
    --min_delta "${MIN_DELTA}"
    --log_every "${LOG_EVERY}"
  )

  if [[ "${MAX_STEPS}" -gt 0 ]]; then
    TRAIN_ARGS+=( --max_steps "${MAX_STEPS}" )
  fi

  # unbuffered + log to file
  python -u fit_llm_proj_bce.py "${TRAIN_ARGS[@]}" | tee "${TRAIN_LOG}"

  # Require output to exist (fail fast)
  if [[ ! -s "${OUT_PTH}" ]]; then
    echo "ERROR: training did not create ${OUT_PTH}"
    echo "Check: ${TRAIN_LOG}"
    exit 1
  fi

  # Decide which checkpoint to evaluate
  EVAL_PTH="${OUT_PTH}"
  if [[ "${USE_BEST}" -eq 1 && -s "${OUT_PTH}.best" ]]; then
    EVAL_PTH="${OUT_PTH}.best"
  fi

  echo "=============================="
  echo "SEED ${SEED}: EVAL main.py (ckpt=${EVAL_PTH})"
  echo "=============================="

  LOGTXT="${OUTDIR}/seed${SEED}.eval.txt"
  python main.py \
    --dataset "${DATASET}" \
    --train_dir "${TRAIN_DIR}" \
    --inference_only true \
    --state_dict_path "${EVAL_PTH}" \
    --maxlen "${MAXLEN}" \
    --norm_first \
    --use_llm \
    --llm_emb_path "${LLM_EMB}" \
    --llm_dim "${LLM_DIM}" | tee "${LOGTXT}"

  OVERALL_LINE="$(grep -E 'test \(NDCG@10:' "${LOGTXT}" | tail -n 1 || true)"
  COLD_LINE="$(grep -E 'cold_test k=0 \(NDCG@10:' "${LOGTXT}" | tail -n 1 || true)"

  OVERALL_NDCG="$(echo "${OVERALL_LINE}" | sed -n 's/.*NDCG@10: \([0-9.]*\), HR@10: \([0-9.]*\).*/\1/p')"
  OVERALL_HR="$(echo "${OVERALL_LINE}" | sed -n 's/.*NDCG@10: \([0-9.]*\), HR@10: \([0-9.]*\).*/\2/p')"

  COLD_NDCG="$(echo "${COLD_LINE}" | sed -n 's/.*NDCG@10: \([0-9.]*\), HR@10: \([0-9.]*\), users: \([0-9]*\).*/\1/p')"
  COLD_HR="$(echo "${COLD_LINE}" | sed -n 's/.*NDCG@10: \([0-9.]*\), HR@10: \([0-9.]*\), users: \([0-9]*\).*/\2/p')"
  COLD_USERS="$(echo "${COLD_LINE}" | sed -n 's/.*NDCG@10: \([0-9.]*\), HR@10: \([0-9.]*\), users: \([0-9]*\).*/\3/p')"

  if [[ -z "${OVERALL_NDCG}" || -z "${COLD_NDCG}" ]]; then
    echo "WARNING: failed to parse metrics for seed ${SEED}. Check ${LOGTXT}"
  fi

  python - <<PY
import json, time
row = {
  "ts": time.time(),
  "seed": ${SEED},
  "dataset": "${DATASET}",
  "base_ckpt": "${CKPT}",
  "llm_emb_path": "${LLM_EMB}",
  "llm_dim": ${LLM_DIM},
  "maxlen": ${MAXLEN},
  "epochs": ${EPOCHS},
  "steps_per_epoch": ${STEPS_PER_EPOCH},
  "max_steps": ${MAX_STEPS},
  "lr": ${LR},
  "batch_size": ${BS},
  "patience": ${PATIENCE},
  "min_delta": ${MIN_DELTA},
  "log_every": ${LOG_EVERY},
  "state_dict_path": "${EVAL_PTH}",
  "state_dict_path_raw": "${OUT_PTH}",
  "state_dict_path_best": "${OUT_PTH}.best",
  "used_best": bool(${USE_BEST}) and ("${EVAL_PTH}" == "${OUT_PTH}.best"),
  "overall_ndcg10": float("${OVERALL_NDCG}") if "${OVERALL_NDCG}" else None,
  "overall_hr10": float("${OVERALL_HR}") if "${OVERALL_HR}" else None,
  "cold_ndcg10": float("${COLD_NDCG}") if "${COLD_NDCG}" else None,
  "cold_hr10": float("${COLD_HR}") if "${COLD_HR}" else None,
  "cold_users": int("${COLD_USERS}") if "${COLD_USERS}" else None,
}
with open("${JSONL}", "a") as f:
  f.write(json.dumps(row) + "\\n")
print("logged:", row)
PY

done

echo "=============================="
echo "DONE. Results JSONL: ${JSONL}"
echo "Next: python scripts/summarize_jsonl.py --path ${JSONL}"