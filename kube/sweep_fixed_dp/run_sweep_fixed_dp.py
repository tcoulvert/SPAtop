#!/usr/bin/env python3
"""W&B sweep wrapper for SPAtop on the FIXED dataset (dp, 2026-07-12).

Adapted from Ellison's /data/spatop/sweep/run_sweep.py. Protocol-identical
except:
  - training_file -> tpm70_wpm30_FB350_fixed (Thomas's corrected data)
  - num_dataloader_workers 8 -> 0   (fork-worker deadlock fix)
  - epochs 100 -> 50 (+ hyperband early termination in the sweep config)
  - logs to node-local /scratch (agent job rsyncs to PVC) -- cephfs stall fix
  - PAIRWISE_BLOCKS=1 env enables block-structured pairwise attention
    (dprim7/SPANet@feat/pairwise-attention)
  - passes WANDB_ENTITY through to the training subprocess: the sweep lives in
    the escheuller-uc-san-diego TEAM entity; without this the resumed run
    would land in the agent user's personal entity.
"""
import json, os, subprocess, sys
import wandb

BASE_CONFIG = {
    "num_encoder_layers": 4,
    "num_branch_embedding_layers": 3,
    "num_branch_encoder_layers": 3,
    "num_jet_embedding_layers": 0,
    "num_jet_encoder_layers": 2,
    "num_detector_layers": 2,
    "num_regression_layers": 3,
    "num_classification_layers": 3,
    "split_symmetric_attention": True,
    "num_attention_heads": 8,
    "transformer_activation": "gelu",
    "linear_block_type": "GRU",
    "transformer_type": "Gated",
    "linear_activation": "gelu",
    "normalization": "LayerNorm",
    "masking": "Filling",
    "skip_connections": True,
    "initial_embedding_skip_connections": True,
    "event_info_file": "/data/spatop/event_files/v11/tt_hadronic_v7_full.yaml",
    "training_file": "/data/spatop/tpm70_wpm30_FB350_fixed/all_merged.h5",
    "normalize_features": True,
    "limit_to_num_jets": 0,
    "balance_jets": False,
    "partial_events": True,
    "balance_particles": False,
    "dataset_limit": 1.0,
    "train_validation_split": 0.95,
    "num_dataloader_workers": 0,
    "mask_sequence_vectors": True,
    "combine_pair_loss": "min",
    "optimizer": "AdamW",
    "focal_gamma": 0.0,
    "learning_rate_cycles": 1,
    "learning_rate_warmup_epochs": 1.0,
    "assignment_loss_scale": 1.0,
    "detection_loss_scale": 1.0,
    "kl_loss_scale": 0.0,
    "regression_loss_scale": 0.0,
    "classification_loss_scale": 0.0,
    "l2_penalty": 0.0002,
    "gradient_clip": 10.0,
    "epochs": 50,
    "num_gpu": 1,
    "verbose_output": True,
}

PAIRWISE_BLOCKS = os.environ.get("PAIRWISE_BLOCKS", "0") == "1"
if PAIRWISE_BLOCKS:
    BASE_CONFIG.update({
        "use_pairwise_interactions": True,
        "pairwise_input_source": "all",
        "num_pairwise_features": 4,
        "pairwise_embedding_dim": 8,
        "pairwise_block_embeddings": True,
    })

run = wandb.init()  # bare: inherits the sweep's entity/project (team!)
config = dict(wandb.config)
run_id = run.id
run_project = run.project
run_entity = run.entity

hidden_dim = config.get("hidden_dim", 128)
full_config = {
    **BASE_CONFIG,
    **config,
    "hidden_dim": hidden_dim,
    "transformer_dim": hidden_dim,
    "initial_embedding_dim": hidden_dim,
    "position_embedding_dim": hidden_dim,
}

os.makedirs("/data/spatop/options_files/sweep_fixed_dp", exist_ok=True)
os.makedirs("/scratch/logs", exist_ok=True)

config_path = f"/data/spatop/options_files/sweep_fixed_dp/config_{run_id}.json"
with open(config_path, "w") as f:
    json.dump(full_config, f, indent=2)

variant = "blocks" if PAIRWISE_BLOCKS else "vanilla"
run_name = f"dpfx_{variant}_{run_id}"
log_dir = "/scratch/logs"

print(f"=== Trial {run.name} ({variant}) | swept: {json.dumps(config, indent=2)}")

# Release the wrapper's hold; SPANet's WandbLogger resumes the SAME run via
# WANDB_RUN_ID so metrics stream live (single active client at a time).
wandb.finish()

proc = subprocess.run(
    ["python", "-m", "spanet.train", "-of", config_path, "-l", log_dir, "-n", run_name],
    env={
        **os.environ,
        "WANDB_ENTITY": run_entity,
        "WANDB_PROJECT": run_project,
        "WANDB_RUN_ID": run_id,
        "WANDB_RESUME": "allow",
    },
)

if proc.returncode != 0:
    print(f"Training failed (exit code {proc.returncode})")
    sys.exit(1)
