# tttt hyperparameter sweep: work instructions

Task: run a Bayesian hyperparameter sweep for **both** the vanilla and the
pairwise-attention arms on the four-top (tttt) dataset, and report which
configuration wins each arm.

## Why this matters

Every tttt number we currently quote came from hyperparameters that were tuned
on **ttbar**, not on tttt. The 15M A/B used `hidden_dim: 64`,
`learning_rate: 7.3e-4`, `dropout: 0.3`, `batch_size: 512`, inherited from the
ttbar pairwise sweep winner (`gs8pex8v`). Pairwise beat vanilla by 13 points in
FR accuracy on that setup.

The obvious objection is that the vanilla arm was simply undertuned, since it
was handed a config selected for a different process on a different
architecture. This sweep exists to answer that objection. Either the gap
survives independent tuning of both arms, or it does not, and both outcomes are
publishable internally.

There is real reason to think tttt wants a different architecture. The original
SPANet repo ships a dedicated four-top config
(https://github.com/Alexanders101/SPANet/blob/master/options_files/tttt.json)
that is much wider and shifts depth out of the shared encoder into the
per-particle branches:

| | ttbar config | official tttt config |
|---|---|---|
| `hidden_dim` | 64 | **128** |
| `num_attention_heads` | 4 | **8** |
| `num_encoder_layers` | 6 | **2** |
| `num_branch_encoder_layers` | 3 | **7** |

Our sweep space must be able to reach that region, or we will not have tested
the hypothesis.

## What already exists (reuse it, do not rewrite it)

The ttbar sweep machinery works and is already Bayesian. Copy it:

```
kube/sweep_fixed_dp/run_sweep_fixed_dp.py    # builds a config per trial, calls spanet.train
kube/sweep_fixed_dp/sweep_vanilla.yaml       # method: bayes + hyperband
kube/sweep_fixed_dp/sweep_blocks.yaml        # identical but for the arm name
```

Both use `method: bayes` with `early_terminate: hyperband, min_iter: 15` and
maximise `validation_average_jet_accuracy`.

Data and event file for tttt:

```
training_file    /data/spatop/tttt_15M/h5/tttt_training.h5      (2.16 GB, 1.63M events)
testing_file     /data/spatop/tttt_15M/h5/tttt_testing.h5       (540 MB)
event_info_file  /data/spatop/event_files/tttt/tttt_hadronic.yaml
```

Reference configs from the existing A/B, useful as a starting point:

```
/data/spatop/tttt_15M/configs/tttt15m_vanilla.json
/data/spatop/tttt_15M/configs/tttt15m_blocks.json
```

## Cluster rules that will bite you

These are not style preferences. Each one has already cost real time.

1. **A100 nodes use a typed resource.** Request `nvidia.com/a100`, NOT
   `nvidia.com/gpu`. A100 nodes advertise `nvidia.com/gpu: 0`, so a generic GPU
   request with A100 affinity sits `Pending` forever and never errors. This
   exact mistake burned four hours recently.
2. **Limit/request ratios must stay under 1.2** for both cpu and memory, or the
   admission controller warns and may deny. Example that passes: requests
   cpu 14 / 56Gi, limits cpu 16 / 64Gi.
3. **The a100 quota is 4** for the whole namespace, shared with everyone. Do not
   run more than 2 concurrent agents per arm without asking.
4. **Prefix every job name with your initials.** `dp-` is Daniel, `ts-` is
   Thomas. Pick your own and use it consistently.
5. **Never delete, kill, or edit a job that is not yours.** If something of
   yours will not schedule, ask rather than making room.
6. **Set `num_dataloader_workers: 0`.** Fork-based workers deadlock at the epoch
   boundary on this cluster. The dataset fits in RAM anyway.
7. **Use your own W&B account and API key**, stored as your own k8s secret. Do
   not reuse the `dp-wandb` secret.

## The cost problem, and how to handle it

A single full-data tttt trial is expensive: the 15M A/B took about **15 hours
for pairwise** and **10 hours for vanilla**, at 15 epochs on 1.63M events. That
is roughly one hour per epoch. Forty trials at that cost is 400+ GPU-hours,
which is not acceptable on a shared quota.

So sweep on a **subsample**, then validate the winners at full data:

- Sweep with `dataset_limit: 0.2` (about 325k events) and `epochs: 10`, with
  hyperband `min_iter: 4`. That is roughly 12 minutes per epoch, so about two
  hours per full-length trial, and hyperband kills the weak ones earlier.
- Budget about 20 trials per arm. With 2 concurrent agents per arm, expect
  roughly half a day of wall time per arm.

**Read this carefully, it is the subtle part.** Sweeping on a subsample assumes
the ranking of configurations is preserved when you scale the data up. Our own
ttbar result shows that assumption can fail: pairwise led vanilla by 1.0 points
at 864k events and *lost* to it by 5.0 points at 15.4M. Ranking is not
guaranteed to be stable across dataset size.

Therefore: **carry the top three configurations per arm into full-data
validation, not just the top one.** If the ranking flips between 20% and 100%,
that is a genuine result worth reporting, not a mistake.

## Step by step

### 0. Verify access before changing anything

```bash
kubectl get pods -n axol1tl
kubectl get resourcequota a100-limit -n axol1tl
```

If kubectl hangs rather than erroring, your OIDC token has expired and it is
silently waiting on a browser. Re-authenticate before going further.

### 1. Copy the sweep machinery

Make `kube/sweep_tttt_<yourinitials>/` holding your own copies of the runner
and the two sweep yamls. Do not edit the ttbar originals; other results depend
on them.

In your copy of the runner, change `event_info_file` and `training_file` to the
tttt paths above, and add `dataset_limit` as a swept-or-fixed parameter.

### 2. Define the search space

Start from the ttbar space and widen it so it can reach the official tttt
architecture:

```yaml
method: bayes
metric:
  name: validation_average_jet_accuracy
  goal: maximize
early_terminate:
  type: hyperband
  min_iter: 4
parameters:
  learning_rate:      {distribution: log_uniform_values, min: 0.0001, max: 0.002}
  hidden_dim:         {values: [64, 128, 256]}
  dropout:            {distribution: uniform, min: 0.0, max: 0.3}
  batch_size:         {values: [256, 512, 1024]}
  num_embedding_layers:     {values: [6, 10, 14]}
  num_encoder_layers:       {values: [2, 4, 6]}
  num_branch_encoder_layers: {values: [3, 5, 7]}
  num_attention_heads:      {values: [4, 8]}
```

The last three are new relative to the ttbar sweep and are the whole point:
they let the search reach the wide, branch-heavy region the official tttt
config occupies.

**Metric warning.** Use `validation_average_jet_accuracy`. Do **not** use
`validation_accuracy`: it is structurally NaN in our multi-topology setup, and
an earlier sweep generation optimised it by accident, which made those trials
effectively a random search.

### 3. Run the two arms as separate sweeps

The only difference between arms is the pairwise switch. Vanilla:

```json
{"use_pairwise_interactions": false}
```

Pairwise:

```json
{"use_pairwise_interactions": true,
 "pairwise_input_source": "all",
 "num_pairwise_features": 4,
 "pairwise_embedding_dim": 8}
```

Keep every other setting, the search space, the trial budget and the seed
policy identical between arms. If the arms differ in anything else, the
comparison is not like-for-like and the result will not survive review.

### 4. Create the sweeps and launch agents

```bash
wandb sweep kube/sweep_tttt_<init>/sweep_vanilla.yaml
wandb sweep kube/sweep_tttt_<init>/sweep_pairwise.yaml
```

Each prints a sweep ID. Launch agents as k8s Jobs modelled on
`kube/spatop-sweepagents-fixed-axol1tl.yml`, with the A100 rules above applied.

### 5. Monitor

```bash
kubectl get jobs -n axol1tl | grep <yourinitials>
kubectl logs -n axol1tl -l job-name=<your-job> --tail=50
```

Watch for trials that die instantly (usually OOM: lower `batch_size` or raise
memory) and for trials that plateau at exactly the same value (usually a config
that collapsed to predicting one class).

### 6. Validate the winners at full data

Take the **top three** configs per arm. Re-run each at `dataset_limit: 1.0`,
`epochs: 15`, matching the existing A/B exactly so the numbers are directly
comparable to the 15.4 / 28.4 FR figures already on record. Evaluate on
`/data/spatop/tttt_15M/h5/tttt_testing.h5`.

## Deliverables

1. A table of all trials per arm: config, best `val_avg_jet_acc`, epochs
   completed, parameter count. The existing
   `reports/plots_dp_fixed/sweep_summary.csv` shows the format.
2. The best configuration per arm, as a JSON config file.
3. Full-data validation numbers for the top three per arm, with per-topology
   accuracy (FR, SRqq, FB) and event purity.
4. A short note answering the actual question: **does the pairwise advantage on
   tttt survive independent tuning of both arms?** State the gap before and
   after tuning.
5. Flag explicitly whether the subsample ranking held at full data.

## Using Claude Code on this

It is genuinely useful for writing the yamls, parsing W&B results and making the
summary tables. Two rules:

- **Never let it launch or delete GPU jobs unreviewed.** Read every `kubectl
  apply` and every resource block yourself first. The typed-a100 mistake above
  was made by an AI assistant that knew the rule and failed to apply it.
- **Do not let it tell you the expected answer before you have measured it.**
  If you want an independent check of the physics, run
  `src/models/fully_resolved_baseline.py` (written by Thomas, no AI involvement)
  rather than anything generated for you.

## Ask before you do these

- Running more than 2 concurrent A100 agents.
- Deleting or modifying anything under `/data/spatop/` that you did not create.
- Anything touching a job whose name does not start with your initials.
