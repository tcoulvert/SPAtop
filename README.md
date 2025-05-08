# SPANet for up to 4 tops

## 1. Check out the GitHub repository
```bash
cd work
git clone https://github.com/tcoulvert/SPAtop/tree/topnet_dev
```

## 2. Install the Python venv (Ensure you have installed python 3.9 or greater)
```bash
python -m venv
pip install -e ./SPAtop
```
If for some reason the code does not run properly, it could be because packages may have changed and broken things. If that is the case, you can install your environment as follows:
```bash
python -m venv
pip install -r requirements.txt
```

## 3. Copy and convert the dataset(s)
Copy the Delphes ROOT TTree datasets from:
- LPC EOS: `/eos/uscms/store/user/tsievert/ttbar_hadronic/ttbar_hadronic_*.root`, or
- non-LPC EOS: `root://cmseos.fnal.gov//store/user/tsievert/ttbar_hadronic/ttbar_hadronic_*.root`

to the `data/delphes/v1/ttbar_hadronic` directory

Convert to training and testing HDF5 files.
```bash
python -m src.data.delphes.convert_to_h5 data/delphes/v1/ttbar_hadronic/sample_*.root --out-file data/delphes/v1/ttbar_hadronic_training.h5
python -m src.data.delphes.convert_to_h5 data/delphes/v1/ttbar_hadronic/sample_*.root --out-file data/delphes/v1/ttbar_hadronic_testing.h5
```

### !!! WARNING !!! : From this step on, this repo hasn't been updated, so don't expect things to work. When the repo is updated, this README will change to reflect that.

## 5. Run the SPANet training
Override options file with `--gpus 0` if no GPUs are available.
```bash
python -m spanet.train -of options_files/delphes/hhh_v2.json [--gpus 0]
```
# SPAtop Training via Kubernetes

## Prerequisites

Training via kubernetes on the cms-ml namespace requires the following:

* `kubectl` configured to target the `cms-ml` namespace
* PersistentVolumeClaim named `spatopvol` containing training data (already created)
* Docker image `gitlab-registry.nrp-nautilus.io/jmduarte/hhh:latest` (to be updated)

---

## Data Organization

Data should be placed under the PVC at:

```
spatopvol/data/delphes/v1/tt_training.h5
```
---

## Configuration Files

SPAtop training requires the following files, each located within the PVC:

| File              | Description              | Location                                  |
| ----------------- | ------------------------ | ----------------------------------------- |
| `tt_hadronic.yml` | Physics process settings | `/spatopvol/event_files/tt_hadronic.yml`  |
| `spatop_v1.json`  | SPANet model parameters  | `/spatopvol/options_files/spatop_v1.json` |

Additionally, the Kubernetes job manifest is required:

* **`spatop-job-train.yml`**: Defines the Job spec for launching the SPAtop training container. Place this file in your local working directory where you run `kubectl` commands.

---

## Launching Training

To start the SPAtop training job, apply the Kubernetes manifest:

```bash
kubectl apply -f spatop-job-train.yml -n cms-ml
```

This command creates a Kubernetes Job that spawns one or more pods to perform the training.

---

## Monitoring Jobs

1. **List Jobs and Pods**

   ```bash
   kubectl get jobs -n cms-ml
   kubectl get pods -l job-name=spatop-job-train -n cms-ml
   ```

2. **Describe a Pod**

   ```bash
   kubectl describe pod <pod-name> -n cms-ml
   ```

3. **Stream Logs**

   ```bash
   kubectl logs -f <pod-name> -n cms-ml
   ```

Repeat these commands to track pod status, resource usage, and training progress.

---

## Cleanup

Once training completes successfully,remove the job and its pods:

```bash
kubectl delete job spatop-job-train -n cms-ml
```

---

## 6. Evaluate the SPANet training
Assuming the output log directory is `spanet_output/version_0`.
Add `--gpu` if a GPU is available.
```bash
python -m spanet.test spanet_output/version_0 -tf data/delphes/v2/hhh_testing.h5 [--gpu]
```

## 7. Evaluate the baseline method
```bash
python -m src.models.test_baseline --test-file data/delphes/v2/hhh_testing.h5
```

# Instructions for CMS data set baseline
The CMS dataset was updated to run with the `v26` setup (`nAK4 >= 4 and HLT selection`). The update includes the possibility to apply the b-jet energy correction. By keeping events with at a least 4 jets, the boosted training can be performed on a maximum number of events and topologies.

List of samples (currently setup validated using 2018):
```
/eos/user/m/mstamenk/CxAOD31run/hhh-6b/cms-samples-spanet/v26/GluGluToHHHTo6B_SM_spanet_v26_2016APV.root
/eos/user/m/mstamenk/CxAOD31run/hhh-6b/cms-samples-spanet/v26/GluGluToHHHTo6B_SM_spanet_v26_2016.root
/eos/user/m/mstamenk/CxAOD31run/hhh-6b/cms-samples-spanet/v26/GluGluToHHHTo6B_SM_spanet_v26_2017.root
/eos/user/m/mstamenk/CxAOD31run/hhh-6b/cms-samples-spanet/v26/GluGluToHHHTo6B_SM_spanet_v26_2018.root
```

To run the framework, first convert the samples (this will allow to use both jets `pt` or `ptcorr`, steerable from the configuration file:
```
mkdir data/cms/v26/
python -m src.data.cms.convert_to_h5 /eos/user/m/mstamenk/CxAOD31run/hhh-6b/cms-samples-spanet/v26/GluGluToHHHTo6B_SM_spanet_v26_2018.root --out-file data/cms/v26/hhh_training.h5
python -m src.data.cms.convert_to_h5 /eos/user/m/mstamenk/CxAOD31run/hhh-6b/cms-samples-spanet/v26/GluGluToHHHTo6B_SM_spanet_v26_2018.root --out-file data/cms/v26/hhh_testing.h5
```

Then training can be done via:

```
python -m spanet.train -of options_files/cms/hhh_v26.json --gpus 1
```

Two config files exist for the event options:
```
event_files/cms/hhh.yaml # regular jet pT
event_files/cms/hhh_bregcorr.yaml # jet pT with b-jet energy correction scale factors applied
```

Note: to run the training with the b-jet energy correction applied, the `log_normalize` of the input variable was removed. Keeping it caused a 'Assignement collision'.
