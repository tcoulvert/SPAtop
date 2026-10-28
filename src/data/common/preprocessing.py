import copy
import glob
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

import awkward as ak
import click
import h5py
import numpy as np
import numba as nb
import uproot
import vector

import matplotlib.pyplot as plt
import mplhep as hep
import hist
from cycler import cycler

################################


plt.style.use(hep.style.CMS)
plt.rcParams.update({'font.size': 20})
cmap_petroff10 = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]
plt.rcParams.update({"axes.prop_cycle": cycler("color", cmap_petroff10)})

from src.data.common.matching import (
    reconstruct_top,
    FullyResolved_top, SemiResolvedQQ_top, SemiResolvedBQ_top, FullyBoosted_top, 
    FullyResolved_overlap, SemiResolved_overlap, FullyBoosted_overlap,
    match_fjet_to_jet,
)
from src.data.delphes import get_delphes_objects as dlp
from src.data.cms import get_cms_objects as cms
from src.data.common.condor_conversion import LPCCITVanillaSubmitter

vector.register_awkward()
vector.register_numba()
ak.numba.register_and_check()

logging.basicConfig(level=logging.INFO)

################################


MIN_JET_PT = 30  # 20
MIN_FJET_PT = 150  # 200
PROJECT_DIR = Path(__file__).resolve().parents[3]
N_JETS = lambda n_tops: 3*n_tops + 4
N_FJETS = lambda n_tops: n_tops + 1

PLOTS = False
RNG = np.random.default_rng(seed=21)

# TVSQCD_EFFS = {'t': 83.21e-2, 'W': 6.90e-2, 'bq': 19.65e-2, '-': 1e-2, 'QCD': 1e-2}  # taken from CMS 2022preEE tt -> 4Q
# WVSQCD_EFFS = {'t': 15.19e-2, 'W': 56.15e-2, 'bq': 9.92e-2, '-': 1e-2, 'QCD': 1e-2}
TVSQCD_EFFS = {'t': 91.53e-2, 'W': 5.54e-2, 'bq': 32.33e-2, '-': 7.24e-2, 'QCD': 1e-2}  # taken from 2016preVFP ttjets, effs ± (<0.006)
WVSQCD_EFFS = {'t': 16.17e-2, 'W': 67.76e-2, 'bq': 13.76e-2, '-': 10.26e-2, 'QCD': 1e-2}

################################


def to_np_array(ak_array, max_n=10, pad=0):
    return ak.fill_none(ak.pad_none(ak_array, max_n, clip=True, axis=-1), pad).to_numpy()


################################
# Recontrsuct hadronic tops
def get_matched_jetfjets_idx(combo_arr, selected_idxs, field, n_targets):
    return ak.fill_none(
        [ak.firsts(combo_arr[ak.local_index(combo_arr, axis=1) == selected_idxs[:, i]][field]['idx']) for i in range(n_targets)], 
        -1
    ).to_numpy().T
def dummy_idx_array(n_evts, n_tops, n_targets):
    return -1*np.ones((n_evts, n_tops - n_targets), dtype=int)

def fr_match_jets(jets, fjets, topquarks, bquarks, wbosons, wquarks_d1, wquarks_d2, n_evts, n_tops, n_targets):
    if n_targets == 0: return (
        dummy_idx_array(n_evts, n_tops, n_targets), dummy_idx_array(n_evts, n_tops, n_targets), dummy_idx_array(n_evts, n_tops, n_targets)
    )

    FR_3jets_idxs = ak.argcartesian([jets, jets, jets], axis=1)
    FR_3jet0, FR_3jet1, FR_3jet2 = ak.unzip(FR_3jets_idxs)
    FR_3jets_mask = (FR_3jet0 != FR_3jet1) & (FR_3jet0 != FR_3jet2) & (FR_3jet1 != FR_3jet2)
    FR_3jet0, FR_3jet1, FR_3jet2 = FR_3jet0[FR_3jets_mask], FR_3jet1[FR_3jets_mask], FR_3jet2[FR_3jets_mask]
    FR_3jets = ak.zip({'bjet': jets[FR_3jet0], 'q1jet': jets[FR_3jet1], 'q2jet': jets[FR_3jet2]})
    FR_combo_jet_idxs = reconstruct_top(
        topquarks, bquarks, wbosons, wquarks_d1, wquarks_d2,
        FR_3jets,
        FullyResolved_top,
        FullyResolved_overlap,
        ak.ArrayBuilder()
    ).snapshot()
    FR_bjet_idxs = get_matched_jetfjets_idx(FR_3jets, FR_combo_jet_idxs, 'bjet', n_targets)
    FR_q1jet_idxs = get_matched_jetfjets_idx(FR_3jets, FR_combo_jet_idxs, 'q1jet', n_targets)
    FR_q2jet_idxs = get_matched_jetfjets_idx(FR_3jets, FR_combo_jet_idxs, 'q2jet', n_targets)

    if n_targets < n_tops:
        FR_bjet_idxs = np.concatenate((FR_bjet_idxs, dummy_idx_array(n_evts, n_tops, n_targets)), axis=1)
        FR_q1jet_idxs = np.concatenate((FR_q1jet_idxs, dummy_idx_array(n_evts, n_tops, n_targets)), axis=1)
        FR_q2jet_idxs = np.concatenate((FR_q2jet_idxs, dummy_idx_array(n_evts, n_tops, n_targets)), axis=1)

    return FR_bjet_idxs, FR_q1jet_idxs, FR_q2jet_idxs

def srqq_match_jets(jets, fjets, topquarks, bquarks, wbosons, wquarks_d1, wquarks_d2, n_evts, n_tops, n_targets):
    if n_targets == 0: return (
        dummy_idx_array(n_evts, n_tops, n_targets), dummy_idx_array(n_evts, n_tops, n_targets)
    )

    SR_1jet1fjet = ak.cartesian({'jet': jets, 'fjet': fjets})
    SRqq_combo_jetfjet_idxs = reconstruct_top(
        topquarks, bquarks, wbosons, wquarks_d1, wquarks_d2,
        SR_1jet1fjet,
        SemiResolvedQQ_top,
        SemiResolved_overlap,
        ak.ArrayBuilder()
    ).snapshot()
    SRqq_bjet_idxs = get_matched_jetfjets_idx(SR_1jet1fjet, SRqq_combo_jetfjet_idxs, 'jet', n_targets)
    SRqq_qqfjet_idxs = get_matched_jetfjets_idx(SR_1jet1fjet, SRqq_combo_jetfjet_idxs, 'fjet', n_targets)

    if n_targets < n_tops:
        SRqq_bjet_idxs = np.concatenate((SRqq_bjet_idxs, dummy_idx_array(n_evts, n_tops, n_targets)), axis=1)
        SRqq_qqfjet_idxs = np.concatenate((SRqq_qqfjet_idxs, dummy_idx_array(n_evts, n_tops, n_targets)), axis=1)

    return SRqq_bjet_idxs, SRqq_qqfjet_idxs

def srbq_match_jets(jets, fjets, topquarks, bquarks, wbosons, wquarks_d1, wquarks_d2, n_evts, n_tops, n_targets):
    if n_targets == 0: return (
        dummy_idx_array(n_evts, n_tops, n_targets), dummy_idx_array(n_evts, n_tops, n_targets)
    )
    
    SR_1jet1fjet = ak.cartesian({'jet': jets, 'fjet': fjets})
    SRbq_combo_jetfjet_idxs = reconstruct_top(
        topquarks, bquarks, wbosons, wquarks_d1, wquarks_d2,
        SR_1jet1fjet,
        SemiResolvedBQ_top,
        SemiResolved_overlap,
        ak.ArrayBuilder()
    ).snapshot()
    SRbq_qjet_idxs = get_matched_jetfjets_idx(SR_1jet1fjet, SRbq_combo_jetfjet_idxs, 'jet', n_targets)
    SRbq_bqfjet_idxs = get_matched_jetfjets_idx(SR_1jet1fjet, SRbq_combo_jetfjet_idxs, 'fjet', n_targets)

    if n_targets < n_tops:
        SRbq_qjet_idxs = np.concatenate((SRbq_qjet_idxs, dummy_idx_array(n_evts, n_tops, n_targets)), axis=1)
        SRbq_bqfjet_idxs = np.concatenate((SRbq_bqfjet_idxs, dummy_idx_array(n_evts, n_tops, n_targets)), axis=1)

    return SRbq_qjet_idxs, SRbq_bqfjet_idxs

def fb_match_jets(jets, fjets, topquarks, bquarks, wbosons, wquarks_d1, wquarks_d2, n_evts, n_tops, n_targets):  
    if n_targets == 0: return dummy_idx_array(n_evts, n_tops, n_targets)
      
    FB_1fjet = ak.combinations(fjets, 1, fields=['fjet'])
    FB_combo_fjet_idxs = reconstruct_top(
        topquarks, bquarks, wbosons, wquarks_d1, wquarks_d2,
        FB_1fjet,
        FullyBoosted_top,
        FullyBoosted_overlap,
        ak.ArrayBuilder()
    ).snapshot()
    FB_bqqfjet_idxs = get_matched_jetfjets_idx(FB_1fjet, FB_combo_fjet_idxs, 'fjet', n_targets)

    if n_targets < n_tops:
        FB_bqqfjet_idxs = np.concatenate((FB_bqqfjet_idxs, dummy_idx_array(n_evts, n_tops, n_targets)), axis=1)

    return FB_bqqfjet_idxs

################################
# Dataset processing
def get_datasets(arrays, file_type, n_tops, n_targets, min_valid_targets):  # noqa: C901
    global N_JETS, N_FJETS
    if callable(N_JETS): N_JETS = N_JETS(n_tops)
    if callable(N_FJETS): N_FJETS = N_FJETS(n_tops)
    if file_type.lower() == "delphes":
        event_mask, jets, fjets, genjets, genfjets = dlp.get_jets(arrays, n_tops, n_targets, N_JETS, N_FJETS)
        event_mask, particles, topquarks, bquarks, wbosons, wquarks_d1, wquarks_d2 = dlp.get_genparts(arrays, n_tops, n_targets, event_mask)
    elif file_type.lower() == "nanoaodv9":
        event_mask, jets, fjets, genjets, genfjets = cms.get_jets(arrays, n_tops, n_targets, N_JETS, N_FJETS)
        event_mask, particles, topquarks, bquarks, wbosons, wquarks_d1, wquarks_d2 = cms.get_genparts(arrays, n_tops, n_targets, event_mask)
    else:
        raise Exception(f"Unknown ROOT file type, currently only accepts Delphes or NanoAODv16 (not case-sensitive)")
    jets, fjets, genjets, genfjets = jets[event_mask], fjets[event_mask], genjets[event_mask], genfjets[event_mask]
    n_evts = np.sum(event_mask)

    # Jet-FatJet overlap
    matched_fjet_jet_idx, matched_fjet_jet_DR  = match_fjet_to_jet(fjets, jets, ak.ArrayBuilder(), ak.ArrayBuilder())
    matched_fjet_jet_idx, matched_fjet_jet_DR = matched_fjet_jet_idx.snapshot(), matched_fjet_jet_DR.snapshot()

    # Matched Jet-Reconstruction indices
    FR_bjet_idxs, FR_q1jet_idxs, FR_q2jet_idxs = fr_match_jets(jets, fjets, topquarks, bquarks, wbosons, wquarks_d1, wquarks_d2, n_evts, n_tops, n_targets)
    SRqq_bjet_idxs, SRqq_qqfjet_idxs = srqq_match_jets(jets, fjets, topquarks, bquarks, wbosons, wquarks_d1, wquarks_d2, n_evts, n_tops, n_targets)
    SRbq_qjet_idxs, SRbq_bqfjet_idxs = srbq_match_jets(jets, fjets, topquarks, bquarks, wbosons, wquarks_d1, wquarks_d2, n_evts, n_tops, n_targets)
    FB_bqqfjet_idxs = fb_match_jets(jets, fjets, topquarks, bquarks, wbosons, wquarks_d1, wquarks_d2, n_evts, n_tops, n_targets)


    ################################
    # Reconstruction dicts
    top_pt = ak.fill_none(ak.pad_none(topquarks.pt, target=n_tops, axis=1, clip=True), -1)
    top_pt_dict = {}
    for i in range(n_tops):
        top_pt_dict[f"top{i+1}_pt"] = top_pt[:, i].to_numpy()

    top_fullyResolved = {}
    for i in range(n_tops):
        top_fullyResolved[f"top{i+1}_b"] = FR_bjet_idxs[:, i]
        top_fullyResolved[f"top{i+1}_q1"] = FR_q1jet_idxs[:, i]
        top_fullyResolved[f"top{i+1}_q2"] = FR_q2jet_idxs[:, i]
        top_fullyResolved[f"top{i+1}_mask"] = (top_fullyResolved[f"top{i+1}_b"] != -1)
    print(f'num FR tops = {sum([ak.sum(top_fullyResolved[f"top{i+1}_mask"]) for i in range(n_tops)])}')

    top_semiResolved_qq = {}
    for i in range(n_tops):
        top_semiResolved_qq[f"top{i+1}_b"] = SRqq_bjet_idxs[:, i]
        top_semiResolved_qq[f"top{i+1}_qq"] = SRqq_qqfjet_idxs[:, i]
        top_semiResolved_qq[f"top{i+1}_mask"] = (top_semiResolved_qq[f"top{i+1}_b"] != -1)
    print(f'num SRqq tops = {sum([ak.sum(top_semiResolved_qq[f"top{i+1}_mask"]) for i in range(n_tops)])}')

    top_semiResolved_bq = {}
    for i in range(n_tops):
        top_semiResolved_bq[f"top{i+1}_q"] = SRbq_qjet_idxs[:, i]
        top_semiResolved_bq[f"top{i+1}_bq"] = SRbq_bqfjet_idxs[:, i]
        top_semiResolved_bq[f"top{i+1}_mask"] = (top_semiResolved_bq[f"top{i+1}_q"] != -1)
    print(f'num SRbq tops = {sum([ak.sum(top_semiResolved_bq[f"top{i+1}_mask"]) for i in range(n_tops)])}')

    top_fullyBoosted = {}
    for i in range(n_tops):
        top_fullyBoosted[f"top{i+1}_bqq"] = FB_bqqfjet_idxs[:, i]
        top_fullyBoosted[f"top{i+1}_mask"] = (top_fullyBoosted[f"top{i+1}_bqq"] != -1)
    print(f'num FB tops = {sum([ak.sum(top_fullyBoosted[f"top{i+1}_mask"]) for i in range(n_tops)])}')



    ################################
    # PNet tagger emulations
    # apply emulated TvsQCD and WvsQCD bools @ 1.0% QCD eff WPs
    if file_type.lower() == "delphes":
        tops_fjet_mask = ak.any([ak.local_index(fjets) == top_fullyBoosted[f"top{i+1}_bqq"] for i in range(n_tops)], axis=0)
        w_fjet_mask = ak.any([ak.local_index(fjets) == top_semiResolved_qq[f"top{i+1}_qq"] for i in range(n_tops)], axis=0)
        bq_fjet_mask = ak.any([ak.local_index(fjets) == top_semiResolved_bq[f"top{i+1}_bq"] for i in range(n_tops)], axis=0)
        apply_effs_by_mask = lambda tag, effs: ak.where(
            tops_fjet_mask, fjets[f"{tag}RN"] < effs['t'], 
            ak.where(
                w_fjet_mask, fjets[f"{tag}RN"] < effs['W'], 
                ak.where(
                    bq_fjet_mask, fjets[f"{tag}RN"] < effs['bq'],
                    fjets[f"{tag}RN"] < (effs['-'] if n_targets > 0 else effs['QCD'])
                )
            )
        )
        fjets["Ttag"] = apply_effs_by_mask("Ttag", TVSQCD_EFFS)
        fjets["Wtag"] = apply_effs_by_mask("Wtag", WVSQCD_EFFS)
    elif file_type.lower() == "nanoaodv9":
        eff_filepath = 'nanoaodv9_tagger_effs.json'
        if os.path.exists(eff_filepath):
            with open(eff_filepath, 'r') as f:
                tagger_effs_map = json.load(f)
        else:
            tagger_effs_map = {
                'Ttag': {'t': {'pass': 0, 'total': 0}, 'W': {'pass': 0, 'total': 0}, 'bq': {'pass': 0, 'total': 0}, '-': {'pass': 0, 'total': 0}},
                'Wtag': {'t': {'pass': 0, 'total': 0}, 'W': {'pass': 0, 'total': 0}, 'bq': {'pass': 0, 'total': 0}, '-': {'pass': 0, 'total': 0}}
            }
        tops_fjet_mask = ak.any([ak.local_index(fjets) == top_fullyBoosted[f"top{i+1}_bqq"] for i in range(n_tops)], axis=0)
        w_fjet_mask = ak.any([ak.local_index(fjets) == top_semiResolved_qq[f"top{i+1}_qq"] for i in range(n_tops)], axis=0)
        bq_fjet_mask = ak.any([ak.local_index(fjets) == top_semiResolved_bq[f"top{i+1}_bq"] for i in range(n_tops)], axis=0)
        qcd_fjet_mask = (~tops_fjet_mask & ~w_fjet_mask & ~bq_fjet_mask)
        mask_map = {'t': tops_fjet_mask, 'W': w_fjet_mask, 'bq': bq_fjet_mask, '-': qcd_fjet_mask}
        for tag, effs_map in tagger_effs_map.items():
            print(tag)
            for genmatch, eff_map in effs_map.items():
                n_pass_tag = int(ak.sum(fjets[tag][mask_map[genmatch]], axis=None))
                n_genmatch = int(ak.sum(mask_map[genmatch], axis=None))
                eff_map['pass'] += n_pass_tag
                eff_map['total'] += n_genmatch
                print(f" - % of Gen-Matched {genmatch} that pass {tag} = {100 * n_pass_tag / n_genmatch:.2f}%")
        with open(eff_filepath, 'w') as f:
            json.dump(tagger_effs_map, f)


    ################################
    # Plots for validating reconstruction
    if PLOTS:
        # jet pt
        plot_destdir = os.path.join(PROJECT_DIR, 'plots', 'jet_pt')
        if not os.path.exists(plot_destdir):
            os.makedirs(plot_destdir)

        jet_pt_axis = hist.axis.Regular(50, 0., 500., name='var', label=r'$p_T$ [GeV]', growth=False, underflow=False, overflow=False)
        for local_idx in range(4):
            jet_i_hist = hist.Hist(jet_pt_axis).fill(var=ak.firsts(jets.pt[jets["idx"] == local_idx]))
            plt.figure()
            hep.histplot(
                jet_i_hist, label=f'sublead jet {local_idx}' if local_idx > 0 else 'lead jet', histtype='step'
            )
            plt.legend()
            plt.yscale('log')
            plt.savefig(os.path.join(plot_destdir, f"reco_jet{local_idx}_pt"))
            plt.savefig(os.path.join(plot_destdir, f"reco_jet{local_idx}_pt.pdf"), format='pdf')
            plt.close()

        # jet-genjet deltaR
        plot_destdir = os.path.join(PROJECT_DIR, 'plots', 'jet_genjet_deltaR')
        if not os.path.exists(plot_destdir):
            os.makedirs(plot_destdir)

        for jet_type, gen_jet_arr, reco_jet_arr in [('ak5', genjets, jets), ('ak8', genfjets, fjets)]:
            
            jet_genjet_deltaR_axis = hist.axis.Regular(25, 0., 1.5 if int(jet_type[2:]) > 5 else 0.5, name='var', label=r'$\Delta R$', growth=False, underflow=False, overflow=False)
            min_deltaR = 998*np.ones(ak.num(gen_jet_arr, axis=0), dtype=float)
            
            for local_gen_idx in range(np.max(np.unique(ak.num(gen_jet_arr)))):
                for local_reco_idx in range(np.max(np.unique(ak.num(reco_jet_arr)))):

                    ij_deltaR = ak.fill_none(
                        ak.firsts(
                            gen_jet_arr[ak.local_index(gen_jet_arr) == local_gen_idx]
                        ).deltaR(
                            ak.firsts(reco_jet_arr[ak.local_index(reco_jet_arr) == local_reco_idx])
                        ), 999
                    )
                    
                    min_deltaR = ak.where(ij_deltaR < min_deltaR, ij_deltaR, min_deltaR)

            jet_genjet_deltaR_hist = hist.Hist(jet_genjet_deltaR_axis).fill(var=min_deltaR)
            plt.figure()
            hep.histplot(
                jet_genjet_deltaR_hist, label=f'min($\Delta R$({jet_type}jet, Gen{jet_type}jet))', histtype='step'
            )
            plt.legend()
            plt.yscale('log')
            plt.savefig(os.path.join(plot_destdir, f"{jet_type}jet_genjet_minDeltaR"))
            plt.savefig(os.path.join(plot_destdir, f"{jet_type}jet_genjet_minDeltaR.pdf"), format='pdf')
            plt.close()


    ################################
    # Cleaning non-reconstructable events
    n_targets_per_event = np.zeros(n_evts, dtype=int)
    for i in range(n_tops):
        n_targets_per_event += ak.to_numpy(
            top_fullyResolved[f"top{i+1}_mask"]
            | top_semiResolved_qq[f"top{i+1}_mask"]
            | top_semiResolved_bq[f"top{i+1}_mask"]
            | top_fullyBoosted[f"top{i+1}_mask"]
        ).astype("int")
    print(f"Viable tops = {np.sum(n_targets_per_event)}")
    min_valid_targets_mask = np.asarray(n_targets_per_event >= min_valid_targets, dtype=bool)

    ################################
    # Sanity checking reconstruction
    # pass

    ################################
    # Store processed data in dataset for training/testing
    # Inputs
    datasets = {}
    datasets["INPUTS/Jets/MASK"] = to_np_array(jets.pt > 0, max_n=N_JETS, pad=False).astype("bool")[min_valid_targets_mask]
    datasets["INPUTS/Jets/pt"] = to_np_array(jets.pt, max_n=N_JETS).astype("float32")[min_valid_targets_mask]
    datasets["INPUTS/Jets/eta"] = to_np_array(jets.eta, max_n=N_JETS).astype("float32")[min_valid_targets_mask]
    datasets["INPUTS/Jets/phi"] = to_np_array(jets.phi, max_n=N_JETS).astype("float32")[min_valid_targets_mask]
    datasets["INPUTS/Jets/sinphi"] = to_np_array(np.sin(jets.phi), max_n=N_JETS).astype("float32")[min_valid_targets_mask]
    datasets["INPUTS/Jets/cosphi"] = to_np_array(np.cos(jets.phi), max_n=N_JETS).astype("float32")[min_valid_targets_mask]
    datasets["INPUTS/Jets/mass"] = to_np_array(jets.mass, max_n=N_JETS).astype("float32")[min_valid_targets_mask]
    datasets["INPUTS/Jets/btag"] = to_np_array(jets["btag"], max_n=N_JETS).astype("bool")[min_valid_targets_mask]
    datasets["INPUTS/Jets/flavor"] = to_np_array(jets["flavor"], max_n=N_JETS).astype("float32")[min_valid_targets_mask]
    datasets["INPUTS/Jets/matchedfj"] = to_np_array(matched_fjet_jet_idx, max_n=N_JETS).astype("int32")[min_valid_targets_mask]
    datasets["INPUTS/Jets/deltaRfj"] = to_np_array(matched_fjet_jet_DR, max_n=N_JETS).astype("float32")[min_valid_targets_mask]

    datasets["INPUTS/BoostedJets/MASK"] = to_np_array(fjets.pt > 0, max_n=N_FJETS).astype("bool")[min_valid_targets_mask]
    datasets["INPUTS/BoostedJets/fj_pt"] = to_np_array(fjets.pt, max_n=N_FJETS).astype("float32")[min_valid_targets_mask]
    datasets["INPUTS/BoostedJets/fj_eta"] = to_np_array(fjets.eta, max_n=N_FJETS).astype("float32")[min_valid_targets_mask]
    datasets["INPUTS/BoostedJets/fj_phi"] = to_np_array(fjets.phi, max_n=N_FJETS).astype("float32")[min_valid_targets_mask]
    datasets["INPUTS/BoostedJets/fj_sinphi"] = to_np_array(np.sin(fjets.phi), max_n=N_FJETS).astype("float32")[min_valid_targets_mask]
    datasets["INPUTS/BoostedJets/fj_cosphi"] = to_np_array(np.cos(fjets.phi), max_n=N_FJETS).astype("float32")[min_valid_targets_mask]
    datasets["INPUTS/BoostedJets/fj_mass"] = to_np_array(fjets.mass, max_n=N_FJETS).astype("float32")[min_valid_targets_mask]
    datasets["INPUTS/BoostedJets/fj_sdmass"] = to_np_array(fjets["sdmass"], max_n=N_FJETS).astype("float32")[min_valid_targets_mask]
    datasets["INPUTS/BoostedJets/fj_Ttag"] = to_np_array(fjets["Ttag"], max_n=N_FJETS).astype("bool")[min_valid_targets_mask]
    datasets["INPUTS/BoostedJets/fj_Wtag"] = to_np_array(fjets["Wtag"], max_n=N_FJETS).astype("bool")[min_valid_targets_mask]
    datasets["INPUTS/BoostedJets/fj_tau21"] = to_np_array(fjets["tau21"], max_n=N_FJETS).astype("float32")[min_valid_targets_mask]
    datasets["INPUTS/BoostedJets/fj_tau32"] = to_np_array(fjets["tau32"], max_n=N_FJETS).astype("float32")[min_valid_targets_mask]
    datasets["INPUTS/BoostedJets/fj_ncharged"] = to_np_array(fjets["ncharged"], max_n=N_FJETS).astype("float32")[min_valid_targets_mask]

    # Targets
    for i in range(n_tops):
        # fully-resolved tops
        datasets[f"TARGETS/FRt{i+1}/MASK"] = top_fullyResolved[f"top{i+1}_mask"][min_valid_targets_mask]
        datasets[f"TARGETS/FRt{i+1}/b"] = top_fullyResolved[f"top{i+1}_b"][min_valid_targets_mask]
        datasets[f"TARGETS/FRt{i+1}/q1"] = top_fullyResolved[f"top{i+1}_q1"][min_valid_targets_mask]
        datasets[f"TARGETS/FRt{i+1}/q2"] = top_fullyResolved[f"top{i+1}_q2"][min_valid_targets_mask]
        datasets[f"TARGETS/FRt{i+1}/pt"] = top_pt_dict[f"top{i+1}_pt"][min_valid_targets_mask]

        # semi-resolved (qq fatjet) tops
        datasets[f"TARGETS/SRqqt{i+1}/MASK"] = top_semiResolved_qq[f"top{i+1}_mask"][min_valid_targets_mask]
        datasets[f"TARGETS/SRqqt{i+1}/b"] = top_semiResolved_qq[f"top{i+1}_b"][min_valid_targets_mask]
        datasets[f"TARGETS/SRqqt{i+1}/qq"] = top_semiResolved_qq[f"top{i+1}_qq"][min_valid_targets_mask]
        datasets[f"TARGETS/SRqqt{i+1}/pt"] = top_pt_dict[f"top{i+1}_pt"][min_valid_targets_mask]

        # semi-resolved (bq fatjet) tops
        datasets[f"TARGETS/SRbqt{i+1}/MASK"] = top_semiResolved_bq[f"top{i+1}_mask"][min_valid_targets_mask]
        datasets[f"TARGETS/SRbqt{i+1}/q"] = top_semiResolved_bq[f"top{i+1}_q"][min_valid_targets_mask]
        datasets[f"TARGETS/SRbqt{i+1}/bq"] = top_semiResolved_bq[f"top{i+1}_bq"][min_valid_targets_mask]
        datasets[f"TARGETS/SRbqt{i+1}/pt"] = top_pt_dict[f"top{i+1}_pt"][min_valid_targets_mask]

        # fully-boosted tops
        datasets[f"TARGETS/FBt{i+1}/MASK"] = top_fullyBoosted[f"top{i+1}_mask"][min_valid_targets_mask]
        datasets[f"TARGETS/FBt{i+1}/bqq"] = top_fullyBoosted[f"top{i+1}_bqq"][min_valid_targets_mask]
        datasets[f"TARGETS/FBt{i+1}/pt"] = top_pt_dict[f"top{i+1}_pt"][min_valid_targets_mask]

    return datasets

def process_file(file_name, file_type, out_file, train_frac, n_tops, n_targets, min_valid_targets):
    try:
        if re.match('root://', file_name):
            current_file_name = 'tmp_'+('train_' if 'training' in out_file else 'test_')+file_name.split('/')[-1]
            subprocess.run(['xrdcp', '-f', file_name, current_file_name])
        else:
            current_file_name = file_name
        with uproot.open(current_file_name) as in_file:
            if file_type.lower() == "delphes":
                events = in_file["Delphes"]
                keys = (
                    [key for key in events.keys() if "Particle/Particle." in key and "fBits" not in key]
                    + [key for key in events.keys() if "Jet/Jet." in key]
                    + [key for key in events.keys() if "FatJet/FatJet." in key and "fBits" not in key]
                    + [key for key in events.keys() if "GenJet/GenJet." in key and "fBits" not in key]
                    + [key for key in events.keys() if "GenFatJet/GenFatJet." in key and "fBits" not in key]
                )
            elif file_type.lower() == "nanoaodv9":
                events = in_file["Events"]
                keys = (
                    [key for key in events.keys() if "GenPart" in key and "fBits" not in key]
                    + [key for key in events.keys() if "Jet" in key]
                    + [key for key in events.keys() if "FatJet" in key and "fBits" not in key]
                    + [key for key in events.keys() if "GenJet" in key and "fBits" not in key]
                    + [key for key in events.keys() if "GenJetAK8." in key and "fBits" not in key]
                )
            else:
                raise Exception(f"Unknown ROOT file type, currently only accepts Delphes or NanoAODv16 (not case-sensitive)")
            print(f'loaded {file_type} tree')


            num_entries = events.num_entries
            if "training" in out_file:
                entry_start = None
                entry_stop = int(train_frac * num_entries)
            else:
                entry_start = int(train_frac * num_entries)
                entry_stop = None
            
            arrays = events.arrays(keys, entry_start=entry_start, entry_stop=entry_stop)
            datasets = get_datasets(arrays, file_type, n_tops, n_targets, min_valid_targets)

        if re.match('root://', file_name): subprocess.run(['rm', '-rf', current_file_name])
        return datasets
    except Exception:
        if re.match('root://', file_name): subprocess.run(['rm', '-rf', current_file_name])
        logging.exception(f"Preprocessing failed for file: {file_name}\n...continuing with other files")
        return 400

def save_file(filepath: str, dataset: dict):
    if len(filepath.split('/')[-1].split('.')) == 1:
        raise Exception(f'No filepath extension, ambiguous how to save file.')
    else:
        filetype = filepath.split('/')[-1].split('.')[-1]
    n_events = sum(len(chunk) for chunk in dataset[list(dataset.keys())[0]])
    print(f"Saving {n_events} events into {filepath}")
        
    if filetype == 'h5':
        with h5py.File(filepath, "w") as output:
            for dataset_name, all_data in dataset.items():
                concat_data = np.concatenate(all_data, axis=0)
                logging.info(f"Dataset name: {dataset_name}")
                logging.info(f"Dataset shape: {concat_data.shape}")
                output.create_dataset(dataset_name, data=concat_data)
    elif filetype == 'root':
        with uproot.recreate(filepath) as output:
            merged_dataset = {}
            for dataset_name, all_data in dataset.items():
                concat_data = np.concatenate(all_data, axis=0)
                logging.info(f"Dataset name: {dataset_name}")
                logging.info(f"Dataset type: {concat_data.dtype}")
                logging.info(f"Dataset shape: {concat_data.shape}")
                merged_dataset[dataset_name] = concat_data
            output['Events'] = merged_dataset
    else: raise NotImplementedError(f'Requested file type saving not implemented, try \'.h5\' or \'.root\'.')
    return n_events

@click.command()
@click.argument("in-files", nargs=-1)
@click.argument("in-file-type", nargs=1)
@click.option(
    "--out-file",
    "out_file",
    default=f"{PROJECT_DIR}/data/delphes/tt_training.h5",
    help="Output file.",
)
@click.option(
    "--split-file-size",
    "split_file_size",
    default=-1,
    help="Size for output files in MB, default is \'-1\' which creates 1 merged output file",
)
@click.option(
    "--file-limit",
    "file_limit",
    default=-1,
    help="Number of output files to make, default is \'-1\' which creates all output files",
)
@click.option("--train-frac", default=0.80, help="Fraction for training.")
@click.option(
    "--n-tops",
    "n_tops",
    default=2,
    type=click.IntRange(1, 4),
    help="Number of top quark targets to include per event",
)
@click.option("--plots", is_flag=True, help="Boolean to make plots.")
@click.option("--condor", is_flag=True, help="Boolean to use condor processing.")
@click.option(
    "--condor-queue",
    default="longlunch",
    help="Queue for condor jobs",
)
@click.option(
    "--condor-memory",
    default="4GB",
    help="RAM per condor jobs",
)
@click.option(
    "--condor-cpus",
    default=1,
    help="Number of CPUS per condor job",
)
@click.option(
    "--condor-disk",
    default="4GB",
    help="Disk memory per condor job",
)
@click.option(
    "--condor-location",
    default="LPC",
    help="Location for condor",
)
@click.option(
    "--condor-files-per-job",
    default=20,
    help="Number of input files per condor job",
)
@click.option(
    "--n-targets",
    "n_targets",
    default=None,
    type=click.IntRange(0, 4),
    help="Number of true hadronic top quarks required per event (default: same as \'n-tops\'; must not exceed it)",
)
@click.option(
    "--min-valid-targets",
    "min_valid_targets",
    default=0,
    type=click.IntRange(0, 4),
    help="Minimum number of valid targets (hadronic tops) per-event, events with fewer valid targets are excluded from the output dataset",
)
def main(in_files, in_file_type, out_file, split_file_size, file_limit, train_frac, n_tops, plots, condor, condor_queue, condor_memory, condor_cpus, condor_disk, condor_location, condor_files_per_job, n_targets, min_valid_targets):
    if plots:
        global PLOTS
        PLOTS = True
    if n_targets is None:
        n_targets = n_tops
    assert n_targets <= n_tops, f"\'n-targets\' needs to be less than or equal to \'n-tops\'"
    
    all_datasets = {}
    expanded_in_files = []

    for file_name in in_files:
        if re.match('root://', file_name):
            print(' '.join(['xrdfs', '//'.join(file_name.split('//')[:2])+'/', 'ls', '-R', '/'+file_name.split('//')[2]]))
            xrdfs_files = subprocess.run(['xrdfs', '//'.join(file_name.split('//')[:2])+'/', 'ls', '-R', '/'+file_name.split('//')[2]], capture_output=True, text=True)
            expanded_in_files.extend(['//'.join(file_name.split('//')[:2])+'/'+xrdfs_file for xrdfs_file in xrdfs_files.stdout.split() if xrdfs_file.endswith('.root')])
        elif os.path.isdir(file_name):
            print(f"found directory: {file_name}")
            for root_file in glob.glob(os.path.join(file_name, '*.root')):
                expanded_in_files.append(root_file)
        else:
            expanded_in_files.append(file_name)
    in_files = expanded_in_files
    out_file_idx = 0
    new_outfile_with_idx = lambda outfile, idx: outfile[:outfile.rfind('.')]+str(idx)+outfile[outfile.rfind('.'):]
    n_failed, n_saved = 0, 0
    if not condor:
        for file_name in in_files:
            datasets = process_file(file_name, in_file_type, out_file, train_frac, n_tops, n_targets, min_valid_targets)
            if isinstance(datasets, int):
                n_failed += 1
                continue
            for dataset_name, data in datasets.items():
                if dataset_name not in all_datasets: all_datasets[dataset_name] = []
                all_datasets[dataset_name].append(data)
            num_events = np.sum([len(all_datasets[dataset_name][i]) for i in range(len(all_datasets[dataset_name]))])
            num_bytes_per_event = np.sum([data.itemsize * data.size / data.shape[0] for data in datasets.values()])
            if split_file_size > 0 and num_events * num_bytes_per_event > split_file_size:
                print(f'Saving file to {new_outfile_with_idx(out_file, out_file_idx)}')
                n_saved += save_file(new_outfile_with_idx(out_file, out_file_idx), all_datasets)
                out_file_idx += 1; all_datasets = {}
            if out_file_idx == file_limit: break
        if all_datasets != {}:
            if split_file_size > 0: n_saved += save_file(new_outfile_with_idx(out_file, out_file_idx), all_datasets)
            else: n_saved += save_file(out_file, all_datasets)
        print(f"Done: {len(in_files) - n_failed} of {len(in_files)} input files converted, {n_saved} events saved")
        if n_failed:
            logging.error(f"{n_failed} of {len(in_files)} input files failed to convert")
            sys.exit(1)
        if n_saved == 0:
            logging.error("No events were saved")
            sys.exit(2)
    elif condor:
        job_filepaths = [
            list(in_files[i*condor_files_per_job:(i+1)*condor_files_per_job]) 
            for i in range(len(in_files)//condor_files_per_job)
        ]
        submitter = LPCCITVanillaSubmitter(job_filepaths, in_file_type, out_file, queue=condor_queue, memory=condor_memory, cpus=condor_cpus, disk=condor_disk, location=condor_location)
        submitter.submit()


if __name__ == "__main__":
    main()
