import copy
import glob
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

TVSQCD_EFFS = {'t': 83.21e-2, 'W': 6.90e-2, 'bq': 19.65e-2, 'QCD': 1e-2}
WVSQCD_EFFS = {'t': 15.19e-2, 'W': 56.15e-2, 'bq': 9.92e-2, 'QCD': 1e-2}

################################


@nb.njit
def random2D(layout_array, RN_array, RN_builder):
    RN_idx = 0
    for dim_1 in layout_array:
        RN_builder.begin_list()
        for dim_2 in dim_1:
            RN_builder.append(RN_array[RN_idx])
            RN_idx += 1
        RN_builder.end_list()
    return RN_builder


def get_bs(particles: ak.Array, n_targets: int):
    init_bs = particles[np.logical_and(
        np.abs(particles.pid) == 5,
        np.abs(particles.pid[particles.m1]) == 6,
    )]
    bevt_mask = (ak.num(init_bs) == n_targets)
    return init_bs, bevt_mask

def get_Ws(particles: ak.Array, n_targets: int):
    init_Ws = particles[np.logical_and(
        np.abs(particles.pid) == 24,
        np.abs(particles.pid[particles.m1]) == 6,
    )]
    Wevt_mask = (ak.num(init_Ws) == n_targets)
    return init_Ws, Wevt_mask

@nb.njit
def get_particle_from_idx(particles: ak.Array, idxs: ak.Array, evt_mask: ak.Array, parts: ak.ArrayBuilder):
    for particles_event, idx_event in zip(particles, idxs):
        parts.begin_list()
        for idx_ in idx_event:
            parts.append(particles_event[idx_])
        parts.end_list()
    return parts

def get_ts(particles: ak.Array, n_targets: int, bs: ak.Array, Ws: ak.Array):

    @nb.njit
    def nbt(
        n_targets: int, 
        particles_pid: ak.Array, 
        bs_m1: ak.Array, Ws_m1: ak.Array, 
        t: ak.ArrayBuilder, evt_mask: ak.ArrayBuilder
    ):
        for particles_pid_event, bs_m1_event, Ws_m1_event in zip(
            particles_pid, bs_m1, Ws_m1
        ):
            badevt = False
            if not (len(bs_m1_event) == n_targets and len(Ws_m1_event) == n_targets): badevt = True
            for b_m1 in bs_m1_event:
                if b_m1 not in Ws_m1_event: badevt = True; break
            for W_m1 in Ws_m1_event:
                if W_m1 not in bs_m1_event: badevt = True; break
            t.begin_list()
            for b_m1 in bs_m1_event:
                t.append(b_m1)
                if b_m1 < 0 or abs(particles_pid_event[b_m1]) != 6: badevt = True
            if badevt: evt_mask.append(0)
            else: evt_mask.append(1)
            t.end_list()
        return t, evt_mask

    t, tevt_mask = nbt(n_targets, particles.pid, bs.m1, Ws.m1, ak.ArrayBuilder(), ak.ArrayBuilder())
    t, tevt_mask = t.snapshot(), tevt_mask.snapshot()
    return particles[t][tevt_mask], tevt_mask

def get_Wds(particles: ak.Array, n_targets: int, Ws: ak.Array):

    @nb.njit
    def nbWd(
        n_targets: int, 
        particles_idx: ak.Array, particles_pid: ak.Array, particles_m1: ak.Array, 
        Ws_idx: ak.Array, 
        d1: ak.ArrayBuilder, d2: ak.ArrayBuilder, evt_mask: ak.ArrayBuilder
    ):
        for particles_idx_event, particles_pid_event, particles_m1_event, Ws_idx_event in zip(
            particles_idx, particles_pid, particles_m1, Ws_idx
        ):
            badevt = False
            if len(Ws_idx_event) != n_targets: badevt = True
            d1.begin_list(); d2.begin_list()
            for w_idx in Ws_idx_event:
                eff_w_idx = w_idx
                d1_ = -1; d2_ = -1
                for i in range(len(particles_idx_event)):  # limits total steps to length of particles per event
                    for part_idx, part_pid, part_m1 in zip(particles_idx_event, particles_pid_event, particles_m1_event):
                        if part_m1 == eff_w_idx:
                            if abs(part_pid) == 24: eff_w_idx = part_idx; break
                            elif abs(part_pid) in list(range(1, 6))+list(range(11, 15)):  # assumes W can't decay to t or tau
                                if d1_ == -1: d1_ = part_idx
                                elif d1_ != part_idx and d2_ == -1: d2_ = part_idx
                                else: badevt = True
                        if (d1_ != -1 and d2_ != -1) or badevt: break
                    if (d1_ != -1 and d2_ != -1) or badevt: break
                d1.append(d1_); d2.append(d2_)
                if d1_ == -1 or d2_ == -1: badevt = True
            d1.end_list(); d2.end_list()
            if badevt: evt_mask.append(0)
            else: evt_mask.append(1)
        return d1, d2, evt_mask

    d1, d2, Wdevt_mask = nbWd(n_targets, particles.idx, particles.pid, particles.m1, Ws.idx, ak.ArrayBuilder(), ak.ArrayBuilder(), ak.ArrayBuilder())
    d1, d2, Wdevt_mask = d1.snapshot(), d2.snapshot(), Wdevt_mask.snapshot()
    return particles[d1][Wdevt_mask], particles[d2][Wdevt_mask], Wdevt_mask


################################
# Targets builder
def get_genparts(arrays, n_tops, n_targets, event_mask):
    ################################
    # Read Delphes file
    part_pid = arrays["GenPart_pdgId"]  # PDG ID
    part_status = arrays["GenPart_status"]
    part_statflag = arrays["GenPart_statusFlags"]
    part_m1 = arrays["GenPart_genPartIdxMother"]
    part_idx = ak.local_index(part_pid)
    # part_d1 = part_idx[np.argsort(part_idx[arrays["GenPart_genPartIdxMother"]])]
    part_pt = arrays["GenPart_pt"]
    part_eta = arrays["GenPart_eta"]
    part_phi = arrays["GenPart_phi"]
    part_mass = arrays["GenPart_mass"]

    
    ################################
    # Build quarks
    particles = ak.zip(
        {
            "pt": part_pt,
            "eta": part_eta,
            "phi": part_phi,
            "mass": part_mass,
            "pid": part_pid,
            "status": part_status,
            "statflag": part_statflag,
            "m1": part_m1,
            "idx": part_idx,
        },
        with_name="Momentum4D",
    )

    ################################
    # Perform pre-selection
    particles = particles[event_mask]

    ################################
    # Find tops and children
    bquarks, bevt_mask = get_bs(particles, n_targets)
    wbosons, Wevt_mask = get_Ws(particles, n_targets)
    topquarks, tevt_mask = get_ts(particles, n_targets, bquarks, wbosons)
    wquarks_d1, wquarks_d2, wdsevt_mask = get_Wds(particles, n_targets, wbosons)

    evt_mask = bevt_mask & Wevt_mask & tevt_mask & wdsevt_mask

    top_sort = ak.argsort(topquarks[evt_mask].pt, ascending=False, axis=1)
    topquarks = topquarks[evt_mask][top_sort]
    bquarks = bquarks[evt_mask][top_sort]
    wbosons = wbosons[evt_mask][top_sort]
    wquarks_d1 = wquarks_d1[evt_mask][top_sort]
    wquarks_d2 = wquarks_d2[evt_mask][top_sort]

    good_tops = np.logical_and(
        np.abs(topquarks.pid) == 6,
        np.logical_and(np.abs(bquarks.pid) == 5, np.abs(wbosons.pid) == 24)
    )
    hadronic_tops = good_tops & np.logical_and(
        np.isin(np.abs(wquarks_d1.pid), list(range(1, 6))),  # assumes W can't decay to t
        np.isin(np.abs(wquarks_d2.pid), list(range(1, 6)))
    )
    print(f"Number of hadronic tops: {ak.sum(hadronic_tops, axis=None)}")
    leptonic_tops = good_tops & np.logical_and(
        np.isin(np.abs(wquarks_d1.pid), list(range(11, 15))),  # assumes W can't decay to tau
        np.isin(np.abs(wquarks_d2.pid), list(range(11, 15)))
    )
    print(f"Number of leptonic tops: {ak.sum(leptonic_tops, axis=None)}")

    ################################
    # Perform hadronic selection
    topquarks = topquarks[hadronic_tops]
    bquarks = bquarks[hadronic_tops]
    wbosons = wbosons[hadronic_tops]
    wquarks_d1 = wquarks_d1[hadronic_tops]
    wquarks_d2 = wquarks_d2[hadronic_tops]

    return evt_mask, particles[evt_mask], topquarks, bquarks, wbosons, wquarks_d1, wquarks_d2


################################
# Inputs builder
def get_jets(arrays, n_tops, n_targets):
    ################################
    # Read Delphes file
    # small-radius jet info
    pt = arrays["Jet_pt"]
    eta = arrays["Jet_eta"]
    phi = arrays["Jet_phi"]
    mass = arrays["Jet_mass"]
    btag = arrays["Jet_btagDeepB"] > 0.6502  # https://btv-wiki.docs.cern.ch/ScaleFactors/Run2UL2016preVFPNanoAODv9/#ak4-b-tagging
    flavor = arrays["Jet_partonFlavour"]

    # large-radius jet info
    fj_pt = arrays["FatJet_pt"]
    fj_eta = arrays["FatJet_eta"]
    fj_phi = arrays["FatJet_phi"]
    fj_mass = arrays["FatJet_mass"]
    fj_TtagRN = random2D(fj_pt, RNG.random(size=ak.sum(ak.num(fj_pt))), ak.ArrayBuilder()).snapshot()  # FatJet_particleNet_TvsQCD
    fj_WtagRN = random2D(fj_pt, RNG.random(size=ak.sum(ak.num(fj_pt))), ak.ArrayBuilder()).snapshot()  # FatJet_particleNet_WvsQCD
    fj_sdmass = arrays["FatJet_msoftdrop"]
    fj_tau21 = np.nan_to_num(arrays["FatJet_tau2"] / arrays["FatJet_tau1"], nan=-1)
    fj_tau32 = np.nan_to_num(arrays["FatJet_tau3"] / arrays["FatJet_tau2"], nan=-1)
    fj_ncharged = np.zeros_like(fj_pt)

    # gen small-radius jet info
    gen_pt = arrays["GenJet_pt"]
    gen_eta = arrays["GenJet_eta"]
    gen_phi = arrays["GenJet_phi"]
    gen_mass = arrays["GenJet_mass"]

    # gen large-radius jet 
    gen_fj_pt = arrays["GenJetAK8_pt"]
    gen_fj_eta = arrays["GenJetAK8_eta"]
    gen_fj_phi = arrays["GenJetAK8_phi"]
    gen_fj_mass = arrays["GenJetAK8_mass"]


    ################################
    # Build (f)jets and gen(f)jets
    jets = ak.zip(
        {
            "pt": pt,
            "eta": eta,
            "phi": phi,
            "mass": mass,
            "idx": ak.local_index(pt),

            "flavor": flavor,
            "btag": btag,
        },
        with_name="Momentum4D",
    )
    fjets = ak.zip(
        {
            "pt": fj_pt,
            "eta": fj_eta,
            "phi": fj_phi,
            "mass": fj_mass,
            "idx": ak.local_index(fj_pt),

            "TtagRN": fj_TtagRN,
            "WtagRN": fj_WtagRN,
            "Ttag": ak.zeros_like(fj_pt),
            "Wtag": ak.zeros_like(fj_pt),
            "sdmass": fj_sdmass,
            "tau21": fj_tau21,
            "tau32": fj_tau32,
            "ncharged": fj_ncharged
        },
        with_name="Momentum4D",
    )
    genjets = ak.zip(
        {
            "pt": gen_pt,
            "eta": gen_eta,
            "phi": gen_phi,
            "mass": gen_mass,
            "idx": ak.local_index(gen_pt),
        },
        with_name="Momentum4D",
    )
    genfjets = ak.zip(
        {
            "pt": gen_fj_pt,
            "eta": gen_fj_eta,
            "phi": gen_fj_phi,
            "mass": gen_fj_mass,
            "idx": ak.local_index(gen_fj_pt),
        },
        with_name="Momentum4D",
    )


    ################################
    # Pre-selection cut(s) and ordering
    #  -> what cuts we apply depends on what phase-space (and benchmark) we're targeting
    event_mask = (ak.num(pt[pt > MIN_JET_PT]) >= 3*n_tops)  # resolved-like training
    print('-'*60)
    print(f'Nevts passing event selection: {ak.sum(event_mask, axis=0)}')

    jet_sort = ak.argsort(jets.pt, ascending=False, axis=-1)
    jets = jets[jet_sort]
    jet_mask = (jets.pt[event_mask] > MIN_JET_PT)
    print(f'Unique Njets passing Jet object selection: {np.unique(ak.num(pt > MIN_JET_PT, axis=1))}')
    print(f'Unique Njets passing event selection and Jet object selection: {np.unique(ak.num(jet_mask, axis=1))}')

    fjet_sort = ak.argsort(fjets.pt, ascending=False, axis=-1)
    fjets = fjets[fjet_sort]
    fjet_mask = (fjets.pt[event_mask] > MIN_FJET_PT)
    print(f'Unique Nfatjets passing FatJet object selection: {np.unique(ak.num(fj_pt > MIN_FJET_PT, axis=1))}')
    print(f'Unique Nfatjets passing event selection and FatJet object selection: {np.unique(ak.num(fjet_mask, axis=1))}')
    print('-'*60)

    global N_JETS, N_FJETS
    if callable(N_JETS): N_JETS = N_JETS(n_tops)
    if callable(N_FJETS): N_FJETS = N_FJETS(n_tops)


    ################################
    # Perform pre-selection and sorting
    # Jets
    jets = jets[event_mask][jet_mask][:, :N_JETS]
    jets['idx'] = ak.local_index(jets.pt, axis=1)

    # FatJets
    fjets = fjets[event_mask][fjet_mask][:, :N_FJETS]
    fjets['idx'] = ak.local_index(fjets.pt, axis=1)

    # Gen Jets
    genjets = genjets[event_mask]

    # Gen FatJets
    genfjets = genfjets[event_mask]

    return event_mask, jets, fjets, genjets, genfjets
