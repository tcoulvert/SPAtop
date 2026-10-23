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


def get_tops(particles: ak.Array):
    return particles[np.abs(particles.pid) == 6]

def get_bs(particles: ak.Array):
    return particles[np.logical_and(
        np.abs(particles.pid) == 5,
        np.abs(particles.pid[particles.m1]) == 6,
    )]

def get_Ws(particles: ak.Array):
    return particles[np.logical_and(
            np.abs(particles.pid) == 24,
            np.abs(particles.pid[particles.m1]) == 6,
        )]

def get_Wds(particles: ak.Array, Ws: ak.Array):
    Wi_d1s, Wi_d2s = [], []
    for Wi in range(Ws.shape[1]):
        Wi_d1s.append(particles[particles.m1 == Ws[:, Wi].idx][:, 0])
        Wi_d2s.append(particles[particles.m1 == Ws[:, Wi].idx][:, 1])
    return np.concatenate(Wi_d1s).T, np.concatenate(Wi_d2s).T


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
    bquarks = get_bs(particles)
    topquarks = particles[bquarks.m1]
    top_sort = ak.argsort(topquarks.pt, ascending=False, axis=-1)
    topquarks = topquarks[top_sort]
    bquarks = bquarks[top_sort]
    wbosons = get_Ws(particles)
    wbosons = wbosons[top_sort]
    wquarks_d1, wquarks_d2 = get_Wds(particles, wbosons)

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

    event_mask = (ak.num(topquarks, axis=1) == n_targets)

    return event_mask, particles[event_mask], topquarks[event_mask], bquarks[event_mask], wbosons[event_mask], wquarks_d1[event_mask], wquarks_d2[event_mask]


################################
# Inputs builder
def get_jets(arrays, n_tops, n_targets):
    ################################
    # Read Delphes file
    # small-radius jet info
    pt = arrays["Jet_pt"]
    eta = arrays["Jet/Jet.Eta"]
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
