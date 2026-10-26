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

def akabs(arr: ak.Array):
    return ak.where(arr >= 0, arr, -1*arr)


def get_particle_from_idx(particles: ak.Array, idxs: ak.Array):
    return ak.Array([
        particles_event[idx_event]
        for particles_event, idx_event in zip(particles, idxs)
    ])

@nb.njit
def nbfinal_particle(
    intermediates_idx: ak.Array, intermediates_pid: ak.Array, intermediates_m1: ak.Array,
    particles_idx: ak.Array, particles_pid: ak.Array, particles_m1: ak.Array,
    final_builder: ak.ArrayBuilder
):
    """Follow each intermediate particle to the last copy of itself in the record.

    A copy is a daughter with the same PID. Either daughter slot may hold it
    (W -> W gamma lists the photon in d1 or d2), and Pythia does not guarantee
    the copy's status code exceeds its parent's (W -> W W at equal status, or a
    status-51 copy under a status-52 parent), so no status condition is used.
    Final-state particles carry d1 = d2 = -1, which must not be used as an index.
    The step cap guards against a malformed record with a cycle.
    """
    for (
        intermediates_idx_event, intermediates_pid_event, intermediates_m1_event,
        particles_idx_event, particles_pid_event, particles_m1_event
    ) in zip(
        intermediates_idx, intermediates_pid, intermediates_m1,
        particles_pid, particles_idx, particles_m1,
    ):
        final_builder.begin_list()
        n_particles = len(particles_pid_event)
        for intermediate_idx, intermediate_pid, intermediate_m1 in zip(
            intermediates_idx_event, intermediates_pid_event, intermediates_m1_event
        ):
            next_idx, next_m1 = intermediate_idx, intermediate_m1
            for _ in range(n_particles):
                for particle_idx, particle_pid, particle_m1 in zip(
                    particles_idx_event, particles_pid_event, particles_m1_event
                ):
                    if particle_m1 >= 0 and particle_m1 == next_idx and particle_pid == intermediate_pid:
                        next_idx = particle_idx
                        next_m1 = particle_m1
                        break
            final_builder.append(next_idx)
        final_builder.end_list()
    return final_builder

def final_particle(intermediates: ak.Array, particles: ak.Array):
    return particles[
        nbfinal_particle(
            intermediates.idx, intermediates.pid, intermediates.m1,
            particles.idx, particles.pid, particles.m1,
            ak.ArrayBuilder()
        ).snapshot()
    ]


def get_ts(particles: ak.Array):
    init_ts = particles[np.logical_and(
        np.abs(particles.pid) == 6,
        np.logical_or(
            np.abs(particles.pid[particles.m1]) == 21,
            np.abs(particles.pid[particles.m1]) < 6,
        )
    )]
    return final_particle(init_ts, particles)

def get_bs(particles: ak.Array):
    init_bs = particles[np.logical_and(
        np.abs(particles.pid) == 5,
        np.abs(particles.pid[particles.m1]) == 6,
    )]
    return final_particle(init_bs, particles)

def get_Ws(particles: ak.Array):
    init_Ws = particles[np.logical_and(
        np.abs(particles.pid) == 24,
        np.abs(particles.pid[particles.m1]) == 6,
    )]
    return final_particle(init_Ws, particles)

def get_Wds(particles: ak.Array, n_targets: int, Ws: ak.Array):

    @nb.njit
    def nbWd(
        n_targets: int, 
        particles_idx: ak.Array, particles_pid: ak.Array, particles_m1: ak.Array, 
        Ws_idx: ak.Array, 
        d1: ak.ArrayBuilder, d2: ak.ArrayBuilder
    ):
        for particles_idx_event, particles_pid_event, particles_m1_event, Ws_idx_event in zip(
            particles_idx, particles_pid, particles_m1, Ws_idx
        ):
            badevt = False
            if len(Ws_idx_event) != n_targets: badevt = True
            n_particles = len(particles_pid_event)
            d1.begin_list(); d2.begin_list()
            for w_idx in Ws_idx_event:
                eff_w_idx = w_idx
                d1_ = -1; d2_ = -1
                for _ in range(n_particles):  # limits total steps to length of particles per event
                    for part_idx, part_pid, part_m1 in zip(particles_idx_event, particles_pid_event, particles_m1_event):
                        if part_m1 == eff_w_idx:
                            if abs(part_pid) == 24: eff_w_idx = part_idx; break
                            elif abs(part_pid) in list(range(1, 6))+list(range(11, 15)):  # assumes W can't decay to t or tau
                                if d1_ == -1: d1_ = part_idx
                                elif d1_ != part_idx and d2_ == -1: d2_ = part_idx
                                else: badevt = True
                        if (d1_ != -1 and d2_ != -1) or badevt: break
                    if (d1_ != -1 and d2_ != -1) or badevt: break
                if (d1_ != -1 and d2_ != -1) and not badevt: d1.append(d1_); d2.append(d2_)
            d1.end_list(); d2.end_list()
        return d1, d2

    init_d1s, init_d2s = nbWd(n_targets, particles.idx, particles.pid, particles.m1, Ws.idx, ak.ArrayBuilder(), ak.ArrayBuilder())
    return (
        final_particle(get_particle_from_idx(particles, init_d1s.snapshot()), particles), 
        final_particle(get_particle_from_idx(particles, init_d2s.snapshot()), particles)
    )


################################
# Targets builder
def get_genparts(arrays, n_tops, n_targets, event_mask, top_type: str="hadronic"):
    ################################
    # Read Delphes file
    part_pid = arrays["GenPart_pdgId"]  # PDG ID
    part_status = arrays["GenPart_status"]
    part_statflag = arrays["GenPart_statusFlags"]
    part_m1 = arrays["GenPart_genPartIdxMother"]
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
            "idx": ak.local_index(part_pid),
        },
        with_name="Momentum4D",
    )

    ################################
    # Perform pre-selection
    particles = particles[event_mask]

    ################################
    # Find tops and children
    topquarks = get_ts(particles)
    bquarks = get_bs(particles)
    wbosons = get_Ws(particles)
    wquarks_d1, wquarks_d2 = get_Wds(particles, n_targets, wbosons)

    evt_mask = ak.to_numpy(
        (ak.num(topquarks, axis=1) == n_targets)
        & (ak.num(bquarks, axis=1) == n_targets)
        & (ak.num(wbosons, axis=1) == n_targets)
        & (ak.num(wquarks_d1, axis=1) == n_targets)
        & (ak.num(wquarks_d2, axis=1) == n_targets)
    ).astype(bool)
    top_sort = ak.argsort(topquarks[evt_mask].pt, ascending=False, axis=-1)
    topquarks = ak.to_regular(topquarks[evt_mask][top_sort], axis=1)
    bquarks = ak.to_regular(bquarks[evt_mask][top_sort], axis=1)
    wbosons = ak.to_regular(wbosons[evt_mask][top_sort], axis=1)
    wquarks_d1 = ak.to_regular(wquarks_d1[evt_mask][top_sort], axis=1)
    wquarks_d2 = ak.to_regular(wquarks_d2[evt_mask][top_sort], axis=1)

    good_tops = (
        (akabs(topquarks.pid) == 6) & (akabs(bquarks.pid) == 5) & (akabs(wbosons.pid) == 24)
    )
    hadronic_tops = good_tops & (  #  assumes W can't decay to t
        (akabs(wquarks_d1.pid) >= 1) & (akabs(wquarks_d1.pid) < 6)
    )
    print(f"Number of hadronic tops: {ak.sum(hadronic_tops, axis=None)}")
    leptonic_tops = good_tops & (  # assumes W can't decay to tau
        (akabs(wquarks_d1.pid) >= 11) & (akabs(wquarks_d1.pid) < 15)
    )
    print(f"Number of leptonic tops: {ak.sum(leptonic_tops, axis=None)}")

    ################################
    # Perform decay channel selection
    if top_type =="hadronic":
        evt_mask[evt_mask] = (ak.num(hadronic_tops, axis=1) == n_targets)
    elif top_type =="leptonic":
        evt_mask[evt_mask] = (ak.num(leptonic_tops, axis=1) == n_targets)
    else:
        evt_mask[evt_mask] = (ak.num(good_tops, axis=1) == n_targets)
    print('-'*60+'\n'+f'Nevts passing Jet (Reco) and Particle (Gen) event pre-selection: {ak.sum(evt_mask, axis=0)}')

    return evt_mask, particles[evt_mask], topquarks, bquarks, wbosons, wquarks_d1, wquarks_d2


################################
# Inputs builder
def get_jets(arrays, n_tops, n_targets, njets, nfjets):
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
    print('-'*60+'\n'+f'Nevts before pre-selection: {ak.num(event_mask, axis=0)}')
    print(f'Nevts passing Jet (Reco) event pre-selection: {ak.sum(event_mask, axis=0)}')

    jet_sort = ak.argsort(jets.pt, ascending=False, axis=-1)
    jets = jets[jet_sort]
    jet_mask = (jets.pt[event_mask] > MIN_JET_PT)

    fjet_sort = ak.argsort(fjets.pt, ascending=False, axis=-1)
    fjets = fjets[fjet_sort]
    fjet_mask = (fjets.pt[event_mask] > MIN_FJET_PT)


    ################################
    # Perform pre-selection and sorting
    # Jets
    jets = jets[event_mask][jet_mask][:, :njets]
    jets['idx'] = ak.local_index(jets.pt, axis=1)

    # FatJets
    fjets = fjets[event_mask][fjet_mask][:, :nfjets]
    fjets['idx'] = ak.local_index(fjets.pt, axis=1)

    # Gen Jets
    genjets = genjets[event_mask]

    # Gen FatJets
    genfjets = genfjets[event_mask]

    return event_mask, jets, fjets, genjets, genfjets
