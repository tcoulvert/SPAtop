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

RNG = np.random.default_rng(seed=21)

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


@nb.njit
def nbfinal_particle(
    intermediates_idx: ak.Array, intermediates_pid: ak.Array, intermediates_d1: ak.Array, intermediates_d2: ak.Array,
    particles_pid: ak.Array, particles_d1: ak.Array, particles_d2: ak.Array,
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
        intermediates_idx_event, intermediates_pid_event, intermediates_d1_event, intermediates_d2_event,
        particles_pid_event, particles_d1_event, particles_d2_event
    ) in zip(
        intermediates_idx, intermediates_pid, intermediates_d1, intermediates_d2,
        particles_pid, particles_d1, particles_d2
    ):
        final_builder.begin_list()
        n_particles = len(particles_pid_event)
        for intermediate_idx, intermediate_pid, intermediate_d1, intermediate_d2 in zip(
            intermediates_idx_event, intermediates_pid_event, intermediates_d1_event, intermediates_d2_event
        ):
            next_idx, next_d1, next_d2 = intermediate_idx, intermediate_d1, intermediate_d2
            for _ in range(n_particles):
                if next_d1 >= 0 and particles_pid_event[next_d1] == intermediate_pid:
                    next_idx = next_d1
                elif next_d2 >= 0 and particles_pid_event[next_d2] == intermediate_pid:
                    next_idx = next_d2
                else:
                    break
                next_d1 = particles_d1_event[next_idx]
                next_d2 = particles_d2_event[next_idx]
            final_builder.append(next_idx)
        final_builder.end_list()
    return final_builder

def final_particle(intermediates: ak.Array, particles: ak.Array):
    return particles[
        nbfinal_particle(
            intermediates.idx, intermediates.pid, intermediates.d1, intermediates.d2,
            particles.pid, particles.d1, particles.d2,
            ak.ArrayBuilder()
        ).snapshot()
    ]

def get_tops(particles: ak.Array):
    init_tops = particles[
        np.logical_and(
            np.abs(particles.pid) == 6,
            np.logical_or(
                np.logical_and(np.abs(particles.pid[particles.d1]) == 5, np.abs(particles.pid[particles.d2]) == 24),
                np.logical_and(np.abs(particles.pid[particles.d2]) == 5, np.abs(particles.pid[particles.d1]) == 24)
            )
        )
    ]
    return final_particle(init_tops, particles)

def get_bs(particles: ak.Array, tops: ak.Array):
    init_bs = ak.where(
        np.abs(particles.pid[tops.d1]) == 5, particles[tops.d1], particles[tops.d2]
    )
    return final_particle(init_bs, particles)

def get_Ws(particles: ak.Array, tops: ak.Array):
    init_Ws = ak.where(
        np.abs(particles.pid[tops.d1]) == 24, particles[tops.d1], particles[tops.d2]
    )
    return final_particle(init_Ws, particles)

def get_Wds(particles: ak.Array, Ws: ak.Array):
    init_Wd1s, init_Wd2s = particles[Ws.d1], particles[Ws.d2]
    return final_particle(init_Wd1s, particles), final_particle(init_Wd2s, particles)


################################
# Targets builder
def get_genparts(arrays, n_tops, n_targets, event_mask, top_type: str="hadronic"):
    ################################
    # Read Delphes file
    part_pid = arrays["Particle/Particle.PID"]  # PDG ID
    part_status = arrays["Particle/Particle.Status"]
    part_m1 = arrays["Particle/Particle.M1"]
    part_m2 = arrays["Particle/Particle.M2"]
    part_d1 = arrays["Particle/Particle.D1"]
    part_d2 = arrays["Particle/Particle.D2"]
    part_pt = arrays["Particle/Particle.PT"]
    part_eta = arrays["Particle/Particle.Eta"]
    part_phi = arrays["Particle/Particle.Phi"]
    part_mass = arrays["Particle/Particle.Mass"]

    
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
            "m1": part_m1,
            "m2": part_m2,
            "d1": part_d1,
            "d2": part_d2,
            "idx": ak.local_index(part_pid),
        },
        with_name="Momentum4D",
    )

    ################################
    # Perform pre-selection
    particles = particles[event_mask]

    ################################
    # Find tops and children
    topquarks = get_tops(particles)
    bquarks = get_bs(particles, topquarks)
    wbosons = get_Ws(particles, topquarks)
    wquarks_d1, wquarks_d2 = get_Wds(particles, wbosons)

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
    pt = arrays["Jet/Jet.PT"]
    eta = arrays["Jet/Jet.Eta"]
    phi = arrays["Jet/Jet.Phi"]
    mass = arrays["Jet/Jet.Mass"]
    btag = arrays["Jet/Jet.BTag"]
    flavor = arrays["Jet/Jet.Flavor"]

    # large-radius jet info
    fj_pt = arrays["FatJet/FatJet.PT"]
    fj_eta = arrays["FatJet/FatJet.Eta"]
    fj_phi = arrays["FatJet/FatJet.Phi"]
    fj_mass = arrays["FatJet/FatJet.Mass"]
    fj_TtagRN = random2D(fj_pt, RNG.random(size=ak.sum(ak.num(fj_pt))), ak.ArrayBuilder()).snapshot()
    fj_WtagRN = random2D(fj_pt, RNG.random(size=ak.sum(ak.num(fj_pt))), ak.ArrayBuilder()).snapshot()
    fj_sdp4 = arrays["FatJet/FatJet.SoftDroppedP4[5]"]
    # first entry (i = 0) is the total SoftDropped Jet 4-momenta
    # from i = 1 to 4 are the pruned subjets 4-momenta
    fj_sdmass2 = (
        fj_sdp4.fE[..., 0] ** 2 - fj_sdp4.fP.fX[..., 0] ** 2 - fj_sdp4.fP.fY[..., 0] ** 2 - fj_sdp4.fP.fZ[..., 0] ** 2
    )
    fj_sdmass = np.sqrt(np.maximum(fj_sdmass2, 0))
    fj_taus = arrays["FatJet/FatJet.Tau[5]"]
    # just saving just tau21 and tau32, can save others if useful
    fj_tau21 = np.nan_to_num(fj_taus[..., 1] / fj_taus[..., 0], nan=-1)
    fj_tau32 = np.nan_to_num(fj_taus[..., 2] / fj_taus[..., 1], nan=-1)
    fj_charge = arrays["FatJet/FatJet.Charge"]
    fj_ehadovereem = arrays["FatJet/FatJet.EhadOverEem"]
    fj_neutralenergyfrac = arrays["FatJet/FatJet.NeutralEnergyFraction"]
    fj_chargedenergyfrac = arrays["FatJet/FatJet.ChargedEnergyFraction"]
    fj_nneutral = arrays["FatJet/FatJet.NNeutrals"]
    fj_ncharged = arrays["FatJet/FatJet.NCharged"]

    # gen small-radius jet info
    gen_pt = arrays["GenJet/GenJet.PT"]
    gen_eta = arrays["GenJet/GenJet.Eta"]
    gen_phi = arrays["GenJet/GenJet.Phi"]
    gen_mass = arrays["GenJet/GenJet.Mass"]

    # gen large-radius jet 
    gen_fj_pt = arrays["GenFatJet/GenFatJet.PT"]
    gen_fj_eta = arrays["GenFatJet/GenFatJet.Eta"]
    gen_fj_phi = arrays["GenFatJet/GenFatJet.Phi"]
    gen_fj_mass = arrays["GenFatJet/GenFatJet.Mass"]


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
            "charge": fj_charge,
            "ehadovereem": fj_ehadovereem,
            "neutralenergyfrac": fj_neutralenergyfrac,
            "chargedenergyfrac": fj_chargedenergyfrac,
            "nneutral": fj_nneutral,
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
