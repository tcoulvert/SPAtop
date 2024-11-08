import logging
from pathlib import Path

import awkward as ak
import click
import h5py
import numpy as np
import uproot
import vector

import matplotlib.pyplot as plt
import mplhep as hep
import hist
from cycler import cycler
plt.style.use(hep.style.CMS)
plt.rcParams.update({'font.size': 20})
cmap_petroff10 = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]
plt.rcParams.update({"axes.prop_cycle": cycler("color", cmap_petroff10)})

from src.data.delphes.matching import (
    match_fjet_to_jet,
    match_top_to_fjet,
    match_top_to_jet,
)

vector.register_awkward()
vector.register_numba()
ak.numba.register_and_check()

logging.basicConfig(level=logging.INFO)

N_JETS = 16
MIN_JET_PT = 10
MIN_FJET_PT = 200
PROJECT_DIR = Path(__file__).resolve().parents[3]


def to_np_array(ak_array, max_n=10, pad=0):
    return ak.fill_none(ak.pad_none(ak_array, max_n, clip=True, axis=-1), pad).to_numpy()

def final_particle(particle_pdgid, mother_pdgid, particles, final_status=-1, intermediate_particles=None):
    if intermediate_particles is None:
        intermediate_particles = particles[np.logical_and(
            np.abs(particles.pid) == particle_pdgid, np.abs(particles.pid[particles.m1]) == mother_pdgid
        )]
    while ak.any(
        ak.any(
            np.logical_and(
                np.abs(particles.pid[intermediate_particles.d1]) == particle_pdgid,
                intermediate_particles.status != final_status
            ), axis=1
        ), axis=0
    ):
        intermediate_particles = ak.where(
            np.logical_and(
                np.abs(particles.pid[intermediate_particles.d1]) == particle_pdgid,
                intermediate_particles.status != final_status
            ),
            particles[intermediate_particles.d1],
            intermediate_particles
        )
    return intermediate_particles


def get_datasets(arrays, n_tops):  # noqa: C901
    part_pid = arrays["Particle/Particle.PID"]  # PDG ID
    part_status = arrays["Particle/Particle.Status"]
    part_m1 = arrays["Particle/Particle.M1"]
    part_d1 = arrays["Particle/Particle.D1"]
    part_d2 = arrays["Particle/Particle.D2"]
    part_pt = arrays["Particle/Particle.PT"]
    part_eta = arrays["Particle/Particle.Eta"]
    part_phi = arrays["Particle/Particle.Phi"]
    part_mass = arrays["Particle/Particle.Mass"]

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

    particles = ak.zip(
        {
            "pt": part_pt,
            "eta": part_eta,
            "phi": part_phi,
            "mass": part_mass,
            "pid": part_pid,
            "status": part_status,
            "m1": part_m1,
            "d1": part_d1,
            "d2": part_d2,
            "idx": ak.local_index(part_pid),
        },
        with_name="Momentum4D",
    )

    tops_condition = np.logical_and(
        np.abs(particles.pid) == 6, np.logical_and(
            np.abs(particles.pid[particles.d1]) == 5, np.abs(particles.pid[particles.d2]) == 24
        )   # do we know the bquarks are going to be daughter 1? yes, confirmed.
    )
    topquarks = ak.to_regular(particles[tops_condition], axis=1)
    topquark_idx_sort = ak.argsort(topquarks.idx, axis=-1)
    topquarks = ak.to_regular(topquarks[topquark_idx_sort])

    bquarks = ak.to_regular(
        final_particle(
            5, 6, particles,
        ), axis=1
    )
    bquarks = ak.to_regular(bquarks[topquark_idx_sort])

    wbosons = ak.to_regular(
        final_particle(
            24, 6, particles, 
        ), axis=1
    )
    wbosons = ak.to_regular(wbosons[topquark_idx_sort])
    
    wquarks_d1 = ak.to_regular(
        final_particle(
            np.abs(ak.to_regular(particles.pid[wbosons.d1], axis=1)), None, particles, 
            intermediate_particles=particles[wbosons.d1]
        ), axis=1
    )
    wquarks_d2 = ak.to_regular(
        final_particle(
            np.abs(ak.to_regular(particles.pid[wbosons.d2], axis=1)), None, particles, 
            intermediate_particles=particles[wbosons.d2]
        ), axis=1
    )

    jets = ak.zip(
        {
            "pt": pt,
            "eta": eta,
            "phi": phi,
            "mass": mass,
            "flavor": flavor,
            "idx": ak.local_index(pt),
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
        },
        with_name="Momentum4D",
    )
    
    top_idx, top_b_idx, top_q1_idx, top_q2_idx = match_top_to_jet(
        bquarks, wquarks_d1, wquarks_d2, jets, 
        ak.ArrayBuilder(), ak.ArrayBuilder(), ak.ArrayBuilder(), ak.ArrayBuilder()
    )
    top_idx, top_b_idx, top_q1_idx, top_q2_idx = (
        top_idx.snapshot(), top_b_idx.snapshot(), top_q1_idx.snapshot(), top_q2_idx.snapshot()
    )
    top_q_idx = ak.where(top_q1_idx > 0, top_q1_idx, top_q2_idx)
    fj_top_idx, fj_top_bqq_idx, fj_top_bq1_idx, fj_top_bq2_idx, fj_top_qq_idx = match_top_to_fjet(
        bquarks, wquarks_d1, wquarks_d2, fjets, 
        ak.ArrayBuilder(), ak.ArrayBuilder(), ak.ArrayBuilder(), ak.ArrayBuilder(), ak.ArrayBuilder()
    )
    fj_top_idx, fj_top_bqq_idx, fj_top_bq1_idx, fj_top_bq1_idx, fj_top_qq_idx = (
        fj_top_idx.snapshot(), fj_top_bqq_idx.snapshot(), fj_top_bq1_idx.snapshot(), fj_top_bq2_idx.snapshot(), fj_top_qq_idx.snapshot()
    )
    fj_top_bq_idx = ak.where(fj_top_bq1_idx > 0, fj_top_bq1_idx, fj_top_bq2_idx)
    matched_fj_idx = match_fjet_to_jet(fjets, jets, ak.ArrayBuilder()).snapshot()
    # print(f"fjets: \n{fjets}\n{'='*60}")
    # print(f"empty at same places fjets-any: \n{ak.all(ak.num(fjets) == ak.num(fj_top_idx))}\n{'='*60}")
    # print(f"empty at same places fjets-bqq: \n{ak.all(ak.num(fjets) == ak.num(fj_top_bqq_idx))}\n{'='*60}")
    # print(f"empty at same places fjets-bq: \n{ak.all(ak.num(fjets) == ak.num(fj_top_bq_idx))}\n{'='*60}")
    # print(f"empty at same places fjets-qq: \n{ak.all(ak.num(fjets) == ak.num(fj_top_qq_idx))}\n{'='*60}")
    print(f"fjet any boosted: \n{fj_top_idx}\n{'='*60}")
    print(f"fjet all boosted: \n{fj_top_bqq_idx}\n{'='*60}")
    print(f"fjet bq boosted: \n{fj_top_bq_idx}\n{'='*60}")
    print(f"fjet qq boosted: \n{fj_top_qq_idx}\n{'='*60}")
    # total_arr = np.where(fj_top_idx > 0, 1, -1) + np.where(fj_top_bqq_idx > 0, 1, -1) + np.where(fj_top_bq_idx > 0, 1, -1) + np.where(fj_top_qq_idx > 0, 1, -1)
    # print(f"only 2 arrays have fatjets: \n{ak.all((total_arr == 0) | (total_arr == -4))}\n{'='*60}")
    # less_total_arr = np.where(fj_top_bqq_idx > 0, 1, -1) + np.where(fj_top_bq_idx > 0, 1, -1) + np.where(fj_top_qq_idx > 0, 1, -1)
    # print(f"only 1 array has fatjets: \n{ak.all((less_total_arr == -1) | (less_total_arr == -3))}\n{'='*60}")
    print(f"jet any: \n{top_idx}\n{'='*60}")
    print(f"jet t->b: \n{top_b_idx}\n{'='*60}")
    print(f"jet w->q: \n{top_q_idx}\n{'='*60}")
    # total_arr = np.where(top_idx > 0, 1, -1) + np.where(top_b_idx > 0, 1, -1) + np.where(top_q_idx > 0, 1, -1)
    # print(f"only 2 arrays have jets: \n{ak.all((total_arr == 1) | (total_arr == -3))}\n{'='*60}")
    # less_total_arr = np.where(top_b_idx > 0, 1, -1) + np.where(top_q_idx > 0, 1, -1)
    # print(f"only 1 array has jets: \n{ak.all((less_total_arr == 0) | (less_total_arr == -2))}\n{'='*60}")
    # print(f"total arr where problems: \n{ak.drop_none(ak.firsts(total_arr[(total_arr != 1) & (total_arr != -3)]))}")
    # print(f"less total arr where problems: \n{ak.drop_none(ak.firsts(less_total_arr[(less_total_arr != 0) & (less_total_arr != -2)]))}")
    # problem_indices = ak.where(
    #     ak.is_none(ak.firsts(total_arr[(total_arr != 1) & (total_arr != -3)])), -1, range(len(ak.firsts(total_arr[(total_arr != 1) & (total_arr != -3)])))
    # )
    # print(f"top_idx where problems: \n{top_idx[problem_indices != -1]}")
    # print(f"top_b_idx where problems: \n{top_b_idx[problem_indices != -1]}")
    # print(f"top_q_idx where problems: \n{top_q_idx[problem_indices != -1]}")
    # print(f"fatjets where problems: \n{fj_top_idx[problem_indices != -1]}")



    print(f"total number of events = {ak.num(top_idx, axis=0)}")

    # two fully resolved tops
    two_fullyResolved = (
        (ak.sum(top_idx == 1, axis=1) == 3) & (ak.sum(top_idx == 2, axis=1) == 3)
    )
    print(f"number of reco. 2 tops fully-resolved events = {ak.sum(two_fullyResolved)}")
    
    # one fully resolved top, one semi-resolved top
    one_fullyResolved_one_bqFjet = (
        (
            (ak.sum(top_idx == 2, axis=1) == 3) 
            & (ak.sum(top_q_idx == 1, axis=1) == 1) 
            & (ak.sum(fj_top_bq_idx == 1, axis=1) == 1)
        ) | (
            (ak.sum(top_idx == 1, axis=1) == 3) 
            & (ak.sum(top_q_idx == 2, axis=1) == 1) 
            & (ak.sum(fj_top_bq_idx == 2, axis=1) == 1)
        )
    )
    print(f"number of reco. 1 top fully-resolved, 1 top semi-resolved (bq) events = {ak.sum(one_fullyResolved_one_bqFjet)}")
    one_fullyResolved_one_qqFjet = (
        (
            (ak.sum(top_idx == 2, axis=1) == 3) 
            & (ak.sum(top_b_idx == 1, axis=1) == 1) 
            & (ak.sum(fj_top_qq_idx == 1, axis=1) == 1)
        ) | (
            (ak.sum(top_idx == 1, axis=1) == 3) 
            & (ak.sum(top_b_idx == 2, axis=1) == 1) 
            & (ak.sum(fj_top_qq_idx == 2, axis=1) == 1)
        )
    )
    print(f"number of reco. 1 top fully-resolved, 1 top semi-resolved (qq) events = {ak.sum(one_fullyResolved_one_qqFjet)}")
    one_fullyResolved_one_semiResolved = one_fullyResolved_one_bqFjet | one_fullyResolved_one_qqFjet
    print(f"number of reco. 1 top fully-resolved, 1 top semi-resolved (bq or qq) events = {ak.sum(one_fullyResolved_one_semiResolved)}")
    
    # two semi-resolved tops
    two_bqFjet = (
        (ak.sum(top_q_idx == 2, axis=1) == 1) 
        & (ak.sum(fj_top_bq_idx == 2, axis=1) == 1)
        & (ak.sum(top_q_idx == 1, axis=1) == 1) 
        & (ak.sum(fj_top_bq_idx == 1, axis=1) == 1)
    )
    print(f"number of reco. 2 tops semi-resolved (bq) events = {ak.sum(two_bqFjet)}")
    one_bqFjet_one_qqFjet = (
        (
            (ak.sum(top_b_idx == 2, axis=1) == 1) 
            & (ak.sum(fj_top_qq_idx == 2, axis=1) == 1)
            & (ak.sum(top_q_idx == 1, axis=1) == 1) 
            & (ak.sum(fj_top_bq_idx == 1, axis=1) == 1)
        ) | (
            (ak.sum(top_b_idx == 1, axis=1) == 1) 
            & (ak.sum(fj_top_qq_idx == 1, axis=1) == 1)
            & (ak.sum(top_q_idx == 2, axis=1) == 1) 
            & (ak.sum(fj_top_bq_idx == 2, axis=1) == 1)
        )
    )
    print(f"number of reco. 1 top semi-resolved (bq), 1 top semi-resolved (qq) events = {ak.sum(one_bqFjet_one_qqFjet)}")
    two_qqFjet = (
        (ak.sum(top_b_idx == 2, axis=1) == 1) 
        & (ak.sum(fj_top_qq_idx == 2, axis=1) == 1)
        & (ak.sum(top_b_idx == 1, axis=1) == 1) 
        & (ak.sum(fj_top_qq_idx == 1, axis=1) == 1)
    )
    print(f"number of reco. 2 tops semi-resolved (qq) events = {ak.sum(two_qqFjet)}")
    two_semiResolved = two_bqFjet | one_bqFjet_one_qqFjet | two_qqFjet
    print(f"number of reco. 2 tops semi-resolved (bq or qq) events = {ak.sum(two_semiResolved)}")

    # one fully resolved top, one fully-boosted top
    one_fullyResolved_one_fullyBoosted = (
        (
            (ak.sum(top_idx == 2, axis=1) == 3)
            & (ak.sum(fj_top_bqq_idx == 1, axis=1) == 1)
        ) | (
            (ak.sum(top_idx == 1, axis=1) == 3)
            & (ak.sum(fj_top_bqq_idx == 2, axis=1) == 1)
        )
    )
    print(f"number of reco. 1 top fully-resolved, 1 top fully-boosted events = {ak.sum(one_fullyResolved_one_fullyBoosted)}")
    
    # one semi-resolved top, one fully-boosted top
    one_bqFjet_one_bqqFjet = (
        (
            (ak.sum(top_q_idx == 2, axis=1) == 1) 
            & (ak.sum(fj_top_bq_idx == 2, axis=1) == 1)
            & (ak.sum(fj_top_bqq_idx == 1, axis=1) == 1)
        ) | (
            (ak.sum(top_q_idx == 1, axis=1) == 1) 
            & (ak.sum(fj_top_bq_idx == 1, axis=1) == 1)
            & (ak.sum(fj_top_bqq_idx == 2, axis=1) == 1)
        )
    )
    print(f"number of reco. 1 top semi-resolved (bq), 1 top fully-boosted events = {ak.sum(one_bqFjet_one_bqqFjet)}")
    one_qqFjet_one_bqqFjet = (
        (
            (ak.sum(top_b_idx == 2, axis=1) == 1) 
            & (ak.sum(fj_top_qq_idx == 2, axis=1) == 1)
            & (ak.sum(fj_top_bqq_idx == 1, axis=1) == 1)
        ) | (
            (ak.sum(top_b_idx == 1, axis=1) == 1) 
            & (ak.sum(fj_top_qq_idx == 1, axis=1) == 1)
            & (ak.sum(fj_top_bqq_idx == 2, axis=1) == 1)
        )
    )
    print(f"number of reco. 1 top semi-resolved (qq), 1 top fully-boosted events = {ak.sum(one_qqFjet_one_bqqFjet)}")
    one_semiResolved_one_fullyBoosted = one_bqFjet_one_bqqFjet | one_qqFjet_one_bqqFjet
    print(f"number of reco. 1 top semi-resolved (bq or qq), 1 top fully-boosted events = {ak.sum(one_semiResolved_one_fullyBoosted)}")
    
    # two fully-boosted tops
    two_fullyBoosted = (
        (
            (ak.sum(top_q_idx == 2, axis=1) == 1) 
            & (ak.sum(fj_top_bq_idx == 2, axis=1) == 1)
            & (ak.sum(fj_top_bqq_idx == 1, axis=1) == 1)
        ) | (
            (ak.sum(top_q_idx == 1, axis=1) == 1) 
            & (ak.sum(fj_top_bq_idx == 1, axis=1) == 1)
            & (ak.sum(fj_top_bqq_idx == 2, axis=1) == 1)
        )
    )
    print(f"number of reco. 2 tops fully-boosted events = {ak.sum(two_fullyBoosted)}")

    def fiducial_mask(quarks, min_pt=10):
        eta_barrel, eta_endcap, max_eta = 1.4442, 1.566, 2.3

        return np.logical_and(
            np.logical_and(
                np.logical_or(
                    np.abs(quarks.eta) < eta_barrel, np.abs(quarks.eta) > eta_endcap
                ), np.abs(quarks.eta) < max_eta
            ), quarks.pt > min_pt
        )
    
    bquark1_fiducial_mask = fiducial_mask(bquarks[ak.local_index(bquarks) == 0])
    bquark2_fiducial_mask = fiducial_mask(bquarks[ak.local_index(bquarks) == 1])
    w1quark1_fiducial_mask = fiducial_mask(wquarks_d1[ak.local_index(wquarks_d1) == 0])
    w1quark2_fiducial_mask = fiducial_mask(wquarks_d2[ak.local_index(wquarks_d2) == 0])
    w2quark1_fiducial_mask = fiducial_mask(wquarks_d1[ak.local_index(wquarks_d1) == 1])
    w2quark2_fiducial_mask = fiducial_mask(wquarks_d2[ak.local_index(wquarks_d2) == 1])
    measureable_events = bquark1_fiducial_mask & bquark2_fiducial_mask & w1quark1_fiducial_mask & w1quark2_fiducial_mask & w2quark1_fiducial_mask & w2quark2_fiducial_mask

    proper_events = (
        two_fullyResolved | one_fullyResolved_one_semiResolved | one_fullyResolved_one_fullyBoosted
        | two_semiResolved | one_semiResolved_one_fullyBoosted
        | two_fullyBoosted
    ) & measureable_events
    print('-'*60)
    # print(f"number of good reco events = {ak.sum(proper_events)}")
    # print(f"num detectable events = {ak.sum(measureable_events)}")

    print(f"fiducial efficiency = {ak.sum(measureable_events) / ak.num(top_idx, axis=0)}")
    print(f"reco. efficiency = {ak.sum(proper_events) / ak.sum(measureable_events)}")
    print(f"total efficiency = {ak.sum(proper_events) / ak.num(top_idx, axis=0)}")
    
    # def weights_pt_hist(data, pt, n_bins: int=35, min_pt: float=0., max_pt: float=350.):
    #     bin_edges = np.array([(i * (max_pt-min_pt) / n_bins) + min_pt for i in range(n_bins+1)])

    #     weights = np.array([1. for _ in range(np.shape(data)[0])])
    #     for i in range(n_bins):
    #         mask_arr = np.logical_and(pt >= bin_edges[i], pt < bin_edges[i+1])
    #         weights[mask_arr] = weights[mask_arr] * (np.sum(data[mask_arr]) / np.sum(mask_arr)) / np.sum(mask_arr)
        
    #     return weights
    
    # fig, ax = plt.subplots()
    # hep.cms.text("Work in Progress", ax=ax)
    # n_bins, min_pt, max_pt = 35, 0., 350.
    # hist_axis = hist.axis.Regular(n_bins, min_pt, max_pt, name='var', label=r'$p_T$', growth=False, underflow=False, overflow=False)
    # bquark1_hist = hist.Hist(hist_axis, storage='weight').fill(var=bquarks.pt[ak.local_index(bquarks) == 0][bquark1_fiducial_mask], weight=weights_pt_hist(min_dR_bquark1[bquark1_fiducial_mask], bquarks.pt[ak.local_index(bquarks) == 0][bquark1_fiducial_mask], n_bins, min_pt, max_pt))
    # bquark2_hist = hist.Hist(hist_axis, storage='weight').fill(var=bquarks.pt[ak.local_index(bquarks) == 1][bquark2_fiducial_mask], weight=weights_pt_hist(min_dR_bquark2[bquark2_fiducial_mask], bquarks.pt[ak.local_index(bquarks) == 1][bquark2_fiducial_mask], n_bins, min_pt, max_pt))
    # w1quark1_hist = hist.Hist(hist_axis, storage='weight').fill(var=wquarks_d1.pt[ak.local_index(wquarks_d1) == 0][w1quark1_fiducial_mask], weight=weights_pt_hist(min_dR_w1quark1[w1quark1_fiducial_mask], wquarks_d1.pt[ak.local_index(wquarks_d1) == 0][w1quark1_fiducial_mask], n_bins, min_pt, max_pt))
    # w1quark2_hist = hist.Hist(hist_axis, storage='weight').fill(var=wquarks_d2.pt[ak.local_index(wquarks_d2) == 0][w1quark2_fiducial_mask], weight=weights_pt_hist(min_dR_w1quark2[w1quark2_fiducial_mask], wquarks_d2.pt[ak.local_index(wquarks_d2) == 0][w1quark2_fiducial_mask], n_bins, min_pt, max_pt))
    # w2quark1_hist = hist.Hist(hist_axis, storage='weight').fill(var=wquarks_d1.pt[ak.local_index(wquarks_d1) == 1][w2quark1_fiducial_mask], weight=weights_pt_hist(min_dR_w2quark1[w2quark1_fiducial_mask], wquarks_d1.pt[ak.local_index(wquarks_d1) == 1][w2quark1_fiducial_mask], n_bins, min_pt, max_pt))
    # w2quark2_hist = hist.Hist(hist_axis, storage='weight').fill(var=wquarks_d2.pt[ak.local_index(wquarks_d2) == 1][w2quark2_fiducial_mask], weight=weights_pt_hist(min_dR_w2quark2[w2quark2_fiducial_mask], wquarks_d2.pt[ak.local_index(wquarks_d2) == 1][w2quark2_fiducial_mask], n_bins, min_pt, max_pt))
    # hep.histplot(
    #     [bquark1_hist, bquark2_hist, w1quark1_hist, w1quark2_hist, w2quark1_hist, w2quark2_hist], 
    #     yerr=True, alpha=0.5, histtype='step', label=['bquark1', 'bquark2', 'w1quark1', 'w1quark2', 'w2quark1', 'w2quark2']
    # )
    # ax.legend()
    # ax.set_ylabel('Average min($\Delta R$)')
    # plt.savefig('avg_min_deltaR_against_pt.png')
    # plt.close()

    # n_pt_bins, min_pt, max_pt = 35, 0., 350.
    # n_dR_bins, min_dR, max_dR = 15, 0., 1.5
    # for plot, pt_data, dR_data, fid_mask in [
    #     ('bquark1', bquarks.pt[ak.local_index(bquarks) == 0], min_dR_bquark1, bquark1_fiducial_mask), ('bquark2', bquarks.pt[ak.local_index(bquarks) == 1], min_dR_bquark2, bquark2_fiducial_mask),
    #     ('w1quark1', wquarks_d1.pt[ak.local_index(wquarks_d1) == 0], min_dR_w1quark1, w1quark1_fiducial_mask), ('w1quark2', wquarks_d2.pt[ak.local_index(wquarks_d2) == 0], min_dR_w1quark2, w1quark2_fiducial_mask),
    #     ('w2quark1', wquarks_d1.pt[ak.local_index(wquarks_d1) == 1], min_dR_w2quark1, w2quark1_fiducial_mask), ('w2quark2', wquarks_d2.pt[ak.local_index(wquarks_d2) == 1], min_dR_w2quark2, w2quark2_fiducial_mask),
    # ]:
    #     fig, ax = plt.subplots()
    #     hep.cms.text("Work in Progress", ax=ax)
        
    #     pt_axis = hist.axis.Regular(n_pt_bins, min_pt, max_pt, name='pt_var', label=r'$p_T$', growth=False, underflow=False, overflow=False)
    #     deltaR_axis = hist.axis.Regular(n_dR_bins, min_dR, max_dR, name='dr_var', label=r'$\Delta R$', growth=False, underflow=False, overflow=False)
    #     hist_data = hist.Hist(pt_axis, deltaR_axis).fill(pt_var=pt_data[fid_mask], dr_var=dR_data[fid_mask])
    #     hep.hist2dplot(
    #         hist_data, ax=ax,
    #     )
    #     plt.savefig(f'{plot}_min_deltaR_against_pt.png')
    #     plt.close()

    

    # print('-'*60)
    # deltaR_cut = 0.5
    # frac_bquark1 = ak.sum(ak.min(bquarks[ak.local_index(bquarks) == 0].deltaR(jets), axis=1)[bquark1_fiducial_mask] < deltaR_cut) / ak.num(top_idx[bquark1_fiducial_mask], axis=0)
    # print(f"frac bquark1 within {deltaR_cut} = {frac_bquark1}")
    # frac_bquark2 = ak.sum(ak.min(bquarks[ak.local_index(bquarks) == 1].deltaR(jets), axis=1)[bquark2_fiducial_mask] < deltaR_cut) / ak.num(top_idx[bquark2_fiducial_mask], axis=0)
    # print(f"frac bquark2 within {deltaR_cut} = {frac_bquark2}")
    # frac_w1quark1 = ak.sum(ak.min(wquarks_d1[ak.local_index(wquarks_d1) == 0].deltaR(jets), axis=1)[w1quark1_fiducial_mask] < deltaR_cut) / ak.num(top_idx[w1quark1_fiducial_mask], axis=0)
    # print(f"frac w1quark1 within {deltaR_cut} = {frac_w1quark1}")
    # frac_w1quark2 = ak.sum(ak.min(wquarks_d2[ak.local_index(wquarks_d2) == 0].deltaR(jets), axis=1)[w1quark2_fiducial_mask] < deltaR_cut) / ak.num(top_idx[w1quark2_fiducial_mask], axis=0)
    # print(f"frac w1quark2 within {deltaR_cut} = {frac_w1quark2}")
    # frac_w2quark1 = ak.sum(ak.min(wquarks_d1[ak.local_index(wquarks_d1) == 1].deltaR(jets), axis=1)[w2quark1_fiducial_mask] < deltaR_cut) / ak.num(top_idx[w2quark1_fiducial_mask], axis=0)
    # print(f"frac w2quark1 within {deltaR_cut} = {frac_w2quark1}")
    # frac_w2quark2 = ak.sum(ak.min(wquarks_d2[ak.local_index(wquarks_d2) == 1].deltaR(jets), axis=1)[w2quark2_fiducial_mask] < deltaR_cut) / ak.num(top_idx[w2quark2_fiducial_mask], axis=0)
    # print(f"frac w2quark2 within {deltaR_cut} = {frac_w2quark2}")
    # expected_resolved_efficiency = frac_bquark1 * frac_bquark2 * frac_w1quark1 * frac_w1quark2 * frac_w2quark1 * frac_w2quark2
    # print(f"expected fully-resolved efficiency at dR {deltaR_cut} = {expected_resolved_efficiency}")

    # print('-'*60)
    # print(f"w2quark1 deltaR with jets: {wquarks_d2[0, 1].deltaR(jets[0])}")
    # print(f"w2quark1.m1 deltaR with jets: {particles[wquarks_d2.m1][0, 1].deltaR(jets[0])}")
    # print(f"w2quark1.m1 status: {particles.status[wquarks_d2.m1][0, 1]}")
    # print(f"w2quark1.m1.m1 deltaR with jets: {particles[particles.m1[wquarks_d2.m1]][0, 1].deltaR(jets[0])}")
    # print(f"w2quark1.m1.m1 status: {particles.status[particles.m1[wquarks_d2.m1]][0, 1]}")

    # print(f"min Delta pT btwn bquark1 and jets = {ak.min(ak.where(bquarks.pt[ak.local_index(bquarks) == 0] - jets.pt > 0, bquarks.pt[ak.local_index(bquarks) == 0] - jets.pt, -(bquarks.pt[ak.local_index(bquarks) == 0] - jets.pt)), axis=1)}")
    # print(f"min Delta pT btwn bquark2 and jets = {ak.min(ak.where(bquarks.pt[ak.local_index(bquarks) == 1] - jets.pt > 0, bquarks.pt[ak.local_index(bquarks) == 1] - jets.pt, -(bquarks.pt[ak.local_index(bquarks) == 1] - jets.pt)), axis=1)}")
    # print(f"min Delta pT btwn w1quark1 and jets = {ak.min(ak.where(wquarks_d1.pt[ak.local_index(wquarks_d1) == 0] - jets.pt > 0, wquarks_d1.pt[ak.local_index(wquarks_d1) == 0] - jets.pt, -(wquarks_d1.pt[ak.local_index(wquarks_d1) == 0] - jets.pt)), axis=1)}")
    # print(f"min Delta pT btwn w1quark2 and jets = {ak.min(ak.where(wquarks_d2.pt[ak.local_index(wquarks_d2) == 0] - jets.pt > 0, wquarks_d2.pt[ak.local_index(wquarks_d2) == 0] - jets.pt, -(wquarks_d2.pt[ak.local_index(wquarks_d2) == 0] - jets.pt)), axis=1)}")
    # print(f"min Delta pT btwn w2quark1 and jets = {ak.min(ak.where(wquarks_d1.pt[ak.local_index(wquarks_d1) == 1] - jets.pt > 0, wquarks_d1.pt[ak.local_index(wquarks_d1) == 1] - jets.pt, -(wquarks_d1.pt[ak.local_index(wquarks_d1) == 1] - jets.pt)), axis=1)}")
    # print(f"min Delta pT btwn w2quark2 and jets = {ak.min(ak.where(wquarks_d2.pt[ak.local_index(wquarks_d2) == 1] - jets.pt > 0, wquarks_d2.pt[ak.local_index(wquarks_d2) == 1] - jets.pt, -(wquarks_d2.pt[ak.local_index(wquarks_d2) == 1] - jets.pt)), axis=1)}")

    # keep events with >= min_jets small-radius jets
    min_jets = 3 * n_tops
    mask_minjets = ak.num(pt[pt > MIN_JET_PT]) >= min_jets
    # sort by pt
    sorted_by_pt = ak.argsort(pt, ascending=False, axis=-1)
    # sorted = ak.concatenate([sorted_by_pt[btag == 1], sorted_by_pt[btag == 0]], axis=-1)
    btag = btag[sorted_by_pt][mask_minjets]
    pt = pt[sorted_by_pt][mask_minjets]
    eta = eta[sorted_by_pt][mask_minjets]
    phi = phi[sorted_by_pt][mask_minjets]
    mass = mass[sorted_by_pt][mask_minjets]
    flavor = flavor[sorted_by_pt][mask_minjets]
    top_idx = top_idx[sorted_by_pt][mask_minjets]
    matched_fj_idx = matched_fj_idx[sorted_by_pt][mask_minjets]

    # keep only top N_JETS
    btag = btag[:, :N_JETS]
    pt = pt[:, :N_JETS]
    eta = eta[:, :N_JETS]
    phi = phi[:, :N_JETS]
    mass = mass[:, :N_JETS]
    flavor = flavor[:, :N_JETS]
    top_idx = top_idx[:, :N_JETS]
    matched_fj_idx = matched_fj_idx[:, :N_JETS]

    # sort by pt
    sorted_by_fj_pt = ak.argsort(fj_pt, ascending=False, axis=-1)
    fj_pt = fj_pt[sorted_by_fj_pt][mask_minjets]
    fj_eta = fj_eta[sorted_by_fj_pt][mask_minjets]
    fj_phi = fj_phi[sorted_by_fj_pt][mask_minjets]
    fj_mass = fj_mass[sorted_by_fj_pt][mask_minjets]
    fj_sdmass = fj_sdmass[sorted_by_fj_pt][mask_minjets]
    fj_tau21 = fj_tau21[sorted_by_fj_pt][mask_minjets]
    fj_tau32 = fj_tau32[sorted_by_fj_pt][mask_minjets]
    fj_charge = fj_charge[sorted_by_fj_pt][mask_minjets]
    fj_ehadovereem = fj_ehadovereem[sorted_by_fj_pt][mask_minjets]
    fj_neutralenergyfrac = fj_neutralenergyfrac[sorted_by_fj_pt][mask_minjets]
    fj_chargedenergyfrac = fj_chargedenergyfrac[sorted_by_fj_pt][mask_minjets]
    fj_nneutral = fj_nneutral[sorted_by_fj_pt][mask_minjets]
    fj_ncharged = fj_ncharged[sorted_by_fj_pt][mask_minjets]
    fj_top_idx = fj_top_idx[sorted_by_fj_pt][mask_minjets]
    fj_top_bqq_idx = fj_top_bqq_idx[sorted_by_fj_pt][mask_minjets]
    fj_top_bq_idx = fj_top_bq_idx[sorted_by_fj_pt][mask_minjets]
    fj_top_qq_idx = fj_top_qq_idx[sorted_by_fj_pt][mask_minjets]

    # keep only top N_FJETS
    N_FJETS = n_tops
    fj_pt = fj_pt[:, :N_FJETS]
    fj_eta = fj_eta[:, :N_FJETS]
    fj_phi = fj_phi[:, :N_FJETS]
    fj_mass = fj_mass[:, :N_FJETS]
    fj_sdmass = fj_sdmass[:, :N_FJETS]
    fj_tau21 = fj_tau21[:, :N_FJETS]
    fj_tau32 = fj_tau32[:, :N_FJETS]
    fj_charge = fj_charge[:, :N_FJETS]
    fj_ehadovereem = fj_ehadovereem[:, :N_FJETS]
    fj_neutralenergyfrac = fj_neutralenergyfrac[:, :N_FJETS]
    fj_chargedenergyfrac = fj_chargedenergyfrac[:, :N_FJETS]
    fj_nneutral = fj_nneutral[:, :N_FJETS]
    fj_ncharged = fj_ncharged[:, :N_FJETS]
    fj_top_idx = fj_top_idx[:, :N_FJETS]
    fj_top_bqq_idx = fj_top_bqq_idx[:, :N_FJETS]
    fj_top_bq_idx = fj_top_bq_idx[:, :N_FJETS]
    fj_top_qq_idx = fj_top_qq_idx[:, :N_FJETS]

    # add top pT info
    top_pt = topquarks[mask_minjets].pt
    top_pt = ak.fill_none(ak.pad_none(top_pt, target=3, axis=1, clip=True), -1)

    top_pt_dict = {}
    for i in range(n_tops):
        top_pt_dict[f"top{i+1}_pt"] = top_pt[:, i]
        top_pt_dict[f"top{i+1}_bqq_pt"] = top_pt[:, i]
        top_pt_dict[f"top{i+1}_bq_pt"] = top_pt[:, i]
        top_pt_dict[f"top{i+1}_qq_pt"] = top_pt[:, i]

    # mask to define zero-padded small-radius jets
    mask = pt > MIN_JET_PT

    # mask to define zero-padded large-radius jets
    fj_mask = fj_pt > MIN_FJET_PT

    # fully-resolved top mask
    top_fullyResolved_mask = {}
    for i in range(n_tops):
        top_fullyResolved_mask[f"top{i+1}"] = (
            ak.sum(top_idx == i+1, axis=1) == 3
        )
    
    # semi-resolved (qq fatjet) top mask
    top_semiResolved_qq_mask = {}
    for i in range(n_tops):
        top_semiResolved_qq_mask[f"top{i+1}"] = (
            (ak.sum(top_b_idx == i+1, axis=1) == 1) 
            & (ak.sum(fj_top_qq_idx == i+1, axis=1) == 1)
        )
    # semi-resolved (bq fatjet) top mask
    top_semiResolved_bq_mask = {}
    for i in range(n_tops):
        top_semiResolved_bq_mask[f"top{i+1}"] = (
            (ak.sum(top_q_idx == i+1, axis=1) == 1) 
            & (ak.sum(fj_top_bq_idx == i+1, axis=1) == 1)
        )

    # fully-boosted top mask
    top_fullyBoosted_mask = {}
    for i in range(n_tops):
        top_fullyBoosted_mask[f"top{i+1}"] = (
            ak.sum(fj_top_bqq_idx == i+1, axis=1) == 1
        )

    # index of small-radius jet if top is reconstructed
    top_jet_idxs = {}
    for i in range(n_tops):
        top_jet_idxs[f"top{i+1}"] = ak.local_index(top_idx)[top_idx == i+1]
        top_jet_idxs[f"top{i+1}_b"] = ak.local_index(top_b_idx)[top_b_idx == i+1]
        top_jet_idxs[f"top{i+1}_q"] = ak.local_index(top_q_idx)[top_q_idx == i+1]
    # h1_bs = ak.local_index(top_idx)[top_idx == 1]
    # h2_bs = ak.local_index(top_idx)[top_idx == 2]
    # if n_tops == 3:
    #     h3_bs = ak.local_index(higgs_idx)[higgs_idx == 3]

    # index of large-radius jet if Higgs is reconstructed
    top_fjet_idxs = {}
    for i in range(n_tops):
        top_fjet_idxs[f"top{i+1}"] = ak.local_index(fj_top_idx)[fj_top_idx == i+1]
        top_fjet_idxs[f"top{i+1}_bqq"] = ak.local_index(fj_top_bqq_idx)[fj_top_bqq_idx == i+1]
        top_fjet_idxs[f"top{i+1}_bq"] = ak.local_index(fj_top_bq_idx)[fj_top_bq_idx == i+1]
        top_fjet_idxs[f"top{i+1}_qq"] = ak.local_index(fj_top_qq_idx)[fj_top_qq_idx == i+1]
    # h1_bb = ak.local_index(fj_higgs_idx)[fj_higgs_idx == 1]
    # h2_bb = ak.local_index(fj_higgs_idx)[fj_higgs_idx == 2]
    # if n_tops == 3:
    #     h3_bb = ak.local_index(fj_higgs_idx)[fj_higgs_idx == 3]

    # check/fix small-radius jet truth (ensure max 3 small-radius jets per top)
    check, check_b, check_q = [], [], []
    for i in range(n_tops):
        check += np.unique(ak.count(top_jet_idxs[f"top{i+1}"], axis=-1)).to_list()
        check_b += np.unique(ak.count(top_jet_idxs[f"top{i+1}_b"], axis=-1)).to_list()
        check_q += np.unique(ak.count(top_jet_idxs[f"top{i+1}_q"], axis=-1)).to_list()
    if 4 in check: 
        logging.warning(" Some tops match to 4 small-radius jets! Check truth")
    if 2 in check_b:
        logging.warning(" Some tops match to having 2 daughter bjets (i.e. 2 bjets directly from tops)! Check truth")
    if 3 in check_q:
        logging.warning(" Some tops match to having 3 W-daughter jets (i.e. 3 jets directly from Ws)! Check truth")
    print(f"All proper numbers of jets: {ak.all(np.array(check) <= 6) & ak.all(np.array(check_b) <= 2) & ak.all(np.array(check_q) <= 4)}")

    # check/fix large-radius jet truth (ensure max 1 large-radius jet per top)
    fj_check, fj_check_bqq, fj_check_bq, fj_check_qq = [], [], [], []
    for i in range(n_tops):
        fj_check += np.unique(ak.count(top_fjet_idxs[f"top{i+1}"], axis=-1)).to_list()
        fj_check_bqq += np.unique(ak.count(top_fjet_idxs[f"top{i+1}_bqq"], axis=-1)).to_list()
        fj_check_bq += np.unique(ak.count(top_fjet_idxs[f"top{i+1}_bq"], axis=-1)).to_list()
        fj_check_qq += np.unique(ak.count(top_fjet_idxs[f"top{i+1}_qq"], axis=-1)).to_list()
    if 2 in fj_check:
        logging.warning(" Some tops match to 2 large-radius jets in fj_check! Check truth")
    if 2 in fj_check_bqq:
        logging.warning(" Some tops match to 2 large-radius jets in fj_check_bqq! Check truth")
    if 2 in fj_check_bq:
        logging.warning(" Some tops match to 2 large-radius jets in fj_check_bq! Check truth")
    if 2 in fj_check_qq:
        logging.warning(" Some tops match to 2 large-radius jets in fj_check_qq! Check truth")
    print(f"All proper numbers of fjets: {ak.all(np.array(fj_check) <= 2) & ak.all(np.array(fj_check_bqq) <= 2) & ak.all(np.array(fj_check_bq) <= 2) & ak.all(np.array(fj_check_qq) <= 2)}")

    for i in range(n_tops):
        top_jet_idxs[f"top{i+1}"] = ak.fill_none(ak.pad_none(top_jet_idxs[f"top{i+1}"], 3, clip=True), -1)
        top_jet_idxs[f"top{i+1}_b"] = ak.fill_none(ak.pad_none(top_jet_idxs[f"top{i+1}_b"], 1, clip=True), -1)
        top_jet_idxs[f"top{i+1}_q"] = ak.fill_none(ak.pad_none(top_jet_idxs[f"top{i+1}_q"], 2, clip=True), -1)

    # h1_bs = ak.fill_none(ak.pad_none(h1_bs, 2, clip=True), -1)
    # h2_bs = ak.fill_none(ak.pad_none(h2_bs, 2, clip=True), -1)
    # if n_tops == 3:
    #     h3_bs = ak.fill_none(ak.pad_none(h3_bs, 2, clip=True), -1)

    for i in range(n_tops):
        top_fjet_idxs[f"top{i+1}"] = ak.fill_none(ak.pad_none(top_fjet_idxs[f"top{i+1}"], 1, clip=True), -1)
        top_fjet_idxs[f"top{i+1}_bqq"] = ak.fill_none(ak.pad_none(top_fjet_idxs[f"top{i+1}_bqq"], 1, clip=True), -1)
        top_fjet_idxs[f"top{i+1}_bq"] = ak.fill_none(ak.pad_none(top_fjet_idxs[f"top{i+1}_bq"], 1, clip=True), -1)
        top_fjet_idxs[f"top{i+1}_qq"] = ak.fill_none(ak.pad_none(top_fjet_idxs[f"top{i+1}_qq"], 1, clip=True), -1)

    # h1_bb = ak.fill_none(ak.pad_none(h1_bb, 1, clip=True), -1)
    # h2_bb = ak.fill_none(ak.pad_none(h2_bb, 1, clip=True), -1)
    # if n_tops == 3:
    #     h3_bb = ak.fill_none(ak.pad_none(h3_bb, 1, clip=True), -1)

    # h1_b1, h1_b2 = h1_bs[:, 0], h1_bs[:, 1]
    # h2_b1, h2_b2 = h2_bs[:, 0], h2_bs[:, 1]
    # if n_tops == 3:
    #     h3_b1, h3_b2 = h3_bs[:, 0], h3_bs[:, 1]

    # mask whether top can be reconstructed as 3 small-radius jet
    # top_fullyResolved_mask = {}
    # for i in range(n_tops):
    #     top_fullyResolved_mask[f"top{i+1}"] = ak.all(top_jet_idxs[f"top{i+1}"] != -1, axis=-1)

    # top_semiResolved_mask = {}
    # for i in range(n_tops):
    #     top_semiResolved_mask[f"top{i+1}"] = ak.all(
    #         np.logical_or(
    #             np.logical_and(
    #                 top_jet_idxs[f"top{i+1}_b"] != -1, 
    #                 top_fjet_idxs[f"top{i+1}_qq"] != -1
    #             ),
    #             np.logical_and(
    #                 top_jet_idxs[f"top{i+1}_q"] != -1, 
    #                 top_fjet_idxs[f"top{i+1}_bq"] != -1
    #             )
    #         ),
    #         axis=-1
    #     )


    # h1_mask = ak.all(h1_bs != -1, axis=-1)
    # h2_mask = ak.all(h2_bs != -1, axis=-1)
    # if n_tops == 3:
    #     h3_mask = ak.all(h3_bs != -1, axis=-1)

    # mask whether Higgs can be reconstructed as 1 large-radius jet
    # h1_fj_mask = ak.all(h1_bb != -1, axis=-1)
    # h2_fj_mask = ak.all(h2_bb != -1, axis=-1)
    # if n_tops == 3:
    #     h3_fj_mask = ak.all(h3_bb != -1, axis=-1)

    datasets = {}
    datasets["INPUTS/Jets/MASK"] = to_np_array(mask, max_n=N_JETS).astype("bool")
    datasets["INPUTS/Jets/pt"] = to_np_array(pt, max_n=N_JETS).astype("float32")
    datasets["INPUTS/Jets/eta"] = to_np_array(eta, max_n=N_JETS).astype("float32")
    datasets["INPUTS/Jets/phi"] = to_np_array(phi, max_n=N_JETS).astype("float32")
    datasets["INPUTS/Jets/sinphi"] = to_np_array(np.sin(phi), max_n=N_JETS).astype("float32")
    datasets["INPUTS/Jets/cosphi"] = to_np_array(np.cos(phi), max_n=N_JETS).astype("float32")
    datasets["INPUTS/Jets/mass"] = to_np_array(mass, max_n=N_JETS).astype("float32")
    datasets["INPUTS/Jets/btag"] = to_np_array(btag, max_n=N_JETS).astype("float32")
    datasets["INPUTS/Jets/flavor"] = to_np_array(flavor, max_n=N_JETS).astype("float32")
    datasets["INPUTS/Jets/matchedfj"] = to_np_array(matched_fj_idx, max_n=N_JETS).astype("int32")

    datasets["INPUTS/BoostedJets/MASK"] = to_np_array(fj_mask, max_n=N_FJETS).astype("bool")
    datasets["INPUTS/BoostedJets/fj_pt"] = to_np_array(fj_pt, max_n=N_FJETS).astype("float32")
    datasets["INPUTS/BoostedJets/fj_eta"] = to_np_array(fj_eta, max_n=N_FJETS).astype("float32")
    datasets["INPUTS/BoostedJets/fj_phi"] = to_np_array(fj_phi, max_n=N_FJETS).astype("float32")
    datasets["INPUTS/BoostedJets/fj_sinphi"] = to_np_array(np.sin(fj_phi), max_n=N_FJETS).astype("float32")
    datasets["INPUTS/BoostedJets/fj_cosphi"] = to_np_array(np.cos(fj_phi), max_n=N_FJETS).astype("float32")
    datasets["INPUTS/BoostedJets/fj_mass"] = to_np_array(fj_mass, max_n=N_FJETS).astype("float32")
    datasets["INPUTS/BoostedJets/fj_sdmass"] = to_np_array(fj_sdmass, max_n=N_FJETS).astype("float32")
    datasets["INPUTS/BoostedJets/fj_tau21"] = to_np_array(fj_tau21, max_n=N_FJETS).astype("float32")
    datasets["INPUTS/BoostedJets/fj_tau32"] = to_np_array(fj_tau32, max_n=N_FJETS).astype("float32")
    datasets["INPUTS/BoostedJets/fj_charge"] = to_np_array(fj_charge, max_n=N_FJETS)
    datasets["INPUTS/BoostedJets/fj_ehadovereem"] = to_np_array(fj_ehadovereem, max_n=N_FJETS)
    datasets["INPUTS/BoostedJets/fj_neutralenergyfrac"] = to_np_array(fj_neutralenergyfrac, max_n=N_FJETS)
    datasets["INPUTS/BoostedJets/fj_chargedenergyfrac"] = to_np_array(fj_chargedenergyfrac, max_n=N_FJETS)
    datasets["INPUTS/BoostedJets/fj_nneutral"] = to_np_array(fj_nneutral, max_n=N_FJETS)
    datasets["INPUTS/BoostedJets/fj_ncharged"] = to_np_array(fj_ncharged, max_n=N_FJETS)

    # fully-resolved tops
    for i in range(n_tops):
        datasets[f"TARGETS/frt{i+1}/mask"] = top_fullyResolved_mask[f"top{i+1}"]
        datasets[f"TARGETS/frt{i+1}/b"] = top_jet_idxs[f"top{i+1}_b"]
        datasets[f"TARGETS/frt{i+1}/q1"] = top_jet_idxs[f"top{i+1}_q"]
        datasets[f"TARGETS/frt{i+1}/q2"] = top_jet_idxs[f"top{i+1}_q"]
        datasets[f"TARGETS/frt{i+1}/pt"] = top_pt_dict[f"top{i+1}_pt"]

    # semi-resolved (qq fatjet) tops
    for i in range(n_tops):
        datasets[f"TARGETS/srqqt{i+1}/mask"] = top_semiResolved_qq_mask[f"top{i+1}"]
        datasets[f"TARGETS/srqqt{i+1}/b"] = top_jet_idxs[f"top{i+1}_b"]
        datasets[f"TARGETS/srqqt{i+1}/qq"] = top_fjet_idxs[f"top{i+1}_qq"]
        datasets[f"TARGETS/srqqt{i+1}/pt"] = top_pt_dict[f"top{i+1}_qq_pt"]

    # semi-resolved (bq fatjet) tops
    for i in range(n_tops):
        datasets[f"TARGETS/srbqt{i+1}/mask"] = top_semiResolved_bq_mask[f"top{i+1}"]
        datasets[f"TARGETS/srbqt{i+1}/q"] = top_jet_idxs[f"top{i+1}_q"]
        datasets[f"TARGETS/srbqt{i+1}/bq"] = top_fjet_idxs[f"top{i+1}_bq"]
        datasets[f"TARGETS/srbqt{i+1}/pt"] = top_pt_dict[f"top{i+1}_bq_pt"]

    # fully-boosted tops
    for i in range(n_tops):
        datasets[f"TARGETS/fbt{i+1}/mask"] = top_semiResolved_bq_mask[f"top{i+1}"]
        datasets[f"TARGETS/fbt{i+1}/bqq"] = top_fjet_idxs[f"top{i+1}_bqq"]
        datasets[f"TARGETS/fbt{i+1}/pt"] = top_pt_dict[f"top{i+1}_bqq_pt"]

    # datasets["TARGETS/h1/mask"] = h1_mask.to_numpy()
    # datasets["TARGETS/h1/b1"] = h1_b1.to_numpy()
    # datasets["TARGETS/h1/b2"] = h1_b2.to_numpy()
    # datasets["TARGETS/h1/pt"] = h1_pt.to_numpy()

    # datasets["TARGETS/h2/mask"] = h2_mask.to_numpy()
    # datasets["TARGETS/h2/b1"] = h2_b1.to_numpy()
    # datasets["TARGETS/h2/b2"] = h2_b2.to_numpy()
    # datasets["TARGETS/h2/pt"] = h2_pt.to_numpy()

    # if n_tops == 3:
    #     datasets["TARGETS/h3/mask"] = h3_mask.to_numpy()
    #     datasets["TARGETS/h3/b1"] = h3_b1.to_numpy()
    #     datasets["TARGETS/h3/b2"] = h3_b2.to_numpy()
    #     datasets["TARGETS/h3/pt"] = h3_pt.to_numpy()

    # datasets["TARGETS/bh1/mask"] = h1_fj_mask.to_numpy()
    # datasets["TARGETS/bh1/bb"] = h1_bb.to_numpy().reshape(h1_fj_mask.to_numpy().shape)
    # datasets["TARGETS/bh1/pt"] = bh1_pt.to_numpy()

    # datasets["TARGETS/bh2/mask"] = h2_fj_mask.to_numpy()
    # datasets["TARGETS/bh2/bb"] = h2_bb.to_numpy().reshape(h2_fj_mask.to_numpy().shape)
    # datasets["TARGETS/bh2/pt"] = bh2_pt.to_numpy()

    # if n_tops == 3:
    #     datasets["TARGETS/bh3/mask"] = h3_fj_mask.to_numpy()
    #     datasets["TARGETS/bh3/bb"] = h3_bb.to_numpy().reshape(h3_fj_mask.to_numpy().shape)
    #     datasets["TARGETS/bh3/pt"] = bh3_pt.to_numpy()

    return datasets


@click.command()
@click.argument("in-files", nargs=-1)
@click.option(
    "--out-file",
    default=f"{PROJECT_DIR}/data/delphes/tt_training.h5",
    help="Output file.",
)
@click.option("--train-frac", default=0.50, help="Fraction for training.")
@click.option(
    "--n-tops",
    "n_tops",
    default=2,
    type=click.IntRange(2, 4),
    help="Number of top quarks per event",
)
def main(in_files, out_file, train_frac, n_tops):
    all_datasets = {}
    for file_name in in_files:
        with uproot.open(file_name) as in_file:
            events = in_file["Delphes"]
            num_entries = events.num_entries
            if "training" in out_file:
                entry_start = None
                entry_stop = int(train_frac * num_entries)
            else:
                entry_start = int(train_frac * num_entries)
                entry_stop = None

            # for key in events.keys():
            #     print(f"{key}\n{'-'*60}")
            # print('='*60)
            # print('='*60)
            # print('='*60)

            keys = (
                [key for key in events.keys() if "Particle/Particle." in key and "fBits" not in key]
                + [key for key in events.keys() if "Jet/Jet." in key]
                + [key for key in events.keys() if "FatJet/FatJet." in key and "fBits" not in key]
            )

            # for key in keys:
            #     print(f"{key}\n{'-'*60}")
            
            arrays = events.arrays(keys, entry_start=entry_start, entry_stop=entry_stop)
            datasets = get_datasets(arrays, n_tops)
            for dataset_name, data in datasets.items():
                if dataset_name not in all_datasets:
                    all_datasets[dataset_name] = []
                all_datasets[dataset_name].append(data)

    with h5py.File(out_file, "w") as output:
        for dataset_name, all_data in all_datasets.items():
            concat_data = np.concatenate(all_data, axis=0)
            logging.info(f"Dataset name: {dataset_name}")
            logging.info(f"Dataset shape: {concat_data.shape}")
            output.create_dataset(dataset_name, data=concat_data)


if __name__ == "__main__":
    main()
