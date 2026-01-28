import glob
import logging
import os
import re
import subprocess
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

from src.data.delphes.matching import (
    match_fjet_to_jet,
    match_top_to_fjet,
    match_top_to_jet,
)

vector.register_awkward()
vector.register_numba()
ak.numba.register_and_check()

logging.basicConfig(level=logging.INFO)

################################


MIN_JET_PT = 10  # 20
MIN_FJET_PT = 100  # 200
PROJECT_DIR = Path(__file__).resolve().parents[3]

PLOTS = True
RNG = np.random.default_rng(seed=21)

TVSQCD_EFFS = {'t':83.21e-2, 'W':6.90e-2, 'bq': 19.65e-2, 'QCD': 1e-2}
WVSQCD_EFFS = {'t':15.19e-2, 'W':56.15e-2, 'bq': 9.92e-2, 'QCD': 1e-2}

################################


def to_np_array(ak_array, max_n=10, pad=0):
    return ak.fill_none(ak.pad_none(ak_array, max_n, clip=True, axis=-1), pad).to_numpy()

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

def final_particle(particle_pdgid, mother_pdgid, particles, final_status=-1, intermediate_particles=None):
    if intermediate_particles is None:
        intermediate_particles = particles[np.logical_and(
            np.abs(particles.pid) == particle_pdgid, np.abs(particles.pid[particles.m1]) == mother_pdgid
        )]
    while ak.any(
        ak.any(
            np.logical_and(
                np.abs(particles.pid[intermediate_particles.d1]) == particle_pdgid,
                intermediate_particles.status < final_status
            ), axis=1
        ), axis=0
    ):
        intermediate_particles = ak.where(
            np.logical_and(
                np.abs(particles.pid[intermediate_particles.d1]) == particle_pdgid,
                intermediate_particles.status < final_status
            ),
            particles[intermediate_particles.d1],
            intermediate_particles
        )
    return intermediate_particles


def get_datasets(arrays, n_tops):  # noqa: C901
    print('='*60+'\n'+'='*60+'\n'+'='*60)
    print(f'num events = {len(arrays["Particle/Particle.PID"])}')

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
    # fj_Ttag = arrays["Jet/Jet.TvsQCD"]
    # fj_Wtag = arrays["Jet/Jet.WvsQCD"]
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
        np.abs(particles.pid) == 6, np.logical_or(
            np.logical_and(
                np.abs(particles.pid[particles.d1]) == 24, np.abs(particles.pid[particles.d2]) == 5
            ),
            np.logical_and(
                np.abs(particles.pid[particles.d1]) == 5, np.abs(particles.pid[particles.d2]) == 24
            )
        )
    )
    topquarks = ak.to_regular(particles[tops_condition], axis=1)
    print(f"2 tops in every event? = {ak.all(ak.num(topquarks) == 2)}")
    topquark_idx_sort = ak.argsort(topquarks.idx, axis=-1)
    topquarks = ak.to_regular(topquarks[topquark_idx_sort])

    bquarks = ak.to_regular(
        final_particle(
            5, 6, particles,
        ), axis=1
    )
    bquarks = ak.to_regular(bquarks[topquark_idx_sort])
    print(f"2 bquarks in every event? = {ak.all(ak.num(bquarks) == 2)}")

    wbosons = ak.to_regular(
        final_particle(
            24, 6, particles, 
        ), axis=1
    )
    wbosons = ak.to_regular(wbosons[topquark_idx_sort])
    print(f"2 wbosons in every event? = {ak.all(ak.num(wbosons) == 2)}")
    
    wquarks_d1 = ak.to_regular(
        final_particle(
            np.abs(ak.to_regular(particles.pid[wbosons.d1], axis=1)), None, particles, 
            intermediate_particles=particles[wbosons.d1]
        ), axis=1
    )
    print(f"2 wquarks_d1 in every event? = {ak.all(ak.num(wquarks_d1) == 2)}")
    wquarks_d2 = ak.to_regular(
        final_particle(
            np.abs(ak.to_regular(particles.pid[wbosons.d2], axis=1)), None, particles, 
            intermediate_particles=particles[wbosons.d2]
        ), axis=1
    )
    print(f"2 wquarks_d2 in every event? = {ak.all(ak.num(wquarks_d2) == 2)}")



    def fid_mask(particle, pt_threshold=8, max_eta=2.3, barrel_endcap_gap=(1.4, 1.6)):
        pt_mask = ak.sum(particle['pt'] > pt_threshold, axis=1) == n_tops
        eta_mask = ak.sum(
            (
                np.abs(particle['eta']) < barrel_endcap_gap[0]
            ) | np.logical_and(
                np.abs(particle['eta']) > barrel_endcap_gap[1],
                np.abs(particle['eta']) < max_eta
            ), axis=1
        ) == n_tops

        return pt_mask & eta_mask

    # fiducial mask for the quarks
    quark_fid_mask = (
        fid_mask(bquarks)
        & fid_mask(wquarks_d1)
        & fid_mask(wquarks_d2)
    )
    print('-'*60)
    print(f'num events with all quarks passing fiducial cuts = {ak.sum(quark_fid_mask)}')

    jets = ak.zip(
        {
            "pt": pt,
            "eta": eta,
            "phi": phi,
            "mass": mass,
            "flavor": flavor,
            "btag": btag,
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
            "TtagRN": fj_TtagRN,
            "WtagRN": fj_WtagRN
        },
        with_name="Momentum4D",
    )

    gen_jets = ak.zip(
        {
            "pt": gen_pt,
            "eta": gen_eta,
            "phi": gen_phi,
            "mass": gen_mass,
            "idx": ak.local_index(gen_pt),
        },
        with_name="Momentum4D",
    )
    gen_fjets = ak.zip(
        {
            "pt": gen_fj_pt,
            "eta": gen_fj_eta,
            "phi": gen_fj_phi,
            "mass": gen_fj_mass,
            "idx": ak.local_index(gen_fj_pt),
        },
        with_name="Momentum4D",
    )


    
    # Fully-Resolved tops
    top_idx, top_b_idx, top_q1_idx, top_q2_idx = match_top_to_jet(
        bquarks, wquarks_d1, wquarks_d2, jets, 
        ak.ArrayBuilder(), ak.ArrayBuilder(), ak.ArrayBuilder(), ak.ArrayBuilder()
    )
    top_idx, top_b_idx, top_q1_idx, top_q2_idx = (
        top_idx.snapshot(), top_b_idx.snapshot(), top_q1_idx.snapshot(), top_q2_idx.snapshot()
    )
    # Fully-Boosted and Semi-Resolved tops
    fj_top_idx, fj_top_bqq_idx, fj_top_qq_idx, fj_top_bq1_idx, fj_top_bq2_idx = match_top_to_fjet(
        topquarks, bquarks, wbosons, wquarks_d1, wquarks_d2, fjets, 
        ak.ArrayBuilder(), ak.ArrayBuilder(), ak.ArrayBuilder(), ak.ArrayBuilder(), ak.ArrayBuilder()
    )
    fj_top_idx, fj_top_bqq_idx, fj_top_qq_idx, fj_top_bq1_idx, fj_top_bq2_idx = (
        fj_top_idx.snapshot(), 
        fj_top_bqq_idx.snapshot(), fj_top_qq_idx.snapshot(), 
        fj_top_bq1_idx.snapshot(), fj_top_bq2_idx.snapshot()
    )
    matched_fj_j_idx, matched_fj_j_DR  = match_fjet_to_jet(fjets, jets, ak.ArrayBuilder(), ak.ArrayBuilder())
    matched_fj_j_idx, matched_fj_j_DR = matched_fj_j_idx.snapshot(), matched_fj_j_DR.snapshot()

    # # keep events with >= min_jets -> what cuts we apply depends on what phase-space (and benchmark) we're targeting
    # event_mask = ak.num(pt[pt > MIN_JET_PT]) >= 3*n_tops  # phase-space cuts for resolved-like training
    # print(f"Num events with >=3 jets per top = {ak.sum(event_mask, axis=0)}")
    # print(f"    -> Num events with >=3 jets per top & quark fiducial mask = {ak.sum(event_mask & quark_fid_mask, axis=0)}")
    # print('-'*60)

    excl_FBandSR_bool = (
        ~ak.all( (fj_top_bqq_idx > 0) & (fj_top_qq_idx > 0) )  # FB and SRqq
        & ~ak.all( (fj_top_bqq_idx > 0) & (fj_top_bq1_idx > 0) )  # FB and SRbq1
        & ~ak.all( (fj_top_bqq_idx > 0) & (fj_top_bq2_idx > 0) )  # FB and SRbq2
        & ~ak.all( (fj_top_qq_idx > 0) & (fj_top_bq1_idx > 0) )  # SRqq and SRbq1
        & ~ak.all( (fj_top_qq_idx > 0) & (fj_top_bq2_idx > 0) )  # SRqq and SRbq2
        & ~ak.all( (fj_top_bq1_idx > 0) & (fj_top_bq2_idx > 0) )  # SRbq1 and SRbq2
    )
    print(f"Exclusive FB and SRs? {excl_FBandSR_bool}")
    

    top_fjets_mask = (fj_top_bqq_idx > 0)
    w_fjets_mask = (fj_top_qq_idx > 0)
    bq_fjets_mask = ( (fj_top_bq1_idx > 0) | (fj_top_bq2_idx > 0) )
    qcd_fjets_mask = ( ~top_fjets_mask & ~w_fjets_mask & ~bq_fjets_mask )

    # apply emulated TvsQCD and WvsQCD bools @ 1.0% QCD eff WPs
    fjets["Ttag"] = ak.where(
        (
            ( top_fjets_mask & (fjets["TtagRN"] < TVSQCD_EFFS['t']) )
            | ( w_fjets_mask & (fjets["TtagRN"] < TVSQCD_EFFS['W']) )
            | ( bq_fjets_mask & (fjets["TtagRN"] < TVSQCD_EFFS['bq']) )
            | ( qcd_fjets_mask & (fjets["TtagRN"] < TVSQCD_EFFS['QCD']) )
        ), 1, 0
    )
    fjets["Wtag"] = ak.where(
        (
            ( top_fjets_mask & (fjets["WtagRN"] < WVSQCD_EFFS['t']) )
            | ( w_fjets_mask & (fjets["WtagRN"] < WVSQCD_EFFS['W']) )
            | ( bq_fjets_mask & (fjets["WtagRN"] < WVSQCD_EFFS['bq']) )
            | ( qcd_fjets_mask & (fjets["WtagRN"] < WVSQCD_EFFS['QCD']) )
        ), 1, 0
    )
    fj_Ttag = fjets["Ttag"]
    fj_Wtag = fjets["Wtag"]

    

    ## Jets ##
    # sort by pt
    sorted_by_pt = ak.argsort(pt, ascending=False, axis=-1)
    btag = btag[sorted_by_pt][event_mask]
    pt = pt[sorted_by_pt][event_mask]
    eta = eta[sorted_by_pt][event_mask]
    phi = phi[sorted_by_pt][event_mask]
    mass = mass[sorted_by_pt][event_mask]
    flavor = flavor[sorted_by_pt][event_mask]
    top_idx = top_idx[sorted_by_pt][event_mask]
    top_b_idx = top_b_idx[sorted_by_pt][event_mask]
    top_q1_idx = top_q1_idx[sorted_by_pt][event_mask]
    top_q2_idx = top_q2_idx[sorted_by_pt][event_mask]
    matched_fj_j_idx = matched_fj_j_idx[sorted_by_pt][event_mask]

    if PLOTS:
        # jet pt
        plot_destdir = os.path.join(PROJECT_DIR, 'plots', 'jet_pt')
        if not os.path.exists(plot_destdir):
            os.makedirs(plot_destdir)

        jet_pt_axis = hist.axis.Regular(50, 0., 500., name='var', label=r'$p_T$ [GeV]', growth=False, underflow=False, overflow=False)
        for local_idx in range(4):
            jet_i_hist = hist.Hist(jet_pt_axis).fill(var=ak.firsts(pt[ak.local_index(pt) == local_idx]))
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

        for jet_type, gen_jet_arr, reco_jet_arr in [('ak5', gen_jets, jets), ('ak8', gen_fjets, fjets)]:
            
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

    # keep only top N_JETS
    N_JETS = 3*n_tops + 4
    btag = btag[:, :N_JETS]
    pt = pt[:, :N_JETS]
    eta = eta[:, :N_JETS]
    phi = phi[:, :N_JETS]
    mass = mass[:, :N_JETS]
    flavor = flavor[:, :N_JETS]
    top_idx = top_idx[:, :N_JETS]
    top_b_idx = top_b_idx[:, :N_JETS]
    top_q1_idx = top_q1_idx[:, :N_JETS]
    top_q2_idx = top_q2_idx[:, :N_JETS]
    matched_fj_j_idx = matched_fj_j_idx[:, :N_JETS]

    ## FatJets ##
    # sort by pt
    sorted_by_fj_pt = ak.argsort(fj_pt, ascending=False, axis=-1)
    fj_pt = fj_pt[sorted_by_fj_pt][event_mask]
    fj_eta = fj_eta[sorted_by_fj_pt][event_mask]
    fj_phi = fj_phi[sorted_by_fj_pt][event_mask]
    fj_mass = fj_mass[sorted_by_fj_pt][event_mask]
    fj_sdmass = fj_sdmass[sorted_by_fj_pt][event_mask]
    fj_Ttag = fj_Ttag[sorted_by_fj_pt][event_mask]
    fj_Wtag = fj_Wtag[sorted_by_fj_pt][event_mask]
    fj_tau21 = fj_tau21[sorted_by_fj_pt][event_mask]
    fj_tau32 = fj_tau32[sorted_by_fj_pt][event_mask]
    fj_charge = fj_charge[sorted_by_fj_pt][event_mask]
    fj_ehadovereem = fj_ehadovereem[sorted_by_fj_pt][event_mask]
    fj_neutralenergyfrac = fj_neutralenergyfrac[sorted_by_fj_pt][event_mask]
    fj_chargedenergyfrac = fj_chargedenergyfrac[sorted_by_fj_pt][event_mask]
    fj_nneutral = fj_nneutral[sorted_by_fj_pt][event_mask]
    fj_ncharged = fj_ncharged[sorted_by_fj_pt][event_mask]
    fj_top_idx = fj_top_idx[sorted_by_fj_pt][event_mask]
    fj_top_bqq_idx = fj_top_bqq_idx[sorted_by_fj_pt][event_mask]
    fj_top_qq_idx = fj_top_qq_idx[sorted_by_fj_pt][event_mask]
    fj_top_bq1_idx = fj_top_bq1_idx[sorted_by_fj_pt][event_mask]
    fj_top_bq2_idx = fj_top_bq2_idx[sorted_by_fj_pt][event_mask]

    # keep only top N_FJETS
    N_FJETS = n_tops
    fj_pt = fj_pt[:, :N_FJETS]
    fj_eta = fj_eta[:, :N_FJETS]
    fj_phi = fj_phi[:, :N_FJETS]
    fj_mass = fj_mass[:, :N_FJETS]
    fj_sdmass = fj_sdmass[:, :N_FJETS]
    fj_Ttag = fj_Ttag[:, :N_FJETS]
    fj_Wtag = fj_Wtag[:, :N_FJETS]
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
    fj_top_qq_idx = fj_top_qq_idx[:, :N_FJETS]
    fj_top_bq1_idx = fj_top_bq1_idx[:, :N_FJETS]
    fj_top_bq2_idx = fj_top_bq2_idx[:, :N_FJETS]

    # add top pT info
    top_pt = topquarks[event_mask].pt
    top_pt = ak.fill_none(ak.pad_none(top_pt, target=n_tops, axis=1, clip=True), -1)
    top_pt_dict = {}
    for i in range(n_tops):
        top_pt_dict[f"top{i+1}_pt"] = top_pt[:, i].to_numpy()

    # mask to define zero-padded small-radius jets
    mask = pt > MIN_JET_PT
    # mask to define zero-padded large-radius jets
    fj_mask = fj_pt > MIN_FJET_PT
    
    # fully-resolved
    top_fullyResolved = {}
    for i in range(n_tops):
        top_fullyResolved[f"top{i+1}_b"] = ak.local_index(top_b_idx)[top_b_idx == i+1]
        top_fullyResolved[f"top{i+1}_q1"] = ak.local_index(top_q1_idx)[top_q1_idx == i+1]
        top_fullyResolved[f"top{i+1}_q2"] = ak.local_index(top_q2_idx)[top_q2_idx == i+1]
        top_fullyResolved[f"top{i+1}_mask"] = ak.fill_none(
            ( ak.sum(top_idx == i+1, axis=1) == 3 )
            & ( top_fullyResolved[f"top{i+1}_b"] != top_fullyResolved[f"top{i+1}_q1"] )
            & ( top_fullyResolved[f"top{i+1}_b"] != top_fullyResolved[f"top{i+1}_q2"] )
            & ( top_fullyResolved[f"top{i+1}_q1"] != top_fullyResolved[f"top{i+1}_q2"] ),
            False
        )
        print(f'top {i+1} - num fully-resolved tops = {ak.sum(top_fullyResolved[f"top{i+1}_mask"])}')
    print(f'num fully-resolved tops = {sum([ak.sum(top_fullyResolved[f"top{i+1}_mask"]) for i in range(n_tops)])}')
    
    # semi-resolved (qq fatjet)
    top_semiResolved_qq = {}
    for i in range(n_tops):
        top_semiResolved_qq[f"top{i+1}_b"] = ak.local_index(top_b_idx)[top_b_idx == i+1]
        top_semiResolved_qq[f"top{i+1}_qq"] = ak.local_index(fj_top_qq_idx)[fj_top_qq_idx == i+1]
        top_semiResolved_qq[f"top{i+1}_mask"] = (
            (ak.sum(top_b_idx == i+1, axis=1) == 1) 
            & (ak.sum(fj_top_qq_idx == i+1, axis=1) == 1)
        )
        print(f'top {i+1} - num qq tops = {ak.sum(top_semiResolved_qq[f"top{i+1}_mask"])}')
    print(f'num qq tops = {sum([ak.sum(top_semiResolved_qq[f"top{i+1}_mask"]) for i in range(n_tops)])}')
    
    # semi-resolved (bq fatjet)
    top_semiResolved_bq = {}
    for i in range(n_tops):
        top_semiResolved_bq[f"top{i+1}_mask"] = (bq2_mask | bq1_mask)
        top_semiResolved_bq[f"top{i+1}_q"] = ak.where(
            bq2_mask, 
            ak.local_index(top_q1_idx)[top_q1_idx == i+1], 
            ak.where(
                bq1_mask,
                ak.local_index(top_q2_idx)[top_q2_idx == i+1], 
                ak.local_index(top_q1_idx)[top_q1_idx == i+1]
            )
        )
        top_semiResolved_bq[f"top{i+1}_bq"] = ak.where(
            bq2_mask, 
            ak.local_index(fj_top_bq2_idx)[fj_top_bq2_idx == i+1], 
            ak.where(
                bq1_mask,
                ak.local_index(fj_top_bq1_idx)[fj_top_bq1_idx == i+1],
                ak.local_index(fj_top_bq2_idx)[fj_top_bq2_idx == i+1]
            )
        )
        bq2_mask = (
            (ak.sum(top_q1_idx == i+1, axis=1) == 1) 
            & (ak.sum(fj_top_bq2_idx == i+1, axis=1) == 1)
        )
        bq1_mask = (
            (ak.sum(top_q2_idx == i+1, axis=1) == 1) 
            & (ak.sum(fj_top_bq1_idx == i+1, axis=1) == 1)
        )
        print(f'top {i+1} - num bq tops = {ak.sum(top_semiResolved_bq[f"top{i+1}_mask"])}')
    print(f'num bq tops = {sum([ak.sum(top_semiResolved_bq[f"top{i+1}_mask"]) for i in range(n_tops)])}')

    # fully-boosted
    top_fullyBoosted = {}
    for i in range(n_tops):
        top_fullyBoosted[f"top{i+1}_bqq"] = ak.local_index(fj_top_bqq_idx)[fj_top_bqq_idx == i+1]
        top_fullyBoosted[f"top{i+1}_mask"] = (
            ak.sum(fj_top_bqq_idx == i+1, axis=1) == 1
        )
        print(f'top {i+1} - num bqq tops = {ak.sum(top_fullyBoosted[f"top{i+1}_mask"])}')
    print(f'num bqq tops = {sum([ak.sum(top_fullyBoosted[f"top{i+1}_mask"]) for i in range(n_tops)])}')
    print(f'num reco tops = {sum([ak.sum(top_fullyResolved[f"top{i+1}_mask"]) for i in range(n_tops)]+[ak.sum(top_semiResolved_qq[f"top{i+1}_mask"]) for i in range(n_tops)]+[ak.sum(top_semiResolved_bq[f"top{i+1}_mask"]) for i in range(n_tops)]+[ak.sum(top_fullyBoosted[f"top{i+1}_mask"]) for i in range(n_tops)])}')

    print('-='*60)


    ## Check data ##
    # check fully-resolved tops
    check_b, check_q1, check_q2 = [], [], []
    for i in range(n_tops):
        check_b += np.unique(ak.count(top_fullyResolved[f"top{i+1}_b"], axis=-1)).to_list()
        check_q1 += np.unique(ak.count(top_fullyResolved[f"top{i+1}_q1"], axis=-1)).to_list()
        check_q2 += np.unique(ak.count(top_fullyResolved[f"top{i+1}_q2"], axis=-1)).to_list()
    if 2 in check_b: 
        logging.warning(" Some fully-resolved tops match to 2 bjets! Check truth")
    if 2 in check_q1:
        logging.warning(" Some fully-resolved tops match to 2 wjet1s! Check truth")
    if 2 in check_q2:
        logging.warning(" Some fully-resolved tops match to 2 wjet2s! Check truth")
    print(f"All proper numbers of jets for fully-resolved: {ak.all(np.array(check_b) < 2) & ak.all(np.array(check_q1) < 2) & ak.all(np.array(check_q2) < 2)}")

    # check semi-resolved (qq fatjet) tops
    check_b, check_qq = [], []
    for i in range(n_tops):
        check_b += np.unique(ak.count(top_semiResolved_qq[f"top{i+1}_b"], axis=-1)).to_list()
        check_qq += np.unique(ak.count(top_semiResolved_qq[f"top{i+1}_qq"], axis=-1)).to_list()
    if 2 in check_b: 
        logging.warning(" Some semi-resolved (qq) tops match to 2 bjets! Check truth")
    if 2 in check_qq:
        logging.warning(" Some semi-resolved (qq) tops match to 2 qq fatjets! Check truth")
    print(f"All proper numbers of fatjets/jets for semi-resolved (qq fatjet): {ak.all(np.array(check_b) < 2) & ak.all(np.array(check_qq) < 2)}")

    # check semi-resolved (bq fatjet) tops
    check_q, check_bq = [], []
    for i in range(n_tops):
        check_q += np.unique(ak.count(top_semiResolved_bq[f"top{i+1}_q"], axis=-1)).to_list()
        check_bq += np.unique(ak.count(top_semiResolved_bq[f"top{i+1}_bq"], axis=-1)).to_list()
    if 2 in check_q: 
        logging.warning(" Some semi-resolved (bq) tops match to 2 wjets! Check truth")
    if 2 in check_bq:
        logging.warning(" Some semi-resolved (bq) tops match to 2 bq fatjets! Check truth")
    print(f"All proper numbers of fatjets/jets for semi-resolved (bq fatjet): {ak.all(np.array(check_q) < 2) & ak.all(np.array(check_bq) < 2)}")

    # check/fix large-radius jet truth (ensure max 1 large-radius jet per top)
    fj_check_bqq = []
    for i in range(n_tops):
        fj_check_bqq += np.unique(ak.count(top_fullyBoosted[f"top{i+1}_bqq"], axis=-1)).to_list()
    if 2 in fj_check_bqq:
        logging.warning(" Some fully-boosted tops match to 2 fatjets! Check truth")
    print(f"All proper numbers of fatjets for fully-boosted: {ak.all(np.array(fj_check_bqq) < 2)}")

    ## Clip data ##
    for i in range(n_tops):
        # fully-resolved
        top_fullyResolved[f"top{i+1}_mask"] = top_fullyResolved[f"top{i+1}_mask"].to_numpy()
        top_fullyResolved[f"top{i+1}_b"] = ak.fill_none(ak.firsts(top_fullyResolved[f"top{i+1}_b"]), -1).to_numpy()
        top_fullyResolved[f"top{i+1}_q1"] = ak.fill_none(ak.firsts(top_fullyResolved[f"top{i+1}_q1"]), -1).to_numpy()
        top_fullyResolved[f"top{i+1}_q2"] = ak.fill_none(ak.firsts(top_fullyResolved[f"top{i+1}_q2"]), -1).to_numpy()

        # semi-resolved (qq fatjet)
        top_semiResolved_qq[f"top{i+1}_mask"] = top_semiResolved_qq[f"top{i+1}_mask"].to_numpy()
        top_semiResolved_qq[f"top{i+1}_b"] = ak.fill_none(ak.firsts(top_semiResolved_qq[f"top{i+1}_b"]), -1).to_numpy()
        top_semiResolved_qq[f"top{i+1}_qq"] = ak.fill_none(ak.firsts(top_semiResolved_qq[f"top{i+1}_qq"]), -1).to_numpy()

        # semi-resolved (bq fatjet)
        top_semiResolved_bq[f"top{i+1}_mask"] = top_semiResolved_bq[f"top{i+1}_mask"].to_numpy()
        top_semiResolved_bq[f"top{i+1}_q"] = ak.fill_none(ak.firsts(top_semiResolved_bq[f"top{i+1}_q"]), -1).to_numpy()
        top_semiResolved_bq[f"top{i+1}_bq"] = ak.fill_none(ak.firsts(top_semiResolved_bq[f"top{i+1}_bq"]), -1).to_numpy()

        # fully-boosted
        top_fullyBoosted[f"top{i+1}_mask"] = top_fullyBoosted[f"top{i+1}_mask"].to_numpy()
        top_fullyBoosted[f"top{i+1}_bqq"] = ak.fill_none(ak.firsts(top_fullyBoosted[f"top{i+1}_bqq"]), -1).to_numpy()

    ## Store processed data in dataset for training/testing ##
    # Store the model inputs
    datasets = {}
    datasets["INPUTS/Jets/MASK"] = to_np_array(mask, max_n=N_JETS).astype("bool")
    datasets["INPUTS/Jets/pt"] = to_np_array(pt, max_n=N_JETS).astype("float32")
    datasets["INPUTS/Jets/eta"] = to_np_array(eta, max_n=N_JETS).astype("float32")
    datasets["INPUTS/Jets/phi"] = to_np_array(phi, max_n=N_JETS).astype("float32")
    datasets["INPUTS/Jets/sinphi"] = to_np_array(np.sin(phi), max_n=N_JETS).astype("float32")
    datasets["INPUTS/Jets/cosphi"] = to_np_array(np.cos(phi), max_n=N_JETS).astype("float32")
    datasets["INPUTS/Jets/mass"] = to_np_array(mass, max_n=N_JETS).astype("float32")
    datasets["INPUTS/Jets/btag"] = to_np_array(btag, max_n=N_JETS).astype("bool")
    datasets["INPUTS/Jets/flavor"] = to_np_array(flavor, max_n=N_JETS).astype("float32")
    datasets["INPUTS/Jets/matchedfj"] = to_np_array(matched_fj_j_idx, max_n=N_JETS).astype("int32")
    datasets["INPUTS/Jets/deltaRfj"] = to_np_array(matched_fj_j_DR, max_n=N_JETS).astype("int32")

    datasets["INPUTS/BoostedJets/MASK"] = to_np_array(fj_mask, max_n=N_FJETS).astype("bool")
    datasets["INPUTS/BoostedJets/fj_pt"] = to_np_array(fj_pt, max_n=N_FJETS).astype("float32")
    datasets["INPUTS/BoostedJets/fj_eta"] = to_np_array(fj_eta, max_n=N_FJETS).astype("float32")
    datasets["INPUTS/BoostedJets/fj_phi"] = to_np_array(fj_phi, max_n=N_FJETS).astype("float32")
    datasets["INPUTS/BoostedJets/fj_sinphi"] = to_np_array(np.sin(fj_phi), max_n=N_FJETS).astype("float32")
    datasets["INPUTS/BoostedJets/fj_cosphi"] = to_np_array(np.cos(fj_phi), max_n=N_FJETS).astype("float32")
    datasets["INPUTS/BoostedJets/fj_mass"] = to_np_array(fj_mass, max_n=N_FJETS).astype("float32")
    datasets["INPUTS/BoostedJets/fj_sdmass"] = to_np_array(fj_sdmass, max_n=N_FJETS).astype("float32")
    datasets["INPUTS/BoostedJets/fj_Ttag"] = to_np_array(fj_Ttag, max_n=N_FJETS).astype("bool")
    datasets["INPUTS/BoostedJets/fj_Wtag"] = to_np_array(fj_Wtag, max_n=N_FJETS).astype("bool")
    datasets["INPUTS/BoostedJets/fj_tau21"] = to_np_array(fj_tau21, max_n=N_FJETS).astype("float32")
    datasets["INPUTS/BoostedJets/fj_tau32"] = to_np_array(fj_tau32, max_n=N_FJETS).astype("float32")
    datasets["INPUTS/BoostedJets/fj_charge"] = to_np_array(fj_charge, max_n=N_FJETS).astype("float32")
    datasets["INPUTS/BoostedJets/fj_ehadovereem"] = to_np_array(fj_ehadovereem, max_n=N_FJETS).astype("float32")
    datasets["INPUTS/BoostedJets/fj_neutralenergyfrac"] = to_np_array(fj_neutralenergyfrac, max_n=N_FJETS).astype("float32")
    datasets["INPUTS/BoostedJets/fj_chargedenergyfrac"] = to_np_array(fj_chargedenergyfrac, max_n=N_FJETS).astype("float32")
    datasets["INPUTS/BoostedJets/fj_nneutral"] = to_np_array(fj_nneutral, max_n=N_FJETS).astype("float32")
    datasets["INPUTS/BoostedJets/fj_ncharged"] = to_np_array(fj_ncharged, max_n=N_FJETS).astype("float32")

    # Store the truth-level info
    for i in range(n_tops):
        # fully-resolved tops
        datasets[f"TARGETS/FRt{i+1}/mask"] = top_fullyResolved[f"top{i+1}_mask"]
        datasets[f"TARGETS/FRt{i+1}/b"] = top_fullyResolved[f"top{i+1}_b"]
        datasets[f"TARGETS/FRt{i+1}/q1"] = top_fullyResolved[f"top{i+1}_q1"]
        datasets[f"TARGETS/FRt{i+1}/q2"] = top_fullyResolved[f"top{i+1}_q2"]
        datasets[f"TARGETS/FRt{i+1}/pt"] = top_pt_dict[f"top{i+1}_pt"]

        # semi-resolved (qq fatjet) tops
        datasets[f"TARGETS/SRqqt{i+1}/mask"] = top_semiResolved_qq[f"top{i+1}_mask"]
        datasets[f"TARGETS/SRqqt{i+1}/b"] = top_semiResolved_qq[f"top{i+1}_b"]
        datasets[f"TARGETS/SRqqt{i+1}/qq"] = top_semiResolved_qq[f"top{i+1}_qq"]
        datasets[f"TARGETS/SRqqt{i+1}/pt"] = top_pt_dict[f"top{i+1}_pt"]

        # semi-resolved (bq fatjet) tops
        datasets[f"TARGETS/SRbqt{i+1}/mask"] = top_semiResolved_bq[f"top{i+1}_mask"]
        datasets[f"TARGETS/SRbqt{i+1}/q"] = top_semiResolved_bq[f"top{i+1}_q"]
        datasets[f"TARGETS/SRbqt{i+1}/bq"] = top_semiResolved_bq[f"top{i+1}_bq"]
        datasets[f"TARGETS/SRbqt{i+1}/pt"] = top_pt_dict[f"top{i+1}_pt"]

        # fully-boosted tops
        datasets[f"TARGETS/FBt{i+1}/mask"] = top_fullyBoosted[f"top{i+1}_mask"]
        datasets[f"TARGETS/FBt{i+1}/bqq"] = top_fullyBoosted[f"top{i+1}_bqq"]
        datasets[f"TARGETS/FBt{i+1}/pt"] = top_pt_dict[f"top{i+1}_pt"]

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
@click.option("--plots", is_flag=True, help="Boolean to make plots.")
def main(in_files, out_file, train_frac, n_tops, plots):
    if plots:
        PLOTS = True
    
    all_datasets = {}
    expanded_in_files = []
    print(f"starting in_files = {in_files}")

    for file_name in in_files:
        if re.match('root://', in_files):
            subprocess.run(['xrdfs', '//'.join(in_files.split('//')[:2])+'/', 'ls', '/'+in_files.split('//')[2], '>', 'xrdfs_output.txt'])
            with open('xrdfs_output.txt', 'r') as f:
                expanded_in_files = ['//'.join(in_files.split('//')[:2])+line.strip() for line in f]
        elif os.path.isdir(file_name):
            print(f"found directory: {file_name}")
            for root_file in glob.glob(os.path.join(file_name, '*.root')):
                expanded_in_files.append(os.path.join(file_name, root_file))
        else:
            expanded_in_files.append(file_name)
    in_files = expanded_in_files
    print(f"ending in_files = {in_files}")
    for file_name in in_files:
        try:
            if re.match('root://', file_name):
                subprocess.run(['xrdcp', '-f', file_name, 'tmp.root'])
                current_file_name = 'tmp.root'
            else:
                current_file_name = file_name
            with uproot.open(current_file_name) as in_file:
                events = in_file["Delphes"]
                num_entries = events.num_entries
                if "training" in out_file:
                    entry_start = None
                    entry_stop = int(train_frac * num_entries)
                else:
                    entry_start = int(train_frac * num_entries)
                    entry_stop = None

                keys = (
                    [key for key in events.keys() if "Particle/Particle." in key and "fBits" not in key]
                    + [key for key in events.keys() if "Jet/Jet." in key]
                    + [key for key in events.keys() if "FatJet/FatJet." in key and "fBits" not in key]
                    + [key for key in events.keys() if "GenJet/GenJet." in key and "fBits" not in key]
                    + [key for key in events.keys() if "GenFatJet/GenFatJet." in key and "fBits" not in key]
                )
                
                arrays = events.arrays(keys, entry_start=entry_start, entry_stop=entry_stop)
                datasets = get_datasets(arrays, n_tops)
                for dataset_name, data in datasets.items():
                    if dataset_name not in all_datasets:
                        all_datasets[dataset_name] = []
                    all_datasets[dataset_name].append(data)
            if re.match('root://', file_name): subprocess.run(['rm', '-rf', 'tmp.root'])
        except:
            logging.info(f"Preprocessing failed for file:\n{file_name}\n\n   ...continuing with other files")

    with h5py.File(out_file, "w") as output:
        for dataset_name, all_data in all_datasets.items():
            concat_data = np.concatenate(all_data, axis=0)
            logging.info(f"Dataset name: {dataset_name}")
            logging.info(f"Dataset shape: {concat_data.shape}")
            output.create_dataset(dataset_name, data=concat_data)


if __name__ == "__main__":
    main()
