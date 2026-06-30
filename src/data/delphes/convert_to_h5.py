import glob
import logging
import os
import re
import subprocess
from multiprocessing import Pool
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
    reconstruct_top,
    FullyResolved_top, SemiResolvedQQ_top, SemiResolvedBQ_top, FullyBoosted_top, 
    match_fjet_to_jet,
)
from src.data.delphes.condor_conversion import LPCVanillaSubmitter

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

# def awkidx(record):
#     return record.idx
# def typer(array_view_type):
#     return array_view_type
# def lower(context, builder, sig, args):
#     return context.compile_internal(builder, awkidx, sig, args)
# ak.behavior["__numba_typer__", "Momentum4D", "idx"] = typer
# ak.behavior["__numba_lower__", "Momentum4D", "idx"] = lower


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

    
    ################################
    # Read Delphes file
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



    ################################
    # Build (f)jets and gen(f)jets
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
            "WtagRN": fj_WtagRN,
            "Ttag": ak.zeros_like(fj_pt),
            "Wtag": ak.zeros_like(fj_pt),
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


    
    ################################
    # Pre-selection cut(s) and ordering
    #  -> what cuts we apply depends on what phase-space (and benchmark) we're targeting

    topquark_mask = (topquarks['pt'] > 0)  # In case we want cuts on quarks, for now cuts do nothing
    bquark_mask = topquark_mask & (bquarks['pt'] > 0)
    wboson_mask = topquark_mask & (wbosons['pt'] > 0)
    w1quark_mask = topquark_mask & wboson_mask & (wquarks_d1['pt'] > 0)
    w2quark_mask = topquark_mask & wboson_mask & (wquarks_d1['pt'] > 0)
    
    event_mask = (
        (ak.sum(topquark_mask & bquark_mask & wboson_mask & w1quark_mask & w2quark_mask, axis=1) > 0)
        & (ak.num(pt[pt > MIN_JET_PT]) >= 3*n_tops)  # resolved-like training
    )

    jet_sort = ak.argsort(pt, ascending=False, axis=-1)
    jet_mask = (pt[event_mask] > MIN_JET_PT)

    fjet_sort = ak.argsort(fj_pt, ascending=False, axis=-1)
    fjet_mask = (fj_pt[event_mask] > MIN_FJET_PT)

    N_JETS = 3*n_tops + 4
    N_FJETS = n_tops + 1


    ################################
    # Perform pre-selection and sorting
    # Jets
    jets = jets[jet_sort][event_mask][jet_mask][:, :N_JETS]
    pt = pt[jet_sort][event_mask][jet_mask][:, :N_JETS]
    eta = eta[jet_sort][event_mask][jet_mask][:, :N_JETS]
    phi = phi[jet_sort][event_mask][jet_mask][:, :N_JETS]
    mass = mass[jet_sort][event_mask][jet_mask][:, :N_JETS]
    btag = btag[jet_sort][event_mask][jet_mask][:, :N_JETS]
    flavor = flavor[jet_sort][event_mask][jet_mask][:, :N_JETS]
    jets['idx'] = ak.local_index(pt, axis=1)

    # FatJets
    fjets = fjets[fjet_sort][event_mask][fjet_mask][:, :N_FJETS]
    fj_pt = fj_pt[fjet_sort][event_mask][fjet_mask][:, :N_FJETS]
    fj_eta = fj_eta[fjet_sort][event_mask][fjet_mask][:, :N_FJETS]
    fj_phi = fj_phi[fjet_sort][event_mask][fjet_mask][:, :N_FJETS]
    fj_mass = fj_mass[fjet_sort][event_mask][fjet_mask][:, :N_FJETS]
    fj_TtagRN = fj_TtagRN[fjet_sort][event_mask][fjet_mask][:, :N_FJETS]
    fj_WtagRN = fj_WtagRN[fjet_sort][event_mask][fjet_mask][:, :N_FJETS]
    fj_sdp4 = fj_sdp4[fjet_sort][event_mask][fjet_mask][:, :N_FJETS]
    fj_sdmass2 = fj_sdmass2[fjet_sort][event_mask][fjet_mask][:, :N_FJETS]
    fj_sdmass = fj_sdmass[fjet_sort][event_mask][fjet_mask][:, :N_FJETS]
    fj_taus = fj_taus[fjet_sort][event_mask][fjet_mask][:, :N_FJETS]
    fj_tau21 = fj_tau21[fjet_sort][event_mask][fjet_mask][:, :N_FJETS]
    fj_tau32 = fj_tau32[fjet_sort][event_mask][fjet_mask][:, :N_FJETS]
    fj_charge = fj_charge[fjet_sort][event_mask][fjet_mask][:, :N_FJETS]
    fj_ehadovereem = fj_ehadovereem[fjet_sort][event_mask][fjet_mask][:, :N_FJETS]
    fj_neutralenergyfrac = fj_neutralenergyfrac[fjet_sort][event_mask][fjet_mask][:, :N_FJETS]
    fj_chargedenergyfrac = fj_chargedenergyfrac[fjet_sort][event_mask][fjet_mask][:, :N_FJETS]
    fj_nneutral = fj_nneutral[fjet_sort][event_mask][fjet_mask][:, :N_FJETS]
    fj_ncharged = fj_ncharged[fjet_sort][event_mask][fjet_mask][:, :N_FJETS]
    fjets['idx'] = ak.local_index(fj_pt)

    # Quarks
    particles = particles[event_mask]
    topquarks = topquarks[event_mask]
    bquarks = bquarks[event_mask]
    wbosons = wbosons[event_mask]
    wquarks_d1 = wquarks_d1[event_mask]
    wquarks_d2 = wquarks_d2[event_mask]


    ################################
    # Recontrsuct tops
    def get_matched_jetfjets_idx(combo_arr, selected_idxs, field):
        if field == '':
            return ak.fill_none(
                [ak.firsts(combo_arr[ak.local_index(combo_arr, axis=1) == selected_idxs[:, i]]['idx']) for i in range(n_tops)], 
                -1
            ).to_numpy().T
        else:
            return ak.fill_none(
                [ak.firsts(combo_arr[ak.local_index(combo_arr, axis=1) == selected_idxs[:, i]][field]['idx']) for i in range(n_tops)], 
                -1
            ).to_numpy().T

    FR_3jets = ak.combinations(jets, 3, fields=['bjet', 'q1jet', 'q2jet'], axis=1)
    FR_combo_jet_idxs = reconstruct_top(
        topquarks, bquarks, wbosons, wquarks_d1, wquarks_d2,
        FR_3jets,
        FullyResolved_top,
        ak.ArrayBuilder()
    ).snapshot()
    FR_bjet_idxs = get_matched_jetfjets_idx(FR_3jets, FR_combo_jet_idxs, 'bjet')
    FR_q1jet_idxs = get_matched_jetfjets_idx(FR_3jets, FR_combo_jet_idxs, 'q1jet')
    FR_q2jet_idxs = get_matched_jetfjets_idx(FR_3jets, FR_combo_jet_idxs, 'q2jet')

    # Semi-ResolvedQQ tops
    SR_1jet1fjet = ak.cartesian({'jet': jets, 'fjet': fjets})
    SRqq_combo_jetfjet_idxs = reconstruct_top(
        topquarks, bquarks, wbosons, wquarks_d1, wquarks_d2,
        SR_1jet1fjet,
        SemiResolvedQQ_top,
        ak.ArrayBuilder()
    ).snapshot()
    SRqq_qqfjet_idxs = get_matched_jetfjets_idx(SR_1jet1fjet, SRqq_combo_jetfjet_idxs, 'fjet')
    SRqq_bjet_idxs = get_matched_jetfjets_idx(SR_1jet1fjet, SRqq_combo_jetfjet_idxs, 'jet')
    
    # Semi-ResolvedBQ tops
    SRbq_combo_jetfjet_idxs = reconstruct_top(
        topquarks, bquarks, wbosons, wquarks_d1, wquarks_d2,
        SR_1jet1fjet,
        SemiResolvedBQ_top,
        ak.ArrayBuilder()
    ).snapshot()
    SRbq_bqfjet_idxs = get_matched_jetfjets_idx(SR_1jet1fjet, SRbq_combo_jetfjet_idxs, 'fjet')
    SRbq_qjet_idxs = get_matched_jetfjets_idx(SR_1jet1fjet, SRbq_combo_jetfjet_idxs, 'jet')
    
    # Fully-Boosted tops
    FB_1fjet = ak.combinations(fjets, 1, fields=['fjet'])
    FB_combo_fjet_idxs = reconstruct_top(
        topquarks, bquarks, wbosons, wquarks_d1, wquarks_d2,
        FB_1fjet,
        FullyBoosted_top,
        ak.ArrayBuilder()
    ).snapshot()
    FB_bqqfjet_idxs = get_matched_jetfjets_idx(FB_1fjet, FB_combo_fjet_idxs, 'fjet')
    
    # Jet-FatJet overlap
    matched_fjet_jet_idx, matched_fjet_jet_DR  = match_fjet_to_jet(fjets, jets, ak.ArrayBuilder(), ak.ArrayBuilder())
    matched_fjet_jet_idx, matched_fjet_jet_DR = matched_fjet_jet_idx.snapshot(), matched_fjet_jet_DR.snapshot()


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
    for i in range(n_tops):
        top_fjet_mask = (ak.local_index(fjets) == top_fullyBoosted[f"top{i+1}_bqq"])
        w_fjet_mask = ( ~top_fjet_mask & (ak.local_index(fjets) == top_semiResolved_qq[f"top{i+1}_qq"]) )
        bq_fjet_mask = ( ~top_fjet_mask & (ak.local_index(fjets) == top_semiResolved_bq[f"top{i+1}_bq"]) )
        qcd_fjet_mask = ( ~top_fjet_mask & ~w_fjet_mask & ~bq_fjet_mask )

        # Emulate PNet AK8 T-tagger
        fjets[top_fjet_mask]["Ttag"] = (fjets["TtagRN"][top_fjet_mask] < TVSQCD_EFFS['t'])
        fjets[w_fjet_mask]["Ttag"] = (fjets["TtagRN"][w_fjet_mask] < TVSQCD_EFFS['W'])
        fjets[bq_fjet_mask]["Ttag"] = (fjets["TtagRN"][bq_fjet_mask] < TVSQCD_EFFS['bq'])
        fjets[qcd_fjet_mask]["Ttag"] = (fjets["TtagRN"][qcd_fjet_mask] < TVSQCD_EFFS['QCD'])
        # Emulate PNet AK8 W-tagger
        fjets[top_fjet_mask]["Wtag"] = (fjets["WtagRN"][top_fjet_mask] < WVSQCD_EFFS['t'])
        fjets[w_fjet_mask]["Wtag"] = (fjets["WtagRN"][w_fjet_mask] < WVSQCD_EFFS['W'])
        fjets[bq_fjet_mask]["Wtag"] = (fjets["WtagRN"][bq_fjet_mask] < WVSQCD_EFFS['bq'])
        fjets[qcd_fjet_mask]["Wtag"] = (fjets["WtagRN"][qcd_fjet_mask] < WVSQCD_EFFS['QCD'])
    fj_Ttag = fjets["Ttag"]
    fj_Wtag = fjets["Wtag"]


    ################################
    # Plots for validating reconstruction
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


    ################################
    # Cleaning non-reconstructable events
    at_least_one_target_mask = np.zeros_like(ak.to_numpy(ak.firsts(pt))).astype("bool")
    for i in range(n_tops):
        at_least_one_target_mask += ak.to_numpy(
            top_fullyResolved[f"top{i+1}_mask"]
            | top_semiResolved_qq[f"top{i+1}_mask"]
            | top_semiResolved_bq[f"top{i+1}_mask"]
            | top_fullyBoosted[f"top{i+1}_mask"]
        ).astype("bool")
    print(f"Viable tops = {np.sum(at_least_one_target_mask)}")
    at_least_one_target_mask = np.flatnonzero(at_least_one_target_mask)

    ################################
    # Sanity checking reconstruction
    # pass

    ################################
    # Store processed data in dataset for training/testing
    # Inputs
    datasets = {}
    datasets["INPUTS/Jets/MASK"] = to_np_array(pt > 0, max_n=N_JETS).astype("bool")[at_least_one_target_mask]
    datasets["INPUTS/Jets/pt"] = to_np_array(pt, max_n=N_JETS).astype("float32")[at_least_one_target_mask]
    datasets["INPUTS/Jets/eta"] = to_np_array(eta, max_n=N_JETS).astype("float32")[at_least_one_target_mask]
    datasets["INPUTS/Jets/phi"] = to_np_array(phi, max_n=N_JETS).astype("float32")[at_least_one_target_mask]
    datasets["INPUTS/Jets/sinphi"] = to_np_array(np.sin(phi), max_n=N_JETS).astype("float32")[at_least_one_target_mask]
    datasets["INPUTS/Jets/cosphi"] = to_np_array(np.cos(phi), max_n=N_JETS).astype("float32")[at_least_one_target_mask]
    datasets["INPUTS/Jets/mass"] = to_np_array(mass, max_n=N_JETS).astype("float32")[at_least_one_target_mask]
    datasets["INPUTS/Jets/btag"] = to_np_array(btag, max_n=N_JETS).astype("bool")[at_least_one_target_mask]
    datasets["INPUTS/Jets/flavor"] = to_np_array(flavor, max_n=N_JETS).astype("float32")[at_least_one_target_mask]
    datasets["INPUTS/Jets/matchedfj"] = to_np_array(matched_fjet_jet_idx, max_n=N_JETS).astype("int32")[at_least_one_target_mask]
    datasets["INPUTS/Jets/deltaRfj"] = to_np_array(matched_fjet_jet_DR, max_n=N_JETS).astype("int32")[at_least_one_target_mask]

    datasets["INPUTS/BoostedJets/MASK"] = to_np_array(fj_pt > 0, max_n=N_FJETS).astype("bool")[at_least_one_target_mask]
    datasets["INPUTS/BoostedJets/fj_pt"] = to_np_array(fj_pt, max_n=N_FJETS).astype("float32")[at_least_one_target_mask]
    datasets["INPUTS/BoostedJets/fj_eta"] = to_np_array(fj_eta, max_n=N_FJETS).astype("float32")[at_least_one_target_mask]
    datasets["INPUTS/BoostedJets/fj_phi"] = to_np_array(fj_phi, max_n=N_FJETS).astype("float32")[at_least_one_target_mask]
    datasets["INPUTS/BoostedJets/fj_sinphi"] = to_np_array(np.sin(fj_phi), max_n=N_FJETS).astype("float32")[at_least_one_target_mask]
    datasets["INPUTS/BoostedJets/fj_cosphi"] = to_np_array(np.cos(fj_phi), max_n=N_FJETS).astype("float32")[at_least_one_target_mask]
    datasets["INPUTS/BoostedJets/fj_mass"] = to_np_array(fj_mass, max_n=N_FJETS).astype("float32")[at_least_one_target_mask]
    datasets["INPUTS/BoostedJets/fj_sdmass"] = to_np_array(fj_sdmass, max_n=N_FJETS).astype("float32")[at_least_one_target_mask]
    datasets["INPUTS/BoostedJets/fj_Ttag"] = to_np_array(fj_Ttag, max_n=N_FJETS).astype("bool")[at_least_one_target_mask]
    datasets["INPUTS/BoostedJets/fj_Wtag"] = to_np_array(fj_Wtag, max_n=N_FJETS).astype("bool")[at_least_one_target_mask]
    datasets["INPUTS/BoostedJets/fj_tau21"] = to_np_array(fj_tau21, max_n=N_FJETS).astype("float32")[at_least_one_target_mask]
    datasets["INPUTS/BoostedJets/fj_tau32"] = to_np_array(fj_tau32, max_n=N_FJETS).astype("float32")[at_least_one_target_mask]
    datasets["INPUTS/BoostedJets/fj_charge"] = to_np_array(fj_charge, max_n=N_FJETS).astype("float32")[at_least_one_target_mask]
    datasets["INPUTS/BoostedJets/fj_ehadovereem"] = to_np_array(fj_ehadovereem, max_n=N_FJETS).astype("float32")[at_least_one_target_mask]
    datasets["INPUTS/BoostedJets/fj_neutralenergyfrac"] = to_np_array(fj_neutralenergyfrac, max_n=N_FJETS).astype("float32")[at_least_one_target_mask]
    datasets["INPUTS/BoostedJets/fj_chargedenergyfrac"] = to_np_array(fj_chargedenergyfrac, max_n=N_FJETS).astype("float32")[at_least_one_target_mask]
    datasets["INPUTS/BoostedJets/fj_nneutral"] = to_np_array(fj_nneutral, max_n=N_FJETS).astype("float32")[at_least_one_target_mask]
    datasets["INPUTS/BoostedJets/fj_ncharged"] = to_np_array(fj_ncharged, max_n=N_FJETS).astype("float32")[at_least_one_target_mask]

    # Targets
    for i in range(n_tops):
        # fully-resolved tops
        datasets[f"TARGETS/FRt{i+1}/MASK"] = top_fullyResolved[f"top{i+1}_mask"][at_least_one_target_mask]
        datasets[f"TARGETS/FRt{i+1}/b"] = top_fullyResolved[f"top{i+1}_b"][at_least_one_target_mask]
        datasets[f"TARGETS/FRt{i+1}/q1"] = top_fullyResolved[f"top{i+1}_q1"][at_least_one_target_mask]
        datasets[f"TARGETS/FRt{i+1}/q2"] = top_fullyResolved[f"top{i+1}_q2"][at_least_one_target_mask]
        datasets[f"TARGETS/FRt{i+1}/pt"] = top_pt_dict[f"top{i+1}_pt"][at_least_one_target_mask]

        # semi-resolved (qq fatjet) tops
        datasets[f"TARGETS/SRqqt{i+1}/MASK"] = top_semiResolved_qq[f"top{i+1}_mask"][at_least_one_target_mask]
        datasets[f"TARGETS/SRqqt{i+1}/b"] = top_semiResolved_qq[f"top{i+1}_b"][at_least_one_target_mask]
        datasets[f"TARGETS/SRqqt{i+1}/qq"] = top_semiResolved_qq[f"top{i+1}_qq"][at_least_one_target_mask]
        datasets[f"TARGETS/SRqqt{i+1}/pt"] = top_pt_dict[f"top{i+1}_pt"][at_least_one_target_mask]

        # semi-resolved (bq fatjet) tops
        datasets[f"TARGETS/SRbqt{i+1}/MASK"] = top_semiResolved_bq[f"top{i+1}_mask"][at_least_one_target_mask]
        datasets[f"TARGETS/SRbqt{i+1}/q"] = top_semiResolved_bq[f"top{i+1}_q"][at_least_one_target_mask]
        datasets[f"TARGETS/SRbqt{i+1}/bq"] = top_semiResolved_bq[f"top{i+1}_bq"][at_least_one_target_mask]
        datasets[f"TARGETS/SRbqt{i+1}/pt"] = top_pt_dict[f"top{i+1}_pt"][at_least_one_target_mask]

        # fully-boosted tops
        datasets[f"TARGETS/FBt{i+1}/MASK"] = top_fullyBoosted[f"top{i+1}_mask"][at_least_one_target_mask]
        datasets[f"TARGETS/FBt{i+1}/bqq"] = top_fullyBoosted[f"top{i+1}_bqq"][at_least_one_target_mask]
        datasets[f"TARGETS/FBt{i+1}/pt"] = top_pt_dict[f"top{i+1}_pt"][at_least_one_target_mask]

        # Auxiliary for dataset-checking
        # datasets[f'AUXILIARY/FRt{i+1}/b_deltaR'] = ak.fill_none(ak.firsts(jets[ak.local_index(jets) == top_fullyResolved[f"top{i+1}_b"]].deltaR(bquarks[:, i])[at_least_one_target_mask]), -1).to_numpy()
        # datasets[f'AUXILIARY/FRt{i+1}/q1_deltaR'] = ak.fill_none(ak.firsts(jets[ak.local_index(jets) == top_fullyResolved[f"top{i+1}_q1"]].deltaR(wquarks_d1[:, i])[at_least_one_target_mask]), -1).to_numpy()
        # datasets[f'AUXILIARY/FRt{i+1}/q2_deltaR'] = ak.fill_none(ak.firsts(jets[ak.local_index(jets) == top_fullyResolved[f"top{i+1}_q2"]].deltaR(wquarks_d2[:, i])[at_least_one_target_mask]), -1).to_numpy()

        # datasets[f'AUXILIARY/SRqqt{i+1}/b_deltaR'] = ak.fill_none(ak.firsts(jets[ak.local_index(jets) == top_semiResolved_qq[f"top{i+1}_b"]].deltaR(bquarks[:, i])[at_least_one_target_mask]), -1).to_numpy()
        # datasets[f'AUXILIARY/SRqqt{i+1}/q1_deltaR'] = ak.fill_none(ak.firsts(fjets[ak.local_index(fjets) == top_semiResolved_qq[f"top{i+1}_qq"]].deltaR(wquarks_d1[:, i])[at_least_one_target_mask]), -1).to_numpy()
        # datasets[f'AUXILIARY/SRqqt{i+1}/q2_deltaR'] = ak.fill_none(ak.firsts(fjets[ak.local_index(fjets) == top_semiResolved_qq[f"top{i+1}_qq"]].deltaR(wquarks_d2[:, i])[at_least_one_target_mask]), -1).to_numpy()
        # datasets[f'AUXILIARY/SRqqt{i+1}/w_deltaR'] = ak.fill_none(ak.firsts(fjets[ak.local_index(fjets) == top_semiResolved_qq[f"top{i+1}_qq"]].deltaR(wbosons[:, i])[at_least_one_target_mask]), -1).to_numpy()

        # datasets[f'AUXILIARY/SRbqt{i+1}/b_deltaR'] = ak.fill_none(ak.firsts(fjets[ak.local_index(fjets) == top_semiResolved_bq[f"top{i+1}_bq"]].deltaR(bquarks[:, i])[at_least_one_target_mask]), -1).to_numpy()
        # datasets[f'AUXILIARY/SRbqt{i+1}/q1_deltaR_q'] = ak.fill_none(ak.firsts(jets[ak.local_index(jets) == top_semiResolved_bq[f"top{i+1}_q"]].deltaR(wquarks_d1[:, i])[at_least_one_target_mask]), -1).to_numpy()
        # datasets[f'AUXILIARY/SRbqt{i+1}/q1_deltaR_bq'] = ak.fill_none(ak.firsts(fjets[ak.local_index(fjets) == top_semiResolved_bq[f"top{i+1}_bq"]].deltaR(wquarks_d1[:, i])[at_least_one_target_mask]), -1).to_numpy()
        # datasets[f'AUXILIARY/SRbqt{i+1}/q2_deltaR_q'] = ak.fill_none(ak.firsts(jets[ak.local_index(jets) == top_semiResolved_bq[f"top{i+1}_q"]].deltaR(wquarks_d2[:, i])[at_least_one_target_mask]), -1).to_numpy()
        # datasets[f'AUXILIARY/SRbqt{i+1}/q2_deltaR_bq'] = ak.fill_none(ak.firsts(fjets[ak.local_index(fjets) == top_semiResolved_bq[f"top{i+1}_bq"]].deltaR(wquarks_d2[:, i])[at_least_one_target_mask]), -1).to_numpy()
        
        # datasets[f'AUXILIARY/FBt{i+1}/b_deltaR'] = ak.fill_none(ak.firsts(fjets[ak.local_index(fjets) == top_fullyBoosted[f"top{i+1}_bqq"]].deltaR(bquarks[:, i])[at_least_one_target_mask]), -1).to_numpy()
        # datasets[f'AUXILIARY/FBt{i+1}/q1_deltaR'] = ak.fill_none(ak.firsts(fjets[ak.local_index(fjets) == top_fullyBoosted[f"top{i+1}_bqq"]].deltaR(wquarks_d1[:, i])[at_least_one_target_mask]), -1).to_numpy()
        # datasets[f'AUXILIARY/FBt{i+1}/q2_deltaR'] = ak.fill_none(ak.firsts(fjets[ak.local_index(fjets) == top_fullyBoosted[f"top{i+1}_bqq"]].deltaR(wquarks_d2[:, i])[at_least_one_target_mask]), -1).to_numpy()
        # datasets[f'AUXILIARY/FBt{i+1}/w_deltaR'] = ak.fill_none(ak.firsts(fjets[ak.local_index(fjets) == top_fullyBoosted[f"top{i+1}_bqq"]].deltaR(wbosons[:, i])[at_least_one_target_mask]), -1).to_numpy()
        # datasets[f'AUXILIARY/FBt{i+1}/t_deltaR'] = ak.fill_none(ak.firsts(fjets[ak.local_index(fjets) == top_fullyBoosted[f"top{i+1}_bqq"]].deltaR(topquarks[:, i])[at_least_one_target_mask]), -1).to_numpy()

    return datasets

def process_file(file_name, out_file, train_frac, n_tops):
    try:
        if re.match('root://', file_name):
            current_file_name = 'tmp_'+('train_' if 'training' in out_file else 'test_')+file_name.split('/')[-1]
            subprocess.run(['xrdcp', '-f', file_name, current_file_name])
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

        if re.match('root://', file_name): subprocess.run(['rm', '-rf', current_file_name])
        return datasets
    except Exception as e:
        if re.match('root://', file_name): subprocess.run(['rm', '-rf', current_file_name])
        if e is KeyboardInterrupt: return 999
        logging.info(f"Preprocessing failed for file:\n{file_name}\n\nwith error:\n{e}\n\n...continuing with other files")
        return 400

def save_file(filepath: str, dataset: dict):
    if len(filepath.split('/')[-1].split('.')) == 1:
        raise Exception(f'No filepath extension, ambiguous how to save file.')
    else:
        filetype = filepath.split('/')[-1].split('.')[-1]
    print(f"Saving {len(dataset[list(dataset.keys())[0]])} events into {filepath}")
        
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

@click.command()
@click.argument("in-files", nargs=-1)
@click.option(
    "--out-file",
    default=f"{PROJECT_DIR}/data/delphes/tt_training.h5",
    help="Output file.",
)
@click.option(
    "--split_file_size",
    default=-1,
    help="Size for output files in MB, default is \'-1\' which creates 1 merged output file",
)
@click.option(
    "--file_limit",
    default=-1,
    help="Number of output files to make, default is \'-1\' which creates all output files",
)
@click.option("--train-frac", default=0.80, help="Fraction for training.")
@click.option(
    "--n-tops",
    "n_tops",
    default=2,
    type=click.IntRange(2, 4),
    help="Number of top quarks per event",
)
@click.option("--plots", is_flag=True, help="Boolean to make plots.")
@click.option("--multip", is_flag=True, help="Boolean to use multiprocessing.")
@click.option("--condor", is_flag=True, help="Boolean to use condor processing.")
@click.option(
    "--condor_files_per_job",
    default=10,
    help="Number of input files per condor job",
)
def main(in_files, out_file, split_file_size, file_limit, train_frac, n_tops, plots, multip, condor, condor_files_per_job):
    if plots:
        PLOTS = True
    
    all_datasets = {}
    expanded_in_files = []
    print(f"starting in_files = {in_files}")

    for file_name in in_files:
        if re.match('root://', file_name):
            print(' '.join(['xrdfs', '//'.join(file_name.split('//')[:2])+'/', 'ls', '-R', '/'+file_name.split('//')[2]]))
            xrdfs_files = subprocess.run(['xrdfs', '//'.join(file_name.split('//')[:2])+'/', 'ls', '-R', '/'+file_name.split('//')[2]], capture_output=True, text=True)
            expanded_in_files.extend(['//'.join(file_name.split('//')[:2])+'/'+xrdfs_file for xrdfs_file in xrdfs_files.stdout.split() if xrdfs_file.endswith('.root')])
        elif os.path.isdir(file_name):
            print(f"found directory: {file_name}")
            for root_file in glob.glob(os.path.join(file_name, '*.root')):
                expanded_in_files.append(os.path.join(file_name, root_file))
        else:
            expanded_in_files.append(file_name)
    in_files = expanded_in_files
    out_file_idx = 0
    new_outfile_with_idx = lambda outfile, idx: outfile[:outfile.rfind('.')]+str(idx)+outfile[outfile.rfind('.'):]
    if not multip and not condor:
        for file_name in in_files:
            datasets = process_file(file_name, out_file, train_frac, n_tops)
            if type(datasets) is int: 
                if datasets == 999: break
                else: continue
            for dataset_name, data in datasets.items():
                if dataset_name not in all_datasets: all_datasets[dataset_name] = []
                all_datasets[dataset_name].append(data)
            num_events = sum(len(all_datasets[dataset_name][i]) for i in range(len(all_datasets[dataset_name])))
            if split_file_size > 0 and num_events > 2_000*split_file_size:
                save_file(new_outfile_with_idx(out_file, out_file_idx), all_datasets)
                out_file_idx += 1; all_datasets = {}
            if out_file_idx == file_limit: break
        if all_datasets != {}:
            if split_file_size > 0: save_file(new_outfile_with_idx(out_file, out_file_idx), all_datasets)
            else: save_file(out_file, all_datasets)
    elif condor:
        job_filepaths = [
            list(in_files[i*condor_files_per_job:(i+1)*condor_files_per_job]) 
            for i in range(len(in_files)//condor_files_per_job)
        ]
        submitter = LPCVanillaSubmitter(job_filepaths, out_file)
        submitter.submit()
    elif multip:
        with Pool(10) as p:
            out_files, train_fracs, n_topses = [out_file]*len(in_files), [train_frac]*len(in_files), [n_tops]*len(in_files)
            results = p.imap_unordered(process_file, zip(in_files, out_files, train_fracs, n_topses))


if __name__ == "__main__":
    main()
