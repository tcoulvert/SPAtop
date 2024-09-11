import logging
from pathlib import Path

import awkward as ak
import click
import h5py
import numpy as np
import uproot
import vector

from src.data.delphes.matching import (
    match_fjet_to_jet,
    # match_higgs_to_fjet,
    # match_higgs_to_jet,
    match_top_to_fjet,
    match_top_to_jet,
)

vector.register_awkward()
vector.register_numba()
ak.numba.register_and_check()

logging.basicConfig(level=logging.INFO)

N_JETS = 16
MIN_JET_PT = 20
MIN_FJET_PT = 200
PROJECT_DIR = Path(__file__).resolve().parents[3]


def to_np_array(ak_array, max_n=10, pad=0):
    return ak.fill_none(ak.pad_none(ak_array, max_n, clip=True, axis=-1), pad).to_numpy()


def get_datasets(arrays, n_tops):  # noqa: C901
    part_pid = arrays["Particle/Particle.PID"]  # PDG ID
    part_m1 = arrays["Particle/Particle.M1"]
    # condition_hbb = np.logical_and(np.abs(part_pid) == 5, part_pid[part_m1] == 25)
    # mask_hbb = ak.count(part_pid[condition_hbb], axis=-1) == 2 * n_tops
    # part_pid = part_pid
    part_pt = arrays["Particle/Particle.PT"]
    part_eta = arrays["Particle/Particle.Eta"]
    part_phi = arrays["Particle/Particle.Phi"]
    part_mass = arrays["Particle/Particle.Mass"]
    part_m1 = arrays["Particle/Particle.M1"]
    part_d1 = arrays["Particle/Particle.D1"]
    part_d2 = arrays["Particle/Particle.D2"]
    # print(f"top d1 = \n{part_pid[part_d1[part_pid == 6]][:, -1]}\n{'='*60}")
    # print(f"all top d1 are bjets? = {ak.all(part_pid[part_d1[part_pid == 6]][:, -1] == 5)}\n{'='*60}")
    # print(f"top d2 = \n{part_pid[part_d2[part_pid == 6]][:, -1]}\n{'='*60}")
    # print(f"all top d2 are Ws? = {ak.all((part_pid[part_d2[part_pid == 6]][:, -1] == 24) | (part_pid[part_d2[part_pid == 6]][:, -1] == -24))}\n{'='*60}")
    # print(f"d1 of W-: \n{part_pid[part_d1[part_pid == -24]][:, -1]}\n{'='*60}")
    # print(f"d1 of W+: \n{part_pid[part_d1[part_pid == 24]][:, -1]}\n{'='*60}")
    # print(f"d2 of W-: \n{part_pid[part_d2[part_pid == -24]][:, -1]}\n{'='*60}")
    # print(f"d2 of W+: \n{part_pid[part_d2[part_pid == 24]][:, -1]}\n{'='*60}")
    # print(f"how many None d1 of W-: \n{ak.sum(ak.is_none(part_pid[part_d1[part_pid == -24]][:, -1]))}\n{'='*60}")
    # print(f"how many None d1 of W+: \n{ak.sum(ak.is_none(part_pid[part_d1[part_pid == 24]][:, -1]))}\n{'='*60}")
    # print(f"how many None d2 of W-: \n{ak.sum(ak.is_none(part_pid[part_d2[part_pid == -24]][:, -1]))}\n{'='*60}")
    # print(f"how many None d2 of W+: \n{ak.sum(ak.is_none(part_pid[part_d2[part_pid == 24]][:, -1]))}\n{'='*60}")

    # small-radius jet info
    pt = arrays["Jet/Jet.PT"]
    eta = arrays["Jet/Jet.Eta"]
    phi = arrays["Jet/Jet.Phi"]
    mass = arrays["Jet/Jet.Mass"]
    btag = arrays["Jet/Jet.BTag"]
    flavor = arrays["Jet/Jet.Flavor"]
    # print(f"btag = \n{btag}\n{'='*60}")
    # print(f"btag of top d1 = \n{btag[part_d1[part_pid == 6]][:, -1]}\n{'='*60}")
    # print(f"W- d1: \n{part_pid[part_d1[part_pid == -24]][:, -1]}\n{'='*60}")
    # print(f"btag of W- d1: \n{btag[part_d1[part_pid == -24]][:, -1]}\n{'='*60}")

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
    bquark_condition = np.logical_and(np.abs(particles.pid) == 5, np.abs(particles.pid[particles.m1]) == 6)
    bquarks = ak.to_regular(particles[bquark_condition], axis=1)
    wbosons_condition = np.logical_and(np.abs(particles.pid) == 24, np.abs(particles.pid[particles.m1]) == 6)
    wbosons = ak.to_regular(particles[wbosons_condition], axis=1)
    wquarks_condition = np.logical_and(
        np.abs(particles.pid[particles.m1]) == 24, np.logical_and(
            particles.pid >= -6, particles.pid <= 6
        )
    )
    wquarks_temp = ak.to_regular(particles[wquarks_condition], axis=1)
    w1_mask = wquarks_temp.m1 < np.tile(ak.singletons(wbosons.idx[:, 1]), (1, 4))
    w2_mask = wquarks_temp.m1 >= np.tile(ak.singletons(wbosons.idx[:, 1]), (1, 4))
    w1quarks = ak.drop_none(ak.mask(wquarks_temp, w1_mask))
    w2quarks = ak.drop_none(ak.mask(wquarks_temp, w2_mask))
    wquarks_d1 = ak.to_regular(
        ak.concatenate([ak.singletons(w1quarks[:, 0]), ak.singletons(w2quarks[:, 0])], axis=1), axis=1
    )
    wquarks_d2 = ak.to_regular(
        ak.concatenate([ak.singletons(w1quarks[:, 1]), ak.singletons(w2quarks[:, 1])], axis=1), axis=1
    )
    # print(f"wquarks_condition: \n{wquarks_condition}")
    # print(f"wquarks: \n{particles[wquarks_condition]}")
    # print(f"wquarks nums: \n{np.unique(ak.num(particles[wquarks_condition]))}")
    # print(f"wquarks.m1: \n{wquarks_temp.m1}")
    # print(f"wbosons.idx: \n{wbosons.idx}\n -> wbosons.idx[:,0]: {wbosons.idx[:, 0]}")
    # print(np.tile(ak.singletons(wbosons.idx[:, 1]), (1, 4)))
    # print(np.shape(np.tile(ak.singletons(wbosons.idx[:, 1]), (1, 4))))
    # print(f"wquarks from w1: \n{wquarks_temp[wquarks_temp.m1 < np.tile(ak.singletons(wbosons.idx[:, 1]), (1, 4))]}")
    # print(f"wquarks from w2: \n{wquarks_temp[wquarks_temp.m1 > np.tile(ak.singletons(wbosons.idx[:, 1]), (1, 4))]}")
    # print(f"wquarks.m1: \n{wquarks_temp.m1}\n{'-'*60}")
    # print(f"w2_idx: \n{np.tile(ak.singletons(wbosons.idx[:, 1]), (1, 4))}\n{'-'*60}")
    # print(f"w1_mask -> wquarks.m1 < w2_idx: \n{w1_mask}\n{'-'*60}")
    # print(f"w2_mask -> wquarks.m1 > w2_idx: \n{w2_mask}\n{'-'*60}")
    # print(f"masks or = {ak.all(w1_mask | w2_mask)}")
    # bad_idxs = ~(w1_mask | w2_mask)
    # print(f"masks not or \n-> w1mask: \n{w1_mask[bad_idxs]}\n-> w2mask: \n{w2_mask[bad_idxs]}")
    # print(f"wquarks.m1[bad_idxs]: \n{wquarks_temp.m1[bad_idxs]}")
    # print(f"wbosons.idx[bad_idxs]: \n{np.tile(ak.singletons(wbosons.idx[:, 1]), (1, 4))[bad_idxs]}")
    # print(f"wquarks.pid[bad_idxs]: \n{wquarks_temp.pid[bad_idxs]}")
    # print(f"wbosons.pid[bad_idxs]: \n{np.tile(ak.singletons(wbosons.pid[:, 1]), (1, 4))[bad_idxs]}")
    # print(f"masks and = {ak.all(~(w1_mask & w2_mask))}")
    # print(f"masks both = {ak.all(w1_mask | w2_mask) & ak.all(~(w1_mask & w2_mask))}")
    # print(f"w1quarks shape: ({ak.num(w1quarks, axis=0)}, {np.unique(ak.num(w1quarks))})")
    # print(f"w2quarks shape: ({ak.num(w2quarks, axis=0)}, {np.unique(ak.num(w2quarks))})")
    # print(f"w1quarks: \n{w1quarks}\n{'-'*60}")
    # print(f"w2quarks: \n{w2quarks}\n{'-'*60}")
    # print(f"w1quarks.d1: \n{w1quarks[:, 0]}")
    # print(f"ak.singletons(w1quarks.d1): \n{ak.singletons(w1quarks[:, 0])}")
    # print(f"concat lists: \n{ak.concatenate([ak.singletons(w1quarks[:, 0]), ak.singletons(w2quarks[:, 0])], axis=1)}")
    # concat_temp = ak.concatenate([ak.singletons(w1quarks[:, 0]), ak.singletons(w2quarks[:, 0])], axis=1)
    # print(f"concat lists shape = ({ak.num(concat_temp, axis=0)}, {np.unique(ak.num(concat_temp))})")
    # wquarks = ak.to_regular(
    #     ak.zip(
    #         {
    #             'd1': ak.concatenate([ak.singletons(w1quarks[:, 0]), ak.singletons(w2quarks[:, 0])], axis=1), 
    #             'd2': ak.concatenate([ak.singletons(w1quarks[:, 1]), ak.singletons(w2quarks[:, 1])], axis=1)
    #         }
    #     )
    # )
    # print(f"topquarks: \n{topquarks}\n{'-'*60}")
    # print(f"num topquarks: \n{ak.num(topquarks)}\n{'-'*60}")
    # print(f"num topquarks = n_tops?: \n{ak.all(ak.num(topquarks) == n_tops)}\n{'-'*60 + '-'*60}")
    # print(f"bquarks: \n{bquarks}\n{'-'*60}")
    # print(f"num bquarks: \n{ak.num(bquarks)}\n{'-'*60}")
    # print(f"num bquarks = n_tops?: \n{ak.all(ak.num(bquarks) == n_tops)}\n{'-'*60 + '-'*60}")
    # print(f"wbosons: \n{wbosons}\n{'-'*60}")
    # print(f"num wbosons: \n{ak.num(wbosons)}\n{'-'*60}")
    # print(f"num wbosons = n_tops?: \n{ak.all(ak.num(wbosons) == n_tops)}\n{'-'*60 + '-'*60}")
    # print(f"wquarks: \n{wquarks}\n{'-'*60}")
    # print(f"num wquarks: \n{ak.num(wquarks)}\n{'-'*60}")
    # print(f"num wquarks = 2 * n_tops?: \n{ak.all(ak.num(wquarks.d1) == n_tops) & ak.all(ak.num(wquarks.d2) == n_tops)}\n{'-'*60}")

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
    # print(f"wquarks pid: \n{wquarks_d1.pid}")
    # print(f"wquarks pid: \n{(wquarks_d1[0]).pid}")
    # for quark in wquarks_d1:
    #     print(quark.pid)
    #     break
    top_idx, top_b_idx, top_q_idx = match_top_to_jet(
        topquarks, bquarks, wbosons, wquarks_d1, wquarks_d2, jets, 
        ak.ArrayBuilder(), ak.ArrayBuilder(), ak.ArrayBuilder()
    )
    top_idx, top_b_idx, top_q_idx = top_idx.snapshot(), top_b_idx.snapshot(), top_q_idx.snapshot()
    matched_fj_idx = match_fjet_to_jet(fjets, jets, ak.ArrayBuilder()).snapshot()
    fj_top_idx = match_top_to_fjet(topquarks, bquarks, wbosons, wquarks_d1, wquarks_d2, fjets, ak.ArrayBuilder()).snapshot()
    fj_top_bqq_idx = match_top_to_fjet(topquarks, bquarks, wbosons, wquarks_d1, wquarks_d2, fjets, ak.ArrayBuilder(), match_type='bqq').snapshot()
    fj_top_bq_idx = match_top_to_fjet(topquarks, bquarks, wbosons, wquarks_d1, wquarks_d2, fjets, ak.ArrayBuilder(), match_type='bq').snapshot()
    fj_top_qq_idx = match_top_to_fjet(topquarks, bquarks, wbosons, wquarks_d1, wquarks_d2, fjets, ak.ArrayBuilder(), match_type='qq').snapshot()
    print(f"fjets: \n{fjets}\n{'='*60}")
    print(f"empty at same places fjets-any: \n{ak.all(ak.num(fjets) == ak.num(fj_top_idx))}\n{'='*60}")
    print(f"empty at same places fjets-bqq: \n{ak.all(ak.num(fjets) == ak.num(fj_top_bqq_idx))}\n{'='*60}")
    print(f"empty at same places fjets-bq: \n{ak.all(ak.num(fjets) == ak.num(fj_top_bq_idx))}\n{'='*60}")
    print(f"empty at same places fjets-qq: \n{ak.all(ak.num(fjets) == ak.num(fj_top_qq_idx))}\n{'='*60}")
    print(f"fjet any boosted: \n{fj_top_idx}\n{'='*60}")
    print(f"fjet all boosted: \n{fj_top_bqq_idx}\n{'='*60}")
    print(f"fjet bq boosted: \n{fj_top_bq_idx}\n{'='*60}")
    print(f"fjet qq boosted: \n{fj_top_qq_idx}\n{'='*60}")
    total_arr = np.where(fj_top_idx > 0, 1, -1) + np.where(fj_top_bqq_idx > 0, 1, -1) + np.where(fj_top_bq_idx > 0, 1, -1) + np.where(fj_top_qq_idx > 0, 1, -1)
    print(f"only 2 arrays have fatjets: \n{ak.all((total_arr == 0) | (total_arr == -4))}\n{'='*60}")
    less_total_arr = np.where(fj_top_bqq_idx > 0, 1, -1) + np.where(fj_top_bq_idx > 0, 1, -1) + np.where(fj_top_qq_idx > 0, 1, -1)
    print(f"only 1 array has fatjets: \n{ak.all((less_total_arr == -1) | (less_total_arr == -3))}\n{'='*60}")
    print(f"jet any: \n{top_idx}\n{'='*60}")
    print(f"jet t->b: \n{top_b_idx}\n{'='*60}")
    print(f"jet w->q: \n{top_q_idx}\n{'='*60}")
    total_arr = np.where(top_idx > 0, 1, -1) + np.where(top_b_idx > 0, 1, -1) + np.where(top_q_idx > 0, 1, -1)
    print(f"only 2 arrays have jets: \n{ak.all((total_arr == 1) | (total_arr == -3))}\n{'='*60}")
    less_total_arr = np.where(top_b_idx > 0, 1, -1) + np.where(top_q_idx > 0, 1, -1)
    print(f"only 1 array has jets: \n{ak.all((less_total_arr == 0) | (less_total_arr == -2))}\n{'='*60}")
    print(f"total arr where problems: \n{ak.drop_none(ak.firsts(total_arr[(total_arr != 1) & (total_arr != -3)]))}")
    print(f"less total arr where problems: \n{ak.drop_none(ak.firsts(less_total_arr[(less_total_arr != 0) & (less_total_arr != -2)]))}")
    problem_indices = ak.where(
        ak.is_none(ak.firsts(total_arr[(total_arr != 1) & (total_arr != -3)])), -1, range(len(ak.firsts(total_arr[(total_arr != 1) & (total_arr != -3)])))
    )
    print(f"top_idx where problems: \n{top_idx[problem_indices != -1]}")
    print(f"top_b_idx where problems: \n{top_b_idx[problem_indices != -1]}")
    print(f"top_q_idx where problems: \n{top_q_idx[problem_indices != -1]}")
    print(f"fatjets where problems: \n{fj_top_idx[problem_indices != -1]}")

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

    # keep only top n_fjets
    n_fjets = n_tops
    fj_pt = fj_pt[:, :n_fjets]
    fj_eta = fj_eta[:, :n_fjets]
    fj_phi = fj_phi[:, :n_fjets]
    fj_mass = fj_mass[:, :n_fjets]
    fj_sdmass = fj_sdmass[:, :n_fjets]
    fj_tau21 = fj_tau21[:, :n_fjets]
    fj_tau32 = fj_tau32[:, :n_fjets]
    fj_charge = fj_charge[:, :n_fjets]
    fj_ehadovereem = fj_ehadovereem[:, :n_fjets]
    fj_neutralenergyfrac = fj_neutralenergyfrac[:, :n_fjets]
    fj_chargedenergyfrac = fj_chargedenergyfrac[:, :n_fjets]
    fj_nneutral = fj_nneutral[:, :n_fjets]
    fj_ncharged = fj_ncharged[:, :n_fjets]
    fj_top_idx = fj_top_idx[:, :n_fjets]
    fj_top_bqq_idx = fj_top_bqq_idx[:, :n_fjets]
    fj_top_bq_idx = fj_top_bq_idx[:, :n_fjets]
    fj_top_qq_idx = fj_top_qq_idx[:, :n_fjets]

    # add H pT info
    top_pt = topquarks[mask_minjets].pt
    top_pt = ak.fill_none(ak.pad_none(top_pt, target=3, axis=1, clip=True), -1)

    top_pt_dict = {}
    for i in range(n_tops):
        top_pt_dict[f"top{i+1}_pt"] = top_pt[:, i]
        top_pt_dict[f"top{i+1}_bqq_pt"] = top_pt[:, i]
        top_pt_dict[f"top{i+1}_bq_pt"] = top_pt[:, i]
        top_pt_dict[f"top{i+1}_qq_pt"] = top_pt[:, i]
    # h1_pt, bh1_pt = top_pt[:, 0], top_pt[:, 0]
    # h2_pt, bh2_pt = top_pt[:, 1], top_pt[:, 1]
    # if n_tops == 3:
    #     h3_pt, bh3_pt = top_pt[:, 2], top_pt[:, 2]

    # mask to define zero-padded small-radius jets
    mask = pt > MIN_JET_PT

    # mask to define zero-padded large-radius jets
    fj_mask = fj_pt > MIN_FJET_PT

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
        # print('='*60 + '\n' + '='*60)
        # print(top_jet_idxs[f"top{i+1}"])
        # print(np.unique(ak.count(top_jet_idxs[f"top{i+1}"], axis=-1)))
        # print(top_jet_idxs[f"top{i+1}"][ak.count(top_jet_idxs[f"top{i+1}"], axis=-1) == 4])
        # print('-'*60)
        # print(top_jet_idxs[f"top{i+1}_b"])
        # print(np.unique(ak.count(top_jet_idxs[f"top{i+1}_b"], axis=-1)))
        # print(top_jet_idxs[f"top{i+1}_b"][ak.count(top_jet_idxs[f"top{i+1}_b"], axis=-1) == 2])
        # print('-'*60)
        # print(top_jet_idxs[f"top{i+1}_q"])
        # print(np.unique(ak.count(top_jet_idxs[f"top{i+1}_q"], axis=-1)))
        # print(top_jet_idxs[f"top{i+1}_q"][ak.count(top_jet_idxs[f"top{i+1}_q"], axis=-1) == 3])
        check += np.unique(ak.count(top_jet_idxs[f"top{i+1}"], axis=-1)).to_list()
        check_b += np.unique(ak.count(top_jet_idxs[f"top{i+1}_b"], axis=-1)).to_list()
        check_q += np.unique(ak.count(top_jet_idxs[f"top{i+1}_q"], axis=-1)).to_list()
    if 4 in check:
        logging.warning(" Some tops match to 4 small-radius jets! Check truth")
    if 2 in check_b:
        logging.warning(" Some tops match to having 2 daughter bjets (i.e. 2 bjets directly from tops)! Check truth")
    if 3 in check_q:
        logging.warning(" Some tops match to having 3 W-daughter jets (i.e. 3 jets directly from Ws)! Check truth")
    print(f"All proper numbers of jets: {ak.all(np.array(check) < 4) & ak.all(np.array(check_b) < 2) & ak.all(np.array(check_q) < 3)}")
    # np.unique(ak.count(h1_bs, axis=-1)).to_list() + np.unique(ak.count(h2_bs, axis=-1)).to_list()
    # if n_tops == 3:
    #     check += np.unique(ak.count(h3_bs, axis=-1)).to_list()
    # if 3 in check:
    #     logging.warning("some Higgs bosons match to 3 small-radius jets! Check truth")

    # check/fix large-radius jet truth (ensure max 1 large-radius jet per higgs)
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
    print(f"All proper numbers of fjets: {ak.all(np.array(fj_check) < 2) & ak.all(np.array(fj_check_bqq) < 2) & ak.all(np.array(fj_check_bq) < 2) & ak.all(np.array(fj_check_qq) < 2)}")
    # fj_check = np.unique(ak.count(h1_bb, axis=-1)).to_list() + np.unique(ak.count(h2_bb, axis=-1)).to_list()
    # if n_tops == 3:
    #     fj_check += np.unique(ak.count(h3_bb, axis=-1)).to_list()
    # if 2 in fj_check:
    #     logging.warning("some Higgs bosons match to 2 large-radius jets! Check truth")

    h1_bs = ak.fill_none(ak.pad_none(h1_bs, 2, clip=True), -1)
    h2_bs = ak.fill_none(ak.pad_none(h2_bs, 2, clip=True), -1)
    if n_tops == 3:
        h3_bs = ak.fill_none(ak.pad_none(h3_bs, 2, clip=True), -1)

    h1_bb = ak.fill_none(ak.pad_none(h1_bb, 1, clip=True), -1)
    h2_bb = ak.fill_none(ak.pad_none(h2_bb, 1, clip=True), -1)
    if n_tops == 3:
        h3_bb = ak.fill_none(ak.pad_none(h3_bb, 1, clip=True), -1)

    h1_b1, h1_b2 = h1_bs[:, 0], h1_bs[:, 1]
    h2_b1, h2_b2 = h2_bs[:, 0], h2_bs[:, 1]
    if n_tops == 3:
        h3_b1, h3_b2 = h3_bs[:, 0], h3_bs[:, 1]

    # mask whether Higgs can be reconstructed as 2 small-radius jet
    h1_mask = ak.all(h1_bs != -1, axis=-1)
    h2_mask = ak.all(h2_bs != -1, axis=-1)
    if n_tops == 3:
        h3_mask = ak.all(h3_bs != -1, axis=-1)

    # mask whether Higgs can be reconstructed as 1 large-radius jet
    h1_fj_mask = ak.all(h1_bb != -1, axis=-1)
    h2_fj_mask = ak.all(h2_bb != -1, axis=-1)
    if n_tops == 3:
        h3_fj_mask = ak.all(h3_bb != -1, axis=-1)

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

    datasets["INPUTS/BoostedJets/MASK"] = to_np_array(fj_mask, max_n=n_fjets).astype("bool")
    datasets["INPUTS/BoostedJets/fj_pt"] = to_np_array(fj_pt, max_n=n_fjets).astype("float32")
    datasets["INPUTS/BoostedJets/fj_eta"] = to_np_array(fj_eta, max_n=n_fjets).astype("float32")
    datasets["INPUTS/BoostedJets/fj_phi"] = to_np_array(fj_phi, max_n=n_fjets).astype("float32")
    datasets["INPUTS/BoostedJets/fj_sinphi"] = to_np_array(np.sin(fj_phi), max_n=n_fjets).astype("float32")
    datasets["INPUTS/BoostedJets/fj_cosphi"] = to_np_array(np.cos(fj_phi), max_n=n_fjets).astype("float32")
    datasets["INPUTS/BoostedJets/fj_mass"] = to_np_array(fj_mass, max_n=n_fjets).astype("float32")
    datasets["INPUTS/BoostedJets/fj_sdmass"] = to_np_array(fj_sdmass, max_n=n_fjets).astype("float32")
    datasets["INPUTS/BoostedJets/fj_tau21"] = to_np_array(fj_tau21, max_n=n_fjets).astype("float32")
    datasets["INPUTS/BoostedJets/fj_tau32"] = to_np_array(fj_tau32, max_n=n_fjets).astype("float32")
    datasets["INPUTS/BoostedJets/fj_charge"] = to_np_array(fj_charge, max_n=n_fjets)
    datasets["INPUTS/BoostedJets/fj_ehadovereem"] = to_np_array(fj_ehadovereem, max_n=n_fjets)
    datasets["INPUTS/BoostedJets/fj_neutralenergyfrac"] = to_np_array(fj_neutralenergyfrac, max_n=n_fjets)
    datasets["INPUTS/BoostedJets/fj_chargedenergyfrac"] = to_np_array(fj_chargedenergyfrac, max_n=n_fjets)
    datasets["INPUTS/BoostedJets/fj_nneutral"] = to_np_array(fj_nneutral, max_n=n_fjets)
    datasets["INPUTS/BoostedJets/fj_ncharged"] = to_np_array(fj_ncharged, max_n=n_fjets)

    datasets["TARGETS/h1/mask"] = h1_mask.to_numpy()
    datasets["TARGETS/h1/b1"] = h1_b1.to_numpy()
    datasets["TARGETS/h1/b2"] = h1_b2.to_numpy()
    datasets["TARGETS/h1/pt"] = h1_pt.to_numpy()

    datasets["TARGETS/h2/mask"] = h2_mask.to_numpy()
    datasets["TARGETS/h2/b1"] = h2_b1.to_numpy()
    datasets["TARGETS/h2/b2"] = h2_b2.to_numpy()
    datasets["TARGETS/h2/pt"] = h2_pt.to_numpy()

    if n_tops == 3:
        datasets["TARGETS/h3/mask"] = h3_mask.to_numpy()
        datasets["TARGETS/h3/b1"] = h3_b1.to_numpy()
        datasets["TARGETS/h3/b2"] = h3_b2.to_numpy()
        datasets["TARGETS/h3/pt"] = h3_pt.to_numpy()

    datasets["TARGETS/bh1/mask"] = h1_fj_mask.to_numpy()
    datasets["TARGETS/bh1/bb"] = h1_bb.to_numpy().reshape(h1_fj_mask.to_numpy().shape)
    datasets["TARGETS/bh1/pt"] = bh1_pt.to_numpy()

    datasets["TARGETS/bh2/mask"] = h2_fj_mask.to_numpy()
    datasets["TARGETS/bh2/bb"] = h2_bb.to_numpy().reshape(h2_fj_mask.to_numpy().shape)
    datasets["TARGETS/bh2/pt"] = bh2_pt.to_numpy()

    if n_tops == 3:
        datasets["TARGETS/bh3/mask"] = h3_fj_mask.to_numpy()
        datasets["TARGETS/bh3/bb"] = h3_bb.to_numpy().reshape(h3_fj_mask.to_numpy().shape)
        datasets["TARGETS/bh3/pt"] = bh3_pt.to_numpy()

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
