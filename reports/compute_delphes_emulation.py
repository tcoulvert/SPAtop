import os

import hist
import matplotlib.pyplot as plt
import mplhep as hep

import awkward as ak
import numpy as np
import uproot
import vector
vector.register_awkward()

################################


CWD = os.getcwd()
PLOT_DIR = os.path.join(CWD, "plots")
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)
FATJET_DR = 0.8
DENSITY = False
FILL_VALUE = -999

PNet_TvsQCD_WP = 0.580  # 1.0% QCD bkg efficiency for 2018
PNet_WvsQCD_WP = 0.940  # 1.0% QCD bkg efficiency for 2018

################################


ttbar_file = uproot.open(
    # "/storage/cms/store/mc/Run3Summer22NanoAODv12/TTto4Q-2Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_v5-v1/2550000/01268018-b8bd-4292-8e59-1b95904a1de8.root"  # 674k events
    "/storage/cms/store/mc/Run3Summer22NanoAODv12/TTto4Q-2Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_v5-v1/2550000/1248e4bd-ef5b-4a60-ab02-ca385991223d.root"  # 304k events
    # "/storage/cms/store/mc/Run3Summer22NanoAODv12/TTto4Q-2Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_v5-v1/2550000/03c75fc4-f6af-440b-aec6-871c7d27faaf.root"  # 54k events
    # "/storage/cms/store/mc/Run3Summer22NanoAODv12/TTto4Q-2Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_v5-v1/2550000/0ae47a9a-770e-41d2-aebd-5fe25d498ec8.root"  # 6.1k events
)
# print(ttbar_file["Events"].keys())
events = ttbar_file["Events"].arrays()

fatjets = ak.zip({
    "rho": events.FatJet_pt,
    "eta": events.FatJet_eta,
    "phi": events.FatJet_phi,
    "mass": events.FatJet_mass,
    "softdrop_mass": events.FatJet_msoftdrop,
    "PNet_TvsQCD": events.FatJet_particleNetWithMass_TvsQCD,
    "PNet_WvsQCD": events.FatJet_particleNetWithMass_WvsQCD,
}, with_name="Momentum4D")
# gen_fatjets = ak.zip({
#     "rho": events.GenJetAK8_pt,
#     "eta": events.GenJetAK8_eta,
#     "phi": events.GenJetAK8_phi,
#     "mass": events.GenJetAK8_mass,
# }, with_name="Momentum4D")
gen_particles = ak.zip({
    "rho": events.GenPart_pt,
    "eta": events.GenPart_eta,
    "phi": events.GenPart_phi,
    "mass": events.GenPart_mass,
    "pdgId": events.GenPart_pdgId,
    "status": events.GenPart_status,
    "statusFlags": events.GenPart_statusFlags,
    "genPartIdxMother": events.GenPart_genPartIdxMother
}, with_name="Momentum4D")

################################


# Finding gen-matched top fatjets
ts_mask = ( (gen_particles.status == 62) & (np.abs(gen_particles.pdgId) == 6) )  # status 62
ts = gen_particles[ts_mask]
bs_mask = ( (np.abs(gen_particles.pdgId) == 5) & (np.abs(gen_particles.pdgId[gen_particles.genPartIdxMother]) == 6) )  # status 71
bs = gen_particles[bs_mask]
ws_mask = ( (np.abs(gen_particles.pdgId) == 24) & (np.abs(gen_particles.pdgId[gen_particles.genPartIdxMother]) == 6) )  # status 52
ws = gen_particles[ws_mask]
wqs_mask = ( (np.abs(gen_particles.pdgId) <= 5) & (np.abs(gen_particles.pdgId[gen_particles.genPartIdxMother]) == 24) )  # status 71
wqs = gen_particles[wqs_mask]

good_ttbar_events_mask = (
    (ak.num(ts) == 2) & (ak.num(bs) == 2)
    & (ak.num(ws) == 2) & (ak.num(wqs) == 4)
)
fatjets = fatjets[good_ttbar_events_mask]
gen_particles = gen_particles[good_ttbar_events_mask]
ts = ts[good_ttbar_events_mask]
bs = bs[good_ttbar_events_mask]
ws = ws[good_ttbar_events_mask]
wqs = wqs[good_ttbar_events_mask]

wq0s = ak.concatenate([wqs[ak.local_index(wqs) == 0], wqs[ak.local_index(wqs) == 0]], axis=1)
wq1s = ak.concatenate([wqs[ak.local_index(wqs) == 1], wqs[ak.local_index(wqs) == 1]], axis=1)
wq2s = ak.concatenate([wqs[ak.local_index(wqs) == 2], wqs[ak.local_index(wqs) == 2]], axis=1)
wq3s = ak.concatenate([wqs[ak.local_index(wqs) == 3], wqs[ak.local_index(wqs) == 3]], axis=1)

################################


# Making fatjet-genquark cartesian maps
def make_fjp(fjs, ps):
    fjp = ak.cartesian({"fj": fjs, "p": ps}, axis=1)
    fjp = ak.with_field(fjp, fjp.fj.deltaR(fjp.p), "fj_p_DR")
    return fjp

fjt = make_fjp(fatjets, ts)
fjb = make_fjp(fatjets, bs)
fjw = make_fjp(fatjets, ws)
fjwq0 = make_fjp(fatjets, wq0s)
fjwq1 = make_fjp(fatjets, wq1s)
fjwq2 = make_fjp(fatjets, wq2s)
fjwq3 = make_fjp(fatjets, wq3s)

################################


t_fj_mask = (fjt.fj_p_DR < FATJET_DR)
b_fj_mask = (fjb.fj_p_DR < FATJET_DR)
w_fj_mask = (fjw.fj_p_DR < FATJET_DR)
wq_fj_mask = (
    ( (fjwq0.fj_p_DR < FATJET_DR) & (fjwq1.fj_p_DR < FATJET_DR) )
    | ( (fjwq0.fj_p_DR < FATJET_DR) & (fjwq2.fj_p_DR < FATJET_DR) )
    | ( (fjwq0.fj_p_DR < FATJET_DR) & (fjwq3.fj_p_DR < FATJET_DR) )
    | ( (fjwq1.fj_p_DR < FATJET_DR) & (fjwq2.fj_p_DR < FATJET_DR) )
    | ( (fjwq1.fj_p_DR < FATJET_DR) & (fjwq3.fj_p_DR < FATJET_DR) )
    | ( (fjwq2.fj_p_DR < FATJET_DR) & (fjwq3.fj_p_DR < FATJET_DR) )
)

# Finding best fatjet for tops -- FB
fj_matched_t_mask = (t_fj_mask & b_fj_mask & w_fj_mask & wq_fj_mask)
best_t_idx = ak.argmin(fjt.fj_p_DR[fj_matched_t_mask], axis=1)
top_matched_fj = ak.firsts(fjt.fj[fj_matched_t_mask][ak.local_index(fjt.fj[fj_matched_t_mask]) == best_t_idx])  # FB

# Finding best fatjet for ws (exclusive from tops) -- SRqq
fj_matched_w_mask = (~t_fj_mask & ~b_fj_mask & w_fj_mask & wq_fj_mask)
best_w_idx = ak.argmin(fjw.fj_p_DR[fj_matched_w_mask], axis=1)
w_matched_fj = ak.firsts(fjw.fj[fj_matched_w_mask][ak.local_index(fjw.fj[fj_matched_w_mask]) == best_w_idx])  # SRqq

################################


def make_efficiency_plots(matched_fjs, plot_field, nbins=100, plot_label="", file_postfix=""):
    plot_hist = hist.Hist(
        hist.axis.Regular(nbins, 0., 1., name="var", label=plot_field)
    ).fill(var=ak.fill_none(matched_fjs[plot_field], FILL_VALUE))

    fig, ax = plt.subplots()
    hep.cms.lumitext(f"2022" + r" (13.6 TeV)", ax=ax)
    hep.cms.text("Simulation", ax=ax)
    hep.histplot(plot_hist, ax=ax, histtype="step", yerr=True, density=True, label=plot_label)
    plt.legend()
    plt.savefig(os.path.join(PLOT_DIR, f"{plot_field}{file_postfix}.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(PLOT_DIR, f"{plot_field}{file_postfix}.png"), bbox_inches='tight')
    plt.close()

# Making efficiency plots for T (tag) and W (mistag) for top fatjets
topfj_eff = ak.sum(top_matched_fj["PNet_TvsQCD"] > PNet_TvsQCD_WP, axis=0) / ak.sum(top_matched_fj["PNet_TvsQCD"] > 0., axis=0)
topfj_misseff = ak.sum(top_matched_fj["PNet_WvsQCD"] > PNet_WvsQCD_WP, axis=0) / ak.sum(top_matched_fj["PNet_WvsQCD"] > 0., axis=0)
make_efficiency_plots(top_matched_fj, "PNet_TvsQCD", file_postfix="topfj_tag_score", label=f"top-matched-fatjet eff @ 1.0% QCD eff = {topfj_eff}")
make_efficiency_plots(top_matched_fj, "PNet_WvsQCD", nbins=80, file_postfix="topfj_mistag_score", label=f"top-matched-fatjet mistag eff @ 1.0% QCD eff = {topfj_misseff}")

# Making efficiency plots for T (mistag) and W (tag) for w fatjets
wfj_eff = ak.sum(w_matched_fj["PNet_WvsQCD"] > PNet_WvsQCD_WP, axis=0) / ak.sum(w_matched_fj["PNet_WvsQCD"] > 0., axis=0)
wfj_misseff = ak.sum(w_matched_fj["PNet_TvsQCD"] > PNet_TvsQCD_WP, axis=0) / ak.sum(w_matched_fj["PNet_TvsQCD"] > 0., axis=0)
make_efficiency_plots(w_matched_fj, "PNet_WvsQCD", nbins=80, file_postfix="wfj_tag_score", label=f"w-matched-fatjet eff @ 1.0% QCD eff = {wfj_eff}")
make_efficiency_plots(w_matched_fj, "PNet_TvsQCD", file_postfix="wfj_mistag_score", label=f"w-matched-fatjet mistag eff @ 1.0% QCD eff = {wfj_misseff}")