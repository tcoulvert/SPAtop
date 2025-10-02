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

################################


ttbar_file = uproot.open("/storage/cms/store/mc/Run3Summer22NanoAODv12/TTto4Q-2Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_v5-v1/2550000/01268018-b8bd-4292-8e59-1b95904a1de8.root")
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
ts_mask = ( (gen_particles.statusFlags == 13) & (np.abs(gen_particles.pdgId) == 6) )
ts = gen_particles[ts_mask]
bs_mask = ( (gen_particles.statusFlags == 13) & (np.abs(gen_particles.pdgId) == 5) & (np.abs(gen_particles.pdgId[gen_particles.genPartIdxMother]) == 6) )
bs = gen_particles[bs_mask]
ws_mask = ( (gen_particles.statusFlags == 13) & (np.abs(gen_particles.pdgId) == 24) & (np.abs(gen_particles.pdgId[gen_particles.genPartIdxMother]) == 6) )
ws = gen_particles[ws_mask]
wqs_mask = ( (gen_particles.statusFlags == 13) & (np.abs(gen_particles.pdgId) <= 5) & (np.abs(gen_particles.pdgId[gen_particles.genPartIdxMother]) == 24) )
wqs = gen_particles[wqs_mask]
wq0s = wqs[ak.local_index(wqs) == 0]
wq1s = wqs[ak.local_index(wqs) == 1]
wq2s = wqs[ak.local_index(wqs) == 2]
wq3s = wqs[ak.local_index(wqs) == 3]

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


# Finding best fatjet for tops -- FB
fj_matched_t_mask = (
    (fjt.fj_p_DR < FATJET_DR) & (fjb.fj_p_DR < FATJET_DR)
    & (fjw.fj_p_DR < FATJET_DR) & ( 
        (fjwq0.fj_p_DR < FATJET_DR) + (fjwq1.fj_p_DR < FATJET_DR)
        + (fjwq2.fj_p_DR < FATJET_DR) + (fjwq3.fj_p_DR < FATJET_DR) >= 2
    )
)
best_t_idx = ak.argmin(fjt.fj_p_DR[fj_matched_t_mask], axis=1)
top_matched_fj = ak.firsts(fjt[fj_matched_t_mask][ak.local_index(fjt[fj_matched_t_mask]) == best_t_idx]).fj  # FB

# Finding best fatjet for ws (exclusive from tops) -- SRqq
fj_matched_w_mask = (
    ~(fjt.fj_p_DR < FATJET_DR) & ~(fjb.fj_p_DR < FATJET_DR)
    & (fjw.fj_p_DR < FATJET_DR) & ( 
        (fjwq0.fj_p_DR < FATJET_DR) + (fjwq1.fj_p_DR < FATJET_DR)
        + (fjwq2.fj_p_DR < FATJET_DR) + (fjwq3.fj_p_DR < FATJET_DR) >= 2
    )
)
best_w_idx = ak.argmin(fjw.fj_p_DR[fj_matched_w_mask], axis=1)
w_matched_fj = ak.firsts(fjw[fj_matched_w_mask][ak.local_index(fjw[fj_matched_w_mask]) == best_t_idx]).fj  # SRqq

################################


def make_efficiency_plots(matched_fjs, plot_field, label_extra="tag_eff"):
    plot_hist = hist.Hist(
        hist.axis.Regular(100, -1., 1., name="var", label=plot_field)
    ).fill(var=matched_fjs[plot_field])

    for density in [False, True]:
        fig, ax = plt.subplots()
        hep.cms.lumitext(f"2022" + r" (13.6 TeV)", ax=ax)
        hep.cms.text("Simulation", ax=ax)
        hep.histplot(plot_hist, ax=ax, linewidth=3, histtype="step", yerr=True, density=density, label=label_extra)
        plt.legend()
        plt.savefig(os.path.join(PLOT_DIR, f"{plot_field}{'_'+label_extra+'_' if label_extra != '' else ''}{'_density' if density else ''}.pdf"), bbox_inches='tight')
        plt.savefig(os.path.join(PLOT_DIR, f"{plot_field}{'_'+label_extra+'_' if label_extra != '' else ''}{'_density' if density else ''}.png"), bbox_inches='tight')
        plt.close()

# Making efficiency plots for T (tag) and W (mistag) for top fatjets
make_efficiency_plots(fj_matched_t_mask, "PNet_TvsQCD")
make_efficiency_plots(fj_matched_t_mask, "PNet_WvsQCD", label_extra="mistag_eff")

# Making efficiency plots for T (mistag) and W (tag) for w fatjets
make_efficiency_plots(w_matched_fj, "PNet_WvsQCD")
make_efficiency_plots(w_matched_fj, "PNet_TvsQCD", label_extra="mistag_eff")