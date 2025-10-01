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
    "rho": events.FatJet.pt,
    "eta": events.FatJet.eta,
    "phi": events.FatJet.phi,
    "mass": events.FatJet.mass,
    "softdrop_mass": events.FatJet.msoftdrop,
    "PNet_TvsQCD": events.FatJet.particleNetWithMass_TvsQCD,
    "PNet_WvsQCD": events.FatJet.particleNetWithMass_WvsQCD,
}, with_name="Momentum4D")
# gen_fatjets = ak.zip({
#     "rho": events.GenJetAK8.pt,
#     "eta": events.GenJetAK8.eta,
#     "phi": events.GenJetAK8.phi,
#     "mass": events.GenJetAK8.mass,
# }, with_name="Momentum4D")
gen_particles = ak.zip({
    "rho": events.GenPart.pt,
    "eta": events.GenPart.eta,
    "phi": events.GenPart.phi,
    "mass": events.GenPart.mass,
    "pdgId": events.GenPart.pdgId,
    "status": events.GenPart.status,
    "statusFlags": events.GenPart.statusFlags,
    "genPartIdxMother": events.GenPart.genPartIdxMother
}, with_name="Momentum4D")


def make_efficiency_plots(fjs, ps, plot_field):
    p_fjs = ak.cartesian({"fj": fjs, "p": ps}, axis=1)
    p_fjs = ak.with_field(p_fjs, p_fjs.fj.deltaR(p_fjs.p), "fj_p_DR")

    best_idx = ak.argmin(p_fjs.fj_p_DR, axis=1)
    best_p_fj = ak.firsts(p_fjs[ak.local_index(p_fjs) == best_idx])
    best_p_fj = ak.where(best_p_fj.fj_t_DR < FATJET_DR, best_p_fj, None)

    plot_hist = hist.Hist(
        hist.axis.Regular(100, -1., 1., name="var", label=plot_field)
    ).fill(var=best_p_fj.fj[plot_field])

    for density in [False, True]:
        fig, ax = plt.subplots()
        hep.cms.lumitext(f"2022" + r" (13.6 TeV)", ax=ax)
        hep.cms.text("Simulation", ax=ax)
        hep.histplot(plot_hist, ax=ax, linewidth=3, histtype="step", yerr=True, density=density)
        plt.savefig(os.path.join(PLOT_DIR, f"{plot_field}{'_density' if density else ''}.pdf"), bbox_inches='tight')
        plt.savefig(os.path.join(PLOT_DIR, f"{plot_field}{'_density' if density else ''}.png"), bbox_inches='tight')

# Finding gen-matched top fatjets
tops_mask = ( (gen_particles.statusFlags == 13) & (np.abs(gen_particles.pdgId) == 6) )
tops = gen_particles[tops_mask]
make_efficiency_plots(fatjets, tops, "PNet_TvsQCD")

# Finding gen-matched w fatjets
w_mask = ( (gen_particles.statusFlags == 13) & (np.abs(gen_particles.pdgId) == 24) )
ws = gen_particles[w_mask]
make_efficiency_plots(fatjets, tops, "PNet_WvsQCD")