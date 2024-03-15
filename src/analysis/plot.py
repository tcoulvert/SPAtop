import h5py as h5
import numpy as np
import matplotlib.pyplot as plt

from src.analysis.boosted import parse_boosted_w_target
from src.analysis.resolved import parse_resolved_w_target

from src.analysis.utils import calc_eff, calc_pur


def calc_pur_eff(target_path, pred_path, bins):
    # open files
    pred_h5 = h5.File(pred_path, 'a')
    target_h5 = h5.File(target_path)

    # handle different pl version
    if 'TARGETS' not in pred_h5.keys():
        pred_h5["INPUTS"] = pred_h5["SpecialKey.Inputs"]
        pred_h5["TARGETS"] = pred_h5["SpecialKey.Targets"]
    
    # generate look up tables
    LUT_boosted_pred, LUT_boosted_target, fjs_reco = parse_boosted_w_target(target_h5, pred_h5)
    LUT_resolved_pred, LUT_resolved_target, _ = parse_resolved_w_target(target_h5, pred_h5, fjs_reco=None)
    LUT_resolved_wOR_pred, LUT_resolved_wOR_target, _ = parse_resolved_w_target(target_h5, pred_h5, fjs_reco=fjs_reco)
    
    # calculate efficiencies and purities for b+r, b, and r
    results = {}
    results['pur_m'], results['purerr_m'] = calc_eff(LUT_boosted_pred, LUT_resolved_wOR_pred, bins)
    results['eff_m'], results['efferr_m'] = calc_pur(LUT_boosted_target, LUT_resolved_wOR_target, bins)
    
    results['pur_b'], results['purerr_b'] = calc_eff(LUT_boosted_pred, None, bins)
    results['eff_b'], results['efferr_b'] = calc_pur(LUT_boosted_target, None, bins)
    
    results['pur_r'], results['purerr_r'] = calc_eff(None, LUT_resolved_pred, bins)
    results['eff_r'], results['efferr_r'] = calc_pur(None, LUT_resolved_target, bins)
    
    return results
    
# I started to use "efficiency" for describing how many gen Higgs were reconstructed
# and "purity" for desrcribing how many reco Higgs are actually gen Higgs
def plot_pur_eff_w_dict(plot_dict, target_path, save_path=None, proj_name=None, bins=None):
    if bins == None:
        bins = np.arange(0, 1000, 50)
    bin_centers = [(bins[i]+bins[i+1])/2 for i in range(bins.size-1)]
    xerr=(bins[1]-bins[0])/2*np.ones(bins.shape[0]-1)
            
    # m: merged (b+r w OR)
    # b: boosted
    # r: resolved
    fig_m, ax_m = plt.subplots(1, 2, figsize=(12, 5))
    fig_b, ax_b = plt.subplots(1, 2, figsize=(12, 5))
    fig_r, ax_r = plt.subplots(1, 2, figsize=(12, 5))
    
    # preset figure labels, titles, limits, etc.
    ax_m[0].set(xlabel=r"Merged Reco H pT (GeV)", ylabel=r"Reconstruction Purity", title=f"Reconstruction Purity vs. Merged Reco H pT")
    ax_m[1].set(xlabel=r"Merged Gen H pT (GeV)", ylabel=r"Reconstruction Efficiency", title=f"Reconstruction Efficiency vs. Merged Gen H pT")
    ax_b[0].set(xlabel=r"Reco Boosted H pT (GeV)", ylabel=r"Reconstruction Purity", title=f"Reconstruction Purity vs. Reco Boosted H pT")
    ax_b[1].set(xlabel=r"Gen Boosted H pT (GeV)", ylabel=r"Reconstruction Efficiency", title=f"Reconstruction Efficiency vs. Gen Boosted H pT")
    ax_r[0].set(xlabel=r"Reco Resolved H pT (GeV)", ylabel=r"Reconstruction Purity", title=f"Reconstruction Purity vs. Reco Resolved H pT")
    ax_r[1].set(xlabel=r"Gen Resolved H pT (GeV)", ylabel=r"Reconstruction Efficiency", title=f"Reconstruction Efficiency vs. Gen Resolved H pT")
    
    
    # plot purities and efficiencies
    for tag, pred_path in plot_dict.items():
        results = calc_pur_eff(target_path, pred_path, bins)
        ax_m[0].errorbar(x=bin_centers, y=results['pur_m'], xerr=xerr, yerr=results['purerr_m'], fmt='o', capsize=5, label=tag)
        ax_m[1].errorbar(x=bin_centers, y=results['eff_m'], xerr=xerr, yerr=results['efferr_m'], fmt='o', capsize=5, label=tag)
        ax_b[0].errorbar(x=bin_centers, y=results['pur_b'], xerr=xerr, yerr=results['purerr_b'], fmt='o', capsize=5, label=tag)
        ax_b[1].errorbar(x=bin_centers, y=results['eff_b'], xerr=xerr, yerr=results['efferr_b'], fmt='o', capsize=5, label=tag)
        ax_r[0].errorbar(x=bin_centers, y=results['pur_r'], xerr=xerr, yerr=results['purerr_r'], fmt='o', capsize=5, label=tag)
        ax_r[1].errorbar(x=bin_centers, y=results['eff_r'], xerr=xerr, yerr=results['efferr_r'], fmt='o', capsize=5, label=tag)
       
    # adjust limits and legends
    ax_m[0].legend()
    ax_m[1].legend()
    ax_m[0].set_ylim([-0.1, 1.1])
    ax_m[1].set_ylim([-0.1, 1.1])
    ax_b[0].legend()
    ax_b[1].legend()
    ax_b[0].set_ylim([-0.1, 1.1])
    ax_b[1].set_ylim([-0.1, 1.1])
    ax_r[0].legend()
    ax_r[1].legend()
    ax_r[0].set_ylim([-0.1, 1.1])
    ax_r[1].set_ylim([-0.1, 1.1])
        
    plt.show()
    
    if save_path is not None:
        fig_m.savefig(f"{save_path}/{proj_name}_merged.png")
        fig_b.savefig(f"{save_path}/{proj_name}_boosted.png")
        fig_r.savefig(f"{save_path}/{proj_name}_resolved.png")
    
    return
    