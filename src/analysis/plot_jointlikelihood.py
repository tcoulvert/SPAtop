import os

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np

from src.analysis.boosted import parse_boosted_w_target
from src.analysis.resolved import parse_resolved_w_target
from src.analysis.semi_resolved import parse_semi_resolved_w_target
from src.analysis.merged import parse_merged_w_target
from src.analysis.utils import calc_eff, calc_pur


def calc_pur_eff(target_path, pred_path, bins_dict, chi2_cut=45, mode: str='pur_eff'):
    # open files
    pred_h5 = h5.File(pred_path, "a")
    target_h5 = h5.File(target_path)

    # handle different pl version
    if "TARGETS" not in pred_h5.keys():
        pred_h5["INPUTS"] = pred_h5["SpecialKey.Inputs"]
        pred_h5["TARGETS"] = pred_h5["SpecialKey.Targets"]

    ## generate look up tables ##
    # boosted
    LUT_boosted_pred, LUT_boosted_target = parse_boosted_w_target(target_h5, pred_h5)

    # semi-resolved
    # qq
    LUT_semiresolved_qq_pred, LUT_semiresolved_qq_target = parse_semi_resolved_w_target(target_h5, pred_h5, 'qq')
    # bq
    LUT_semiresolved_bq_pred, LUT_semiresolved_bq_target = parse_semi_resolved_w_target(target_h5, pred_h5, 'bq')

    # resolved
    LUT_resolved_pred, LUT_resolved_target = parse_resolved_w_target(target_h5, pred_h5, chi2_cut=chi2_cut)

    # merged
    LUT_merged_pred, LUT_merged_target = parse_merged_w_target(target_h5, pred_h5)


    ## calculate efficiencies and purities for b+r, b, and r (and srqq, srbq if available) ##
    results = {}
    if 'pur' in mode.lower():
        results["pur_m"], results["purerr_m"] = calc_eff(LUT_merged_pred, bins_dict['all'])
        results["pur_b"], results["purerr_b"] = calc_eff(LUT_boosted_pred, bins_dict['FB'])
        results["pur_r"], results["purerr_r"] = calc_eff(LUT_resolved_pred, bins_dict['FR'])
        results["pur_srqq"], results["purerr_srqq"] = calc_eff(LUT_semiresolved_qq_pred, bins_dict['SRqq'])
        results["pur_srbq"], results["purerr_srbq"] = calc_eff(LUT_semiresolved_bq_pred, bins_dict['SRbq'])
    if 'eff' in mode.lower():
        results["eff_m"], results["efferr_m"] = calc_pur(LUT_merged_target, bins_dict['all'])
        results["eff_b"], results["efferr_b"] = calc_pur(LUT_boosted_target, bins_dict['FB'])
        results["eff_r"], results["efferr_r"] = calc_pur(LUT_resolved_target, bins_dict['FR'])
        results["eff_srqq"], results["efferr_srqq"] = calc_pur(LUT_semiresolved_qq_target, bins_dict['SRqq'])
        results["eff_srbq"], results["efferr_srbq"] = calc_pur(LUT_semiresolved_bq_target, bins_dict['SRbq'])


    print("Number of Boosted Prediction:", np.array([pred for event in LUT_boosted_pred for pred in event]).shape[0])
    print("Number of Resolved Prediction:", np.array([pred for event in LUT_resolved_pred for pred in event]).shape[0])
    print("Number of Semi-Resolved-qq Prediction:", np.array([pred for event in LUT_semiresolved_qq_pred for pred in event]).shape[0])
    print("Number of Semi-Resolved-bq Prediction:", np.array([pred for event in LUT_semiresolved_bq_pred for pred in event]).shape[0])

    return results


# I started to use "efficiency" for describing how many gen tops were reconstructed
# and "purity" for desrcribing how many reco tops are actually gen tops
def plot_pur_eff_w_dict(
    plot_dict, target_path, save_path=None, proj_name=None, 
    bins_dict={
        'FR': np.arange(0, 300, 10),
        'SRqq': np.arange(100, 400, 10),
        'SRbq': np.arange(100, 400, 10),
        'FB': np.arange(200, 1000, 50),
        'all': np.arange(0, 1000, 50),
    }
):

    plot_bins_dict = {
        key: np.append(bins, 2 * bins[-1] - bins[-2])
        for key, bins in bins_dict.items()
    }
    bin_centers_dict = {
        key: [(plot_bins[i] + plot_bins[i + 1]) / 2 for i in range(plot_bins.size - 1)]
        for key, plot_bins in plot_bins_dict.items()
    }
    xerr_dict = {
        key: (plot_bins[1] - plot_bins[0]) / 2 * np.ones(plot_bins.shape[0] - 1)
        for key, plot_bins in plot_bins_dict.items()
    }

    # m: merged (b++sr+r)
    # b: boosted
    # r: resolved
    fig_m, ax_m = plt.subplots(1, 2, figsize=(15, 5))
    fig_b, ax_b = plt.subplots(1, 2, figsize=(15, 5))
    fig_r, ax_r = plt.subplots(1, 2, figsize=(15, 5))
    fig_srqq, ax_srqq = plt.subplots(1, 2, figsize=(15, 5))
    fig_srbq, ax_srbq = plt.subplots(1, 2, figsize=(15, 5))

    ## preset figure labels, titles, limits, etc. ##
    # merged
    ax_m[0].set(
        xlabel=r"All categories Reco top pT (GeV)",
        ylabel=r"Reconstruction Purity",
        title=f"Reconstruction Purity vs. All category Reco top pT",
    )
    ax_m[1].set(
        xlabel=r"All categories Gen top pT (GeV)",
        ylabel=r"Reconstruction Efficiency",
        title=f"Reconstruction Efficiency vs. All category Gen top pT",
    )
    # boosted
    ax_b[0].set(
        xlabel=r"Reco Boosted top pT (GeV)",
        ylabel=r"Reconstruction Purity",
        title=f"Reconstruction Purity vs. Reco Boosted top pT",
    )
    ax_b[1].set(
        xlabel=r"Gen Boosted top pT (GeV)",
        ylabel=r"Reconstruction Efficiency",
        title=f"Reconstruction Efficiency vs. Gen Boosted top pT",
    )
    # resolved
    ax_r[0].set(
        xlabel=r"Reco Resolved top pT (GeV)",
        ylabel=r"Reconstruction Purity",
        title=f"Reconstruction Purity vs. Reco Resolved top pT",
    )
    ax_r[1].set(
        xlabel=r"Gen Resolved top pT (GeV)",
        ylabel=r"Reconstruction Efficiency",
        title=f"Reconstruction Efficiency vs. Gen Resolved top pT",
    )
    # semi-resolved qq
    ax_srqq[0].set(
        xlabel=r"Reco Semi-Resolved top pT (GeV)",
        ylabel=r"Reconstruction Purity",
        title=f"Reconstruction Purity vs. Reco Semi-Resolved top pT",
    )
    ax_srqq[1].set(
        xlabel=r"Gen Semi-Resolved top pT (GeV)",
        ylabel=r"Reconstruction Efficiency",
        title=f"Reconstruction Efficiency vs. Gen Semi-Resolved top pT",
    )
    # semi-resolved bq
    ax_srbq[0].set(
        xlabel=r"Reco Semi-Resolved top pT (GeV)",
        ylabel=r"Reconstruction Purity",
        title=f"Reconstruction Purity vs. Reco Semi-Resolved top pT",
    )
    ax_srbq[1].set(
        xlabel=r"Gen Semi-Resolved top pT (GeV)",
        ylabel=r"Reconstruction Efficiency",
        title=f"Reconstruction Efficiency vs. Gen Semi-Resolved top pT",
    )


    ## plot purities and efficiencies ##
    for tag, pred_path in plot_dict.items():

        tag_label = tag
        if 'chi2' in tag:
            tag_list = tag.split('_')
            tag = tag_list[0]
            chi2_cut = int(tag_list[1])

            print("Processing", tag_label)
            results = calc_pur_eff(target_path, pred_path, bins_dict, chi2_cut=chi2_cut)
        else:
            print("Processing", tag_label)
            results = calc_pur_eff(target_path, pred_path, bins_dict)


        # merged
        ax_m[0].errorbar(x=bin_centers_dict['all'], y=results["pur_m"], xerr=xerr_dict['all'], yerr=results["purerr_m"], fmt="o", capsize=5, label=tag_label)
        ax_m[1].errorbar(x=bin_centers_dict['all'], y=results["eff_m"], xerr=xerr_dict['all'], yerr=results["efferr_m"], fmt="o", capsize=5, label=tag_label)
        # boosted
        ax_b[0].errorbar(x=bin_centers_dict['FB'], y=results["pur_b"], xerr=xerr_dict['FB'], yerr=results["purerr_b"], fmt="o", capsize=5, label=tag_label)
        ax_b[1].errorbar(x=bin_centers_dict['FB'], y=results["eff_b"], xerr=xerr_dict['FB'], yerr=results["efferr_b"], fmt="o", capsize=5, label=tag_label)
        # resolved
        ax_r[0].errorbar(x=bin_centers_dict['FR'], y=results["pur_r"], xerr=xerr_dict['FR'], yerr=results["purerr_r"], fmt="o", capsize=5, label=tag_label)
        ax_r[1].errorbar(x=bin_centers_dict['FR'], y=results["eff_r"], xerr=xerr_dict['FR'], yerr=results["efferr_r"], fmt="o", capsize=5, label=tag_label)
        # semi-resolved qq
        ax_srqq[0].errorbar(x=bin_centers_dict['SRqq'], y=results["pur_srqq"], xerr=xerr_dict['SRqq'], yerr=results["purerr_srqq"], fmt="o", capsize=5, label=tag_label)
        ax_srqq[1].errorbar(x=bin_centers_dict['SRqq'], y=results["eff_srqq"], xerr=xerr_dict['SRqq'], yerr=results["efferr_srqq"], fmt="o", capsize=5, label=tag_label)
        # semi-resolved bq
        ax_srbq[0].errorbar(x=bin_centers_dict['SRbq'], y=results["pur_srbq"], xerr=xerr_dict['SRbq'], yerr=results["purerr_srbq"], fmt="o", capsize=5, label=tag_label)
        ax_srbq[1].errorbar(x=bin_centers_dict['SRbq'], y=results["eff_srbq"], xerr=xerr_dict['SRbq'], yerr=results["efferr_srbq"], fmt="o", capsize=5, label=tag_label)


    ## adjust limits and legends ##
    # merged
    ax_m[0].legend()
    ax_m[1].legend()
    ax_m[0].set_ylim([-0.1, 1.1])
    ax_m[1].set_ylim([-0.1, 1.1])
    # boosted
    ax_b[0].legend()
    ax_b[1].legend()
    ax_b[0].set_ylim([-0.1, 1.1])
    ax_b[1].set_ylim([-0.1, 1.1])
    # resolved
    ax_r[0].legend()
    ax_r[1].legend()
    ax_r[0].set_ylim([-0.1, 1.1])
    ax_r[1].set_ylim([-0.1, 1.1])
    # semi-resolved qq
    ax_srqq[0].legend()
    ax_srqq[1].legend()
    ax_srqq[0].set_ylim([-0.1, 1.1])
    ax_srqq[1].set_ylim([-0.1, 1.1])
    # semi-resolved bq
    ax_srbq[0].legend()
    ax_srbq[1].legend()
    ax_srbq[0].set_ylim([-0.1, 1.1])
    ax_srbq[1].set_ylim([-0.1, 1.1])

    plt.show()

    if save_path is not None:
        # merged
        fig_m.savefig(os.path.join(save_path, proj_name+'_merged.pdf'))
        # boosted
        fig_b.savefig(os.path.join(save_path, proj_name+'_boosted.pdf'))
        # resolved
        fig_r.savefig(os.path.join(save_path, proj_name+'_resolved.pdf'))
        # semi-resolved qq
        fig_srqq.savefig(os.path.join(save_path, proj_name+'_semiresolved_qq.pdf'))
        # semi-resolved bq
        fig_srbq.savefig(os.path.join(save_path, proj_name+'_semiresolved_bq.pdf'))

    return
