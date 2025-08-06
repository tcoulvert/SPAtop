import os

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np

from src.analysis.boosted import parse_boosted_w_target
from src.analysis.resolved import parse_resolved_w_target
from src.analysis.semi_resolved import parse_semi_resolved_w_target
from src.analysis.utils import calc_eff, calc_pur


def calc_pur_eff(target_path, pred_path, bins):
    # open files
    pred_h5 = h5.File(pred_path, "a")
    target_h5 = h5.File(target_path)

    # handle different pl version
    if "TARGETS" not in pred_h5.keys():
        pred_h5["INPUTS"] = pred_h5["SpecialKey.Inputs"]
        pred_h5["TARGETS"] = pred_h5["SpecialKey.Targets"]

    SR_condition = any("SR" in key for key in pred_h5["TARGETS"].keys())

    ## generate look up tables ##
    # boosted
    LUT_boosted_pred, LUT_boosted_target, vfjs_reco = parse_boosted_w_target(target_h5, pred_h5)
    # semi-resolved
    if SR_condition:
        LUT_semiresolved_qq_pred, LUT_semiresolved_qq_target, fjs_reco_qq = parse_semi_resolved_w_target(target_h5, pred_h5, 'qq', vfjs_reco=None)
        LUT_semiresolved_qq_wOR_pred, LUT_semiresolved_qq_wOR_target, _ = parse_semi_resolved_w_target(target_h5, pred_h5, 'qq', vfjs_reco=vfjs_reco)
        LUT_semiresolved_bq_pred, LUT_semiresolved_bq_target, fjs_reco_bq = parse_semi_resolved_w_target(target_h5, pred_h5, 'bq', vfjs_reco=None)
        LUT_semiresolved_bq_wOR_pred, LUT_semiresolved_bq_wOR_target, _ = parse_semi_resolved_w_target(target_h5, pred_h5, 'bq', vfjs_reco=vfjs_reco)
    else:
        LUT_semiresolved_qq_pred, LUT_semiresolved_qq_target = None, None
        LUT_semiresolved_bq_pred, LUT_semiresolved_bq_target = None, None
    # resolved
    LUT_resolved_pred, LUT_resolved_target, _ = parse_resolved_w_target(target_h5, pred_h5, fjs_reco=None)
    if SR_condition:
        LUT_resolved_wOR_pred, LUT_resolved_wOR_target, _ = parse_resolved_w_target(target_h5, pred_h5, fjs_reco=[vfjs_reco, fjs_reco_qq, fjs_reco_bq])
    else:
        LUT_resolved_wOR_pred, LUT_resolved_wOR_target, _ = parse_resolved_w_target(target_h5, pred_h5, fjs_reco=vfjs_reco)

    # make no_OR LUTs
    LUT_resolved_pred_no_OR = []
    for event in LUT_resolved_wOR_pred:
        event_no_OR = []
        for predFRt in event:
            if predFRt[2] == 0:
                event_no_OR.append(predFRt)
        LUT_resolved_pred_no_OR.append(event_no_OR)

    LUT_resolved_target_no_OR = []
    for event in LUT_resolved_wOR_target:
        event_no_OR = []
        for targetFRt in event:
            if targetFRt[2] == 0:
                event_no_OR.append(targetFRt)
        LUT_resolved_target_no_OR.append(event_no_OR)

    if SR_condition:
        LUT_semiresolved_qq_pred_no_OR = []
        for event in LUT_semiresolved_qq_wOR_pred:
            event_no_OR = []
            for predSRt in event:
                if predSRt[2] == 0:
                    event_no_OR.append(predSRt)
            LUT_semiresolved_qq_pred_no_OR.append(event_no_OR)

        LUT_semiresolved_qq_target_no_OR = []
        for event in LUT_semiresolved_qq_wOR_target:
            event_no_OR = []
            for targetSRt in event:
                if targetSRt[2] == 0:
                    event_no_OR.append(targetSRt)
            LUT_semiresolved_qq_target_no_OR.append(event_no_OR)

        LUT_semiresolved_bq_pred_no_OR = []
        for event in LUT_semiresolved_bq_wOR_pred:
            event_no_OR = []
            for predSRt in event:
                if predSRt[2] == 0:
                    event_no_OR.append(predSRt)
            LUT_semiresolved_bq_pred_no_OR.append(event_no_OR)

        LUT_semiresolved_bq_target_no_OR = []
        for event in LUT_semiresolved_bq_wOR_target:
            event_no_OR = []
            for targetSRt in event:
                if targetSRt[2] == 0:
                    event_no_OR.append(targetSRt)
            LUT_semiresolved_bq_target_no_OR.append(event_no_OR)


    ## calculate efficiencies and purities for b+r, b, and r (and srqq, srbq if available) ##
    results = {}
    # merged
    results["pur_m"], results["purerr_m"] = calc_eff(LUT_boosted_pred, LUT_semiresolved_qq_wOR_pred, LUT_semiresolved_bq_wOR_pred, LUT_resolved_wOR_pred, bins)
    results["eff_m"], results["efferr_m"] = calc_pur(LUT_boosted_target, LUT_semiresolved_qq_wOR_target, LUT_semiresolved_bq_wOR_target, LUT_resolved_wOR_target, bins)
    # boosted
    results["pur_b"], results["purerr_b"] = calc_eff(LUT_boosted_pred, None, None, None, bins)
    results["eff_b"], results["efferr_b"] = calc_pur(LUT_boosted_target, None, None, None, bins)
    # resolved
    results["pur_r"], results["purerr_r"] = calc_eff(None, None, None, LUT_resolved_pred, bins)
    results["eff_r"], results["efferr_r"] = calc_pur(None, None, None, LUT_resolved_target, bins)
    # resolved no OR
    results["pur_r_or"], results["purerr_r_or"] = calc_eff(None, None, None, LUT_resolved_pred_no_OR, bins)
    results["eff_r_or"], results["efferr_r_or"] = calc_pur(None, None, None, LUT_resolved_target_no_OR, bins)
    # semi-resolved
    if SR_condition:
        # semi-resolved qq
        results["pur_srqq"], results["purerr_srqq"] = calc_eff(None, LUT_semiresolved_qq_pred, None, None, bins)
        results["eff_srqq"], results["efferr_srqq"] = calc_pur(None, LUT_semiresolved_qq_target, None, None, bins)
        # semi-resolved qq no OR
        results["pur_srqq_or"], results["purerr_srqq_or"] = calc_eff(None, LUT_semiresolved_qq_pred_no_OR, None, None, bins)
        results["eff_srqq_or"], results["efferr_srqq_or"] = calc_pur(None, LUT_semiresolved_qq_target_no_OR, None, None, bins)
        # semi-resolved bq
        results["pur_srbq"], results["purerr_srbq"] = calc_eff(None, None, LUT_semiresolved_bq_pred, None, bins)
        results["eff_srbq"], results["efferr_srbq"] = calc_pur(None, None, LUT_semiresolved_bq_target, None, bins)
        # semi-resolved bq no OR
        results["pur_srbq_or"], results["purerr_srbq_or"] = calc_eff(None, None, LUT_semiresolved_bq_pred_no_OR, None, bins)
        results["eff_srbq_or"], results["efferr_srbq_or"] = calc_pur(None, None, LUT_semiresolved_bq_target_no_OR, None, bins)


    print("Number of Boosted Prediction:", np.array([pred for event in LUT_boosted_pred for pred in event]).shape[0])
    print(
        "Number of Resolved Prediction before OR:",
        np.array([pred for event in LUT_resolved_pred for pred in event]).shape[0],
    )
    print(
        "Number of Resolved Prediction after OR:",
        np.array([pred for event in LUT_resolved_pred_no_OR for pred in event]).shape[0],
    )
    if SR_condition:
        print(
            "Number of Semi-Resolved-qq Prediction before OR:",
            np.array([pred for event in LUT_semiresolved_qq_pred for pred in event]).shape[0],
        )
        print(
            "Number of Semi-Resolved-qq Prediction after OR:",
            np.array([pred for event in LUT_semiresolved_qq_pred_no_OR for pred in event]).shape[0],
        )
        print(
            "Number of Semi-Resolved-bq Prediction before OR:",
            np.array([pred for event in LUT_semiresolved_bq_pred for pred in event]).shape[0],
        )
        print(
            "Number of Semi-Resolved-bq Prediction after OR:",
            np.array([pred for event in LUT_semiresolved_bq_pred_no_OR for pred in event]).shape[0],
        )

    return results


# I started to use "efficiency" for describing how many gen tops were reconstructed
# and "purity" for desrcribing how many reco tops are actually gen tops
def plot_pur_eff_w_dict(plot_dict, target_path, save_path=None, proj_name=None, bins=None):
    if bins is None:
        bins = np.arange(0, 1050, 50)

    plot_bins = np.append(bins, 2 * bins[-1] - bins[-2])
    bin_centers = [(plot_bins[i] + plot_bins[i + 1]) / 2 for i in range(plot_bins.size - 1)]
    xerr = (plot_bins[1] - plot_bins[0]) / 2 * np.ones(plot_bins.shape[0] - 1)

    # m: merged (b+r w OR)
    # b: boosted
    # r: resolved
    fig_m, ax_m = plt.subplots(1, 2, figsize=(12, 5))
    fig_b, ax_b = plt.subplots(1, 2, figsize=(12, 5))
    fig_r, ax_r = plt.subplots(1, 2, figsize=(12, 5))
    fig_r_or, ax_r_or = plt.subplots(1, 2, figsize=(12, 5))
    fig_srqq, ax_srqq = plt.subplots(1, 2, figsize=(12, 5))
    fig_srqq_or, ax_srqq_or = plt.subplots(1, 2, figsize=(12, 5))
    fig_srbq, ax_srbq = plt.subplots(1, 2, figsize=(12, 5))
    fig_srbq_or, ax_srbq_or = plt.subplots(1, 2, figsize=(12, 5))

    ## preset figure labels, titles, limits, etc. ##
    # merged
    ax_m[0].set(
        xlabel=r"Merged Reco top pT (GeV)",
        ylabel=r"Reconstruction Purity",
        title=f"Reconstruction Purity vs. Merged Reco top pT",
    )
    ax_m[1].set(
        xlabel=r"Merged Gen top pT (GeV)",
        ylabel=r"Reconstruction Efficiency",
        title=f"Reconstruction Efficiency vs. Merged Gen top pT",
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
    ax_r_or[0].set(
        xlabel=r"Reco Resolved top pT (GeV)",
        ylabel=r"Reconstruction Purity",
        title=f"Resolved Purity After OR  vs. Reco Resolved top pT",
    )
    ax_r_or[1].set(
        xlabel=r"Gen Resolved top pT (GeV)",
        ylabel=r"Reconstruction Efficiency",
        title=f"Resolved Efficiency After OR vs. Gen Resolved top pT",
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
    ax_srqq_or[0].set(
        xlabel=r"Reco Semi-Resolved top pT (GeV)",
        ylabel=r"Reconstruction Purity",
        title=f"Semi-Resolved Purity After OR  vs. Reco Semi-Resolved top pT",
    )
    ax_srqq_or[1].set(
        xlabel=r"Gen Semi-Resolved top pT (GeV)",
        ylabel=r"Reconstruction Efficiency",
        title=f"Semi-Resolved Efficiency After OR vs. Gen Semi-Resolved top pT",
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
    ax_srbq_or[0].set(
        xlabel=r"Reco Semi-Resolved top pT (GeV)",
        ylabel=r"Reconstruction Purity",
        title=f"Semi-Resolved Purity After OR  vs. Reco Semi-Resolved top pT",
    )
    ax_srbq_or[1].set(
        xlabel=r"Gen Semi-Resolved top pT (GeV)",
        ylabel=r"Reconstruction Efficiency",
        title=f"Semi-Resolved Efficiency After OR vs. Gen Semi-Resolved top pT",
    )


    SR_condition = False
    ## plot purities and efficiencies ##
    for tag, pred_path in plot_dict.items():
        print("Processing", tag)
        results = calc_pur_eff(target_path, pred_path, bins)
        SR_condition = any('sr' in key.split('_')[-1] for key in results.keys())

        # merged
        ax_m[0].errorbar(
            x=bin_centers, y=results["pur_m"], xerr=xerr, yerr=results["purerr_m"], fmt="o", capsize=5, label=tag
        )
        ax_m[1].errorbar(
            x=bin_centers, y=results["eff_m"], xerr=xerr, yerr=results["efferr_m"], fmt="o", capsize=5, label=tag
        )
        # boosted
        ax_b[0].errorbar(
            x=bin_centers, y=results["pur_b"], xerr=xerr, yerr=results["purerr_b"], fmt="o", capsize=5, label=tag
        )
        ax_b[1].errorbar(
            x=bin_centers, y=results["eff_b"], xerr=xerr, yerr=results["efferr_b"], fmt="o", capsize=5, label=tag
        )
        # resolved
        ax_r[0].errorbar(
            x=bin_centers, y=results["pur_r"], xerr=xerr, yerr=results["purerr_r"], fmt="o", capsize=5, label=tag
        )
        ax_r[1].errorbar(
            x=bin_centers, y=results["eff_r"], xerr=xerr, yerr=results["efferr_r"], fmt="o", capsize=5, label=tag
        )
        ax_r_or[0].errorbar(
            x=bin_centers, y=results["pur_r_or"], xerr=xerr, yerr=results["purerr_r_or"], fmt="o", capsize=5, label=tag
        )
        ax_r_or[1].errorbar(
            x=bin_centers, y=results["eff_r_or"], xerr=xerr, yerr=results["efferr_r_or"], fmt="o", capsize=5, label=tag
        )
        # semi-resolved
        if SR_condition:
            # semi-resolved qq
            ax_srqq[0].errorbar(
                x=bin_centers, y=results["pur_srqq"], xerr=xerr, yerr=results["purerr_srqq"], fmt="o", capsize=5, label=tag
            )
            ax_srqq[1].errorbar(
                x=bin_centers, y=results["eff_srqq"], xerr=xerr, yerr=results["efferr_srqq"], fmt="o", capsize=5, label=tag
            )
            ax_srqq_or[0].errorbar(
                x=bin_centers, y=results["pur_srqq_or"], xerr=xerr, yerr=results["purerr_srqq_or"], fmt="o", capsize=5, label=tag
            )
            ax_srqq_or[1].errorbar(
                x=bin_centers, y=results["eff_srqq_or"], xerr=xerr, yerr=results["efferr_srqq_or"], fmt="o", capsize=5, label=tag
            )
            # semi-resolved bq
            ax_srbq[0].errorbar(
                x=bin_centers, y=results["pur_srbq"], xerr=xerr, yerr=results["purerr_srbq"], fmt="o", capsize=5, label=tag
            )
            ax_srbq[1].errorbar(
                x=bin_centers, y=results["eff_srbq"], xerr=xerr, yerr=results["efferr_srbq"], fmt="o", capsize=5, label=tag
            )
            ax_srbq_or[0].errorbar(
                x=bin_centers, y=results["pur_srbq_or"], xerr=xerr, yerr=results["purerr_srbq_or"], fmt="o", capsize=5, label=tag
            )
            ax_srbq_or[1].errorbar(
                x=bin_centers, y=results["eff_srbq_or"], xerr=xerr, yerr=results["efferr_srbq_or"], fmt="o", capsize=5, label=tag
            )


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
    ax_r_or[0].legend()
    ax_r_or[1].legend()
    ax_r_or[0].set_ylim([-0.1, 1.1])
    ax_r_or[1].set_ylim([-0.1, 1.1])
    # semi-resolved qq
    ax_srqq[0].legend()
    ax_srqq[1].legend()
    ax_srqq[0].set_ylim([-0.1, 1.1])
    ax_srqq[1].set_ylim([-0.1, 1.1])
    ax_srqq_or[0].legend()
    ax_srqq_or[1].legend()
    ax_srqq_or[0].set_ylim([-0.1, 1.1])
    ax_srqq_or[1].set_ylim([-0.1, 1.1])
    # semi-resolved bq
    ax_srbq[0].legend()
    ax_srbq[1].legend()
    ax_srbq[0].set_ylim([-0.1, 1.1])
    ax_srbq[1].set_ylim([-0.1, 1.1])
    ax_srbq_or[0].legend()
    ax_srbq_or[1].legend()
    ax_srbq_or[0].set_ylim([-0.1, 1.1])
    ax_srbq_or[1].set_ylim([-0.1, 1.1])

    plt.show()

    if save_path is not None:
        # merged
        fig_m.savefig(os.path.join(save_path, proj_name+'_merged.pdf'))
        # boosted
        fig_b.savefig(os.path.join(save_path, proj_name+'_boosted.pdf'))
        # resolved
        fig_r.savefig(os.path.join(save_path, proj_name+'_resolved.pdf'))
        fig_r_or.savefig(os.path.join(save_path, proj_name+'_resolved_wOR.pdf'))
        # semi-resolved qq
        fig_srqq.savefig(os.path.join(save_path, proj_name+'_semiresolved_qq.pdf'))
        fig_srqq_or.savefig(os.path.join(save_path, proj_name+'_semiresolved_qq_wOR.pdf'))
        # semi-resolved bq
        fig_srbq.savefig(os.path.join(save_path, proj_name+'_semiresolved_bq.pdf'))
        fig_srbq_or.savefig(os.path.join(save_path, proj_name+'_semiresolved_bq_wOR.pdf'))

    return
