import awkward as ak
import numba as nb
import numpy as np
import vector
vector.register_awkward()

from src.analysis.plot import calc_pur_eff, plot_pur_eff_w_dict


target_file = "/storage/af/user/tsievert/topNet/SPAtop/data/delphes/v4/tt_hadronic_testing_SLIMMED.h5"
pred_files = {
    'SPAtop': "/storage/af/user/tsievert/topNet/SPAtop/data/delphes/v4/tt_hadronic_predict.h5",
    'chi2': "/storage/af/user/tsievert/topNet/SPAtop/data/delphes/v4/tt_hadronic_baseline.h5"
}

bins = np.arange(200, 1000, 100)
bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(bins.size - 1)]
xerr = (bins[1] - bins[0]) / 2 * np.ones(bins.shape[0] - 1)

plot_pur_eff_w_dict(pred_files, target_file, save_path="/storage/af/user/tsievert/topNet/SPAtop/reports", proj_name='SPAtop', bins=bins)
