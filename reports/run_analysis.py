import json
import os
import sys

import numpy as np
import vector
vector.register_awkward()

from src.analysis.plot import calc_pur_eff, plot_pur_eff_w_dict, plot_pur_w_dict

if len(sys.argv) < 2: raise Exception('You need to provide path to config.')
with open(sys.argv[1], 'r') as f:
    config = json.load(f)

if not os.path.exists(config['save_path']):
    os.makedirs(config['save_path'])

bins_dict = {
    'FR': np.arange(0, 300, 10),
    'SRqq': np.arange(100, 400, 10),
    'SRbq': np.arange(100, 400, 10),
    'FB': np.arange(200, 1000, 50),
    'all': np.arange(0, 1000, 50),
}

# plot_pur_eff_w_dict(config['pred_files'], config['target_file'], config['save_path'], proj_name='SPAtop', bins_dict=bins_dict)
plot_pur_w_dict(config['pred_files'], config['target_file'], config['save_path'], proj_name='SPAtop', bins_dict=bins_dict)
