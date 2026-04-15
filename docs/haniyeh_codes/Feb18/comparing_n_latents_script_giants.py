import model_template as model
import pandas as pd
import numpy as np
import pollux as plx
from pollux.models.transforms import LinearTransform
import matplotlib.pyplot as plt
from matplotlib import rcParams
import time

datafile_path = '/Users/honeyeah/Codes/LuxProject/data/giants_rel_most_rel.pkl'
df_stars = pd.read_pickle(datafile_path)
n_stars = 500
n_latents = 20
labels = ['teff', 'logg', 'fe_h', 'mg_h', 'n_h', 'si_h', 'ni_h', 'ca_h', 'ce_h']


rows = []

for n_lat in range(12, 36):
    t0 = time.time()

    predict_train_values, train_data, predict_test_values, test_data, \
    predict_test_values_flux, all_data = model.sdss_gaia_model(
        path_to_pickle=datafile_path,
        label_names=labels,
        n_stars=n_stars,
        n_latents=n_lat,
        which_output='all'
    )

    delta_t = time.time() - t0

    predict_test_unprocessed = test_data.unprocess(predict_test_values_flux)
    test_data_unprocessed = all_data[n_stars // 2:]

    y_true_all = test_data_unprocessed["label"].data
    y_pred_all = predict_test_unprocessed["label"].data

    # global RMSE
    total_rmse = np.sqrt(np.mean((y_pred_all - y_true_all) ** 2))

    # --- build row ---
    row = {
        "n_latent": n_lat,
        "delta_t": delta_t,
        "rmse": total_rmse,
    }

    # per-label RMSE
    for j, name in enumerate(labels):
        res = y_pred_all[:, j] - y_true_all[:, j]
        row[f"rmse_{name}"] = np.sqrt(np.mean(res**2))

    rows.append(row)

# --- write CSV ---
df = pd.DataFrame(rows)
df.to_csv("latent_scan_results_2.csv", index=False)
