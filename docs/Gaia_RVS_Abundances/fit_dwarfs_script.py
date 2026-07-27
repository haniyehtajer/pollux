import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.distributions.constraints import real
import pandas as pd

import pollux as plx
from pollux.models.transforms import LinearTransform

from process_rvs import process_spectra

jax.config.update("jax_enable_x64", True)

from matplotlib import rcParams

params = {
   'axes.labelsize': 15,
   'font.size': 15,
   'legend.fontsize': 14,
   'xtick.minor.visible': True,
   'ytick.minor.visible': True,
   'xtick.labelsize': 15,
   'ytick.labelsize': 15,
   'text.usetex': True, #to use TeX in your labels
   'font.family':'serif',
   'axes.titlesize': 15,
   'xtick.direction': 'in',
   'ytick.direction': 'in',
   'xtick.top': True,
   'ytick.right': True,
   'xtick.major.size': 6,
   'xtick.minor.size': 3,
   'ytick.major.size': 6,
   'ytick.minor.size': 3
   }
rcParams.update(params)


df_stars = pd.read_pickle('data/training_set_dwarfs.pkl')
df_gaia_rvs = pd.read_pickle('data/test_set_KDTree_dwarfs_rvs_fluxes.pkl')

n_all_stars = len(df_stars)
#n_stars = 500  # number of simulated stars to generate in the train and test sets
n_stars = n_all_stars
labels = ['teff', 'logg', 'fe_h', 'mg_h', 'n_h', 'si_h', 'ni_h', 'ca_h', 'm_h_atm', 'alpha_m_atm']
n_labels = len(labels)  # number of labels to generate per star
#n_latents = n_labels * 4  # size of the latent vector per star
n_latents = 21
n_flux = 2401  # number of spectral flux pixels per star

rng = np.random.default_rng(seed=8675309)

A = np.zeros((n_labels, n_latents))
A[0, 0] = 1.0
A[1, 1] = 1.0

B = rng.normal(scale=0.1, size=(n_flux, n_latents))
B[:, 0] = B[:, 0] + 4 * np.exp(-0.5 * (np.arange(n_flux) - n_flux / 2) ** 2 / 5**2)
B[:, 1] = B[:, 1] + 2 * np.exp(-0.5 * (np.arange(n_flux) - n_flux / 4) ** 2 / 3**2)

df_stars = df_stars[:n_stars]
#df_gaia_rvs = df_gaia_rvs[:500]
print("len test set stars = ", len(df_stars))
print("len all stars = ", len(df_gaia_rvs))

# List of error column names corresponding to labels
params_errors = [f"e_{p}" if not p.startswith('e_') else p for p in labels]

# Stack columns for labels
df_stars_labels = np.column_stack([df_stars[p] for p in labels])

# Stack columns for errors
df_stars_label_errs = np.column_stack([df_stars[e] for e in params_errors])

df_stars = df_stars.reset_index(drop=True)


# Apply Ca mask

fluxes_with_ca_mask = []
flux_errs_with_ca_mask = []

lambdas = np.linspace(846, 870, 2401)

for i in range(len(df_stars['flux'])):
	flux_single,flux_err_single = process_spectra(lambdas,df_stars['flux'].iloc[i],df_stars['flux_error'].iloc[i])

	fluxes_with_ca_mask.append(flux_single)
	flux_errs_with_ca_mask.append(flux_err_single)

stars_dict = {
    'label': df_stars_labels,
    'label_err': df_stars_label_errs,
    'flux': fluxes_with_ca_mask,
    'flux_err': flux_errs_with_ca_mask
}


stars_dict['flux'] = np.vstack([f.filled(1) if isinstance(f, np.ma.MaskedArray) else f
                              for f in stars_dict['flux']])

# Convert flux_err list to 2D array
stars_dict['flux_err'] = np.vstack([f.filled(9999) if isinstance(f, np.ma.MaskedArray) else f
                                  for f in stars_dict['flux_err']])


all_data = plx.data.PolluxData(
    flux=plx.data.OutputData(
        stars_dict["flux"],
        err=stars_dict["flux_err"],
        preprocessor=plx.data.ShiftScalePreprocessor.from_data(stars_dict["flux"]),
    ),
    label=plx.data.OutputData(
        stars_dict["label"],
        err=stars_dict["label_err"],
        preprocessor=plx.data.ShiftScalePreprocessor.from_data(stars_dict["label"]),
    )
)

preprocessed_data = all_data.preprocess()

train_data = preprocessed_data[: n_stars // 2]
test_data = preprocessed_data[n_stars // 2 :]

model = plx.LuxModel(latent_size=n_latents)

model.register_output("label", LinearTransform(output_size=n_labels))
model.register_output("flux", LinearTransform(output_size=n_flux))

opt_pars, svi_results = model.optimize(
    train_data,
    rng_key=jax.random.PRNGKey(112358),
    optimizer=numpyro.optim.Adam(1e-3),
    num_steps=10_000,
    svi_run_kwargs={"progress_bar": False},
)

svi_results.losses.block_until_ready()

plt.plot(svi_results.losses[-1000000:])
plt.savefig("loss_plot.pdf", dpi=300, bbox_inches="tight")
plt.close()

predict_train_values = model.predict_outputs(opt_pars["latents"], opt_pars)


fixed_pars = {
    "label": {"data": {"A": opt_pars["label"]["data"]["A"]}},
    "flux": {"data": {"A": opt_pars["flux"]["data"]["A"]}},
}


pt_style = {"ls": "none", "ms": 2.0, "alpha": 0.5, "marker": "o", "color": "k"}

fig, axes = plt.subplots(3, 4, figsize=(12, 10), layout="constrained")
axes = axes.flatten()

for i in range(predict_train_values["label"].shape[1]):
    axes[i].plot(
        predict_train_values["label"][:, i], train_data["label"].data[:, i], **pt_style
    )
    axes[i].set(xlabel=f"Predicted label {i} ({labels[i]})", ylabel=f"True label {i}")
    axes[i].axline([0, 0], slope=1, color="tab:green", zorder=-100)

for j in range(len(labels), len(axes)):
    axes[j].axis("off")

_ = fig.suptitle("Training set: predicted vs. true labels", fontsize=22)
plt.savefig("train_labels.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# processing for all gaia rvs
all_fluxes_with_ca_mask = []
all_flux_errs_with_ca_mask = []


for i in range(len(df_gaia_rvs['flux'])):
	all_flux_single,all_flux_err_single = process_spectra(lambdas,df_gaia_rvs['flux'].iloc[i],df_gaia_rvs['flux_error'].iloc[i])

	all_fluxes_with_ca_mask.append(all_flux_single)
	all_flux_errs_with_ca_mask.append(all_flux_err_single)
	

all_flux_dict = {
    'flux': all_fluxes_with_ca_mask,
    'flux_err': all_flux_errs_with_ca_mask
}


all_flux_dict['flux'] = np.vstack([f.filled(1) if isinstance(f, np.ma.MaskedArray) else f
                              for f in all_flux_dict['flux']])

# Convert flux_err list to 2D array
all_flux_dict['flux_err'] = np.vstack([f.filled(9999) if isinstance(f, np.ma.MaskedArray) else f
                                  for f in all_flux_dict['flux_err']])

all_flux_data = plx.data.PolluxData(
    flux=plx.data.OutputData(
        all_flux_dict["flux"],
        err=all_flux_dict["flux_err"],
        # Reuse the exact preprocessor from your training set
        preprocessor=all_data["flux"].preprocessor, 
    )
)

preprocessed_all_flux_data = all_flux_data.preprocess()

test_flux_only = plx.data.PolluxData(flux=test_data["flux"])

test_opt_pars_flux, _ = model.optimize(
    test_flux_only,
    rng_key=jax.random.PRNGKey(12345),
    optimizer=numpyro.optim.Adam(1e-3),
    num_steps=10_000,
    fixed_pars=fixed_pars,
    names=["flux"],
    svi_run_kwargs={"progress_bar": False},
)

predict_test_values_flux = model.predict_outputs(
    test_opt_pars_flux["latents"], fixed_pars
)

fig, axes = plt.subplots(3, 4, figsize=(14, 10), layout="constrained")
axes = axes.flatten()

for i in range(predict_test_values_flux["label"].shape[1]):
    axes[i].plot(
        predict_test_values_flux["label"][:, i], test_data["label"].data[:, i], **pt_style
    )
    axes[i].set_xlabel(f"Predicted {labels[i]}", fontsize=20)
    axes[i].set_ylabel(f"True {labels[i]}", fontsize=20)
    axes[i].axline([0, 0], slope=1, color="tab:green", zorder=-100)
    axes[i].set_xlim(-3, 3)
    axes[i].set_ylim(-3, 3)

for j in range(len(labels), len(axes)):
    axes[j].axis("off")

_ = fig.suptitle(f"Flux only: predicted vs. true labels, dwarfs, no mask, n_latents = {n_latents}", fontsize=22)
plt.savefig("test_labels_standardized.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)


predict_test_unprocessed = test_data.unprocess(predict_test_values_flux)
test_data_unprocessed = all_data[n_stars // 2:]

fig, axes = plt.subplots(3, 4, figsize=(14, 10), layout="constrained")
axes = axes.flatten()

for i in range(predict_test_values_flux["label"].shape[1]):
    axes[i].plot(
        predict_test_unprocessed["label"].data[:, i], test_data_unprocessed["label"].data[:, i], **pt_style
    )
    axes[i].set_xlabel(f"Predicted {labels[i]}", fontsize=20)
    axes[i].set_ylabel(f"True {labels[i]}", fontsize=20)
    axes[i].axline([0, 0], slope=1, color="tab:green", zorder=-100)

for j in range(len(labels), len(axes)):
    axes[j].axis("off")
    
_ = fig.suptitle(f"Flux only: predicted vs. true labels, dwarfs, no mask, n_latents = {n_latents}", fontsize=22)
plt.savefig("test_labels_unprocessed.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)


split_index = n_stars // 2
test_source_ids = df_stars['source_id'].iloc[split_index:].values

results = {}
df_dict = {}

# 2. Add the source_ids to your dictionary before the loop
df_dict["source_id"] = test_source_ids

for i, name in enumerate(labels):
    y_true = test_data_unprocessed["label"].data[:, i]
    y_pred = predict_test_unprocessed["label"].data[:, i]

    res = y_pred - y_true

    results[name] = {
        "std_res": np.std(res),
        "RMSE": np.sqrt(np.mean(res**2)),
        "MAE": np.mean(np.abs(res)),
        "Bias": np.mean(res),
        "R2": 1 - np.sum(res**2) / np.sum((y_true - np.mean(y_true))**2),
    }

    df_dict[f"{name}_true"] = y_true
    df_dict[f"{name}_pred"] = y_pred
    df_dict[f"{name}_residual"] = res

per_star_df = pd.DataFrame(df_dict)


# save test results

per_star_df.to_csv("data/test_results_dwarfs.csv", index=False)

plt.figure(figsize=(18, 10))

# Changed to a 2x5 grid so all 10 labels fit perfectly with no empty subplots
fig, axes = plt.subplots(3, 4, figsize=(18, 10), layout="constrained")
axes = axes.flatten()

for i in range(len(labels)):
    name = labels[i]
    axes[i].scatter(per_star_df[f"{name}_true"], per_star_df[f"{name}_residual"], s=3)
    
    axes[i].axhline(0, color="k")
    axes[i].set_xlabel(f"True {name}")
    axes[i].set_ylabel("Residual")
    axes[i].set_title(f"RMSE = {results[name]['RMSE']:.2f}, R2 = {results[name]['R2']:.2f}")

for j in range(len(labels), len(axes)):
    axes[j].axis("off")

plt.suptitle(f"dwarfs, n_latents = {n_latents}")
plt.savefig("test_residuals.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# FIX: Use preprocessed_all_flux_data, not all_flux_data
flux_only_data = plx.data.PolluxData(flux=preprocessed_all_flux_data["flux"])

opt_pars_all_data, _ = model.optimize(
    flux_only_data,
    rng_key=jax.random.PRNGKey(12345),
    optimizer=numpyro.optim.Adam(1e-3),
    num_steps=10_000,
    fixed_pars=fixed_pars,
    names=["flux"],
    svi_run_kwargs={"progress_bar": False},
)

predict_all_values_flux = model.predict_outputs(
    opt_pars_all_data["latents"], opt_pars
)


predict_all_unprocessed_flux = all_data.unprocess(predict_all_values_flux)

pred_data = predict_all_unprocessed_flux["label"].data[:]

data_to_save = pred_data[:, :len(labels)]

df_output = pd.DataFrame(data_to_save, columns=labels)

df_output['source_id'] = df_gaia_rvs['source_id']

df_output.to_csv("data/dwarfs_predicted_no_mask.csv", index=False)

df_output['fe_mg'] = df_output['fe_h'] - df_output['mg_h']
 
plt.figure()
plt.scatter(df_output['mg_h'], df_output['fe_mg'], s=5)

# If you don't have labeled artists, plt.legend() will throw a warning, 
# so you may want to remove it or add a label to the scatter plot.
plt.legend() 

plt.ylim(-0.8, 0.5)
plt.xlim(-0.8, 0.6)
plt.xlabel('[Mg/H]')
plt.ylabel('[Fe/Mg]')
plt.savefig("fe_mg_scatter.pdf", dpi=300, bbox_inches="tight")
plt.close()