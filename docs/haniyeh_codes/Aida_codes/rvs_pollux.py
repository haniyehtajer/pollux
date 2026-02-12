import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py as h5
import numpyro
import numpyro.distributions as dist
from numpyro.distributions.constraints import real

import pollux as plx
from pollux.models.transforms import LinearTransform

from modules.process_rvs import *

jax.config.update("jax_enable_x64", True)



df = pd.read_csv('files/sdss_snr200_rvs_snr100.csv',index_col=False)
#df = df[(df['logg']>1.5) & (df['logg']<3.5)] # giants 
df = df[df['logg']>4] # dwarfs
df = df.drop_duplicates(subset='source_id')

# let's do first 100 stars
df = df.head(500)
source_ids = df['source_id'].astype('int64').values


fp = h5.File("files/gaia-dr3-rvs-all.hdf5", "r")
wl = np.linspace(846, 870, 2401) # nm
bool = np.isin(fp["source_id"][:], source_ids)
indices = [i for i, x in enumerate(bool) if x]

source_ids_ordered=[]
flux=[]
flux_err=[]
for index in indices:

	source_id_single = fp["source_id"][index]
	flux_single,flux_err_single = process_spectra(wl,fp["flux"][index],fp["flux_error"][index])

	flux.append(flux_single)
	flux_err.append(flux_err_single)
	source_ids_ordered.append(source_id_single)

flux = np.array(flux)
flux_err = np.array(flux_err)
ivar = 1/flux_err**2

# re-order label dataframe
df['source_id'] = pd.Categorical(df['source_id'], categories=source_ids_ordered, ordered=True)
df = df.sort_values('source_id')

# constructs training set label matrices
Teff = df['teff'].values
logg = df['logg'].values
fe_h = df['fe_h'].values
mg_h = df['mg_h'].values
label = np.vstack((Teff,logg,fe_h,mg_h)).T

e_Teff = df['e_teff'].values
e_logg = df['e_logg'].values
e_fe_h = df['e_fe_h'].values
e_mg_h = df['e_mg_h'].values
e_label = np.vstack((e_Teff,e_logg,e_fe_h,e_mg_h)).T





all_data = plx.data.PolluxData(
    flux=plx.data.OutputData(
        flux,
        err=flux_err,
        preprocessor=plx.data.ShiftScalePreprocessor.from_data(flux),
    ),
    label=plx.data.OutputData(
        label,
        err=e_label, 
        preprocessor=plx.data.ShiftScalePreprocessor.from_data(label),
    ),
)

preprocessed_data = all_data.preprocess()




n_stars = len(df)
n_flux = flux.shape[1]  # len(df)
n_labels = 4
print(f"{n_stars=}, {n_flux=}, {n_labels=}")

train_data = preprocessed_data[: n_stars // 2]
test_data = preprocessed_data[n_stars // 2 :]
test_data_unprocessed = all_data[n_stars // 2:]
print(len(train_data), len(test_data))




model = plx.LuxModel(latent_size=8)

print(all_data.keys())  # noqa: T201
model.register_output("label", LinearTransform(output_size=n_labels))
model.register_output("flux", LinearTransform(output_size=n_flux))

# no regularization
trans = LinearTransform(
    output_size=n_labels, #param_priors={"A": dist.ImproperUniform(real, (), ())}
)

rngs = jax.random.split(jax.random.PRNGKey(42), 3)




# For this demo, we'll generate outputs for 10 objects
latents = jax.random.normal(rngs[0], shape=(10, model.latent_size))
pars = {
    "label": {"A": jax.random.normal(rngs[1], shape=(n_labels, model.latent_size))},
    "flux": {"A": jax.random.normal(rngs[2], shape=(n_flux, model.latent_size))},
}
outputs = model.predict_outputs(latents, pars)
print(outputs["label"].shape, outputs["flux"].shape)
print(train_data["label"].data.shape, train_data["flux"].data.shape)




opt_pars, svi_results = model.optimize(
    train_data,
    rng_key=jax.random.PRNGKey(112358),
    optimizer=numpyro.optim.Adam(1e-2),
    num_steps=10_000,
    svi_run_kwargs={"progress_bar": True},
)
print(opt_pars)
print(svi_results.losses.block_until_ready())

plt.figure()
plt.plot(svi_results.losses[-1000:])
plt.show()



predict_train_values = model.predict_outputs(opt_pars["latents"], opt_pars)
pt_style = {"ls": "none", "ms": 2.0, "alpha": 0.5, "marker": "o", "color": "k"}

fig, axes = plt.subplots(1, n_labels, figsize=(4 * n_labels, 4), layout="constrained")
for i in range(n_labels):
    axes[i].plot(
        predict_train_values["label"][:, i], train_data["label"].data[:, i], **pt_style
    )
    axes[i].set(xlabel=f"Predicted label {i}", ylabel=f"True label {i}")
    axes[i].axline([0, 0], slope=1, color="tab:green", zorder=-100)
_ = fig.suptitle("Training set: predicted vs. true labels", fontsize=22)
plt.show()



# Compare for test set:
fixed_pars = {
    "label": {"data": {"A": opt_pars["label"]["data"]["A"]}},
    "flux": {"data": {"A": opt_pars["flux"]["data"]["A"]}},
}
flux_only_data = plx.data.PolluxData(flux=test_data["flux"])
test_opt_pars, test_svi_results = model.optimize(
    flux_only_data,
    rng_key=jax.random.PRNGKey(12345),
    optimizer=numpyro.optim.Adam(1e-2),
    num_steps=10_000,
    names=["flux"],
    fixed_pars=fixed_pars,
)
test_svi_results.losses.block_until_ready()

predict_test_values = model.predict_outputs(test_opt_pars["latents"], fixed_pars)
predict_test_unprocessed = test_data.unprocess(predict_test_values)

fig, axes = plt.subplots(1, n_labels, figsize=(4 * n_labels, 4), layout="constrained")
for i in range(n_labels):
    axes[i].plot(
        predict_test_unprocessed["label"][:, i], test_data_unprocessed["label"].data[:, i], **pt_style
    )
    axes[i].set(xlabel=f"Predicted label {i}", ylabel=f"True label {i}")
    axes[i].axline([0, 0], slope=1, color="tab:green", zorder=-100)
_ = fig.suptitle("Test set: predicted vs. true labels", fontsize=22)
plt.show()