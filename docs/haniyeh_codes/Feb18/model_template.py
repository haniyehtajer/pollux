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

def sdss_gaia_model(path_to_pickle, label_names, n_stars, n_latents,
                    rand_seed = 8675309, which_output='all'):
    '''
    Docstring for sdss_gaia_model
    
    :param path_to_pickle: pickle data file path
    :param label_names: name of labels, e.g. ['teff', 'logg', 'fe_h']
    :param n_stars: number of stars to use
    :param n_latents: number of latent parameters
    :param rand_seed: random seed number
    :param which_output: which output do you want?
                        options are: "all", "train", "test"
    '''

    df_stars = pd.read_pickle(path_to_pickle)
    n_all_stars = len(df_stars)
    if n_stars is None:
        n_stars = n_all_stars

    labels = label_names
    n_labels = len(labels)  # number of labels per star
    n_flux = 2401

    rng = np.random.default_rng(seed=rand_seed)

    A = np.zeros((n_labels, n_latents))
    A[0, 0] = 1.0
    A[1, 1] = 1.0

    B = rng.normal(scale=0.1, size=(n_flux, n_latents))
    B[:, 0] = B[:, 0] + 4 * np.exp(-0.5 * (np.arange(n_flux) - n_flux / 2) ** 2 / 5**2)
    B[:, 1] = B[:, 1] + 2 * np.exp(-0.5 * (np.arange(n_flux) - n_flux / 4) ** 2 / 3**2)

    # List of error column names corresponding to labels
    params_errors = [f"e_{p}" if not p.startswith('e_') else p for p in labels]

    # Stack columns for labels
    df_stars_labels = np.column_stack([df_stars[p] for p in labels])

    # Stack columns for errors
    df_stars_label_errs = np.column_stack([df_stars[e] for e in params_errors])

    fluxes_with_ca_mask = []
    flux_errs_with_ca_mask = []

    # Wl range from GAIA
    lambdas = np.linspace(846, 870, 2401)

    # Mask Ca triplets
    for i in range(len(df_stars['flux'])):
        flux_single,flux_err_single = process_spectra(lambdas,df_stars['flux'].iloc[i],df_stars['flux_error'].iloc[i])

        fluxes_with_ca_mask.append(flux_single)
        flux_errs_with_ca_mask.append(flux_err_single)

    # Define stars dictionary
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
            ),
    )

    preprocessed_data = all_data.preprocess()

    train_data = preprocessed_data[: n_stars // 2]
    test_data = preprocessed_data[n_stars // 2 :]

    model = plx.LuxModel(latent_size=n_latents)

    model.register_output("label", LinearTransform(output_size=n_labels))
    model.register_output("flux", LinearTransform(output_size=n_flux))


    iterative_result = model.optimize_iterative(
        train_data,
        max_cycles=50,
        tol=1e-6,
        rng_key=jax.random.PRNGKey(112358),
        progress=False,
    )

    opt_pars = iterative_result.params

    fixed_pars = {k: v for k, v in opt_pars.items() if k != "latents"}

    predict_train_values = model.predict_outputs(opt_pars["latents"], opt_pars)

    fixed_pars = {
        "label": {"data": {"A": opt_pars["label"]["data"]["A"]}},
        "flux": {"data": {"A": opt_pars["flux"]["data"]["A"]}},
    }

    test_opt_pars, test_svi_results = model.optimize(
        test_data,
        rng_key=jax.random.PRNGKey(12345),
        optimizer=numpyro.optim.Adam(1e-3),
        num_steps=10_000,
        fixed_pars=fixed_pars,
        svi_run_kwargs={"progress_bar": False},
    )
    test_svi_results.losses.block_until_ready()

    predict_test_values = model.predict_outputs(test_opt_pars["latents"], fixed_pars)

    flux_only_data = plx.data.PolluxData(flux=test_data["flux"])

    test_opt_pars_flux, _ = model.optimize(
        flux_only_data,
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
    if which_output == "all":    
        return predict_train_values, train_data, predict_test_values, test_data, predict_test_values_flux, all_data
    elif which_output == "train":
        return predict_train_values, train_data
    elif which_output == "test":
        return predict_test_values, test_data
    elif which_output == "test-flux-only":
        return predict_test_values_flux, test_data











