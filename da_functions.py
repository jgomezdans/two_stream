import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from IPython.display import display

# Lots of different places that widgets could come from...


from eoldas_ng import Prior

from tip_helpers import retrieve_albedo
from eoldas_machinery import tip_inversion, regularised_tip_inversion

logo = plt.imread("nceologo200.png")

sites = """AU-Tum (Tumbarumba)
BR-Cax (Caxiuana Forest-Almeirim)
CA-Ca1 (BC-Campbell River 1949 Douglas-fir)
DE-Geb (Gebesee)
DE-Hai (Hainich)
ES-LMa (Las Majadas del Tietar)
FI-Hyy (Hyytiala)
FR-Lam (Lamasquere)
IT-SRo (San Rossore)
RU-Ylr (Yakutsk-Larch)
SE-Abi (Abisko)
US-Bar (Bartlett Experimental Forest)
US-Bo1 (Bondville)
US-Brw (Barrow)
US-Dk2 (Duke Forest Hardwoods)
US-Ha2 (Harvard Forest Hemlock Site)
US-MMS (Morgan Monroe State Forest)
US-Me2 (Metolius Intermediate Pine)
US-Me3 (Metolius Second Young Pine)
US-Ne1 (Mead - irrigated continuous maize site)
US-Ne2 (Mead - irrigated maize-soybean rotation site)
US-Ne3 (Mead - rainfed maize-soybean rotation site)
US-Ton (Tonzi Ranch)
ZA-Kru (Skukuza)""".split(
    "\n"
)


sites = """AU-Tum
BR-Cax
CA-Ca1
DE-Geb
DE-Hai
ES-LMa
FI-Hyy
FR-Lam
IT-SRo
RU-Ylr
US-Brw
US-Bar
US-Bo1
US-Dk2
US-Ha2
US-Ne1
US-Ne2
US-Ne3
US-Me2
US-Me3
US-MMS
US-Ton
SE-Abi
ZA-Kru""".split(
    "\n"
)

tip_params = [
    r"$\omega_{vis}\; [-]$",
    r"$d_{vis}\; [-]$",
    r"$\alpha_{vis}\; [-]$",
    r"$\omega_{nir}\; [-]$",
    r"$d_{nir}\; [-]$",
    r"$\alpha_{nir}\; [-]$",
    r"$LAI_{eff}\;[m^{2}m^{-2}]$",
]


def only_one_set(*args):
    return sum(args) == 1


def plot_albedo(albedo, x):

    plt.plot(x, albedo, "o-", mfc="none")
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")


def visualise_twostream(
    omega_vis=False,
    d_vis=False,
    a_vis=False,
    omega_nir=False,
    d_nir=False,
    a_nir=False,
    lai_vis=False,
    lai_nir=False,
    vis_emu_pkl="tip_vis_emulator_real.pkl",
    nir_emu_pkl="tip_nir_emulator_real.pkl",
):

    n = 20
    param_vis = np.array([0.17, 1, 0.1, 2])
    param_nir = np.array([0.7, 2, 0.18, 2])

    gp_vis = pickle.load(open(vis_emu_pkl, "r"))
    gp_nir = pickle.load(open(nir_emu_pkl, "r"))
    if not only_one_set(
        omega_vis, d_vis, a_vis, omega_nir, d_nir, a_nir, lai_vis, lai_nir
    ):

        raise ValueError(
            "You have either set up more than one variable or none"
        )
    x = np.ones((n, 4)) * param_vis
    if omega_vis:
        x[:, 0] = np.linspace(0, 0.9, n)
        albedo_vis = gp_vis.predict(x, do_unc=False)[0]
        plot_albedo(albedo_vis, np.linspace(0, 0.9, n))
        return
    elif d_vis:
        x[:, 1] = np.linspace(0.2, 4, n)
        albedo_vis = gp_vis.predict(x, do_unc=False)[0]
        plot_albedo(albedo_vis, np.linspace(0.2, 4, n))
        return
    elif a_vis:
        x[:, 2] = np.linspace(0, 0.9, n)
        albedo_vis = gp_vis.predict(x, do_unc=False)[0]
        plot_albedo(albedo_vis, np.linspace(0, 0.9, n))
        return
    elif lai_vis:
        x[:, 3] = np.linspace(0, 6, n)
        albedo_vis = gp_vis.predict(x, do_unc=False)[0]
        plot_albedo(albedo_vis, np.linspace(0, 6, n))
        return

    x = np.ones((n, 4)) * param_nir
    if omega_nir:
        x[:, 0] = np.linspace(0, 0.9, n)
        albedo_nir = gp_nir.predict(x, do_unc=False)[0]
        plot_albedo(albedo_nir, np.linspace(0, 0.9, n))
        return
    elif d_nir:
        x[:, 1] = np.linspace(0.2, 4, n)
        albedo_nir = gp_nir.predict(x, do_unc=False)[0]
        plot_albedo(albedo_nir, np.linspace(0.2, 4, n))
        return
    elif a_nir:
        x[:, 2] = np.linspace(0, 0.9, n)
        albedo_nir = gp_nir.predict(x, do_unc=False)[0]
        plot_albedo(albedo_nir, np.linspace(0, 0.9, n))
        return
    elif lai_nir:
        x[:, 3] = np.linspace(0, 6, n)
        albedo_nir = gp_nir.predict(x, do_unc=False)[0]
        plot_albedo(albedo_nir, np.linspace(0, 6, n))
        return


def visualise_albedos(fluxnet_site, year):
    """A function that will visualise albedo data for a particular year and site"""
    observations, mask, bu, passer_snow = retrieve_albedo(
        year, fluxnet_site, [0.05, 0.07]
    )
    passer = mask[:, 1] == 1
    doys = mask[:, 0]
    plt.figure(figsize=(12, 6))
    plt.plot(
        doys[passer], observations[passer, 0], "o", label="Visible Albedo"
    )
    plt.plot(
        doys[passer],
        observations[passer, 1],
        "o",
        label="Near-infrarred Albedo",
    )
    plt.vlines(
        doys[passer],
        observations[passer, 0] + 1.96 * bu[passer, 0],
        observations[passer, 0] - 1.96 * bu[passer, 0],
    )
    plt.vlines(
        doys[passer],
        observations[passer, 1] + 1.96 * bu[passer, 1],
        observations[passer, 1] - 1.96 * bu[passer, 1],
    )
    plt.legend(loc="best", numpoints=1, fancybox=True, shadow=True)
    plt.ylabel("Bi-hemispherical reflectance [-]")
    plt.xlabel("DoY/%d" % year)
    plt.xlim(1, 368)
    plt.ylim(0, 0.7)
    ax1 = plt.gca()
    ax1.figure.figimage(logo, 60, 60, alpha=0.1, zorder=1)

    # Hide the right and top spines
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax1.yaxis.set_ticks_position("left")
    ax1.xaxis.set_ticks_position("bottom")


def single_observation_inversion(
    fluxnet_site, year, green_leaves=False, n_tries=5
):
    """A function to do invert each individual observation in a time series. """

    retval_s, state, obs = tip_inversion(
        year, fluxnet_site, green_leaves=green_leaves, n_tries=n_tries
    )
    mu = state.operators["Prior"].mu
    cinv = state.operators["Prior"].inv_cov
    c = np.array(np.sqrt(np.linalg.inv(cinv.todense()).diagonal())).squeeze()

    fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(14, 12))
    axs = axs.flatten()
    fig.suptitle("%s (%d)" % (fluxnet_site, year), fontsize=18)
    params = [
        "omega_vis",
        "d_vis",
        "a_vis",
        "omega_nir",
        "d_nir",
        "a_nir",
        "lai",
    ]
    post_sd = np.sqrt(np.array(retval_s["post_cov"].todense()).squeeze())
    post_sd = np.where(post_sd > c, c, post_sd)

    for i, p in enumerate(tip_params):
        axs[i].axhspan(
            mu[(i * 46) : ((i + 1) * 46)][0]
            + c[(i * 46) : ((i + 1) * 46)][0],
            mu[(i * 46) : ((i + 1) * 46)][0]
            - c[(i * 46) : ((i + 1) * 46)][0],
            color="0.9",
        )

        axs[i].plot(state.state_grid, mu[(i * 46) : ((i + 1) * 46)], "--")
        axs[i].plot(
            state.state_grid,
            retval_s["real_map"][params[i]],
            "o-",
            mfc="none",
        )

        axs[i].vlines(
            state.state_grid,
            retval_s["real_map"][params[i]]
            - post_sd[(i * 46) : ((i + 1) * 46)],
            retval_s["real_map"][params[i]]
            + post_sd[(i * 46) : ((i + 1) * 46)],
            lw=0.8,
            colors="0.1",
            alpha=0.5,
        )

        if i in [1, 4, 6]:
            axs[i].set_ylim(0, 6)
        else:
            axs[i].set_ylim(0, 1)
        axs[i].set_ylabel(tip_params[i])

    fwd = np.array(obs.fwd_modelled_obs)
    axs[7].plot(obs.observations[:, 0], fwd[:, 0], "k+", label="VIS")
    axs[7].plot(obs.observations[:, 1], fwd[:, 1], "rx", label="NIR")
    axs[7].set_xlabel("Measured BHR [-]")
    axs[7].set_ylabel("Predicted BHR [-]")
    axs[7].plot([0, 0.9], [0, 0.9], "k--", lw=0.5)
    axs[7].legend(loc="best")

    axs[8].vlines(
        obs.mask[:, 0],
        obs.observations[:, 0] - 1.96 * obs.bu[:, 0],
        obs.observations[:, 0] + 1.96 * obs.bu[:, 0],
    )
    axs[8].plot(obs.mask[:, 0], obs.observations[:, 0], "o")

    axs[9].vlines(
        obs.mask[:, 0],
        obs.observations[:, 1] - 1.96 * obs.bu[:, 1],
        obs.observations[:, 1] + 1.96 * obs.bu[:, 1],
    )
    axs[9].plot(obs.mask[:, 0], obs.observations[:, 1], "o")

    axs[8].set_ylabel("BHR VIS [-]")
    axs[9].set_ylabel("BHR NIR [-]")
    axs[8].set_xlabel("DoY [d]")
    axs[9].set_xlabel("DoY [d]")

    for i in range(10):

        if i != 7:
            axs[i].set_xlim(1, 370)
        # Hide the right and top spines
        axs[i].spines["right"].set_visible(False)
        axs[i].spines["top"].set_visible(False)
        # Only show ticks on the left and bottom spines
        axs[i].yaxis.set_ticks_position("left")
        axs[i].xaxis.set_ticks_position("bottom")

    fig.figimage(
        logo, fig.bbox.xmax - 500, fig.bbox.ymax - 250, alpha=0.4, zorder=1
    )
    plt.savefig(
        "single_obs_inversion_%s_%04d.pdf" % (fluxnet_site, year),
        dpi=300,
        boox_inches="tight",
    )


def regularised_inversion(
    fluxnet_site,
    year,
    green_leaves,
    gamma_lai,
    n_tries=5,
    albedo_unc=[0.05, 0.07],
):

    retval_s, state, obs = tip_inversion(
        year, fluxnet_site, green_leaves=green_leaves, n_tries=n_tries
    )
    mu = state.operators["Prior"].mu
    cinv = state.operators["Prior"].inv_cov
    c = np.array(np.sqrt(np.linalg.inv(cinv.todense()).diagonal())).squeeze()
    post_sd = np.sqrt(np.array(retval_s["post_cov"].todense()).squeeze())
    post_sd_single = np.where(post_sd > c, c, post_sd)

    retval, state, obs = regularised_tip_inversion(
        year,
        fluxnet_site,
        [1e-3, 0, 0.1, 1e-3, 0, 0.1, gamma_lai],
        x0=retval_s["real_map"],
        green_leaves=green_leaves,
        n_tries=n_tries,
        albedo_unc=albedo_unc,
    )
    mu = state.operators["Prior"].mu
    cinv = state.operators["Prior"].inv_cov
    c = np.array(np.sqrt(np.linalg.inv(cinv.todense()).diagonal())).squeeze()
    post_sd = np.sqrt(np.array(retval_s["post_cov"].todense()).squeeze())
    post_sd = np.where(post_sd > c, c, post_sd)

    fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(14, 12))
    axs = axs.flatten()
    fig.suptitle("%s (%d)" % (fluxnet_site, year), fontsize=18)
    params = [
        "omega_vis",
        "d_vis",
        "a_vis",
        "omega_nir",
        "d_nir",
        "a_nir",
        "lai",
    ]
    post_sd = np.sqrt(np.array(retval["post_cov"].todense()).squeeze())
    post_sd = np.where(post_sd > c, c, post_sd)

    for i, p in enumerate(tip_params):

        # axs[i].axhspan(mu[(i*46):((i+1)*46)][0]+c[(i*46):((i+1)*46)][0],
        #               mu[(i*46):((i+1)*46)][0] - c[(i * 46):((i + 1) * 46)][0], color="0.9" )

        axs[i].fill_between(
            state.state_grid,
            retval["real_map"][params[i]]
            - post_sd[(i * 46) : ((i + 1) * 46)],
            retval["real_map"][params[i]]
            + post_sd[(i * 46) : ((i + 1) * 46)],
            lw=0.8,
            color="0.8",
        )
        axs[i].vlines(
            state.state_grid,
            retval_s["real_map"][params[i]]
            - post_sd_single[(i * 46) : ((i + 1) * 46)],
            retval_s["real_map"][params[i]]
            + post_sd_single[(i * 46) : ((i + 1) * 46)],
            lw=0.8,
            colors="0.1",
            alpha=0.5,
        )
        axs[i].plot(
            state.state_grid, retval["real_map"][params[i]], "o-", mfc="none"
        )
        axs[i].plot(state.state_grid, retval_s["real_map"][params[i]], "--")
        if i in [1, 4, 6]:
            axs[i].set_ylim(0, 6)
        else:
            axs[i].set_ylim(0, 1)
        axs[i].set_ylabel(tip_params[i])

    fwd = np.array(obs.fwd_modelled_obs)
    axs[7].plot(obs.observations[:, 0], fwd[:, 0], "k+", label="VIS")
    axs[7].plot(obs.observations[:, 1], fwd[:, 1], "rx", label="NIR")
    axs[7].set_xlabel("Measured BHR [-]")
    axs[7].set_ylabel("Predicted BHR [-]")
    axs[7].plot([0, 0.9], [0, 0.9], "k--", lw=0.5)
    axs[7].legend(loc="best")

    axs[8].vlines(
        obs.mask[:, 0],
        obs.observations[:, 0] - 1.96 * obs.bu[:, 0],
        obs.observations[:, 0] + 1.96 * obs.bu[:, 0],
    )
    axs[8].plot(obs.mask[:, 0], obs.observations[:, 0], "o")

    axs[9].vlines(
        obs.mask[:, 0],
        obs.observations[:, 1] - 1.96 * obs.bu[:, 1],
        obs.observations[:, 1] + 1.96 * obs.bu[:, 1],
    )
    axs[9].plot(obs.mask[:, 0], obs.observations[:, 1], "o")

    axs[8].set_ylabel("BHR VIS [-]")
    axs[9].set_ylabel("BHR NIR [-]")
    axs[8].set_xlabel("DoY [d]")
    axs[9].set_xlabel("DoY [d]")

    for i in range(10):

        if i != 7:
            axs[i].set_xlim(1, 370)
        # Hide the right and top spines
        axs[i].spines["right"].set_visible(False)
        axs[i].spines["top"].set_visible(False)
        # Only show ticks on the left and bottom spines
        axs[i].yaxis.set_ticks_position("left")
        axs[i].xaxis.set_ticks_position("bottom")

    fig.figimage(
        logo, fig.bbox.xmax - 500, fig.bbox.ymax - 250, alpha=0.4, zorder=1
    )  # if __name__ == "__main__":
    plt.savefig(
        "regularised_model_%s_%04d.pdf" % (fluxnet_site, year),
        dpi=300,
        bbox_inches="tight",
    )


def prior_experiment(
    fluxnet_site,
    year,
    gamma_lai,
    green_leaves=False,
    inflation=2.0,
    n_tries=5,
):

    retval_s, state, obs = tip_inversion(
        year - 1, fluxnet_site, green_leaves=green_leaves, n_tries=4
    )

    post_sd = np.sqrt(np.array(retval_s["post_cov"].todense()).squeeze())
    main_diag = np.array(1.0 / ((inflation * post_sd) ** 2))
    prior = Prior(
        state.pack_from_dict(retval_s["real_map"]),
        sp.lil_matrix(np.diag(main_diag)),
    )

    mu = state.operators["Prior"].mu
    cinv = state.operators["Prior"].inv_cov
    c = np.array(np.sqrt(np.linalg.inv(cinv.todense()).diagonal())).squeeze()
    post_sd = np.sqrt(np.array(retval_s["post_cov"].todense()).squeeze())
    post_sd_single = np.where(post_sd > c, c, post_sd)

    retval, state, obs = regularised_tip_inversion(
        year,
        fluxnet_site,
        [1e-3, 0, 0.1, 1e-3, 0, 0.1, gamma_lai],
        x0=retval_s["real_map"],
        green_leaves=green_leaves,
        n_tries=n_tries,
        prior=prior,
    )

    fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(14, 12))
    axs = axs.flatten()
    fig.suptitle("%s (%d)" % (fluxnet_site, year), fontsize=18)
    params = [
        "omega_vis",
        "d_vis",
        "a_vis",
        "omega_nir",
        "d_nir",
        "a_nir",
        "lai",
    ]
    post_sd = np.sqrt(np.array(retval["post_cov"].todense()).squeeze())
    post_sd = np.where(post_sd > c, c, post_sd)

    for i, p in enumerate(tip_params):

        # axs[i].axhspan(mu[(i*46):((i+1)*46)][0]+c[(i*46):((i+1)*46)][0],
        #               mu[(i*46):((i+1)*46)][0] - c[(i * 46):((i + 1) * 46)][0], color="0.9" )

        axs[i].fill_between(
            state.state_grid,
            retval["real_map"][params[i]]
            - post_sd[(i * 46) : ((i + 1) * 46)],
            retval["real_map"][params[i]]
            + post_sd[(i * 46) : ((i + 1) * 46)],
            lw=0.8,
            color="0.8",
        )
        axs[i].vlines(
            state.state_grid,
            retval_s["real_map"][params[i]]
            - post_sd_single[(i * 46) : ((i + 1) * 46)],
            retval_s["real_map"][params[i]]
            + post_sd_single[(i * 46) : ((i + 1) * 46)],
            lw=0.8,
            colors="0.1",
            alpha=0.5,
        )
        axs[i].plot(
            state.state_grid, retval["real_map"][params[i]], "o-", mfc="none"
        )
        axs[i].plot(state.state_grid, retval_s["real_map"][params[i]], "--")
        if i in [1, 4, 6]:
            axs[i].set_ylim(0, 6)
        else:
            axs[i].set_ylim(0, 1)
        axs[i].set_ylabel(tip_params[i])

    fwd = np.array(obs.fwd_modelled_obs)
    axs[7].plot(obs.observations[:, 0], fwd[:, 0], "k+", label="VIS")
    axs[7].plot(obs.observations[:, 1], fwd[:, 1], "rx", label="NIR")
    axs[7].set_xlabel("Measured BHR [-]")
    axs[7].set_ylabel("Predicted BHR [-]")
    axs[7].plot([0, 0.9], [0, 0.9], "k--", lw=0.5)
    axs[7].legend(loc="best")

    axs[8].vlines(
        obs.mask[:, 0],
        obs.observations[:, 0] - 1.96 * obs.bu[:, 0],
        obs.observations[:, 0] + 1.96 * obs.bu[:, 0],
    )
    axs[8].plot(obs.mask[:, 0], obs.observations[:, 0], "o")

    axs[9].vlines(
        obs.mask[:, 0],
        obs.observations[:, 1] - 1.96 * obs.bu[:, 1],
        obs.observations[:, 1] + 1.96 * obs.bu[:, 1],
    )
    axs[9].plot(obs.mask[:, 0], obs.observations[:, 1], "o")

    axs[8].set_ylabel("BHR VIS [-]")
    axs[9].set_ylabel("BHR NIR [-]")
    axs[8].set_xlabel("DoY [d]")
    axs[9].set_xlabel("DoY [d]")

    for i in range(10):

        if i != 7:
            axs[i].set_xlim(1, 370)
        # Hide the right and top spines
        axs[i].spines["right"].set_visible(False)
        axs[i].spines["top"].set_visible(False)
        # Only show ticks on the left and bottom spines
        axs[i].yaxis.set_ticks_position("left")
        axs[i].xaxis.set_ticks_position("bottom")

    fig.figimage(
        logo, fig.bbox.xmax - 500, fig.bbox.ymax - 250, alpha=0.4, zorder=1
    )  # if __name__ == "__main__":


if __name__ == "__main__":
    prior_experiment("US-MMS", 2010, False, 1.0)
