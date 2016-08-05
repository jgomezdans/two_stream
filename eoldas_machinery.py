#!/usr/bin/env python
"""
Functions and stuff that run eoldas inversions using the TIP.
"""
import cPickle
from collections import OrderedDict

import eoldas_ng
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline

from tip_helpers import StandardStateTIP, ObservationOperatorTIP
from tip_helpers import bernards_prior, retrieve_albedo
from tip_inversion import single_inversion
# Define some useful globals
optimisation_options = {'gtol':1e-6,
                            'maxcor':50, 'maxiter':1500, "disp":20}
state_grid = np.arange(1, 366, 8)

state_config = OrderedDict()
state_config['omega_vis'] = eoldas_ng.VARIABLE
state_config['d_vis'] = eoldas_ng.VARIABLE
state_config['a_vis'] = eoldas_ng.VARIABLE
state_config['omega_nir'] = eoldas_ng.VARIABLE
state_config['d_nir'] = eoldas_ng.VARIABLE
state_config['a_nir'] = eoldas_ng.VARIABLE
state_config['lai'] = eoldas_ng.VARIABLE




def tip_inversion ( year, fluxnet_site, albedo_unc=[0.05, 0.07], green_leaves=False,
                    prior_type="TIP_standard",
                    vis_emu_pkl="tip_vis_emulator_real.pkl",
                    nir_emu_pkl="tip_nir_emulator_real.pkl", n_tries=2,
                    progressbar=None):
    """The JRC-TIP inversion using eoldas. This function sets up the
    invesion machinery for a particular FLUXNET site and year (assuming
    these are present in the database!)

    Parameters
    ----------
    year : int
        The year
    fluxnet_site: str
        The code of the FLUXNET site (e.g. US-Bo1)
    albedo_unc: list
        A 2-element list, containg the relative uncertainty
    prior_type: str
        Not used yet
    vis_emu_pkl: str
        The emulator file for the visible band.
    nir_emu_pkl: str
        The emulator file for the NIR band.
    n_tries: int
        Number of restarts for the minimisation. Best one (e.g. lowest
        cost) is chosen

    Returns
    -------
    Good stuff
    """
    # Start by setting up the state 
    the_state = StandardStateTIP ( state_config, state_grid, 
                                  optimisation_options=optimisation_options)

    # Load and prepare the emulators for the TIP
    gp_vis = cPickle.load(open(vis_emu_pkl, 'r'))
    gp_nir = cPickle.load(open(nir_emu_pkl, 'r'))
    # Retieve observatiosn and ancillary stuff from database
    observations, mask, bu, passer_snow = retrieve_albedo ( year, fluxnet_site,
                                                       albedo_unc )    
    # Set up the observation operator
    obsop = ObservationOperatorTIP ( state_grid, the_state, observations,
                mask, [gp_vis, gp_nir], bu )
    the_state.add_operator("Obs", obsop)
    # Set up the prior
    ### prior = the_prior(the_state, prior_type )
    prior = bernards_prior ( passer_snow, use_soil_corr=True,
                             green_leaves=green_leaves)
    the_state.add_operator ("Prior", prior )
    # Now, we will do the function minimisation with `n_tries` different starting
    # points. We choose the one with the lowest cost...



    retval = single_inversion( year, fluxnet_site)
    x_dict = {}
    for i,k in enumerate ( ['omega_vis', 'd_vis', 'a_vis', 'omega_nir', 'd_nir', 'a_nir', 'lai']):
        x_dict[k] = retval[:,i]

    results = []
    for i in xrange(n_tries):
        if n_tries > 1:
            x0 = np.random.multivariate_normal( prior.mu, np.array(np.linalg.inv(prior.inv_cov.todense())))
            x_dict = the_state._unpack_to_dict ( x0 )
        retval = the_state.optimize(x_dict, do_unc=True)
        results.append ( ( the_state.cost_history['global'][-1],
                         retval ) )
        if progressbar is not None:
            progressbar.value = progressbar.value + 1
    best_solution = np.array([ x[0] for x in results]).argmin()
    print [ x[0] for x in results]
    print "Chosen cost: %g" % results[best_solution][0]
    return results[best_solution][1], the_state, obsop


def regularised_tip_inversion ( year, fluxnet_site, gamma, x0, albedo_unc=[0.05, 0.07], green_leaves=False,
                    prior_type="TIP_standard",
                    vis_emu_pkl="tip_vis_emulator_real.pkl",
                    nir_emu_pkl="tip_nir_emulator_real.pkl", n_tries=2, progressbar=None):
    """The JRC-TIP inversion using eoldas. This function sets up the
    invesion machinery for a particular FLUXNET site and year (assuming
    these are present in the database!)

    Parameters
    ----------
    year : int
        The year
    fluxnet_site: str
        The code of the FLUXNET site (e.g. US-Bo1)
    albedo_unc: list
        A 2-element list, containg the relative uncertainty
    prior_type: str
        Not used yet
    vis_emu_pkl: str
        The emulator file for the visible band.
    nir_emu_pkl: str
        The emulator file for the NIR band.
    n_tries: int
        Number of restarts for the minimisation. Best one (e.g. lowest
        cost) is chosen

    Returns
    -------
    Good stuff
    """
    # Start by setting up the state
    the_state = StandardStateTIP ( state_config, state_grid,
                                  optimisation_options=optimisation_options)

    # Load and prepare the emulators for the TIP
    gp_vis = cPickle.load(open(vis_emu_pkl, 'r'))
    gp_nir = cPickle.load(open(nir_emu_pkl, 'r'))
    # Retieve observatiosn and ancillary stuff from database
    observations, mask, bu, passer_snow = retrieve_albedo ( year, fluxnet_site,
                                                       albedo_unc )
    # Set up the observation operator
    obsop = ObservationOperatorTIP ( state_grid, the_state, observations,
                mask, [gp_vis, gp_nir], bu )
    the_state.add_operator("Obs", obsop)
    # Set up the prior
    ### prior = the_prior(the_state, prior_type )
    prior = bernards_prior ( passer_snow, use_soil_corr=True,
                             green_leaves=green_leaves)
    the_state.add_operator ("Prior", prior )
    # Now, we will do the function minimisation with `n_tries` different starting
    # points. We choose the one with the lowest cost...

    smoother = eoldas_ng.TemporalSmoother ( state_grid, gamma, required_params =
                                                            ["omega_vis", "d_vis", "a_vis",
                                                            "omega_nir", "d_nir", "a_nir", "lai"] )

    the_state.add_operator ( "Smooth", smoother)


    x_dict = x0
    results = []
    for i in xrange(n_tries):
        if n_tries > 1:
            x0 = np.random.multivariate_normal( prior.mu, np.array(np.linalg.inv(prior.inv_cov.todense())))
            x_dict = the_state._unpack_to_dict ( x0 )
        retval = the_state.optimize(x_dict, do_unc=True)
        results.append ( ( the_state.cost_history['global'][-1],
                         retval ) )
        if progressbar is not None:
            progressbar.value = progressbar.value + 1

    best_solution = np.array([ x[0] for x in results]).argmin()
    print [ x[0] for x in results]
    print "Chosen cost: %g" % results[best_solution][0]
    return results[best_solution][1], the_state, obsop



if __name__ == "__main__":
    params = ['omega_vis', 'd_vis', 'a_vis', 'omega_nir', 'd_nir', 'a_nir', 'lai']
    site = "DE-Geb"
    for year in [2008]:
        retval_s, state, obs = tip_inversion( year, site, green_leaves=False)
        retval, state, obs = regularised_tip_inversion( year, site, [1e-3, 0, 0.1, 1e-3, 0, 0.1, 1  ],
                                                        x0=retval_s['real_map'], green_leaves=False)
        fig = plt.figure()
        fig.suptitle("%s (%d)" % (site, year))

        for i,p in enumerate ( params ):
            plt.subplot(5,2, i+1 )
            # plt.fill_between ( state.state_grid, retval['real_ci5pc'][p],
            #                   retval['real_ci95pc'][p], color="0.8" )
            # plt.fill_between ( state.state_grid, retval['real_ci25pc'][p],
            #                   retval['real_ci75pc'][p], color="0.6" )

            plt.plot ( state.state_grid, retval['real_map'][p], '-' )
            plt.plot ( state.state_grid, retval_s['real_map'][p], '--')
            if p in ["d_vis", "d_nir", "lai"]:
                plt.ylim(0,6)
            else:
                plt.ylim(0,1)
        plt.subplot(5,2,8)
        fwd = np.array(obs.fwd_modelled_obs)
        plt.plot ( obs.observations[:,0], fwd[:,0], 'k+')
        plt.plot ( obs.observations[:,1], fwd[:,1], 'rx')
        plt.plot([0,0.7], [0, 0.7], 'k--')
        plt.plot([2.5e-3, 0.7+0.7*0.07],[2.5e-3, 0.7+0.7*0.07], 'k--', lw=0.5)
        plt.plot([-2.5e-3, 0.7-0.7*0.07],[-2.5e-3, 0.7-0.7*0.07], 'k--', lw=0.5)
        plt.subplot(5,2,9)
        plt.vlines(obs.mask[:, 0], obs.observations[:, 0] - 1.96 * obs.bu[:, 0],
                   obs.observations[:, 0] + 1.96 * obs.bu[:, 0])
        plt.plot(obs.mask[:, 0], obs.observations[:, 0], 'o')
        plt.subplot(5,2,10)
        plt.vlines(obs.mask[:, 0], obs.observations[:, 1] - 1.96 * obs.bu[:, 1],
                   obs.observations[:, 1] + 1.96 * obs.bu[:, 1])
        plt.plot ( obs.mask[:,0], obs.observations[:,1], 'o')



    plt.show()
