#!/usr/bin/env python
"""
Functions and stuff that run eoldas inversions using the TIP.
"""
import cPickle
from collections import OrderedDict
import numpy as np

import eoldas_ng
from tip_helpers import StandardStateTIP, ObservationOperatorTIP
from tip_helpers import the_prior, bernards_prior
from get_albedo import Observations


# Define some useful globals
optimisation_options = {'ftol': 1./10000, 'gtol':1e-12, 
                            'maxcor':300, 'maxiter':1500 }
state_grid = np.arange(1, 366, 8)

state_config = OrderedDict()
state_config['omega_vis'] = eoldas_ng.VARIABLE
state_config['d_vis'] = eoldas_ng.VARIABLE
state_config['a_vis'] = eoldas_ng.VARIABLE
state_config['omega_nir'] = eoldas_ng.VARIABLE
state_config['d_nir'] = eoldas_ng.VARIABLE
state_config['a_nir'] = eoldas_ng.VARIABLE
state_config['lai'] = eoldas_ng.VARIABLE


def retrieve_albedo ( year, fluxnet_site, albedo_unc, albedo_db="albedo.sql"  ):
    """
    A method to retrieve albedo from the practical's albedo database. The method 
    takes the year, the fluxnet site code, a two element list with the associated
    uncertainy factor as a percentage (e.g. Pinty's would be given as [0.05, 0.07])
    for the "good" and "other" quality flags.
    """
    obs = Observations ( albedo_db )
    albedo_data = obs.query( year, fluxnet_site )
    passer = albedo_data.albedo_qa != 255
    passer = np.where ( np.logical_or ( albedo_data.bhr_vis < 0.01, 
                                       albedo_data.bhr_nir < 0.01 ), 
                                        False, passer )
    
    doys = albedo_data.doy[passer]
    observations = np.c_ [ albedo_data.bhr_vis[passer], 
                     albedo_data.bhr_nir[passer] ]
    full_inversions = albedo_data.albedo_qa[passer] == 0
    backup_inversions = albedo_data.albedo_qa[passer] == 1
    is_snow = albedo_data.snow_qa == 1
    mask = np.c_[ doys, doys.astype(np.int)*0 + 1 ]
    bu = observations*0.0
    for i in xrange(2):
        bu[full_inversions, i] = np.min ( np.c_[
            np.ones_like (doys[full_inversions])*2.5e-3, 
            observations[full_inversions,i]*albedo_unc[0]], axis=1)
        bu[backup_inversions, i] = np.min ( np.c_[
            np.ones_like (doys[backup_inversions])*2.5e-3, 
            observations[backup_inversions,i]*albedo_unc[1]], axis=1)
    passer_snow = passer.copy()
    passer_snow[passer] = is_snow[passer]

    # Observation uncertainty is 5% and 7% for flags 0 and 1, resp
    # Min of 2.5e-3
    return observations, mask, bu, passer_snow


def tip_inversion ( year, fluxnet_site,  
        albedo_unc=[0.05, 0.07], prior_type="TIP_standard",
        vis_emu_pkl="tip_vis_albedo_transformed.pkl",
        nir_emu_pkl="tip_nir_albedo_transformed.pkl",
        n_tries=15 ):

    # Start by setting up the state 
    the_state = StandardStateTIP ( state_config, state_grid, verbose=True,
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
    #prior = the_prior(the_state, prior_type )
    prior = bernards_prior ( passer_snow, use_soil_corr=False  )
    the_state.add_operator ("Prior", prior )
    # Now, we will do the function minimisation with `n_tries` different starting
    # points. We choose the one with the lowest cost...
    x_dict = OrderedDict()
    results = []
    for i in xrange(n_tries):
        for p, v in the_state.default_values.iteritems():
            x_dict[p] = np.random.rand(len(state_grid))

        retval = the_state.optimize(x_dict, do_unc=True)
        results.append ( ( the_state.cost_history['global'][-1],
                         retval ) )
    best_solution = np.array([ x[0] for x in results]).argmin()
    return results[best_solution][1]





def regularised_tip_inversion ( year, fluxnet_site, regularisation, 
        albedo_unc=[0.05, 0.07], prior_type="TIP_standard",
        vis_emu_pkl="tip_vis_albedo_transformed.pkl",
        nir_emu_pkl="tip_nir_albedo_transformed.pkl" ):
    """
    A regularised TIP inversion using eoldas
    """


    the_state = StandardStateTIP ( state_config, state_grid, verbose=True,
                                  optimisation_options=optimisation_options)


    gp_vis = cPickle.load(open(vis_emu_pkl, 'r'))
    gp_nir = cPickle.load(open(nir_emu_pkl, 'r'))

    observations, mask, bu, no_snow = retrieve_albedo ( year, fluxnet_site, 
                                                   albedo_unc )

    obsop = ObservationOperatorTIP ( state_grid, the_state, observations,
                mask, [gp_vis, gp_nir], bu )

    x_dict = OrderedDict()
    for p, v in the_state.default_values.iteritems():
        x_dict[p] = np.random.rand(len(state_grid))#np.ones_like ( state_grid )*v

    the_state.add_operator("Obs", obsop)
    smoother = eoldas_ng.TemporalSmoother (state_grid, 
                        np.array([7e2, 1e3, 1e3, 7e2, 1e3, 1e3, 1e3]), 
                        required_params= ['omega_vis', 'd_vis', 'a_vis',
                                              'omega_nir', 'd_nir', 'a_nir', 'lai'])
    the_state.add_operator ( "Smooth", smoother)
    prior = the_prior(the_state, prior_type )
    the_state.add_operator ("Prior", prior )
    retval = the_state.optimize(x_dict, do_unc=True)

if __name__ == "__main__":
    retval = tip_inversion ( 2010, "US-Bo1")
