#!/usr/bin/env python
"""
A Python/GP version of the TIP
"""
import cPickle
from collections import OrderedDict

import eoldas_ng
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline

from get_albedo import Observations
from tip_helpers import retrieve_albedo


the_bounds= [[0.01, 0.95], [0.1, 4], [0, 0.95], [0.01, 0.95],
             [0.1, 4], [0, 0.95], [0.01, 10]]

optimisation_options = {'ftol': 10.*np.finfo(float).eps, 'gtol':1e-12,
                            'maxcor':600, 'maxiter':1500}

####### Polychromatic leaves

mu_prior = np.array ([0.17, 1, 0.1, 0.7, 2., 0.18, 2.])
mu_prior_snow = np.array ([0.17, 1, 0.50, 0.7, 2., 0.35, 2.])

prior_cov = np.diag ( [ 0.12, 0.7, 0.0959, 0.15, 1.5, 0.2, 5.])
prior_cov[2,5] = 0.8862*0.0959*0.2
prior_cov[5,2] = 0.8862*0.0959*0.2

prior_cov_snow = np.diag ( [ 0.12, 0.7, 0.346, 0.15, 1.5, 0.25, 5.])
prior_cov_snow[2,5] = 0.8670*0.346*0.25
prior_cov_snow[5,2] = 0.8670*0.346*0.25

iprior_cov = np.linalg.inv ( prior_cov )
iprior_cov_snow = np.linalg.inv ( prior_cov_snow )

####### Green leaves

mu_prior = np.array ([0.13, 1, 0.1, 0.77, 2., 0.18, 2.])
mu_prior_snow = np.array ([0.13, 1, 0.50, 0.77, 2., 0.35, 2.])

prior_cov = np.diag ( [ 0.0140, 0.7, 0.0959, 0.0140, 1.5, 0.2, 5.])
prior_cov[2,5] = 0.8862*0.0959*0.2
prior_cov[5,2] = 0.8862*0.0959*0.2

prior_cov_snow = np.diag ( [ 0.0140, 0.7, 0.346, 0.0140, 1.5, 0.25, 5.])
prior_cov_snow[2,5] = 0.8670*0.346*0.25
prior_cov_snow[5,2] = 0.8670*0.346*0.25

iprior_cov = np.linalg.inv ( prior_cov )
iprior_cov_snow = np.linalg.inv ( prior_cov_snow )


def tip_single_inversion ( x0, albedo, bu, mu, inv_cov_prior, gp_vis, gp_nir ):

    x0 = mu
    albedo.squeeze()
    def cost (x):


        cost = 0.
        der_cost = np.zeros(7)
        cost += 0.5*(x-mu).dot ( inv_cov_prior).dot(x-mu)
        der_cost = der_cost + (x-mu).dot(inv_cov_prior)

        vis_posns = np.array ([0,1,2,6])
        nir_posns = np.array ([3,4,5,6])
        fwd_vis, dfwd_vis = gp_vis.predict (np.atleast_2d(x[vis_posns]), do_unc=False)
        fwd_nir, dfwd_nir = gp_nir.predict (np.atleast_2d(x[nir_posns]), do_unc=False)
        fwd_vis = fwd_vis.squeeze()
        fwd_nir = fwd_nir.squeeze()
        d = fwd_vis - albedo[0]
        der_cost[vis_posns] = der_cost[vis_posns] + d*dfwd_vis/(bu[0]**2)
        cost += 0.5*d*d/(bu[0]**2)

        d = fwd_nir - albedo[1]
        der_cost[nir_posns] = der_cost[nir_posns] + d * dfwd_nir / (bu[1] ** 2)
        cost += 0.5 * d * d / (bu[1] ** 2)
        return cost, der_cost

    retval = minimize( cost, x0, method="L-BFGS-B", jac=True, bounds=the_bounds,
                       options=optimisation_options )

    return retval





def single_inversion ( year, site ):
    n_tries = 2
    observations, mask, bu, passer_snow = retrieve_albedo( year, site, [0.05, 0.07])

    vis_emu_pkl = "tip_vis_emulator_real.pkl"
    nir_emu_pkl = "tip_nir_emulator_real.pkl"
    gp_vis = cPickle.load(open(vis_emu_pkl, 'r'))
    gp_nir = cPickle.load(open(nir_emu_pkl, 'r'))
    x0 = mu_prior
    state = np.zeros((46,7))

    for j,tstep in  enumerate(np.arange(1, 366, 8)):
        state[j,:] = mu_prior

        if tstep in mask[:,0]:
            i = mask[:,0] == tstep
            is_ok = mask[i,1]
            if is_ok == 1:
                if passer_snow[i]:
                    mu = mu_prior_snow
                    inv_cov_prior = iprior_cov_snow
                    cov_prior = prior_cov_snow
                else:
                    mu = mu_prior
                    inv_cov_prior = iprior_cov
                    cov_prior = prior_cov

            cost_list = np.zeros(n_tries)
            solutions = np.zeros((n_tries, 7))
            for trial in xrange(n_tries):
                 if trial > 0:
                     while True:
                         x0 = np.random.multivariate_normal(mu, cov_prior, 1)
                         if np.all ( x0 > 0 ):
                             break
                 retval = tip_single_inversion(x0, observations[i, :].squeeze(), bu[i,:].squeeze(),
                                           mu, inv_cov_prior, gp_vis, gp_nir)

                 cost_list[trial] = retval.fun
                 solutions[trial, :] = retval.x

            best_sol = cost_list.argmin()
            x0 = solutions[best_sol, :]
            state[j,:] = solutions[best_sol, :]

    return state
if __name__ == "__main__":

    retval = single_inversion( 2010, "ES-LMa")
    x_dict = {}
    for i,k in enumerate ( ['omega_vis', 'd_vis', 'a_vis', 'omega_nir', 'd_nir', 'a_nir', 'lai']):
        x_dict[k] = retval[:,i]

