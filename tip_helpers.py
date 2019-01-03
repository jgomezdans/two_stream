#!/usr/bin/env python
"""
Some helper classes and methods to use the eoldas_ng engine on the JRC-TIP inversions
"""
__author__  = "J Gomez-Dans"
__version__ = "1.0 (01.08.2016)"
__email__   = "j.gomez-dans@ucl.ac.uk"

import numpy as np
from collections import OrderedDict
import eoldas_ng
import scipy.sparse as sp
from eoldas_ng import State

import time
from eoldas_ng import State, MetaState, CONSTANT, VARIABLE, FIXED
from get_albedo import Observations



def retrieve_albedo(year, fluxnet_site, albedo_unc, albedo_db="albedo.sql"):
    """
    A method to retrieve albedo from the practical's albedo database. The method
    takes the year, the fluxnet site code, a two element list with the associated
    uncertainy factor as a percentage (e.g. Pinty's would be given as [0.05, 0.07])
    for the "good" and "other" quality flags.
    """
    obs = Observations(albedo_db)
    albedo_data = obs.query(year, fluxnet_site)
    passer = albedo_data.albedo_qa != 255
    passer = np.where(np.logical_or(albedo_data.bhr_vis < 0.01,
                                    albedo_data.bhr_nir < 0.01),
                      False, passer)

    doys = albedo_data.doy[passer]
    observations = np.c_[albedo_data.bhr_vis[passer],
                         albedo_data.bhr_nir[passer]]
    full_inversions = albedo_data.albedo_qa[passer] == 0
    backup_inversions = albedo_data.albedo_qa[passer] == 1
    is_snow = albedo_data.snow_qa == 1
    mask = np.c_[doys, doys.astype(np.int) * 0 + 1]
    bu = observations * 0.0
    for i in range(2):
        bu[full_inversions, i] = np.max(np.c_[
                                            np.ones_like(doys[full_inversions]) * 2.5e-3,
                                            observations[full_inversions, i] * albedo_unc[0]], axis=1)
        bu[backup_inversions, i] = np.max(np.c_[
                                              np.ones_like(doys[backup_inversions]) * 2.5e-3,
                                              observations[backup_inversions, i] * albedo_unc[1]], axis=1)
    passer_snow = passer.copy()
    passer_snow[passer] = is_snow[passer]
    # Observation uncertainty is 5% and 7% for flags 0 and 1, resp
    # Min of 2.5e-3
    # bu = bu*0. + 0.005
    return observations, mask, bu, passer_snow


def get_problem_size ( x_dict, state_config, state_grid=None ):
    """This function reports
    1. The number of parameters `n_params`
    2. The size of the state

    Parameters
    -----------
    x_dict: dict
        An (ordered) dictionary with the state
    state_config: dict
        A state configuration ordered dictionary
    """
    n_params = 0
    for param, typo in state_config.items():
        if typo == CONSTANT:
            n_params += 1
        elif typo == VARIABLE:
            if state_grid is None:
                n_elems = x_dict[param].size
            else:
                n_elems = state_grid.sum()
            n_params += n_elems
    return n_params, n_elems

class StandardStateTIP ( State ):
    """A standard state configuration for Bernard Pinty's TwoStream model.
    This class has some default values that are as used in the standard
    JRC-TIP parameter inversion machinery."""
    def __init__ ( self, state_config, state_grid, \
                 optimisation_options=None, \
                 output_name=None, verbose=False ):
        """Takes state configuration, state grid, and some common optional 
        parameters.
        """
        self.state_config = state_config
        self.state_grid = state_grid
        self.n_elems =  self.state_grid.size
        # We now test whether the state mask is operational. We need to
        # check the type of the array in order not to break compatibility
        # In the future, this is just wasteful!
        if ( (self.state_grid.dtype == np.dtype ( np.bool )) and 
                    ( self.n_elems != self.state_grid.sum())):
            self.n_elems_masked = self.state_grid.sum()
        else:
            self.n_elems_masked = self.n_elems
        # self.n_elems_masked is the number of "valid" grid cells

        # Now define the default values
        self.default_values = OrderedDict ()
        self.default_values['omega_vis'] = 0.17
        self.default_values['d_vis'] = 2.
        self.default_values['a_vis'] = .1
        self.default_values['omega_nir'] = 0.7
        self.default_values['d_nir'] = 2 # Say?
        self.default_values['a_nir'] = 0.18 # Say?
        self.default_values['lai'] = 2
        
        self.metadata = MetaState()
        self.metadata.add_variable ("omega_vis", "None", 
                    "Leaf single scattering albedo in visible", "omega_vis" )
        self.metadata.add_variable ("d_vis", "None", 
                    "Leaf anisotropy in visible", "d_vis" )                                    
        self.metadata.add_variable ("a_vis", "None", 
                    "Soil albedo in visible", "a_vis" )                                    
        self.metadata.add_variable ("omega_nir", "None", 
                    "Leaf single scattering albedo in NIR", "omega_nir" )
        self.metadata.add_variable ("d_nir", "None", 
                    "Leaf anisotropy in NIR", "d_nir" )                                    
        self.metadata.add_variable ("a_NIR", "None", 
                    "Soil albedo in NIR", "a_nir" )                                    
        self.metadata.add_variable ("lai", "None", 
                    "Effective LAI", "lai_e" )                                    


        self.operators = {}
        self.n_params = self._state_vector_size ()
        self.parameter_min = OrderedDict()
        self.parameter_max = OrderedDict()
        min_vals = [ 0.001, 0.001, 0.001, 0.001, 0.001, 0.001,0.01 ]
        max_vals = [ 0.99, 5., 0.99, 0.99, 5., 0.99, 6 ]

        for i, param in enumerate ( state_config.keys() ):
            self.parameter_min[param] = min_vals[i]
            self.parameter_max[param] = max_vals[i]

        self.verbose = verbose
        self.bounds = []
        for ( i, param ) in enumerate ( self.state_config.keys() ):
            self.bounds.append ( [ self.parameter_min[param],
                                   self.parameter_max[param]] )
            # Define parameter transformations
        transformations = {}
                #'lai': lambda x: np.exp ( -3.*x/2. ), 
                #'d_nir': lambda x: np.exp ( -x ), 
                #'d_vis': lambda x: np.exp ( -x ) }
        inv_transformations = {}
                #'lai': lambda x: -1.5*np.log ( x ), 
                #'d_nir': lambda x: -np.log ( x ), 
                #'d_vis': lambda x: -np.log( x ) }
        
        
        self.set_transformations ( transformations, inv_transformations )

        
        self._set_optimisation_options ( optimisation_options )
        self._create_output_file ( output_name )
        self.cost_history = { 'global': [],
                              'iteration': [] }
        self.iterations = 0
        self.iterations_vector = []
        

class ObservationOperatorTIP ( object ):
    """An operator calculating the "fit to the observations" using the 
    Pinty's Two Stream RT model. The model here is used through emulation
    with Gaussian Processes. The layout of the code is slightly different
    to the one we use typically because 
    1. Each band has a different emulator,
    2. Each band uses a subset of the parameters,
    3. Only two emulators are required for the entire time series.
    
    TODO The coding could be more flexible, as we could potentially use
    this for other ObsOps (e.g. SAR, passive microwaves, etc.)
    """
    def __init__ ( self, state_grid, state, observations, mask, emulators, bu ):
        """
         observations is an array with `n_observations`, `n_bands`, and in 
         reality, `n_bands` should always be 2 (VIS & NIR). The mask just
         has two columns, the DoY and a bunch of 1s or 0s (the former
         indicating it's OK, the latter indicating it's a bad observation).
         `bu` is the uncertainty, which will be in this case the same size
         as `observations`
         """
        self.state = state
        self.observations = observations
        try:
            self.n_obs, self.n_bands = self.observations.shape
        except:
            raise ValueError("Typically, obs should be (n_obs, n_bands)")
        self.mask = mask
        assert ( self.n_obs ) == mask.shape[0]
        self.state_grid = state_grid
        self.nt = self.state_grid.shape[0]
        self.emulators = emulators
        self.bu = bu


    
    def der_cost ( self, x_dict, state_config ):

        """The cost function and its partial derivatives. One important thing
        to note is that GPs have been parameterised in transformed space, 
        whereas `x_dict` is in "real space". So when we go over the parameter
        dictionary, we need to transform back to linear units. TODO Clearly, it
        might be better to have cost functions that report whether they need
        a dictionary in true or transformed units!

        Parameters
        -----------
        x_dict: ordered dict
            The state as a dictionary
        state_config: oredered dict
            The configuration dictionary
        
        Returns
        --------
        cost: float
            The value of the cost function
        der_cost: array
            An array with the partial derivatives of the cost function        
        """
        i = 0
        cost = 0.
        n, n_elems = get_problem_size ( x_dict, state_config )
        der_cost = np.zeros ( n )
        x_params = np.empty ( ( len( list(x_dict.keys())), \
                self.nt ) )
        j = 0
        ii = 0
        
        the_derivatives = np.zeros ( ( len( list(x_dict.keys())), \
                self.nt ) )
        
        for param, typo in state_config.items():
        
            if typo == FIXED or  typo == CONSTANT:
                #if self.state.transformation_dict.has_key ( param ):
                    #x_params[ j, : ] = self.state.transformation_dict[param] ( x_dict[param] )
                #else:
                x_params[ j, : ] = x_dict[param]
                
            elif typo == VARIABLE:
                #if self.state.transformation_dict.has_key ( param ):
                    #x_params[ j, : ] = self.state.transformation_dict[param] ( x_dict[param] )
                #else:
                x_params[ j, : ] = x_dict[param]
            j += 1
        self.fwd_modelled_obs = []
        istart_doy = self.state_grid[0] - 1

        # At this point, we have an array of parameters. Some will need to be
        # ferried over to the VIS GP, and the others to the NIR GP
        # 
        # At this stage, we could run the GPs for all times, and the loop below
        # would do the comparisons, gradients, etc.
        # In other words, we would prepare the class so that calc_mismatch
        # uses the GP output from here
        self.fwd_albedo_vis, self.dfwd_albedo_vis= self.emulators[0].predict (
                            x_params[[0,1,2,6], :].T, do_unc=False )
        self.fwd_albedo_nir, self.dfwd_albedo_nir = self.emulators[1].predict ( 
                            x_params[[3,4,5,6], :].T, do_unc=False )
        for itime, tstep in enumerate ( self.state_grid[1:] ):
            # Select all observations between istart_doy and tstep
            sel_obs = np.where ( np.logical_and ( self.mask[:, 0] > istart_doy,
                self.mask[:, 0] <= tstep ), True, False )
            if sel_obs.sum() == 0:
                # We have no observations, go to next period!
                istart_doy = tstep # Update istart_doy
                continue
            # Now, test the QA flag, field 2 of the mask...
            sel_obs = np.where ( np.logical_and ( self.mask[:, 1], sel_obs ),
                True, False )
            if sel_obs.sum() == 0:
                # We have no observations, go to next period!
                istart_doy = tstep # Update istart_doy
                continue
            # In this bit, we need a loop to go over this period's observations
            # And add the cost/der_cost contribution from each.
            
            for this_obs_loc in sel_obs.nonzero()[0]:
                this_obs, bu = self.time_step ( this_obs_loc )
                this_cost, this_der, fwd_model, this_gradient = \
                    self.calc_mismatch ( itime, this_obs, bu )
                self.fwd_modelled_obs.append ( fwd_model ) # Store fwd model
                cost += this_cost
                the_derivatives[ :, itime] += this_der
            # Advance istart_doy to the end of this period
            istart_doy = tstep
        
        j = 0
        for  i, (param, typo) in enumerate ( state_config.items()) :
            if typo == CONSTANT:
                der_cost[j] = the_derivatives[i, : ].sum()
                j += 1
            elif typo == VARIABLE:
                n_elems = x_dict[param].size
                der_cost[j:(j+n_elems) ] = the_derivatives[i, :]
                j += n_elems
        
        return cost, der_cost
    
    def time_step ( self, this_loc ):
        """Returns relevant information on the observations for a particular time step.
        """
        this_obs = self.observations[ this_loc, :]
        bu = self.bu[this_loc]
        return this_obs, bu
    
    def time_step2 ( self, this_loc ):
        """Needed for the Hessian calculations"""
        this_obs = self.observations[ this_loc, :]
        return self.emulators, this_obs, [None, None]
    
    def calc_mismatch2 ( self, xs, obs, bu, *this_extra ):
        """Needed for the Hessian calculations"""
        cost = 0.
        der_cost = np.zeros(7)
        gradient = np.zeros(7)
        derivs = np.zeros(7)
        vis_posns = np.array ( [0,1,2,6 ])
        nir_posns = np.array ( [3,4,5,6 ] )

        fwd_albedo_vis, dfwd_albedo_vis = self.emulators[0].predict ( 
                    np.atleast_2d(xs[vis_posns]), do_unc=False)
        fwd_albedo_nir, dfwd_albedo_nir = self.emulators[1].predict ( 
            np.atleast_2d(xs[nir_posns]), do_unc=False)
        dfwd_albedo_vis = dfwd_albedo_vis.squeeze()
        dfwd_albedo_nir = dfwd_albedo_nir.squeeze()
        # first vis...
        d = fwd_albedo_vis - obs[0]
        gradient[vis_posns] = dfwd_albedo_vis 
        derivs[vis_posns] = d*dfwd_albedo_vis/(bu[0]**2)
        cost += 0.5*d*d/(bu[0]**2)

        # Now NIR
        d = fwd_albedo_nir - obs[1]
        gradient[nir_posns] = dfwd_albedo_nir 
        
        derivs[nir_posns] += d*dfwd_albedo_nir/(bu[1]**2)
        cost += 0.5*d*d/(bu[1]**2)
        der_cost = derivs # I *think*
        # der_cost 

        return None, derivs, None, None
    
    
    def calc_mismatch ( self, itime, obs, bu ):
        """Simplified mismatch function assuming that both the forward model
        and associated Jacobian of the forward model have already been 
        calculated and are stored in e.g. `self.fwd_albedo{vis,nir}`. There are
        some assumptions about the data storage made at this stage!"""
        
        cost = 0.
        der_cost = np.zeros(7)
        gradient = np.zeros(7)
        derivs = np.zeros(7)
        vis_posns = np.array ( [0,1,2,6 ])
        nir_posns = np.array ( [3,4,5,6 ] )


        # First visible

        d = self.fwd_albedo_vis[itime] - obs[0]
        gradient[vis_posns] = self.dfwd_albedo_vis[itime] 
        derivs[vis_posns] = d*self.dfwd_albedo_vis[itime]/(bu[0]**2)
        cost += 0.5*d*d/(bu[0]**2)

        # Then NIR

        d = self.fwd_albedo_nir[itime] - obs[1]
        gradient[nir_posns] = self.dfwd_albedo_nir[itime]
        derivs[nir_posns] += d*self.dfwd_albedo_nir[itime]/(bu[1]**2)
        cost += 0.5*d*d/(bu[1]**2)
        der_cost = derivs # I *think*
        # der_cost 
        fwd = [self.fwd_albedo_vis[itime], self.fwd_albedo_nir[itime]]
        return cost.sum(), der_cost, fwd, gradient
    
    
    def der_der_cost ( self, x_dict, state_config, state, epsilon=1.0e-2 ):
        """Numerical approximation to the Hessian. This approximation is quite
        simple, and is based on a finite differences of the individual terms of 
        the cost function. Note that this method shares a lot with the `der_cost`
        method in the same function, and a refactoring is probably required, or
        even better, a more "analytic" expression making use of the properties of
        GPs to calculate the second derivatives.
                
                
        Parameters
        -----------
        x_dict: ordered dict
            The state as a dictionary
        state_config: oredered dict
            The configuration dictionary
        state: State
            The state is required in some cases to gain access to parameter
            transformations.
        Returns
        ---------
        Hess: sparse matrix
            The hessian for the cost function at `x`
        """
        
        
        i = 0
        cost = 0.
        
        n, n_elems = get_problem_size ( x_dict, state_config )
        
        der_cost = np.zeros ( n )
        h = sp.lil_matrix ( (n,n))
        x_params = np.empty ( ( len( list(x_dict.keys())), self.nt ) )
        j = 0
        ii = 0
        the_derivatives = np.zeros ( ( len( list(x_dict.keys())), self.nt ) )
        param_pattern = np.zeros ( len( list(state_config.items())))
        for param, typo in state_config.items():
        
            if typo == FIXED:  
                #if self.state.transformation_dict.has_key ( param ):
                    #x_params[ j, : ] = self.state.transformation_dict[param] ( x_dict[param] )
                #else:
                x_params[ j, : ] = x_dict[param]
                param_pattern[j] = FIXED
            elif typo == CONSTANT:
                x_params[ j, : ] = x_dict[param]
                param_pattern[j] = CONSTANT
                
            elif typo == VARIABLE:
                #if self.state.transformation_dict.has_key ( param ):
                    #x_params[ j, : ] = self.state.transformation_dict[param] ( x_dict[param] )
                #else:
                x_params[ j, : ] = x_dict[param]
                param_pattern[j] = VARIABLE

            j += 1
        
        n_const = np.sum ( param_pattern == CONSTANT )
        n_var = np.sum ( param_pattern == VARIABLE )
        n_grid = self.nt # don't ask...
        istart_doy = self.state_grid[0] - 1
        for itime, tstep in enumerate ( self.state_grid[1:] ):
            # Select all observations between istart_doy and tstep
            sel_obs = np.where ( np.logical_and ( self.mask[:, 0] > istart_doy, \
                self.mask[:, 0] <= tstep ), True, False )
            if sel_obs.sum() == 0:
                # We have no observations, go to next period!
                istart_doy = tstep # Update istart_doy
                continue
            # Now, test the QA flag, field 2 of the mask...
            sel_obs = np.where ( np.logical_and ( self.mask[:, 1], sel_obs ), \
                True, False )
            if sel_obs.sum() == 0:
                # We have no observations, go to next period!
                istart_doy = tstep # Update istart_doy
                continue
            # In this bit, we need a loop to go over this period's observations
            # And add the cost/der_cost contribution from each.
            for this_obs_loc in sel_obs.nonzero()[0]:
                
                this_obs, bu = self.time_step ( this_obs_loc )
                xs = x_params[:, itime]*1
                dummy, df_0, dummy_fwd, dummy_gradient = self.calc_mismatch2 ( 
                    xs, this_obs,bu )
                iloc = 0
                iiloc = 0
                for i,fin_diff in enumerate(param_pattern):
                    if fin_diff == 1: # FIXED
                        continue                    
                    xxs = xs[i]*1
                    xs[i] += epsilon
                    dummy, df_1, dummy_fwd, dummy_gradient = self.calc_mismatch2 (
                        xs, this_obs, bu )                    # Calculate d2f/d2x
                    hs =  (df_1 - df_0)/epsilon
                    if fin_diff == 2: # CONSTANT
                        iloc += 1
                    elif fin_diff == 3: # VARIABLE
                        iloc = n_const + iiloc*n_grid + itime
                        iiloc += 1
                    jloc = 0
                    jjloc = 0
                    for j,jfin_diff in enumerate(param_pattern):
                        if jfin_diff == FIXED: 
                            continue
                        if jfin_diff == CONSTANT: 
                            jloc += 1
                        elif jfin_diff == VARIABLE: 
                            jloc = n_const + jjloc*n_grid + itime
                            jjloc += 1
                        h[iloc, jloc] += hs[j]     
                    xs[i] = xxs
            # Advance istart_doy to the end of this period
            istart_doy = tstep

        return sp.lil_matrix ( h.T )
        

def bernards_prior ( snow_qa, unc_multiplier=1.,green_leaves=True,
                     use_soil_corr=True, N=46 ):
    """The "official" JRC prior, with a couple of options. The prior takes
    only the presence of snow flag as a required input, and returns an eoldas
    Prior object already configured. The other options can be used to change
    the prior uncertainty, to swich on and off the soil/snow correlation terms,
    or to change the size of the state grid.

    The description the parameters is in the original JRC-TIP papers (Pinty
    et al, 2011)
    """
    ##########################################################################
    # Prior means for NO SNOW case                                           #
    ##########################################################################
    prior_mean = OrderedDict ()
    prior_mean['omega_vis'] = 0.13
    prior_mean['d_vis'] = 1 # original coords is 1
    prior_mean['a_vis'] = .1
    prior_mean['omega_nir'] = 0.77
    prior_mean['d_nir'] = 2.
    prior_mean['a_nir'] = 0.18 
    prior_mean['lai'] = 2.

    ##########################################################################
    # Prior means for SNOW case                                              #
    ##########################################################################

    prior_mean_snow = OrderedDict ()
    prior_mean_snow['omega_vis'] = 0.13
    prior_mean_snow['d_vis'] = 1
    prior_mean_snow['a_vis'] = 0.50
    prior_mean_snow['omega_nir'] = 0.77
    prior_mean_snow['d_nir'] = 2 
    prior_mean_snow['a_nir'] = 0.350
    prior_mean_snow['lai'] = 2.
    ##########################################################################
    # Prior standard deviation  for NO SNOW case                             #
    ##########################################################################
    prior_inv_cov= OrderedDict ()
    prior_inv_cov['omega_vis'] = np.array ([.0140])*unc_multiplier
    prior_inv_cov['d_vis'] = np.array ([0.7])*unc_multiplier # original 0.7
    prior_inv_cov['a_vis'] = np.array ([0.0959])*unc_multiplier
    prior_inv_cov['omega_nir'] = np.array ([0.0140] )*unc_multiplier
    prior_inv_cov['d_nir'] = np.array ([1.5])*unc_multiplier # original 1.5
    prior_inv_cov['a_nir'] = np.array ([.2])*unc_multiplier
    prior_inv_cov['lai'] = np.array ([5])*unc_multiplier # original 5
    ##########################################################################
    # Prior standard deviation  for SNOW case                             #
    ##########################################################################
    prior_inv_cov_snow= OrderedDict ()
    prior_inv_cov_snow['omega_vis'] = np.array ([.0140])*unc_multiplier
    prior_inv_cov_snow['d_vis'] = np.array ([0.7])*unc_multiplier
    prior_inv_cov_snow['a_vis'] = np.array ([0.346])*unc_multiplier
    prior_inv_cov_snow['omega_nir'] = np.array ([0.0140])*unc_multiplier
    prior_inv_cov_snow['d_nir'] = np.array ([1.5])*unc_multiplier
    prior_inv_cov_snow['a_nir'] = np.array ([.25])*unc_multiplier
    prior_inv_cov_snow['lai'] = np.array ([5])*unc_multiplier

    if not green_leaves:
        prior_mean['omega_nir'] = 0.7
        prior_mean['omega_vis'] = 0.17
        prior_mean_snow['omega_vis'] = 0.17
        prior_mean_snow['omega_nir'] = 0.7
        prior_inv_cov['omega_vis'] = 0.12
        prior_inv_cov['omega_nir'] = 0.15
        prior_inv_cov_snow['omega_vis'] = 0.12
        prior_inv_cov_snow['omega_nir'] = 0.15
        
    soil_cov = 0.8862*(prior_inv_cov['a_vis'] * prior_inv_cov['a_nir'])
    snow_cov = 0.8670*(prior_inv_cov_snow['a_vis'] * prior_inv_cov_snow['a_nir'])

    mu_prior = np.zeros(7*N) # 7 parameters, N time steps
    main_diag = np.zeros(7*N) # 7 parameters, N time steps
    for i, parameter in enumerate(prior_mean.keys()):
        xx = np.where (snow_qa, prior_mean_snow[parameter],
                        prior_mean[parameter])
        mu_prior[i*N:((i+1)*N)] = xx
        xx = np.where (snow_qa, prior_inv_cov_snow[parameter],
                        prior_inv_cov[parameter])
        # This needs to be squared for the main diagonal, as variances go there right?
        main_diag[i*N:((i+1)*N)] = xx**2
    C = np.diag( main_diag )
    if use_soil_corr:
        for i in range(N):
            if snow_qa[i]:
                C[2*N + i, 5*N + i] = snow_cov
                C[5*N + i, 2*N + i] = snow_cov
            else:
                C[2*N + i, 5*N + i] = soil_cov
                C[5*N + i, 2*N + i] = soil_cov
            
    Cinv = np.linalg.inv(C)
    Cinv = sp.lil_matrix (Cinv)
    prior = eoldas_ng.Prior (mu_prior, Cinv)
    return prior

if __name__ == "__main__":
    import pickle
    from collections import OrderedDict
    import numpy as np

    from eoldas_ng import *
    from tip_helpers import StandardStateTIP, ObservationOperatorTIP
    from get_albedo import Observations

    state_config = OrderedDict()
    state_config['omega_vis'] = VARIABLE
    state_config['d_vis'] = VARIABLE
    state_config['a_vis'] = VARIABLE
    state_config['omega_nir'] = VARIABLE
    state_config['d_nir'] = VARIABLE
    state_config['a_nir'] = VARIABLE
    state_config['lai'] = VARIABLE

    optimisation_options = {'ftol': 1./10000, 'gtol':1e-12, 
                            'maxcor':300, 'maxiter':1500 }
    state_grid = np.arange(0, 366, 8)

    the_state = StandardStateTIP ( state_config, state_grid, verbose=True,
                                  optimisation_options=optimisation_options)


    gp_vis = pickle.load(open("tip_vis_albedo_transformed.pkl", 'r'))
    gp_nir = pickle.load(open("tip_nir_albedo_transformed.pkl", 'r'))

    obs = Observations ( "albedo.sql" )
    albedo_data = obs.query( 2009, "US-Bo1" )
    passer = albedo_data.albedo_qa != 255
    doys = albedo_data.doy[passer]
    observations = np.c_ [ albedo_data.bhr_vis[passer], 
                     albedo_data.bhr_nir[passer] ]
    full_inversions = albedo_data.albedo_qa[passer] == 0
    backup_inversions = albedo_data.albedo_qa[passer] == 1
    no_snow = albedo_data.snow_qa[passer] == 0
    mask = np.c_[ doys, doys.astype(np.int)*0 + 1 ]
    bu = observations*0.0
    for i in range(2):
        bu[full_inversions, i] = np.min ( np.c_[
            np.ones_like (doys[full_inversions])*2.5e-3, 
            observations[full_inversions,i]*0.05], axis=1)
        bu[backup_inversions, i] = np.min ( np.c_[
            np.ones_like (doys[backup_inversions])*2.5e-3, 
            observations[backup_inversions,i]*0.05], axis=1)

    # Observation uncertainty is 5% and 7% for flags 0 and 1, resp
    # Min of 2.5e-3

    #bu = np.sqrt(bu)
    obsop = ObservationOperatorTIP ( state_grid, the_state, observations,
                mask, [gp_vis, gp_nir], bu )

    x_dict = OrderedDict()
    for p, v in the_state.default_values.items():
        x_dict[p] = np.random.rand(len(state_grid))#np.ones_like ( state_grid )*v

    the_state.add_operator("Obs", obsop)
    smoother = eoldas_ng.TemporalSmoother (state_grid, 
                        np.array([7e2, 1e3, 1e3, 7e2, 1e3, 1e3, 1e3]), 
                        required_params= ['omega_vis', 'd_vis', 'a_vis',
                                              'omega_nir', 'd_nir', 'a_nir', 'lai'])
    the_state.add_operator ( "Smooth", smoother)
    #prior = the_prior(the_state)
    #the_state.add_operator ("Prior", prior )
    retval = the_state.optimize(x_dict, do_unc=True)

