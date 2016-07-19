#!/usr/bin/env python
import numpy as np


import eoldas_ng 

#!/usr/bin/env python
"""
eoldas_ng helper functions/classes
=======================================

Here we provide  a set of helper functions/classes that provide some often
required functionality. None of this is *actually* required to run the system,
but it should make the library more user friendly.

``StandardStatePROSAIL`` and ``StandardStateSEMIDISCRETE``
------------------------------------------------------------

Potentially, there will be other classes like these two. These two classes
simplify the usage of two standard RT models in defining the state. This means
that they'll provide parameter boundaries and standard transformations.

``CrossValidation``
---------------------

A simple cross-validation framework for models that have a ``gamma`` term. Done
by 

``SequentialVariational``
---------------------------

The world-famous sequential variational approach to big state space assimilation.


"""

__author__  = "J Gomez-Dans"
__version__ = "1.0 (29.12.2014)"
__email__   = "j.gomez-dans@ucl.ac.uk"

import platform
import numpy as np
import time

from collections import OrderedDict
from eoldas_ng import State, MetaState

class StandardStateTIP ( State ):
    """A standard state configuration for the PROSAIL model"""
    def __init__ ( self, state_config, state_grid, \
                 optimisation_options=None, \
                 output_name=None, verbose=False ):
        
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
        self.default_values['omega_vis'] = 1.6
        self.default_values['d_vis'] = 20.
        self.default_values['a_vis'] = 1.
        self.default_values['omega_nir'] = 0.01
        self.default_values['d_nir'] = 0.018 # Say?
        self.default_values['a_nir'] = 0.03 # Say?
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
        min_vals = [ 0.8, 0.2, 0.0, 0.0, 0.0043, 0.0017,0.0001, 0, 0., -1., -1.]
        max_vals = [2.5, 77., 5., 1., 0.0753, 0.0331, 10., 90., 2., 1.]

        for i, param in enumerate ( state_config.keys() ):
            self.parameter_min[param] = min_vals[i]
            self.parameter_max[param] = max_vals[i]

        self.verbose = verbose
        self.bounds = []
        for ( i, param ) in enumerate ( self.state_config.iterkeys() ):
            self.bounds.append ( [ self.parameter_min[param]*0.9, \
                self.parameter_max[param]*1.1 ] )
            # Define parameter transformations
        transformations = {
                'lai': lambda x: np.exp ( -x/2. ), \
                'cab': lambda x: np.exp ( -x/100. ), \
                'car': lambda x: np.exp ( -x/100. ), \
                'cw': lambda x: np.exp ( -50.*x ), \
                'cm': lambda x: np.exp ( -100.*x ), \
                'ala': lambda x: x/90. }
        inv_transformations = {
                'lai': lambda x: -2*np.log ( x ), \
                'cab': lambda x: -100*np.log ( x ), \
                'car': lambda x: -100*np.log( x ), \
                'cw': lambda x: (-1/50.)*np.log ( x ), \
                'cm': lambda x: (-1/100.)*np.log ( x ), \
                'ala': lambda x: 90.*x }

        
        self.set_transformations ( transformations, inv_transformations )

        
        self._set_optimisation_options ( optimisation_options )
        self._create_output_file ( output_name )
        

class ObservationOperatorTIP ( object ):
    """A GP-based observation operator"""
    def __init__ ( self, state_grid, state, observations, mask, emulators, bu, \
            band_pass=None, bw=None ):
        """
         observations is an array with n_bands, nt observations. nt has to be the 
         same size as state_grid (can have dummny numbers in). mask is nt*4 
         (mask, vza, sza, raa) array.
         
         
        """
        self.state = state
        self.observations = observations
        try:
            self.n_bands, self.n_obs = self.observations.shape
        except:
            raise ValueError, "Typically, obs should be (n_obs, n_bands)"
        self.mask = mask
        assert ( self.n_obs ) == mask.shape[0]
        self.state_grid = state_grid
        self.nt = self.state_grid.shape[0]
        self.emulators = emulators
        self.bu = bu
        self.band_pass = band_pass
        self.bw = bw

        
    
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
        x_params = np.empty ( ( len( x_dict.keys()), \
                self.nt ) )
        j = 0
        ii = 0
        
        the_derivatives = np.zeros ( ( len( x_dict.keys()), \
                self.nt ) )
        
        for param, typo in state_config.iteritems():
        
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
        istart_doy = self.state_grid[0]
        
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
                this_obsop, this_obs, this_extra = self.time_step ( \
                    this_obs_loc )
                this_cost, this_der, fwd_model, this_gradient = \
                    self.calc_mismatch ( this_obsop, x_params[:, itime], \
                    this_obs, self.bu, *this_extra )
                self.fwd_modelled_obs.append ( fwd_model ) # Store fwd model
                cost += this_cost
                the_derivatives[ :, itime] += this_der
            # Advance istart_doy to the end of this period
            istart_doy = tstep
            
        j = 0
        for  i, (param, typo) in enumerate ( state_config.iteritems()) :
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
        tag = np.round( self.mask[ this_loc, 1:].astype (np.int)/5.)*5
        tag = tuple ( (tag[:2].astype(np.int)).tolist() )
        this_obs = self.observations[ this_loc, :]
        return self.emulators[tag], this_obs, [ self.band_pass, self.bw ]
    
    def calc_mismatch ( self, gp, x, obs, bu, band_pass, bw ):
        this_cost, this_der, fwd, gradient = fwd_model ( gp, x, obs, bu, \
            band_pass, bw )
        return this_cost, this_der, fwd, gradient
    
    
    def der_der_cost ( self, x_dict, state_config, state, epsilon=1.0e-5 ):
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
        x_params = np.empty ( ( len( x_dict.keys()), self.nt ) )
        j = 0
        ii = 0
        the_derivatives = np.zeros ( ( len( x_dict.keys()), self.nt ) )
        param_pattern = np.zeros ( len( state_config.items()))
        for param, typo in state_config.iteritems():
        
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
        istart_doy = self.state_grid[0]
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
                
                this_obsop, this_obs, this_extra = self.time_step ( \
                    this_obs_loc )
                xs = x_params[:, itime]*1
                dummy, df_0, dummy_fwd, dummy_gradient = self.calc_mismatch ( this_obsop, \
                    xs, this_obs, self.bu, *this_extra )
                iloc = 0
                iiloc = 0
                for i,fin_diff in enumerate(param_pattern):
                    if fin_diff == 1: # FIXED
                        continue                    
                    xxs = xs[i]*1
                    xs[i] += epsilon
                    dummy, df_1, dummy_fwd, dummy_gradient = self.calc_mismatch ( this_obsop, \
                        xs, this_obs, self.bu, *this_extra )                    # Calculate d2f/d2x
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
        



