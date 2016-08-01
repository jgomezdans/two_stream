# Experiments

1. Single TIP inversion
2. Single TIP inversion with weakened prior
    2.1. No soil correlation
3. Regularised TIP inversion
    3.1 Change regularisation for LAI (too high, too low, 
    just right)
4. Use previous year(s) as prior

# TIP package 
The 2stream inversion package (TIP) from JRC. Emulated. Regularised. Improved.


## The TwoStream Fortran code

The 2stream fortran code has been obtained from the JRC site. You can create a ``f2py`` Python bindings to the code by issuing the following commands in the shell:

    # Creates the bindings interface file
    f2py --overwrite-signature -m TwoSInterface -h TwoSInterface.pyf *.f90 only: twostream_solver
    # Compiles the so file so that Python knows what to do. Options to make it fast are probably unnecessary
    f2py -c --fcompiler=gnu95 --opt="-O3" --arch="-march=native" --f90exec=/usr/bin/gfortran --f77exec=/usr/bin/gfortran TwoSInterface.pyf *.f90

In Python, you'd call the library by importing the ``twostream_solver`` function:

    from TwoSInterface import twostream_solver

where you need to pass it:


            real(kind=kind(1.0d0)) intent(in) :: leaf_reflectance
            real(kind=kind(1.0d0)) intent(in) :: leaf_transmittance
            real(kind=kind(1.0d0)) intent(in) :: background_reflectance
            real(kind=kind(1.0d0)) intent(in) :: true_lai
            real(kind=kind(1.0d0)) intent(in) :: structure_factor_zeta
            real(kind=kind(1.0d0)) intent(in) :: structure_factor_zetastar
            real(kind=kind(1.0d0)) intent(in) :: sun_zenith_angle_degrees

and you would get in return...

            real(kind=kind(1.0d0)) intent(out) :: collim_alb_tot
            real(kind=kind(1.0d0)) intent(out) :: collim_tran_tot
            real(kind=kind(1.0d0)) intent(out) :: collim_abs_tot
            real(kind=kind(1.0d0)) intent(out) :: isotrop_alb_tot
            real(kind=kind(1.0d0)) intent(out) :: isotrop_tran_tot
            real(kind=kind(1.0d0)) intent(out) :: isotrop_abs_tot


