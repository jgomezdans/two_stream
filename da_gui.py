import numpy as np
import matplotlib.pyplot as plt

# Lots of different places that widgets could come from...
try:
    from ipywidgets import interact, FloatSlider, IntSlider, Dropdown
except ImportError:
    try:
        from IPython.html.widgets import interact, FloatSlider, IntSlider, Dropdown
    except ImportError:
        try:
            from IPython.html.widgets import (interact,
                                              FloatSliderWidget as FloatSlider,
                                              IntSliderWidget as IntSlider)
        except ImportError:
            pass


from tip_helpers import retrieve_albedo


sites ="""AU-Tum
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
ZA-Kru""".split("\n")


@interact ( fluxnet_site=Dropdown(options=sites), year=IntSlider(min=2004,max=2013, step=1))
def plot_albedos ( fluxnet_site, year ):
    observations, mask, bu, passer_snow = retrieve_albedo(year, fluxnet_site,
                                                     [0.05, 0.07])
    passer = mask[:,1] == 1
    doys = mask[:, 0]
    plt.figure ( figsize=(12,6))
    plt.plot ( doys[passer], observations[passer, 0], 'o', label="Visible Albedo")
    plt.plot ( doys[passer], observations[passer, 1], 'o', label="Near-infrarred Albedo")
    plt.vlines ( doys[passer], observations[passer, 0] + 1.96*bu[passer,0],
                 observations[passer, 0] - 1.96 * bu[passer, 0])
    plt.vlines ( doys[passer], observations[passer, 1] + 1.96*bu[passer,1],
                 observations[passer, 1] - 1.96 * bu[passer, 1])
    plt.legend(loc="best", numpoints=1, fancybox=True, shadow=True)
    plt.ylabel("Bi-hemispherical reflectance [-]")
    plt.xlabel("DoY/%d" % year )
    plt.xlim ( 1, 368)
    plt.ylim ( 0, 0.7)
    ax1 = plt.gca()
    # Hide the right and top spines
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')


if __name__ == "__main__":
    print plot_albedos ( "DE-Geb", 2004)