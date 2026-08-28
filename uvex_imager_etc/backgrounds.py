'''
    Contains utility functions to create sky backgrounds
    for the imaging channels
'''
import os
import numpy as np
from numpy import pi as PI
import warnings

from synphot import GaussianFlux1D, ConstFlux1D, Empirical1D, SourceSpectrum, Observation
from scipy.interpolate import RectBivariateSpline

from astropy.coordinates import get_sun, GeocentricTrueEcliptic
import astropy.units as u
from astropy.constants import h, c

background_dir = os.path.join(os.path.dirname(__file__), 'background_data')

def make_nuv_background(uvex, coord, obstime, diag=False):
    """
    Function to generate the standard total per-pixel background
    for an NUV observation, incorporating Galactic and Zodiacal contributions.
    
    Parameters
    ----------
    uvex : UVEX
        UVEX object containing telescope configuration
    
    coord : SkyCoord
        Coordinates for the observation
        
    obstime : Time
        Observation time
    
    Returns
    -------
    sky_nuv : Quantity
        The NUV per-pixel background count rate in e/s
    """
    nuv_band = uvex.nuv_bandpass
    
    sky_coord = coord.icrs
    gal_coord = coord.galactic
    
    zodi_spec = make_zodi_spec(uvex, sky_coord, obstime)
    galactic_spec = make_galactic_spec(uvex, gal_coord.b, 'nuv')
    
    zodi_rate, gal_rate = np.array([]) * u.ct / u.s, np.array([]) * u.ct / u.s
    n_back = np.maximum(coord.size, obstime.size)
    for i in range(n_back):
        # TODO: Make this more efficient
        nuv_zodi = Observation(zodi_spec[i], nuv_band)
        zodi_rate = np.append(zodi_rate, nuv_zodi.countrate(area=uvex.AREA))
        
        if i < coord.size:
            # Only generate multiple gal_rate if multiple coordinates are present
            nuv_galactic = Observation(galactic_spec[i], nuv_band)
            gal_rate = np.append(gal_rate, nuv_galactic.countrate(area=uvex.AREA))
    
    # Response to Cherenkov photons
    wave = np.arange(1000, 10000) * u.AA
    cherenkov_spectrum = gen_cherenkov_spectrum(uvex, wave)
    received_cherenkov_nuv = uvex.nuv_cherenkov_bandpass(wave) * cherenkov_spectrum
    cherenkov_rate = received_cherenkov_nuv.sum() * u.ct
    
    # Total NUV rate
    sky_nuv = zodi_rate + gal_rate + cherenkov_rate

    if diag:
        print(f'Diagnostics for make_nuv_background')
        print(f'---')
        print(f'Zodi rate: ', (', ').join([f'{z:8.4e}' for z in zodi_rate]))
        print(f'Galactic Diffuse rate: ', (', ').join([f'{g:8.4e}' for g in gal_rate]))
        print(f'Cherenkov Rate: {cherenkov_rate:8.4e}')
        print(f'Total NUV background:', (', ').join([f'{s:8.2e}' for s in sky_nuv]))
        print(f'---')
    
    return sky_nuv * u.electron / u.ct


def make_fuv_background(uvex, coord, obstime, diag=False):
    """
    Function to generate the standard total background for a FUV observation,
    incorporating Lyman-alpha, Galactic and Zodiacal contributions
    
    Parameters
    ----------
    uvex : UVEX
        UVEX object containing telescope configuration
    
    coord : SkyCoord
        Coordinates for the observation
        
    obstime : Time
        Observation time
    
    Returns
    -------
    sky_nuv : Quantity
        The NUV per-pixel background count rate in e/s
    """
    fuv_band = uvex.fuv_bandpass
    
    sky_coord = coord.icrs
    gal_coord = coord.galactic
    
    lya_spec = make_lyman_spec(uvex)
    fuv_lya = Observation(lya_spec, fuv_band)
    lya_rate = fuv_lya.countrate(area=uvex.AREA)
    
    zodi_spec = make_zodi_spec(uvex, sky_coord, obstime)
    galactic_spec = make_galactic_spec(uvex, gal_coord.b, 'fuv')
    
    zodi_rate, gal_rate = np.array([]) * u.ct / u.s, np.array([]) * u.ct / u.s
    n_back = np.maximum(coord.size, obstime.size)
    for i in range(n_back):
        # TODO: Make this more efficient
        fuv_zodi = Observation(zodi_spec[i], fuv_band, force='extrap')
        zodi_rate = np.append(zodi_rate, fuv_zodi.countrate(area=uvex.AREA))

        if i < coord.size:
            # Only generate multiple gal_rate if multiple coordinates are present
            fuv_galactic = Observation(galactic_spec[i], fuv_band)
            gal_rate = np.append(gal_rate, fuv_galactic.countrate(area=uvex.AREA))
    
    # Response to Cherenkov photons
    wave = np.arange(1000, 10000) * u.AA
    cherenkov_spectrum = gen_cherenkov_spectrum(uvex, wave)
    received_cherenkov_fuv = uvex.fuv_cherenkov_bandpass(wave) * cherenkov_spectrum
    cherenkov_rate = received_cherenkov_fuv.sum() * u.ct
    
    # Total FUV rate
    sky_fuv = lya_rate + zodi_rate + gal_rate + cherenkov_rate
    
    if diag:
        print(f'Diagnostics for make_fuv_background')
        print(f'---')
        print(f'Zodi rate: ', (', ').join([f'{z:8.4e}' for z in zodi_rate]))
        print(f'Galactic Diffuse Rate: ', (', ').join([f'{g:8.4e}' for g in gal_rate]))
        print(f'Lyman-alpha Rate: {lya_rate:8.4e}')
        print(f'Cherenkov rate: {cherenkov_rate:8.4e}')
        print(f'Total FUV background: ', (', ').join([f'{s:8.2e}' for s in sky_fuv]))
        
        print(f'---')
    
    return sky_fuv * u.electron / u.ct


def gen_cherenkov_spectrum(uvex, wave):
    """
    Generates a Cherenkov spectrum.
    
    Many things are currently hard coded and will
    need to be revised later. The big one is the particle flux, which is currently
    pegged at 5 particles per cm2 per s.
    
    TODO: Get hard-coded parameters from uvex_response
    
    Parameters
    ----------
    uvex : UVEX
        UVEX object containing telescope configuration
    """
    
    # For MgF2 n=1.38?
    # n=1.5 is Fused silica (good for the dichroic)
    n_dichroic = 1.5
    
    # This is the conversion scale factor for the dichroic, respectively
    scale_dichroic = (2 * PI / 137) * (1 - 1/(n_dichroic**2))
    
    # Dimensions of the dichroic
    dichroic_width = 21 * u.cm
    dichroic_length = 26 * u.cm
    dichroic_thick = 1 * u.cm
    dichroic_distance = 10 * u.cm
    
    # Particle flux
    flux = 5 / (u.cm**2 * u.s)

    # Particle flux, assume face on, through the dichroic
    dichroic_area = dichroic_width * dichroic_length
    dichroic_particles = flux * dichroic_area
    
    pix_area = (uvex.PIX_UM.to(u.cm))**2
    sphere_area = 4 * PI * (dichroic_distance**2)
    
    # Fraction collected per pixel (assuming spherical emission from distance of dichroic)
    pix_scaling = pix_area / sphere_area

    dl = 1 * u.AA
    spec_dichroic = (dl.to(u.cm)) * scale_dichroic / (wave.to(u.cm))**2
    spec_dichroic *= pix_scaling * dichroic_particles * dichroic_thick
    
    return spec_dichroic


def make_lyman_spec(uvex, kr=None):
    """
    Generate Lyman-Alpha emission at 1216 Angstroms.
    
    R == 1e6 / (4pi) ph / cm2 / s / sr
    == 3.15 × 10−17 erg cm-2 s-1 arcsec-2 per COS handbook

    Parameters
    ----------
    uvex : UVEX
        UVEX object containing telescope configuration
    
    kr : float
        Default is kr=2, or 2e3 R of LyAlpha.
        Set here to override default value

    Returns
    -------
    SourceSpectrum
    """
    if kr is None: kr = uvex.lya_kr
    
    mean = 1216 * u.AA
    ph_ergs = (h * c / mean).to(u.erg)
    
    R = kr*1e3 * 3.15e-17 * u.erg / u.cm**2 / u.s / u.arcsec**2
    R_pixel = R * uvex.PIXEL
    #ph_flux = R_pixel / (ph_ergs * 1*u.AA)
    
    # Apply scattered light scaling
    R_pixel *= (1. + uvex.scattered_light_scaling_lya)
    
    return SourceSpectrum(GaussianFlux1D, mean=mean, fwhm=0.1*u.AA, total_flux=R_pixel)


def make_galactic_spec(uvex, lat, band):
    """
    Takes as input the galactic latitude and generates a flat (in photon units)
    spectrum with flux scaled from Murthy (2014)
    
    NOTE: Not applicable for observations in the plane

    Parameters
    -----------
    uvex : UVEX
        UVEX object containing telescope configuration
    
    lat: Quantity or array of Quantities
        Galactic latitude in angular units
        
    band : string
        'nuv' or 'fuv'
    
    Returns
    -------
    SourceSpectrum object
    """
    if np.any(np.abs(lat) < 15*u.deg):
        warnings.warn("Galactic background invalid for observations in the plane")
    
    # Get the scaling for each latitude
    south = lat < -0*u.deg
    north = lat >= 0*u.deg

    gal_flux = np.zeros(lat.shape)
    if band == 'fuv':
        gal_flux[north] = 93.4 + 133.2 / np.sin(np.abs(lat[north]))
        gal_flux[south] = -205.5 + 401.8 / np.sin(np.abs(lat[south]))
    elif band == 'nuv':
        gal_flux[north] = 257.5 + 185.1 / np.sin(np.abs(lat[north]))
        gal_flux[south] = 66.7 + 356.3 / np.sin(np.abs(lat[south]))

    gal_flux *= u.ph / (u.cm**2 * u.sr * u.s * u.AA)
    
    # Convert to per-pixel units
    gal_flux = gal_flux.to(u.ph /(u.cm**2 * u.Angstrom * u.arcsec**2 * u.s)) * uvex.PIXEL
    
    # Apply scattered light scaling
    gal_flux *= (1. + uvex.scattered_light_scaling_galactic)
    
    if gal_flux.size == 1: gal_flux = [gal_flux]
    return [SourceSpectrum(ConstFlux1D, amplitude=f) for f in gal_flux]


def make_zodi_spec(uvex, coord, obstime):
    """
    Return the Zodiacal spectrum appropriately scaled for the given coordinates and time
    
    Parameters
    -----------
    uvex : UVEX
        UVEX object containing telescope configuration
    
    coord : SkyCoord
        
    obstime : Time
        
    band : string
        'nuv' or 'fuv'
    
    Returns
    -------
    SourceSpectrum object
    """
    # Convert to ecliptic coords:
    sun = get_sun(obstime).transform_to(GeocentricTrueEcliptic(equinox=obstime))
    target = coord.transform_to(GeocentricTrueEcliptic(equinox=obstime))
    
    lat = np.abs(target.lat.deg)
    high_lat = lat > 75
    low_lat = lat <= 75
    
    # Compute longitude wrt the sun:
    lon = np.abs((target.lon - sun.lon).wrap_at(180 * u.deg).deg)
    
    # Set Zodi scaling
    zodi_model = load_zodi_spatial()
    scale = np.zeros(lat.shape)
    scale[high_lat] = 72
    scale[low_lat] = scale[low_lat] = zodi_model(lat[low_lat], lon[low_lat])

    if scale.size == 1: scale = [scale]
    return zodi_spec(uvex, scale=scale)


def load_zodi_spatial():
    """
    Loads the 2D spatial map of the Zodiacal background
    
    Data from Leinert 1997, table 17
    """
    data = np.genfromtxt(os.path.join(background_dir, 'Leinert97_table17.txt'))
    lon = data[1:, 0]
    lat = data[0, 1:]
    zodi = data[1:, 1:]
    
    model_t = RectBivariateSpline(lat, lon, zodi.T, kx=1, ky=1)
    model = lambda xnew, ynew: model_t.ev(xnew, ynew)

    return model


def zodi_spec(uvex, scale = np.array([77])):
    """
    Creates an appropriately-scaled Zodiacal light spectrum
    
    From here
    https://cads.iiap.res.in/tools/zodiacalCalc/Documentation

    They provied the tabulated Zodi spectrum:
    https://cads.iiap.res.in/download/simulations/scaled_zodiacal_spec.txt

    From here
    Colina et al
    http://adsabs.harvard.edu/abs/1996AJ....112..307C

    According to this, the flux has been scaled so that at 5000 Ang
    the flux has units of 252 W / m2 / sr / micon

    This takes as input the flux density as read off from Table 17 of
    https://aas.aanda.org/articles/aas/pdf/1998/01/ds1449.pdf

    scale = scale in units of [1e-8 W / m2 / sr / micron at 500 nm]
    
    Default is for polar zodiacal emission, which is 77 in the above units.

    Toward the ecliptic plane this number can grow to be >1000

    For a Sun avoidance of 45 degrees this looks like a value of 200 - 900
    based strongly on the heliocentric longitdue. However, if you try
    72, 300, and 1000 it looks like you'll probably span this space.
    
    Optional Parameters
    -------------------
    scale : array
        See above for definition. Default is 77 (suitable for NEP)

    Returns
    -------
    SourceSpectrum object
    """
    # Load Zodi spectrum
    data = np.genfromtxt(os.path.join(background_dir, 'scaled_zodiacal_spec.txt'))
    wave = data[:, 0] * u.AA
    
    # Scale term:
    scale = scale * u.ph /(u.cm**2 * u.Angstrom * u.sr * u.s)
    
    flux = data[:, 1].reshape(1,len(data[:, 1])).repeat(len(scale),0) * scale.reshape(len(scale),1)

    # Convert to per-pixel units
    flux = flux.to(u.ph /(u.cm**2 * u.Angstrom * u.arcsec**2 * u.s)) * uvex.PIXEL
    
    # Apply scattered light scaling
    flux *= (1. + uvex.scattered_light_scaling_zodi)

    return [SourceSpectrum(Empirical1D, points=wave, lookup_table=f) for f in flux]

