import numpy as np
import warnings

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.stats import signal_to_noise_oir_ccd

from synphot import SourceSpectrum, Observation
from synphot.models import ConstFlux1D

from . import uvex
from . import backgrounds

class ETC():
    '''
        Class to hold ETC-related properties and perform ETC operations.
        
        Parameters
        ----------
        coordinate : SkyCoord
            Source coordinates as SkyCoord object.
            Defaults to an 'average' location 15° out of Galactic Plane
        
        obstime : Time
            Time of observation for each source.
            Defaults to arbitrary time of 2030-06-01 09:00:00
        
        source : Quantity or SourceSpectrum
            A flux Quantity (such as magnitude) or a synphot SourceSpectrum object
        
        telescope : UVEX
            UVEX object containing a particular telescope configuration
    '''
    def __init__(self, source=None, coordinate=None, obstime=None, telescope=None):
        # Standard observing dwell definition
        self.nuv_exposure = 300*u.s
        self.fuv_exposure = 900*u.s
        self.n_nuv = 3
        self.n_fuv = 1
        self.default_coord = SkyCoord(120., 15., unit=u.deg, frame='galactic')
        self.default_obstime = Time('2030-06-01 09:00:00', scale='utc', format='iso')
        
        # Initialize input counts
        self.n_source = 0
        self.n_coord = 0
        self.n_obstime = 0
        
        # Ingest sources
        if source is not None:
            self.set_source(source, regen=False)
        else:
            # No need to define a source for limiting magnitude calculations
            self.source = None
            self.source_info = 'None'
        
        # Set source locations (used for calculating background)
        # Default 'average' location 15-deg out of Galactic Plane
        if coordinate is None: coordinate = self.default_coord
        self.set_coord(coordinate, regen=False)
        
        # Set the observation times (used for calculating background)
        if obstime is None: obstime = self.default_obstime
        self.set_obstime(obstime, regen=False)
        
        if telescope is None: telescope = uvex.UVEX()
        self.set_telescope(telescope, regen=False)
        
        # Initialize source and background count rates
        self.source_count_rate = {}
        self.background_count_rate = {}
        
        # TODO: Add functionality to switch certain background effects on and off
        # Dark current, sky components, Cherenkov, scattered light?

    # Functions
    def get_info(self):
        '''
            Returns current information about ETC setup
        '''
        print(f'UVEX version: {self.telescope.get_caldb()}')
        if self.n_source > 1: n_s = f' x {self.n_source}'
        else: n_s = ''
        print(f'Source: {self.source_info}{n_s}')
        print(f'Source position: {self.coord}')
        print(f'Observation time: {self.obstime}')
    
    def get_source_count_rate(self, band='nuv'):
        '''
            Returns source count rate in given band
        '''
        if band not in self.source_count_rate: self._calc_source_count_rate()
        return self.source_count_rate[band]
    
    def get_background_count_rate(self, band='nuv'):
        '''
            Returns background count rate in given band
        '''
        if band not in self.background_count_rate: self._calc_background_count_rate()
        return self.background_count_rate[band]
    
    def _calc_source_count_rate(self):
        '''
            Calculate and set the count rate for all sources
        '''
        if self.source is None:
            raise ValueError("No source defined. Please use set_source() to define the input source.")
        
        nuv_rate, fuv_rate = np.array([]), np.array([])
        for s in self.source:
            # TODO: Make this more efficient
            nuv_obs = Observation(s, self.telescope.nuv_bandpass)
            nuv_rate = np.append(nuv_rate, nuv_obs.countrate(area=self.telescope.AREA).value)
            fuv_obs = Observation(s, self.telescope.fuv_bandpass)
            fuv_rate = np.append(fuv_rate, fuv_obs.countrate(area=self.telescope.AREA).value)
        self.source_count_rate['nuv'] = nuv_rate * u.electron / u.s
        self.source_count_rate['fuv'] = fuv_rate * u.electron / u.s
    
    def _calc_background_count_rate(self):
        '''
            Calculate and set the background rate for all observation locations and times
        '''
        # Calculate backgrounds
        self.background_count_rate['nuv'] = backgrounds.make_nuv_background(self.telescope, self.coord, self.obstime)
        self.background_count_rate['fuv'] = backgrounds.make_fuv_background(self.telescope, self.coord, self.obstime)
    
    def _req_source(self, k, exposure, bgd_rate, read_noise, neff):
        """
        Isolate source flux to get at least SNR of k in exposure seconds

        Parameters
        -----------
        k : float
            Desired SNR
        exposure: float
            Exposure in seconds
        bgd_rate : float
            Combined sky and dark current
        read_noise : float
            Read noise per pixel
        neff : float
            Effective number of pixels
        """
        c = neff * k**2 * (read_noise**2 + exposure*(bgd_rate))
        source =  (k**2 + np.sqrt(k**4 + 4*c))/ (2*exposure)
        return source * u.ct / u.s
        
        
    def _calc_exposure(self, k, src_rate, bgd_rate, read_noise, neff):
        """
        Compute the time to get to a given significance (k) given the source rate,
        the background rate, the read noise, and the number
        of effective background pixels. Inversion of the standard CCD SNR equation.
        
        Parameters
        -----------
        k : float
            Desired SNR
        src_rate : float
            Source count rate
        bgd_rate : float
            Combined sky and dark current
        read_noise : float
            Read noise per pixel
        neff : float
            Effective number of pixels
        """
        denom = 2 * src_rate**2

        nom1 = (k**2) * (src_rate + neff*bgd_rate)
        nom2 = ( k**4 *(src_rate + neff*bgd_rate)**2 +
                        4 * k**2 * src_rate**2 * neff * read_noise**2)**(0.5)
        exposure = (nom1 + nom2) / denom
        return exposure * u.s

    def get_snr(self, exptime=None, n_frames=None, n_dwells=None, band='nuv'):
        """
        Calculate the SNR of an observation of a source with UVEX.

        Parameters
        ----------
        exptime : Quantity
            Exposure time
        
        n_frames : int
            Number of exposures to stack
            
        n_dwells : int
            A specific number of standard UVEX survey dwells.
            This automatically sets exptime and n_frames: exptime and n_frame inputs are ignored in this case.
            Dwells are defined using ETC properties.
        
        band : 'nuv' or 'fuv'
            The UVEX band in which to calculate SNR
        
        Returns
        -------
        float array
            The signal to noise ratio
        """
        # Determine inputs
        band = band.lower()
        if not ((band == 'nuv') | (band == 'fuv')):
            raise ValueError(f"band must be 'nuv' or 'fuv'; got {band}")
        
        if n_dwells is not None:
            if band == 'nuv':
                exptime = self.nuv_exposure
                n_frames = self.n_nuv * n_dwells
            elif band == 'fuv':
                exptime = self.fuv_exposure
                n_frames = self.n_fuv * n_dwells
        else:
            if not isinstance(exptime, u.quantity.Quantity):
                raise ValueError("Exptime must be a Quantity.")
            if not isinstance(n_frames, int):
                raise ValueError("n_frames must be a positive integer.")
        
        # Load appropriate read noise and dark current from telescope
        dark_current = self.telescope.DARK_CURRENT[band]
        read_noise = self.telescope.READ_NOISE[band]
        npix = self.telescope.NPIX
        
        # Trigger generation of count rates if necessary
        if band not in self.source_count_rate: self._calc_source_count_rate()
        if band not in self.background_count_rate: self._calc_background_count_rate()
        
        source = self.source_count_rate[band]
        sky = self.background_count_rate[band]
    
        snr = signal_to_noise_oir_ccd(exptime.to(u.s).value,
            source.value,
            sky.value,
            dark_eps=dark_current.value,
            rd=read_noise.value,
            npix=npix,
            gain=1.)
        snr *= np.sqrt(n_frames)
    
        return snr
    
    
    def get_limiting_mag(self, snr=5., exptime=None, n_frames=None, n_dwells=None, band='nuv'):
        """
        Get the limiting magnitude at a certain location and time for given SNR and exposure
        
        Does not require any source information to be loaded - length of output will be relative
        to length of coord/obstime, not number of sources

        Parameters
        ----------
        snr : float
            Desired signal-to-noise ratio
        
        exptime : Quantity
            Exposure time
        
        n_frames : int
            Number of exposures to stack
            
        n_dwells : int
            Sets exptime and n_frames to a specific number of dwells
            exptime and n_frame inputs are ignored in this case
            Dwells are defined using ETC properties
        
        band : 'nuv' or 'fuv'
            The UVEX band in which to calculate limiting magnitude
        
        Returns
        -------
        m_limit : float array
            The limiting magnitude for each position/observation time
        """
        # Determine inputs
        band = band.lower()
        if not ((band == 'nuv') | (band == 'fuv')):
            raise ValueError(f"band must be 'nuv' or 'fuv'; got {band}")
        
        if band == 'nuv': bandpass = self.telescope.nuv_bandpass
        elif band == 'fuv': bandpass = self.telescope.fuv_bandpass
        
        if n_dwells is not None:
            if band == 'nuv':
                exptime = self.nuv_exposure
                n_frames = self.n_nuv * n_dwells
            elif band == 'fuv':
                exptime = self.fuv_exposure
                n_frames = self.n_fuv * n_dwells
        else:
            if not isinstance(exptime, u.quantity.Quantity):
                raise ValueError("Exptime must be a Quantity.")
            if not isinstance(n_frames, int):
                raise ValueError("n_frames must be a positive integer.")
        
        # Load appropriate read noise and dark current from telescope
        dark_current = self.telescope.DARK_CURRENT[band].value
        read_noise = self.telescope.READ_NOISE[band].value
        npix = self.telescope.NPIX
        
        # Trigger generation of count rates if necessary
        if band not in self.background_count_rate: self._calc_background_count_rate()
        
        # Get reference count rate
        m_ref = 22*u.ABmag
        sp = SourceSpectrum(ConstFlux1D, amplitude=m_ref)
        obs_band = Observation(sp, bandpass)
        ref_rate = obs_band.countrate(area=self.telescope.AREA)
        
        # Get the required source rate per exposure
        per_exp_snr = snr/np.sqrt(n_frames)
        req_rate = self._req_source(per_exp_snr, exptime.to(u.s).value,
                                    self.background_count_rate[band].value + dark_current,
                                    read_noise, npix)
        ratio = req_rate / ref_rate
        m_limit = m_ref - (2.5*np.log10(ratio))*u.mag
        
        return m_limit
    
    
    def get_exposure(self, snr=5., band='nuv'):
        """
        Get the required exposure time in a single observation to detect a
        point source to a given SNR

        Parameters
        ----------
        snr : float
            Desired signal-to-noise ratio
        
        band : 'nuv' or 'fuv'
            The UVEX band in which to calculate exposure time/dwells
        
        Returns
        -------
        exptime : float array
            The required exposure time in seconds for each source
        """
        # Determine inputs
        band = band.lower()
        if not ((band == 'nuv') | (band == 'fuv')):
            raise ValueError(f"band must be 'nuv' or 'fuv'; got {band}")
        
        # Load appropriate read noise and dark current from telescope
        dark_current = self.telescope.DARK_CURRENT[band].value
        read_noise = self.telescope.READ_NOISE[band].value
        npix = self.telescope.NPIX
        
        # Trigger generation of count rates if necessary
        if band not in self.source_count_rate: self._calc_source_count_rate()
        if band not in self.background_count_rate: self._calc_background_count_rate()
        
        # Get the required single-exposure time
        exptime = self._calc_exposure(snr, self.source_count_rate[band].value,
                                      self.background_count_rate[band].value + dark_current,
                                      read_noise, npix)
        
        return exptime

    def get_dwells(self, snr=5., band='nuv'):
        """
        Get the required number of standard UVEX observing dwells to detect a
        point source to a given SNR

        Parameters
        ----------
        snr : float
            Desired signal-to-noise ratio
        
        band : 'nuv' or 'fuv'
            The UVEX band in which to calculate the needed number of dwells
        
        Returns
        -------
        n_dwells : int array
            The required number of dwells for each source
        """
        # Determine inputs
        band = band.lower()
        if not ((band == 'nuv') | (band == 'fuv')):
            raise ValueError(f"band must be 'nuv' or 'fuv'; got {band}")
        if band == 'nuv':
            exposure = self.nuv_exposure
        elif band == 'fuv':
            exposure = self.fuv_exposure
        
        # Get the required number of standard dwells
        snr_per_frame = self.get_snr(exposure, n_frames=1, band=band)
        n_frames = np.ceil((snr / snr_per_frame)**2)
        
        if band == 'fuv':
            n_dwells = n_frames
        else:
            n_dwells = np.ceil(n_frames / 3)
        
        return n_dwells
    
    
    def set_source(self, source, regen=True):
        '''
        Setter for the input source
        
        Parameters
        ----------
        source : Quantity or SourceSpectrum
            A flux Quantity (such as magnitude) or a synphot SourceSpectrum object
        
        regen : bool
            Whether to immediately regenerate source and background count rates.
            Defaults to True; only False in case of ETC initialization.
        '''
        if isinstance(source, u.quantity.Quantity):
            # Source is provided as a quantity - treat as a flat spectrum
            # TODO: Add capacity to generate a range of spectrum types for given magnitudes
            if source.size == 1:
                self.source = [SourceSpectrum(ConstFlux1D, amplitude=source)]
                self.n_source = 1
            else:
                self.source = [SourceSpectrum(ConstFlux1D, amplitude=s) for s in source]
                self.n_source = len(source)
                # Check against coord and obstime - if number of sources provided is different
                # to a pre-existing number of coord or obstime > 1, warn and reset coord and obstime to defaults
                if (self.n_coord > 1) and (self.n_coord != self.n_source):
                    warnings.warn("Incompatible number of coordinates for this number of sources; resetting to default coordinates")
                    self.set_coord(self.default_coord, regen=False)
                if (self.n_obstime > 1) and (self.n_obstime != self.n_source):
                    warnings.warn("Incompatible number of obs times for this number of sources; resetting to default obs times")
                    self.set_obstime(self.default_obstime, regen=False)
            self.source_info = f'Constant spectrum at {source}'
        elif isinstance(source, SourceSpectrum):
            # Directly assign the spectrum
            self.source = [source]
            self.n_source = 1
            self.source_info = 'User-defined spectrum'
        elif isinstance(source, list) | isinstance(source, np.ndarray):
            if isinstance(source[0], SourceSpectrum):
                # Directly assign list/array of spectra
                self.source = source
                self.n_source = len(source)
                # Check against coord and obstime - if number of sources provided is different
                # to a pre-existing number of coord or obstime > 1, warn and reset coord and obstime to defaults
                if (self.n_coord > 1) and (self.n_coord != self.n_source):
                    warnings.warn("Incompatible number of coordinates for this number of sources; resetting to default coordinates")
                    self.set_coord(self.default_coord, regen=False)
                if (self.n_obstime > 1) and (self.n_obstime != self.n_source):
                    warnings.warn("Incompatible number of obs times for this number of sources; resetting to default obs times")
                    self.set_obstime(self.default_obstime, regen=False)
            self.source_info = 'User-defined spectra'
        else:
            raise ValueError("Source must be a flux Quantity or synphot SourceSpectrum (or list thereof)")
        
        if regen:
            # Regenerate source and background count rates
            self._calc_source_count_rate()
            self._calc_background_count_rate()
    
    def set_coord(self, coordinate, regen=True):
        '''
        Setter for source coordinates
        
        Parameters
        ----------
        coordinate : SkyCoord
            Source coordinates as SkyCoord object
        
        regen : bool
            Whether to immediately regenerate background count rates
            Defaults to True; only False in case of ETC initialization
        '''
        if not isinstance(coordinate, SkyCoord):
            raise ValueError("Coordinate must be a `SkyCoord` object.")
        if coordinate.size > 1 and self.n_source > 1 and coordinate.size != self.n_source:
                raise ValueError("Length of coordinate must be 1 or equal to number of sources.")
        if coordinate.size > 1 and self.n_obstime > 1 and coordinate.size != self.n_source:
                raise ValueError("Length of coordinate must be 1 or equal to number of obs times.")
        self.coord = coordinate
        self.n_coord = coordinate.size
        
        if regen:
            # Regenerate background count rates
            self._calc_background_count_rate()
    
    def set_obstime(self, obstime, regen=True):
        '''
        Setter for observation times
        
        Parameters
        ----------
        obstime : Time
            Time of observation for each source
        
        regen : bool
            Whether to immediately regenerate background count rates.
            Defaults to True; only False in case of ETC initialization
        '''
        if not isinstance(obstime, Time):
            raise ValueError("Obstime must be a `Time` object.")
        if obstime.size > 1 and self.n_source > 1 and obstime.size != self.n_source:
            raise ValueError("Length of obstime must be 1 or equal to number of sources.")
        if obstime.size > 1 and self.n_coord > 1 and obstime.size != self.n_coord:
            raise ValueError("Length of obstime must be 1 or equal to number of coordinates.")
        self.obstime = obstime
        self.n_obstime = self.obstime.size
            
        if regen:
            # Regenerate background count rates
            self._calc_background_count_rate()
    
    def set_telescope(self, telescope, regen=True):
        '''
        Setter for UVEX telescope configuration
        
        Parameters
        ----------
        telescope : UVEX
            UVEX configuration object
        
        regen : bool
            Whether to immediately regenerate source and background count rates.
            Defaults to True; only False in case of ETC initialization
        '''
        if not isinstance(telescope, uvex.UVEX):
            raise ValueError("Telescope must be a `UVEX` object.")
        self.telescope = telescope
    
        if regen:
            # Regenerate source and background count rates
            self._calc_source_count_rate()
            self._calc_background_count_rate()
