Quickstart
==========

Here is how to use the ETC via a worked example. A more detailed demonstration in Jupyter notebook form can be found in uvex-imager-etc/notebooks/example.ipynb. 


Set up the ETC with a source
----------------------------

The :class:`~uvex_imager_etc.etc.ETC` object can be initialized with a coordinate (an Astropy :class:`~astropy.coordinates.SkyCoord` object), an observation time (an Astropy :class:`~astropy.time.Time` object), and a source. A source can be defined either as a synphot :class:`~synphot.spectrum.SourceSpectrum`, or by passing a :class:`~astropy.units.Quantity` in flux or AB magnitude units, in which case the ETC will assume a flat source spectrum in those units. 

.. code-block:: python
 
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.time import Time
 
    from uvex_imager_etc.etc import ETC

    # A single 24th AB-magnitude source, observed at a chosen 
    # sky position and time (used to compute background rates).
    source = 24.0 * u.ABmag
    coord = SkyCoord(120.0, 15.0, unit=u.deg, frame="galactic")
    obstime = Time("2030-06-01 09:00:00")

    etc = ETC(
        source=source,
        coordinate=coord,
        obstime=obstime
    )
    etc.get_info()

If the coordinate or observation time are not set, they will default to values that produce a fairly typical sky background.

Perform ETC calculations
------------------------

The :class:`~uvex_imager_etc.etc.ETC` has a number of functions that can be called to perform sensitivity calculations for the provided source, coordinate and time inputs. 

:func:`~uvex_imager_etc.etc.get_snr` can be called either by specifying the number of standard UVEX survey visits (known as 'dwells') or by explicitly defining an exposure time and number of frames taken. 

.. code-block:: python

    snr_dwell = etc.get_snr(n_dwells=1, band="nuv")
    print(f"NUV SNR in a single UVEX dwell: {snr_dwell}")

    snr_obs = etc.get_snr(exptime=300 * u.s, n_frames=1, band="nuv")
    print(f"NUV SNR in a single 300s frame: {snr_obs}")

The reverse operation can be performed using :func:`~uvex_imager_etc.etc.get_dwells` to compute the number of dwells needed to reach the required depth, or :func:`~uvex_imager_etc.etc.get_exposure` to get the exposure time of a single observation. 

.. code-block:: python

    dwells = etc.get_dwells(snr=20, band="fuv")
    print(f"Number of dwells needed for SNR=20 in FUV: {dwells}")

    exptime = etc.get_exposure(snr=5, band="fuv")
    print(f"FUV exposure needed for SNR=5: {exptime}")

At a given location and time, :func:`~uvex_imager_etc.etc.limiting_mag` can be used to find the limiting magnitude at a given SNR for a certain number of dwells or exposure time. An input source is not required for this function, although it is for the above functions. 

.. code-block:: python

    lim_mag_dwells = etc.get_limiting_mag(snr=5, n_dwells=2, band='fuv')
    print(f"FUV limiting magnitude for SNR=5 after 2 dwells: {lim_mag_dwells}")

    lim_mag_exptime = etc.get_limiting_mag(snr=10, exptime=300*u.s, n_frames=6, band='nuv')
    print(f"NUV limiting magnitude for SNR=10 after 6 x 300s frames: {lim_mag_exptime}")

Multiple coordinates, observation times, and sources can be passed to the ETC at once. If any one of these properties has length 1, the same value will be used for all calculations. However, for any containing multiple entries, they must all have the same length so it is unambiguous which should be grouped together. For example, a single source object could be observed at three coordinates and three observation times (and three results will be provided by the above functions).

Setter functions allow setting different combinations of inputs, although will not allow an incompatible number to be set.

.. code-block:: python

   new_coord = SkyCoord([100.,120.,140.], [30.,15.,-40.], unit=u.deg, frame='galactic')
   new_obstime = Time(['2030-03-01','2030-04-01','2030-05-01'], scale='utc', format='iso')

   etc.set_coord(new_coord)
   etc.set_obstime(new_obstime)

Specify telescope version
-------------------------

The ETC will automatically use the most recent CALDB version in uvex_imager_etc/response_files, and we encourage regularly checking for new versions from the UVEX website. In order to replicate old results, you may wish to specify a particular version of UVEX, using the :class:`~uvex_imager_etc.etc.UVEX` class.

.. code-block:: python
 
    from uvex_imager_etc.uvex import UVEX
    uvex_old = UVEX(caldb='20260820_v0.1a')

    etc_old = ETC(
        source=source,
        coordinate=coord,
        obstime=obstime,
        telescope=uvex_old
    )
    etc_old.get_info()

See the :doc:`api` reference for the full set of methods.