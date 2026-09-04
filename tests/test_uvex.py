"""
Tests for uvex_imager_etc.uvex.UVEX

Covers CALDB discovery/selection, error handling for bad CALDB names, and
the derived telescope properties (AREA, PIXEL, bandpasses, read noise/dark
current).
"""
import os
import warnings

import numpy as np
import pytest
import astropy.units as u
from synphot import SpectralElement

from uvex_imager_etc.uvex import UVEX, response_files_dir


def _available_caldbs():
    return sorted(f for f in os.listdir(response_files_dir) if not f.startswith("."))


def _expected_latest_caldb():
    """
    Re-derive, independently of UVEX's own implementation, which CALDB
    "should" be selected by default: the directory (or directories) whose
    embedded YYYYMMDD date is most recent, tie-broken by the lexicographically
    greatest suffix (e.g. 'v0.1c' beats 'v0.1b' beats 'v0.1a').
    """
    caldbs = _available_caldbs()
    dates = [c[:8] for c in caldbs]
    latest_date = max(dates)
    candidates = sorted(c for c, d in zip(caldbs, dates) if d == latest_date)
    return candidates[-1], sum(d == latest_date for d in dates)


class TestCaldbSelection:
    def test_default_selects_the_most_recent_caldb(self):
        expected, n_matches = _expected_latest_caldb()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            t = UVEX()
        assert str(t.get_caldb()) == expected
        if n_matches > 1:
            # Multiple CALDBs share the latest date -> UVEX should warn about
            # the ambiguity it resolved.
            assert any("Multiple CALDBs" in str(w.message) for w in caught)

    def test_explicit_caldb_is_honored_without_warning(self, caldb_version):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            t = UVEX(caldb=caldb_version)
        assert str(t.get_caldb()) == caldb_version
        assert len(caught) == 0

    def test_unknown_caldb_raises_value_error(self):
        with pytest.raises(ValueError):
            UVEX(caldb="not_a_real_caldb_version")

    def test_get_caldb_matches_selected_version(self, telescope, caldb_version):
        assert str(telescope.get_caldb()) == caldb_version


class TestTelescopeProperties:
    def test_area_matches_epd(self, telescope):
        expected_area = np.pi * (telescope.EPD * 0.5) ** 2
        assert u.isclose(telescope.AREA, expected_area)

    def test_pixel_is_positive_solid_angle(self, telescope):
        assert telescope.PIXEL.unit.is_equivalent(u.arcsec**2)
        assert telescope.PIXEL.value > 0

    def test_bandpasses_are_spectral_elements(self, telescope):
        for attr in ("nuv_bandpass", "fuv_bandpass", "nuv_cherenkov_bandpass", "fuv_cherenkov_bandpass"):
            assert isinstance(getattr(telescope, attr), SpectralElement)

    @pytest.mark.parametrize("band", ["nuv", "fuv"])
    def test_read_noise_and_dark_current_present_and_positive(self, telescope, band):
        assert telescope.READ_NOISE[band].unit.is_equivalent(u.electron)
        assert telescope.READ_NOISE[band].value > 0
        assert telescope.DARK_CURRENT[band].unit.is_equivalent(u.electron / u.s)
        assert telescope.DARK_CURRENT[band].value > 0

    def test_npix_is_a_fixed_positive_constant(self, telescope):
        assert telescope.NPIX > 0

    def test_default_lya_kr_is_positive(self, telescope):
        assert telescope.lya_kr > 0
