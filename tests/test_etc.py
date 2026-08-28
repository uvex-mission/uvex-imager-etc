"""
Tests for uvex_imager_etc.etc.ETC

Organized around:
  * construction / input validation (source, coordinate, obstime, telescope)
  * setters and their consistency-checking warnings
  * the core calculations (get_snr, get_exposure, get_limiting_mag, get_dwells)
  * get_info() output

Where possible, numerical calculations are checked via self-consistency
(e.g. get_exposure() followed by get_snr() should return to the requested
SNR) and monotonicity (brighter source / longer exposure -> higher SNR)
rather than hard-coded reference values, since those are robust to minor
changes in the underlying CALDB while still catching real regressions.
"""
import numpy as np
import pytest
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
from synphot import SourceSpectrum
from synphot.models import ConstFlux1D

from uvex_imager_etc.etc import ETC


# ---------------------------------------------------------------------------
# Construction / input validation
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_construction_has_no_source(self, telescope):
        etc = ETC(telescope=telescope)
        assert etc.source is None
        assert etc.n_source == 0
        assert etc.source_info == "None"

    def test_default_coordinate_and_obstime_are_used(self, telescope):
        etc = ETC(telescope=telescope)
        assert etc.n_coord == 1
        assert etc.n_obstime == 1

    def test_scalar_quantity_source(self, telescope, const_source):
        etc = ETC(source=const_source, telescope=telescope)
        assert etc.n_source == 1
        assert "Constant spectrum" in etc.source_info

    def test_array_quantity_source(self, telescope, source_array):
        etc = ETC(source=source_array, telescope=telescope)
        assert etc.n_source == len(source_array)

    def test_sourcespectrum_source(self, telescope):
        spec = SourceSpectrum(ConstFlux1D, amplitude=22 * u.ABmag)
        etc = ETC(source=spec, telescope=telescope)
        assert etc.n_source == 1
        assert etc.source_info == "User-defined spectrum"

    def test_list_of_sourcespectra(self, telescope):
        specs = [
            SourceSpectrum(ConstFlux1D, amplitude=22 * u.ABmag),
            SourceSpectrum(ConstFlux1D, amplitude=23 * u.ABmag),
        ]
        etc = ETC(source=specs, telescope=telescope)
        assert etc.n_source == 2

    def test_invalid_source_type_raises(self, telescope):
        with pytest.raises(ValueError):
            ETC(source=123, telescope=telescope)

    def test_invalid_telescope_type_raises(self):
        with pytest.raises(ValueError):
            ETC(telescope="not-a-telescope")

    def test_multi_coord_and_obstime_are_accepted(self, telescope, multi_coord, multi_obstime, const_source):
        etc = ETC(
            source=const_source,
            coordinate=multi_coord,
            obstime=multi_obstime,
            telescope=telescope,
        )
        assert etc.n_coord == len(multi_coord)
        assert etc.n_obstime == len(multi_obstime)


# ---------------------------------------------------------------------------
# Setters
# ---------------------------------------------------------------------------

class TestSetters:
    def test_set_coord_rejects_non_skycoord(self, etc_with_source):
        with pytest.raises(ValueError):
            etc_with_source.set_coord("not a coordinate")

    def test_set_coord_rejects_length_mismatch(self, telescope, source_array):
        etc = ETC(source=source_array, telescope=telescope)  # n_source == 4
        bad_coord = SkyCoord([1.0, 2.0], [3.0, 4.0], unit=u.deg, frame="galactic")  # length 2
        with pytest.raises(ValueError):
            etc.set_coord(bad_coord)

    def test_set_obstime_rejects_non_time(self, etc_with_source):
        with pytest.raises(ValueError):
            etc_with_source.set_obstime("not a time")

    def test_set_obstime_rejects_length_mismatch(self, telescope, source_array):
        etc = ETC(source=source_array, telescope=telescope)  # n_source == 4
        bad_obstime = Time(["2030-01-01", "2030-02-01"], format="iso")  # length 2
        with pytest.raises(ValueError):
            etc.set_obstime(bad_obstime)

    def test_set_telescope_rejects_wrong_type(self, etc_with_source):
        with pytest.raises(ValueError):
            etc_with_source.set_telescope(object())

    def test_set_source_with_incompatible_coord_count_warns_and_resets(self, telescope, multi_coord, source_array):
        etc = ETC(coordinate=multi_coord, telescope=telescope)  # n_coord == 3, no source yet
        assert etc.n_coord == 3
        with pytest.warns(UserWarning, match="Incompatible number of coordinates"):
            etc.set_source(source_array)  # n_source == 4 != n_coord == 3
        # Coordinates should have been reset back to the single default coordinate.
        assert etc.n_coord == 1

    def test_set_source_updates_count_rate(self, telescope):
        etc = ETC(source=20.0 * u.ABmag, telescope=telescope)
        bright_rate = etc.get_source_count_rate(band="nuv")[0]
        etc.set_source(25.0 * u.ABmag)
        faint_rate = etc.get_source_count_rate(band="nuv")[0]
        assert bright_rate > faint_rate


# ---------------------------------------------------------------------------
# get_snr / get_exposure / get_limiting_mag / get_dwells
# ---------------------------------------------------------------------------

class TestSnrAndExposure:
    @pytest.mark.parametrize("method,kwargs", [
        ("get_snr", dict(exptime=300 * u.s, n_frames=1)),
        ("get_snr", dict(n_dwells=1)),
        ("get_exposure", dict(snr=5.0)),
        ("get_limiting_mag", dict(snr=5.0, n_dwells=1)),
        ("get_dwells", dict(snr=5.0)),
    ])
    def test_invalid_band_raises(self, etc_with_source, method, kwargs):
        with pytest.raises(ValueError):
            getattr(etc_with_source, method)(band="not-a-band", **kwargs)

    def test_band_argument_is_case_insensitive(self, etc_with_source):
        lower = etc_with_source.get_snr(n_dwells=1, band="nuv")
        upper = etc_with_source.get_snr(n_dwells=1, band="NUV")
        assert lower == pytest.approx(upper)

    def test_get_snr_requires_quantity_exptime(self, etc_with_source):
        with pytest.raises(ValueError):
            etc_with_source.get_snr(exptime=300, n_frames=1, band="nuv")  # missing units

    def test_get_snr_requires_integer_n_frames(self, etc_with_source):
        with pytest.raises(ValueError):
            etc_with_source.get_snr(exptime=300 * u.s, n_frames=1.5, band="nuv")

    @pytest.mark.parametrize("band", ["nuv", "fuv"])
    def test_snr_increases_with_exposure_time(self, etc_with_source, band):
        snr_short = etc_with_source.get_snr(exptime=100 * u.s, n_frames=1, band=band)
        snr_long = etc_with_source.get_snr(exptime=1000 * u.s, n_frames=1, band=band)
        assert snr_long > snr_short

    @pytest.mark.parametrize("band", ["nuv", "fuv"])
    def test_snr_increases_with_n_frames(self, etc_with_source, band):
        snr_one = etc_with_source.get_snr(exptime=300 * u.s, n_frames=1, band=band)
        snr_many = etc_with_source.get_snr(exptime=300 * u.s, n_frames=9, band=band)
        assert snr_many > snr_one
        # Combining N identical frames should scale SNR by sqrt(N).
        assert snr_many / snr_one == pytest.approx(3.0, rel=1e-6)

    @pytest.mark.parametrize("band", ["nuv", "fuv"])
    @pytest.mark.parametrize("target_snr", [3.0, 8.0, 20.0])
    def test_get_exposure_is_consistent_with_get_snr(self, etc_with_source, band, target_snr):
        exptime = etc_with_source.get_exposure(snr=target_snr, band=band)
        assert exptime.unit.is_equivalent(u.s)
        assert exptime.value > 0
        recovered_snr = etc_with_source.get_snr(exptime=exptime, n_frames=1, band=band)
        assert recovered_snr == pytest.approx(target_snr, rel=1e-2)

    def test_get_exposure_increases_for_higher_target_snr(self, etc_with_source):
        t_low = etc_with_source.get_exposure(snr=3.0, band="nuv")
        t_high = etc_with_source.get_exposure(snr=15.0, band="nuv")
        assert t_high > t_low


class TestLimitingMagnitude:
    def test_works_without_a_source(self, etc_no_source):
        mag = etc_no_source.get_limiting_mag(snr=5.0, exptime=300 * u.s, n_frames=1, band="nuv")
        assert mag.unit.is_equivalent(u.ABmag)
        assert np.isfinite(mag.value)

    def test_lower_snr_requirement_gives_fainter_limit(self, etc_no_source):
        mag_snr5 = etc_no_source.get_limiting_mag(snr=5.0, exptime=300 * u.s, n_frames=1, band="nuv")
        mag_snr20 = etc_no_source.get_limiting_mag(snr=20.0, exptime=300 * u.s, n_frames=1, band="nuv")
        # A lower required SNR means fainter (numerically larger AB magnitude) sources are detectable.
        assert mag_snr5.value > mag_snr20.value

    def test_longer_exposure_gives_fainter_limit(self, etc_no_source):
        mag_short = etc_no_source.get_limiting_mag(snr=5.0, exptime=100 * u.s, n_frames=1, band="nuv")
        mag_long = etc_no_source.get_limiting_mag(snr=5.0, exptime=2000 * u.s, n_frames=1, band="nuv")
        assert mag_long.value > mag_short.value

    def test_n_dwells_path_matches_manual_exptime_path(self, etc_no_source):
        via_dwells = etc_no_source.get_limiting_mag(snr=5.0, n_dwells=1, band="nuv")
        via_manual = etc_no_source.get_limiting_mag(
            snr=5.0,
            exptime=etc_no_source.nuv_exposure,
            n_frames=etc_no_source.n_nuv * 1,
            band="nuv",
        )
        assert via_dwells.value == pytest.approx(via_manual.value)


class TestDwells:
    def test_returns_positive_value(self, etc_with_source):
        n_dwells = etc_with_source.get_dwells(snr=5.0, band="nuv")
        assert np.all(np.asarray(n_dwells) >= 1)

    def test_more_dwells_required_for_higher_snr(self, etc_with_source):
        low = etc_with_source.get_dwells(snr=3.0, band="nuv")
        high = etc_with_source.get_dwells(snr=30.0, band="nuv")
        assert np.all(np.asarray(high) >= np.asarray(low))

class TestGetInfo:
    def test_prints_expected_fields(self, etc_with_source, capsys):
        etc_with_source.get_info()
        out = capsys.readouterr().out
        assert "UVEX version" in out
        assert "Source:" in out
        assert "Source position:" in out
        assert "Observation time:" in out

# ---------------------------------------------------------------------------
# End-to-end smoke test
# ---------------------------------------------------------------------------

class TestEndToEndWorkflow:
    def test_multi_source_multi_coord_workflow(self, telescope):
        source_mags = [10.,11.,12.,13.,14.,15.] * u.ABmag
        etc = ETC(source=source_mags, telescope=telescope)

        nuv_src_rate = etc.get_source_count_rate(band="nuv")
        fuv_src_rate = etc.get_source_count_rate(band="fuv")
        nuv_bg_rate = etc.get_background_count_rate(band="nuv")
        fuv_bg_rate = etc.get_background_count_rate(band="fuv")

        assert len(nuv_src_rate) == len(source_mags)
        assert len(fuv_src_rate) == len(source_mags)
        assert np.all(nuv_src_rate.value > 0)
        assert np.all(fuv_src_rate.value > 0)
        # Background rate is per-pointing; a single default coordinate/obstime
        # was used, so it should be a single value.
        assert nuv_bg_rate.size == 1
        assert fuv_bg_rate.size == 1

        # Brighter (numerically smaller magnitude) sources should have a
        # higher count rate than fainter ones.
        assert np.all(np.diff(nuv_src_rate.value) <= 0)

    def test_get_snr_and_get_exposure_round_trip_for_full_workflow(self, telescope, default_coord, default_obstime):
        etc = ETC(
            source=SourceSpectrum(ConstFlux1D, amplitude=24 * u.ABmag),
            coordinate=default_coord,
            obstime=default_obstime,
            telescope=telescope,
        )
        snr_dwell_nuv = etc.get_snr(n_dwells=1, band="NUV")
        assert np.isfinite(snr_dwell_nuv)

        exptime_fuv = etc.get_exposure(snr=5, band="FUV")
        assert exptime_fuv.value > 0
