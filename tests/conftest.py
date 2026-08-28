"""
Shared fixtures for the uvex_imager_etc test suite.

Design notes
------------
- Loading a `UVEX` instance parses several large bandpass text files from disk,
  so we build it once per test session (`telescope` fixture) instead of once
  per test.
- We resolve "whichever CALDB is currently the latest" once (via
  `caldb_version`) and then pass that *specific* version explicitly to every
  other `UVEX(...)` call in the suite. This keeps the rest of the suite fast
  and warning-free (the default/no-`caldb` code path emits a `UserWarning`
  whenever more than one CALDB shares the latest date, which is currently the
  case), while still remaining correct if CALDB directories are added or
  removed in the future.
"""
import warnings

import pytest
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord

from uvex_imager_etc.uvex import UVEX
from uvex_imager_etc.etc import ETC


@pytest.fixture(scope="session")
def caldb_version():
    """Whatever CALDB version `UVEX()` resolves to by default, as a plain str."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = UVEX()
    return str(t.get_caldb())


@pytest.fixture(scope="session")
def telescope(caldb_version):
    """A single, reused UVEX telescope configuration, pinned to a known CALDB."""
    return UVEX(caldb=caldb_version)


@pytest.fixture
def default_coord():
    """Single coordinate matching ETC's own default (15 deg out of the plane)."""
    return SkyCoord(120.0, 15.0, unit=u.deg, frame="galactic")


@pytest.fixture
def default_obstime():
    return Time("2030-06-01 09:00:00", scale="utc", format="iso")


@pytest.fixture
def multi_coord():
    """Three coordinates spanning north/south Galactic latitude."""
    return SkyCoord([100.0, 120.0, 140.0], [30.0, 15.0, -40.0], unit=u.deg, frame="galactic")


@pytest.fixture
def multi_obstime():
    return Time(
        ["2030-02-01 09:00:00", "2030-06-01 09:00:00", "2030-09-01 09:00:00"],
        scale="utc",
        format="iso",
    )


@pytest.fixture
def const_source():
    """A single constant-flux source at 24 ABmag, matching the example notebook."""
    return 24.0 * u.ABmag


@pytest.fixture
def source_array():
    """Several constant-flux sources of increasing faintness."""
    return [23.0, 24.0, 25.0, 26.0] * u.ABmag


@pytest.fixture
def etc_with_source(telescope, const_source):
    """A minimal, fast-to-construct ETC with a single source and pinned telescope."""
    return ETC(source=const_source, telescope=telescope)


@pytest.fixture
def etc_no_source(telescope):
    """An ETC with no source loaded (valid for limiting-magnitude calculations)."""
    return ETC(telescope=telescope)
