"""Shared pytest fixtures for EpexPredictor tests."""

import asyncio
import tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from predictor.model.priceregion import PriceRegion


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_storage_dir():
    """Create a temporary directory for data storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_region():
    """Return a sample price region for testing."""
    return PriceRegion.DE


@pytest.fixture
def sample_datetime():
    """Return a sample datetime for testing."""
    return datetime(2025, 11, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def sample_date_range(sample_datetime):
    """Return a sample date range for testing."""
    start = sample_datetime
    end = start + timedelta(days=7)
    return start, end


@pytest.fixture
def sample_weather_data():
    """Create sample weather data DataFrame.

    Column names match the actual implementation in weatherstore.py:
    - wind_{i} for wind speed at 80m
    - temp_{i} for temperature at 2m
    - irradiance_{i} for global tilted irradiance
    """
    dates = pd.date_range(
        start="2025-11-01",
        end="2025-11-02",
        freq="15min",
        tz="UTC"
    )
    data = {
        "wind_0": [5.0] * len(dates),
        "wind_1": [6.0] * len(dates),
        "temp_0": [10.0] * len(dates),
        "temp_1": [11.0] * len(dates),
        "irradiance_0": [100.0] * len(dates),
        "irradiance_1": [110.0] * len(dates),
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = "time"
    return df


@pytest.fixture
def sample_price_data():
    """Create sample price data DataFrame."""
    dates = pd.date_range(
        start="2025-11-01",
        end="2025-11-02",
        freq="15min",
        tz="UTC"
    )
    # Simulate realistic price patterns (higher during day, lower at night)
    prices = []
    for dt in dates:
        hour = dt.hour
        if 6 <= hour <= 20:
            prices.append(8.0 + (hour - 12) * 0.5)  # Day prices
        else:
            prices.append(4.0)  # Night prices

    df = pd.DataFrame({"price": prices}, index=dates)
    df.index.name = "time"
    return df


@pytest.fixture
def sample_aux_data():
    """Create sample auxiliary data DataFrame.

    Uses Fourier features for time-of-day encoding (tod_sin, tod_cos, tod_sin2, tod_cos2)
    instead of the old 96 one-hot time slot columns.
    """
    import math

    dates = pd.date_range(
        start="2025-11-01",
        end="2025-11-02",
        freq="15min",
        tz="UTC"
    )

    # Compute Fourier features for time of day
    tod_values = [(d.hour * 60 + d.minute) / (24 * 60) for d in dates]

    data = {
        "holiday": [0.0] * len(dates),
        "day_0": [1 if d.weekday() == 0 else 0 for d in dates],
        "day_1": [1 if d.weekday() == 1 else 0 for d in dates],
        "day_2": [1 if d.weekday() == 2 else 0 for d in dates],
        "day_3": [1 if d.weekday() == 3 else 0 for d in dates],
        "day_4": [1 if d.weekday() == 4 else 0 for d in dates],
        "day_5": [1 if d.weekday() == 5 else 0 for d in dates],
        "tod_sin": [math.sin(2 * math.pi * t) for t in tod_values],
        "tod_cos": [math.cos(2 * math.pi * t) for t in tod_values],
        "tod_sin2": [math.sin(4 * math.pi * t) for t in tod_values],
        "tod_cos2": [math.cos(4 * math.pi * t) for t in tod_values],
        "sr_influence": [60] * len(dates),
        "ss_influence": [60] * len(dates),
    }

    df = pd.DataFrame(data, index=dates)
    df.index.name = "time"
    return df


@pytest.fixture
def mock_aiohttp_response():
    """Create a mock aiohttp response."""
    def _create_response(json_data, status=200):
        mock_response = AsyncMock()
        mock_response.status = status
        mock_response.json = AsyncMock(return_value=json_data)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        return mock_response
    return _create_response


@pytest.fixture
def extended_price_data():
    """Create extended price data for lagged features testing."""
    dates = pd.date_range(
        start="2025-10-20", end="2025-11-02", freq="15min", tz="UTC"
    )
    df = pd.DataFrame({"price": [8.0] * len(dates)}, index=dates)
    df.index.name = "time"
    return df


@pytest.fixture
def mocked_predictor(
    sample_region, sample_weather_data, sample_price_data, sample_aux_data, extended_price_data
):
    """Create a PricePredictor with all stores mocked."""
    from predictor.model.pricepredictor import PricePredictor

    predictor = PricePredictor(sample_region)
    predictor.weatherstore.get_data = AsyncMock(return_value=sample_weather_data)
    predictor.pricestore.get_data = AsyncMock(return_value=sample_price_data)
    predictor.pricestore.get_known_data = MagicMock(return_value=extended_price_data)
    predictor.auxstore.get_data = AsyncMock(return_value=sample_aux_data)
    return predictor
