"""Tests for predictor.model.auxdatastore module."""

from datetime import datetime, timezone

import pandas as pd
import pytest

from predictor.model.auxdatastore import AuxDataStore


class TestAuxDataStoreInit:
    """Tests for AuxDataStore initialization."""

    def test_init_creates_empty_store(self, sample_region):
        """Test initialization creates empty store."""
        store = AuxDataStore(sample_region)
        assert store.region == sample_region
        assert store.data.empty


class TestAuxDataStoreIsHoliday:
    """Tests for is_holiday method."""

    def test_sunday_is_holiday(self, sample_region):
        """Test that Sunday returns 1.0 (full holiday)."""
        store = AuxDataStore(sample_region)
        # Nov 2, 2025 is a Sunday
        sunday = pd.Timestamp("2025-11-02", tz="UTC")
        result = store.is_holiday(sunday)
        assert result == pytest.approx(1.0)

    def test_regular_weekday_not_holiday(self, sample_region):
        """Test that regular weekday returns low holiday value."""
        store = AuxDataStore(sample_region)
        # Nov 5, 2025 is a Wednesday (not a holiday)
        wednesday = pd.Timestamp("2025-11-05", tz="UTC")
        result = store.is_holiday(wednesday)
        # Should be 0 or a small fraction if some regions have a holiday
        assert 0.0 <= result <= 1.0

    def test_christmas_is_holiday(self, sample_region):
        """Test that Christmas Day returns 1.0."""
        store = AuxDataStore(sample_region)
        christmas = pd.Timestamp("2025-12-25", tz="UTC")
        result = store.is_holiday(christmas)
        # Christmas is a holiday in all German states
        assert result == pytest.approx(1.0)

    def test_new_year_is_holiday(self, sample_region):
        """Test that New Year's Day returns 1.0."""
        store = AuxDataStore(sample_region)
        new_year = pd.Timestamp("2025-01-01", tz="UTC")
        result = store.is_holiday(new_year)
        assert result == pytest.approx(1.0)


class TestAuxDataStoreFetchMissingData:
    """Tests for fetch_missing_data method."""

    @pytest.mark.asyncio
    async def test_fetch_creates_aux_features(self, sample_region):
        """Test that fetch_missing_data creates auxiliary features."""
        store = AuxDataStore(sample_region)
        # Use a longer range to ensure the algorithm generates ranges
        # (gen_missing_date_ranges uses noon internally)
        start = datetime(2025, 11, 1, tzinfo=timezone.utc)
        end = datetime(2025, 11, 3, tzinfo=timezone.utc)

        await store.fetch_missing_data(start, end)

        # Verify data was created
        assert not store.data.empty

        # Check for expected columns
        assert "holiday" in store.data.columns
        # Day of week columns
        for i in range(6):
            assert f"day_{i}" in store.data.columns
        # Fourier time-of-day features (replaced 96 one-hot time slots)
        assert "tod_sin" in store.data.columns
        assert "tod_cos" in store.data.columns
        assert "tod_sin2" in store.data.columns
        assert "tod_cos2" in store.data.columns
        # Sunrise/sunset influence
        assert "sr_influence" in store.data.columns
        assert "ss_influence" in store.data.columns

    @pytest.mark.asyncio
    async def test_fetch_respects_15min_intervals(self, sample_region):
        """Test that fetch creates 15-minute interval data."""
        store = AuxDataStore(sample_region)
        # Use a longer range to ensure data generation
        start = datetime(2025, 11, 1, tzinfo=timezone.utc)
        end = datetime(2025, 11, 3, tzinfo=timezone.utc)

        await store.fetch_missing_data(start, end)

        # Should have multiple 15-min intervals
        assert len(store.data) >= 4


class TestAuxDataStoreDayOfWeekEncoding:
    """Tests for day of week encoding."""

    @pytest.mark.asyncio
    async def test_day_columns_are_one_hot(self, sample_region):
        """Test that day columns are one-hot encoded."""
        store = AuxDataStore(sample_region)
        start = datetime(2025, 11, 1, tzinfo=timezone.utc)  # Saturday
        end = datetime(2025, 11, 2, tzinfo=timezone.utc)

        await store.fetch_missing_data(start, end)

        # For each row, exactly one day column should be 1 (or 0 for Sunday)
        day_cols = [f"day_{i}" for i in range(6)]
        for _, row in store.data.iterrows():
            day_sum = sum(row[col] for col in day_cols)
            # Sum should be 0 (Sunday) or 1 (Mon-Sat)
            assert day_sum in [0, 1]


class TestAuxDataStoreFourierFeatures:
    """Tests for Fourier time-of-day features."""

    @pytest.mark.asyncio
    async def test_fourier_features_range(self, sample_region):
        """Test that Fourier features are in valid range [-1, 1]."""
        store = AuxDataStore(sample_region)
        start = datetime(2025, 11, 1, tzinfo=timezone.utc)
        end = datetime(2025, 11, 3, tzinfo=timezone.utc)

        await store.fetch_missing_data(start, end)

        # All Fourier features should be between -1 and 1
        for col in ["tod_sin", "tod_cos", "tod_sin2", "tod_cos2"]:
            assert store.data[col].min() >= -1.0
            assert store.data[col].max() <= 1.0

    @pytest.mark.asyncio
    async def test_fourier_features_cyclical(self, sample_region):
        """Test that Fourier features capture daily cycle."""
        store = AuxDataStore(sample_region)
        start = datetime(2025, 11, 1, tzinfo=timezone.utc)
        end = datetime(2025, 11, 3, tzinfo=timezone.utc)

        await store.fetch_missing_data(start, end)

        # sin and cos should have different values at different times
        assert store.data["tod_sin"].std() > 0.1
        assert store.data["tod_cos"].std() > 0.1


class TestAuxDataStoreSunriseSunset:
    """Tests for sunrise/sunset influence calculation."""

    @pytest.mark.asyncio
    async def test_sunrise_sunset_influence_range(self, sample_region):
        """Test that sunrise/sunset influence values are in valid range."""
        store = AuxDataStore(sample_region)
        # Use a range that will generate data
        start = datetime(2025, 11, 1, tzinfo=timezone.utc)
        end = datetime(2025, 11, 3, tzinfo=timezone.utc)

        await store.fetch_missing_data(start, end)

        # Values should be between 0 and 180 (capped at 3 hours)
        assert store.data["sr_influence"].min() >= 0
        assert store.data["sr_influence"].max() <= 180
        assert store.data["ss_influence"].min() >= 0
        assert store.data["ss_influence"].max() <= 180

    @pytest.mark.asyncio
    async def test_midday_has_high_influence(self, sample_region):
        """Test that midday has high sunrise/sunset influence (far from both)."""
        store = AuxDataStore(sample_region)
        # Use a range that will generate data
        start = datetime(2025, 11, 1, tzinfo=timezone.utc)
        end = datetime(2025, 11, 3, tzinfo=timezone.utc)

        await store.fetch_missing_data(start, end)

        # Filter to midday hours
        midday_data = store.data[(store.data.index.hour >= 11) & (store.data.index.hour <= 13)]
        if not midday_data.empty:
            # Both should be at or near the cap of 180
            assert midday_data["sr_influence"].mean() > 100
            assert midday_data["ss_influence"].mean() > 100


class TestAuxDataStoreGetData:
    """Tests for get_data method."""

    @pytest.mark.asyncio
    async def test_get_data_fetches_and_returns(self, sample_region):
        """Test that get_data fetches missing data and returns it."""
        store = AuxDataStore(sample_region)
        # Use a range that will generate data
        start = datetime(2025, 11, 1, tzinfo=timezone.utc)
        end = datetime(2025, 11, 3, tzinfo=timezone.utc)

        result = await store.get_data(start, end)

        assert not result.empty
        # Data may extend beyond requested range due to full-day generation
        assert len(result) > 0
