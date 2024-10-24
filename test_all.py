import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Circle

from data_import import HEADER_DIREZ_VENTO_BINNED
from wind_rose import plot_wind_rose


# Assume the plot_wind_rose function has already been imported
# from the existing code

# Define directions and percentages for the test
@pytest.fixture
def wind_data():
    data = {
        HEADER_DIREZ_VENTO_BINNED: ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'],
        'percentage': [0, 20, 40, 60, 80, 70, 50, 30],  # Example distribution
        'mean_NO2': [0, 10, 20, 30, 40, 50, 80, 100]  # Example NO₂ values
    }
    return pd.DataFrame(data)


def test_wind_rose_plot(wind_data):
    """
    Test function to plot wind rose with a fixed distribution of wind directions and NO2 values.
    """
    # Call the plot function
    plot_wind_rose(wind_data, None)
    pass
