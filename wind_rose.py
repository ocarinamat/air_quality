from cmath import isnan
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Circle
from matplotlib.cm import ScalarMappable

from data_import import HEADER_GRADI_VENTO, HEADER_VALORE_INQ, HEADER_DIREZ_VENTO_BINNED

# Define the 8 wind directions
directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']


# Function to assign wind direction to bins
def assign_wind_direction(deg):
    """
    Assigns wind direction to one of 8 cardinal directions based on degrees.
    Handles values from -inf to inf using modular arithmetic.
    """
    deg = deg % 360  # Wrap the degree values (ensure any value from -inf to inf works)
    bin_idx = int((deg + 22.5) // 45) % 8
    return directions[bin_idx]


# Function to compute average NO2 by wind direction
def compute_avg_NO2_by_direction(df):
    """
    Computes the average NO₂ concentration for each wind direction.
    """
    df_avg = df.groupby(HEADER_DIREZ_VENTO_BINNED)[HEADER_VALORE_INQ].mean().reset_index()
    df_avg.rename(columns={HEADER_VALORE_INQ: 'mean_NO2'}, inplace=True)
    return df_avg


# Function to compute percentage of time wind blew from each direction
def compute_wind_direction_percentage(df):
    """
    Computes the percentage of time the wind blew from each direction.
    """
    df_counts = df[HEADER_DIREZ_VENTO_BINNED].value_counts(normalize=True).reset_index()
    df_counts.columns = [HEADER_DIREZ_VENTO_BINNED, 'percentage']
    df_counts['percentage'] *= 100  # Convert to percentage
    return df_counts


# Function to generate the summary DataFrame
def generate_wind_aggregate_data(df_input, address: str | None):
    """
    Generates a summary DataFrame containing both the mean NO₂ for each wind direction
    and the percentage of time the wind blew from each direction.
    """

    if address is not None:
        df = (df_input[df_input["Indirizzo"] == address]).copy()
    else:
        df = df_input.copy()
    df[HEADER_DIREZ_VENTO_BINNED] = df[HEADER_GRADI_VENTO].apply(assign_wind_direction)
    df_avg = compute_avg_NO2_by_direction(df)
    df_percent = compute_wind_direction_percentage(df)
    df_summary = pd.merge(df_avg, df_percent, on=HEADER_DIREZ_VENTO_BINNED)
    return df_summary


def plot_wind_rose(df_summary, output_png_path: Path | None):
    """
    Plots a wind rose diagram with the average NO₂ and percentage of time
    the wind blew in each direction using Matplotlib.
    """
    # Prepare the data for the wind rose plot
    df_summary = df_summary.set_index(HEADER_DIREZ_VENTO_BINNED).reindex(directions)  # Reorder directions

    # Extract relevant data
    wind_directions = df_summary.index.values  # Directions (e.g., N, NE, etc.)
    percentages = df_summary['percentage'].values  # Percentage of time wind blew in each direction
    mean_NO2 = df_summary['mean_NO2'].values  # Mean NO₂ concentration

    max_perc = np.nanmax(percentages)
    max_radius = max_perc / 100

    # Convert the wind directions (strings) into corresponding angles for plotting
    direction_angles = np.arange(90, -360+90, -45)

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))
    arrow_start = 0.1

    # Plot concentric circles with percentage labels
    circle_every = 10
    for radius in range(0,int(max_perc+ 1),circle_every):
        circle = Circle((0, 0), (radius / 100) + arrow_start, color='gray', fill=False, linestyle='--', linewidth=0.5)
        ax.add_patch(circle)
        text_dist = arrow_start + (radius / 100)
        text_legend_angle = np.deg2rad(90*3/4)
        ax.text(text_dist*np.cos(text_legend_angle),
                text_dist*np.sin(text_legend_angle),
                f'{radius}%', ha='left', va='bottom', fontsize=8, color='gray')

    # Create a color map
    cmap = plt.cm.seismic  # You can change this colormap as needed
    norm = plt.Normalize(vmin=0, vmax=50)  # Normalizing between 0 and 50 µg/m³ for NO₂ values

    # Plot arrows for each direction
    for i, direction in enumerate(wind_directions):
        if isnan(percentages[i]):
            percentages[i] = 0
            mean_NO2[i] = np.nan

        angle = direction_angles[i]
        mean_no2 = mean_NO2[i]
        percentage = percentages[i]

        # Calculate the arrow length based on percentage
        arrow_length = percentage / 100

        # Convert angle to radians
        angle_rad = np.deg2rad(angle)

        arr_x = arrow_start * np.cos(angle_rad)
        arr_y = arrow_start * np.sin(angle_rad)
        end_x = arrow_length * np.cos(angle_rad)
        end_y = arrow_length * np.sin(angle_rad)

        # Plot the arrow using FancyArrow (adjust width and starting point)
        outline_mod = 1.1
        ax.add_patch(FancyArrow(arr_x, arr_y, end_x, end_y,
            width=0.1*max_radius*outline_mod, color='black', head_width=0.15*max_radius*outline_mod, head_length=0.05*max_radius*outline_mod,
            length_includes_head=True,
        ))
        ax.add_patch(FancyArrow(arr_x, arr_y, end_x, end_y,
            width=0.1*max_radius, color=cmap(norm(mean_no2)), head_width=0.15*max_radius, head_length=0.05*max_radius,
            length_includes_head=True,
        ))

        # Annotate with the mean NO₂ value
        text_border = 0.1
        ax.text(
            (arrow_start + arrow_length + text_border) * np.cos(angle_rad),
            (arrow_start + arrow_length + text_border) * np.sin(angle_rad),
            f"{mean_no2:.1f} µg/m³",
            color='black', fontsize=10, ha='center', va='center'
        )
        ax.text(
            (arrow_start + max_radius + text_border*1.4) * np.cos(angle_rad),
            (arrow_start + max_radius + text_border*1.4) * np.sin(angle_rad),
            f"{direction}",
            color='black', fontsize=15, ha='center', va='center',
        )

    # Add a color bar on the right
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # We don't have data associated with the ScalarMappable
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('NO₂ Concentration (µg/m³)', rotation=270, labelpad=15)

    # Customize the plot appearance
    ax.set_aspect('equal')
    max_show_radius = max_radius+ 0.3
    ax.set_xlim(-max_show_radius, max_show_radius)
    ax.set_ylim(-max_show_radius, max_show_radius)
    ax.axis('off')  # Hide axes
    ax.set_title('Concentrazione media di NO₂ per provenienza del vento')

    if output_png_path is not None:
        output_png_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_png_path, bbox_inches='tight')
    else:
        plt.show()


# Main function
def main():
    # Example data with NO2 values and wind direction in degrees
    data = {
        HEADER_VALORE_INQ: [25, 40, 35, 22, 50, 42, 30, 29, 37, 45, 38, 32, 41, 28, 47, 33],
        HEADER_GRADI_VENTO: [350, 10, 80, 140, 200, 280, 310, 45, 135, 225, 270, 350, 15, 30, 190, 220]
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Generate the summary DataFrame
    df_summary = generate_wind_aggregate_data(df)

    # Output the summary
    print(df_summary)

    # Plot wind rose
    plot_wind_rose(df_summary)


# Run the main function
if __name__ == "__main__":
    main()
