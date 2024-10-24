from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from scipy.stats import linregress

from data_import import importa_tutto, HEADER_VALORE_INQ
from wind_rose import generate_wind_aggregate_data, plot_wind_rose


def add_traces_media_per_mese(fig, df_input: pd.DataFrame, suffix: str, box_plot: bool):
    df = df_input.copy()
    df['Data fine'] = pd.to_datetime(df['Data fine'])
    df['year_month'] = df['Data fine'].dt.to_period('M')

    if box_plot:
        df['year_month_str'] = df['year_month'].astype(str)

        # Add box traces for each address
        for address in df['Indirizzo'].unique():
            df_address = df[df['Indirizzo'] == address]
            fig.add_trace(
                go.Box(
                    x=df_address['year_month_str'],
                    y=df_address[HEADER_VALORE_INQ],
                    name=f"{address}{suffix}",
                    # boxmean='sd'  # Shows the mean and standard deviation in the box plot
                )
            )
    else:
        df_grouped = df.groupby(['Indirizzo', 'year_month'])[HEADER_VALORE_INQ].agg(['mean', 'std', 'count']).reset_index()
        df_grouped['year_month_str'] = df_grouped['year_month'].astype(str)

        # Add traces for each address
        for address in df_grouped['Indirizzo'].unique():
            df_address = df_grouped[df_grouped['Indirizzo'] == address]
            fig.add_trace(
                go.Scatter(
                    x=df_address['year_month_str'],
                    y=df_address['mean'],
                    mode='markers+lines',
                    line=dict(width=1),
                    marker=dict(size=6),
                    name=f"{address}{suffix}",
                    error_y=dict(
                        type='data',
                        array=df_address['std'] if SHOW_ERROR else None,
                        visible=SHOW_ERROR
                    )
                )
            )

def media_per_mese(df_input: pd.DataFrame, output_html_path: Path, box_plot: bool) -> None:
    # Create an empty figure
    fig = go.Figure()
    add_traces_media_per_mese(fig, df_input, '', box_plot)
    # Update layout
    fig.update_layout(
        title='NO2 mensile per stazione',
        xaxis_title='Mese',
        yaxis_title='NO2',
        yaxis_tickmode='linear',  # Ensure linear tick mode for the y-axis
        yaxis_dtick=5,  # Add y-axis ticks every 5 units
        xaxis_tickangle=45,
        legend_title='Stazione',
        yaxis_range=YAXIS_RANGE,
    )
    # Export the plot to an interactive HTML file
    fig.write_html(output_html_path)

def media_per_mese_giorno_notte(df_input: pd.DataFrame, output_html_path: Path, ore_notte: set[int], box_plot: bool) -> None:
    # Create an empty figure
    fig = go.Figure()

    # Add traces for both day and night, appending "_day" and "_night"
    ore_giorno = set(range(24)).difference(ore_notte)
    df_day = filter_data_by_hour(df_input, ore_giorno)
    df_night = filter_data_by_hour(df_input, ore_notte)
    add_traces_media_per_mese(fig, df_day, '_day', box_plot)
    add_traces_media_per_mese(fig, df_night, '_night', box_plot)

    # Update layout
    fig.update_layout(
        title='NO2 mensile per stazione - Giorno vs Notte',
        xaxis_title='Mese',
        yaxis_title='NO2 Valore',
        yaxis_tickmode='linear',  # Ensure linear tick mode for the y-axis
        yaxis_dtick=5,  # Add y-axis ticks every 5 units
        xaxis_tickangle=45,
        legend_title='Stazione',
        yaxis_range=YAXIS_RANGE,
    )

    # Export the plot to an interactive HTML file
    fig.write_html(output_html_path)

def add_traces_media_per_ora(fig, df: pd.DataFrame, suffix: str, box_plot: bool):
    df = df.copy()
    df['Data fine'] = pd.to_datetime(df['Data fine'])
    df['year'] = df['Data fine'].dt.year
    df['hour'] = df['Data fine'].dt.hour

    if box_plot:
        # Add box traces for each address
        for address in df['Indirizzo'].unique():
            df_address = df[df['Indirizzo'] == address]
            fig.add_trace(
                go.Box(
                    x=df_address['hour'],
                    y=df_address[HEADER_VALORE_INQ],
                    name=f"{address}{suffix}",
                    boxmean='sd'
                )
            )
    else:
        # Group data to calculate mean, std, and margin of error
        df_grouped = df.groupby(['Indirizzo', 'hour'])[HEADER_VALORE_INQ].agg(['mean', 'std', 'count']).reset_index()

        # Add line traces with markers for each address
        for address in df_grouped['Indirizzo'].unique():
            df_address = df_grouped[df_grouped['Indirizzo'] == address]
            fig.add_trace(
                go.Scatter(
                    x=df_address['hour'],
                    y=df_address['mean'],
                    mode='markers+lines',
                    name=f"{address}{suffix}",
                    line=dict(width=1),
                    marker=dict(size=6),
                    error_y=dict(
                        type='data',
                        array=df_address['std'],
                        visible=SHOW_ERROR
                    )
                )
            )

def media_per_ora(df_input: pd.DataFrame, output_html_path: Path, box_plot) -> None:
    # Create an empty figure
    fig = go.Figure()
    add_traces_media_per_ora(fig, df_input, '', box_plot)

    # Update layout
    fig.update_layout(
        title='NO2 per ogni ora del giorno',
        xaxis_title='Ora',
        yaxis_title='NO2',
        yaxis_tickmode='linear',  # Ensure linear tick mode for the y-axis
        yaxis_dtick=5,  # Add y-axis ticks every 5 units
        xaxis_tickmode='linear',
        yaxis_range=YAXIS_RANGE,
        xaxis_tick0=0,
        xaxis_dtick=1,
        legend_title='Stazione',
    )

    # Export the plot to an interactive HTML file
    fig.write_html(output_html_path)

def media_per_ora_estate_inverno(df_input: pd.DataFrame, output_html_path: Path, mesi_estate: set[str], mesi_inverno: set[str], box_plot) -> None:
    # Create an empty figure
    fig = go.Figure()

    # Add traces for both summer and winter, appending "_summer" and "_winter"
    df_summer = filter_data_by_month(df_input, mesi_estate)
    df_winter = filter_data_by_month(df_input, mesi_inverno)
    add_traces_media_per_ora(fig, df_summer, '_estate', box_plot)
    add_traces_media_per_ora(fig, df_winter, '_inverno', box_plot)

    # Update layout
    fig.update_layout(
        title='NO2 per ogni ora del giorno - Estate vs Inverno',
        xaxis_title='Ora',
        yaxis_title='NO2 Valore',
        yaxis_tickmode='linear',  # Ensure linear tick mode for the y-axis
        yaxis_dtick=5,  # Add y-axis ticks every 5 units
        xaxis_tickmode='linear',
        xaxis_tick0=0,
        xaxis_dtick=1,
        legend_title='Stazione',
        yaxis_range=YAXIS_RANGE,
    )

    # Export the plot to an interactive HTML file
    fig.write_html(output_html_path)

# Functions to filter data by month and hour
def filter_data_by_month(df_input: pd.DataFrame, allowed_months: set[str]) -> pd.DataFrame:
    df = df_input.copy()
    df['Data fine'] = pd.to_datetime(df['Data fine'])
    df['month_name'] = df['Data fine'].dt.strftime('%b')
    df_filtered = df[df['month_name'].isin(allowed_months)]
    return df_filtered

def filter_data_by_hour(df_input: pd.DataFrame, allowed_hours: set[int]) -> pd.DataFrame:
    df = df_input.copy()
    df['Data fine'] = pd.to_datetime(df['Data fine'])
    df['hour'] = df['Data fine'].dt.hour
    df_filtered = df[df['hour'].isin(allowed_hours)]
    return df_filtered

def df_estate_inverno(df_input, mesi_estate, mesi_inverno: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Ensure 'Data fine' is in datetime format
    df = df_input.copy()
    df['Data fine'] = pd.to_datetime(df['Data fine'])
    df['year'] = df['Data fine'].dt.year
    df['month_name'] = df['Data fine'].dt.strftime('%b')

    # Filter data for summer and winter
    df_summer = df[df['month_name'].isin(mesi_estate)]
    df_winter = df[df['month_name'].isin(mesi_inverno)]
    return df_summer, df_winter

def media_estate_inverno_divisa_per_anno(df_input, output_html_path: Path,
                                         mesi_estate,
                                         mesi_inverno,
                                         box_plot):
    """
    Produces two html plotly plots:
        - On the x-axis, the year. On the y-axis, the average value over the summer months (mesi_estate) for every Indirizzo.
        - On the x-axis, the year. On the y-axis, the average value over the winter months (mesi_inverno) for every Indirizzo.
    If box_plot is True, it produces a box plot; otherwise, a line plot.
    """
    # Ensure 'Data fine' is in datetime format
    df = df_input.copy()
    df['Data fine'] = pd.to_datetime(df['Data fine'])
    df['year'] = df['Data fine'].dt.year
    df['month_name'] = df['Data fine'].dt.strftime('%b')

    # Filter data for summer and winter
    df_summer = df[df['month_name'].isin(mesi_estate)]
    df_winter = df[df['month_name'].isin(mesi_inverno)]

    # Create figure
    fig = go.Figure()

    # Function to add traces based on plot type (box or line)
    def add_traces_by_type(fig, df_season, season_name):
        if box_plot:
            # Add box plot traces
            for address in df_season['Indirizzo'].unique():
                df_address = df_season[df_season['Indirizzo'] == address]
                fig.add_trace(
                    go.Box(
                        x=df_address['year'],
                        y=df_address[HEADER_VALORE_INQ],
                        name=f"{address} {season_name}",
                    )
                )

            # Perform linear regression on all data points
            x_values = df_season['year']
            y_values = df_season[HEADER_VALORE_INQ]
        else:
            # Group data by year and Indirizzo for line plots
            df_grouped = df_season.groupby(['Indirizzo', 'year'])[HEADER_VALORE_INQ].mean().reset_index()

            # Add line plot traces with markers for each address
            for address in df_grouped['Indirizzo'].unique():
                df_address = df_grouped[df_grouped['Indirizzo'] == address]
                fig.add_trace(
                    go.Scatter(
                        x=df_address['year'],
                        y=df_address[HEADER_VALORE_INQ],
                        mode='markers+lines',
                        name=f"{address} {season_name}",
                        line=dict(width=1),
                        marker=dict(size=6)
                    )
                )

            # Perform linear regression on mean values
            x_values = df_grouped['year']
            y_values = df_grouped[HEADER_VALORE_INQ]

        # Compute the regression line
        slope, intercept, r_value, p_value, std_err = linregress(list(x_values), list(y_values))
        regression_line = intercept + slope * np.array(sorted(x_values.unique()))

        # Add regression line as a dashed line
        fig.add_trace(
            go.Scatter(
                x=sorted(x_values.unique()),
                y=regression_line,
                mode='lines',
                line=dict(dash='dash', color='red'),
                name=f"Trend {season_name} (p-value: {p_value:.4f})",
                showlegend=True
            )
        )

    # Add traces to the summer and winter figures
    add_traces_by_type(fig, df_summer, str(mesi_estate))
    add_traces_by_type(fig, df_winter, str(mesi_inverno))

    # Update layout for the plot
    fig.update_layout(
        title=f'Valori estivi/invernali per anno',
        xaxis_title='Anno',
        yaxis_title='NO2',
        yaxis_tickmode='linear',  # Ensure linear tick mode for the y-axis
        yaxis_dtick=5,  # Add y-axis ticks every 5 units
        xaxis_tickmode='linear',
        xaxis_dtick=1,
        legend_title='Stazione',
        yaxis_range=YAXIS_RANGE,
    )
    fig.write_html(output_html_path)

def plot_days_above_threshold(df_input: pd.DataFrame, threshold: float, output_html_path: Path):
    """
    Plots the number of days per month where the mean daily value exceeds a given threshold.
    """
    df = df_input.copy()
    df['Data fine'] = pd.to_datetime(df['Data fine'])
    df['year_month'] = df['Data fine'].dt.to_period('M')
    df['date'] = df['Data fine'].dt.date

    # Group by date and address, calculate daily mean, then filter by threshold
    df_daily = df.groupby(['Indirizzo', 'date'])[HEADER_VALORE_INQ].mean().reset_index()
    df_daily['above_threshold'] = df_daily[HEADER_VALORE_INQ] > threshold

    # Convert 'date' to datetime and extract the year-month
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    df_daily['year_month'] = df_daily['date'].dt.to_period('M')

    # Count the days above the threshold for each 'Indirizzo' and 'year_month'
    df_count = df_daily.groupby(['Indirizzo', 'year_month'])['above_threshold'].sum().reset_index()

    # Ensure all year-month combinations for each 'Indirizzo' are represented
    all_stations = df_daily['Indirizzo'].unique()
    all_months = pd.period_range(df_daily['date'].min(), df_daily['date'].max(), freq='M')

    # Create a complete index of all stations and all months
    full_index = pd.MultiIndex.from_product([all_stations, all_months], names=['Indirizzo', 'year_month'])

    # Reindex the DataFrame to include all months and stations, filling missing values with 0
    df_count = df_count.set_index(['Indirizzo', 'year_month']).reindex(full_index, fill_value=None).reset_index()

    # Rename the 'above_threshold' column to 'days_above_threshold'
    df_count.rename(columns={'above_threshold': 'days_above_threshold'}, inplace=True)

    # Create the figure
    fig = go.Figure()

    # Add traces for each station
    for address in df_count['Indirizzo'].unique():
        df_address = df_count[df_count['Indirizzo'] == address]
        fig.add_trace(
            go.Scatter(
                x=df_address['year_month'].astype(str),
                y=df_address['days_above_threshold'],
                mode='markers+lines',
                name=address,
                line=dict(width=1),
                marker=dict(size=6)
            )
        )

    # Update layout
    fig.update_layout(
        title=f'Numero di giorni al mesi con valore medio oltre {threshold}',
        xaxis_title='Mese',
        yaxis_title='Numero di Giorni',
        xaxis_tickangle=45,
        legend_title='Stazione',
        yaxis_range=[0, df_count['days_above_threshold'].max() + 5],  # Adjust y-axis to fit data
    )

    # Export the plot to an interactive HTML file
    fig.write_html(output_html_path)


def plot_histogram_daily_values(df_input: pd.DataFrame, output_html_path: Path):
    """
    Plots a histogram of the mean daily values for each station, over each year.
    """
    df = df_input.copy()
    df['Data fine'] = pd.to_datetime(df['Data fine'])
    df['year'] = df['Data fine'].dt.year
    df['date'] = df['Data fine'].dt.date

    # Group by date and address to get the daily mean values
    df_daily = df.groupby(['Indirizzo', 'date', 'year'])[HEADER_VALORE_INQ].mean().reset_index()

    # Create the figure
    fig = go.Figure()

    # Add histogram traces for each station and year
    for address in df_daily['Indirizzo'].unique():
        for year in df_daily['year'].unique():
            df_address_year = df_daily[(df_daily['Indirizzo'] == address) & (df_daily['year'] == year)]
            fig.add_trace(
                go.Histogram(
                    x=df_address_year[HEADER_VALORE_INQ],
                    name=f"{address} - {year}",
                    opacity=0.6,
                    nbinsx=30  # Adjust the number of bins for better visualization
                )
            )

    # Update layout for better readability
    fig.update_layout(
        title='Histogram of Mean Daily Values per Station and Year',
        xaxis_title='Mean Daily NO2 Value',
        yaxis_title='Frequency',
        barmode='overlay',  # Overlay histograms for comparison
        legend_title='Station and Year',
    )

    # Export the plot to an interactive HTML file
    fig.write_html(output_html_path)

if __name__ == "__main__":
    san_teodoro = "dati_san_teodoro"
    tutta_genova = "dati_genova"
    SHOW_ERROR = False
    inquinante = "BIOSSIDO AZOTO"
    low_limit = 1
    YAXIS_RANGE = [0, None]
    unificazione_via_bari_e_san_francesco_da_paola = True
    mesi_estate = {'Jun', 'Jul', 'Aug'}
    mesi_inverno = {'Dec', 'Jan', 'Feb'}
    base_path: Path = Path('/Users/mattia/Documents/qualita_aria/')

    for data_file_base_name in [san_teodoro, tutta_genova]:
        df = importa_tutto(base_name_inquinanti=data_file_base_name, base_path=base_path)

        # Wind roses
        addresses = pd.unique(df["Indirizzo"])
        for address in addresses:
            wind_agg = generate_wind_aggregate_data(df, address=address)
            plot_wind_rose(wind_agg, base_path / "output" / data_file_base_name / "analisi_vento" / f"{address}.png")

        for box_plot in [True, False]:
            output_path = base_path / "output" / data_file_base_name / ("whisker_box" if box_plot else "medie")
            output_path.mkdir(exist_ok=True, parents=True)

            box_str = "_box" if box_plot else ""
            media_per_mese(df, output_path / f'{data_file_base_name}_NO2_per_mese{box_str}.html', box_plot=box_plot)

            media_per_mese_giorno_notte(df, output_path / f'{data_file_base_name}_NO2_per_mese_giorno_notte{box_str}.html',
                                        ore_notte={22, 23, 0, 1, 2, 3, 4, 5, 6}, box_plot=box_plot)

            media_per_ora(df, output_path / f'{data_file_base_name}_NO2_per_ora{box_str}.html', box_plot=box_plot)
            media_per_ora_estate_inverno(df, output_path / f'{data_file_base_name}_NO2_per_ora_estate_inverno{box_str}.html',
                                         mesi_estate=mesi_estate,
                                         mesi_inverno = mesi_inverno, box_plot=box_plot)

            media_estate_inverno_divisa_per_anno(df,
                                                 output_html_path=output_path / f'media_estate_inverno_divisa_per_anno{box_str}.html',
                                                 mesi_estate=mesi_estate,
                                                 mesi_inverno = mesi_inverno,
                                                 box_plot=box_plot)

            media_estate_inverno_divisa_per_anno(df,
                                                 output_html_path=output_path / f'media_agosto_gennaio_divisa_per_anno{box_str}.html',
                                                 mesi_estate={'Aug'},
                                                 mesi_inverno={'Jan'},
                                                 box_plot=box_plot)

            if not box_plot:
                plot_days_above_threshold(df, 50, output_html_path=output_path / f'{data_file_base_name}_n_giorni_oltre_soglia_50.html')
                plot_histogram_daily_values(df, output_html_path=output_path / f'histogram_daily_values.html')