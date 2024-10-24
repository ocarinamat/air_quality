import pandas as pd
from pathlib import Path

from datetime import datetime

HEADER_GRADI_VENTO = "vento_gradi_nord_provenienza"
HEADER_VALORE_INQ = "Valore"
HEADER_DIREZ_VENTO_BINNED = "vento_dir_binned"

def unifica_via_bari_e_san_francesco_da_paola(df: pd.DataFrame, nuovo_nome) -> pd.DataFrame:
    # Find all unique names of locations
    nomi_quartieri = set(df["Indirizzo"].unique())
    # Identify the stations with the specified names
    stazioni_san_teodoro = [str(x) for x in nomi_quartieri if ("Via Bari" in str(x) or "San Francesco da Paola" in str(x))]

    # Ensure that exactly two stations are found
    assert len(stazioni_san_teodoro) == 2, "Expected exactly 2 stations, found: " + str(len(stazioni_san_teodoro))

    # Use loc to correctly modify the "Indirizzo" for the selected rows
    df.loc[df["Indirizzo"].isin(stazioni_san_teodoro), 'Indirizzo'] = nuovo_nome

    return df

def check_hourly_diff(inizio: pd.Series, fine: pd.Series) -> bool:
    diff = fine - inizio
    # Check that all differences are 1h. So we can use Data fine.
    return len(diff.unique()) == 1

def remove_wrong_measurements(df: pd.DataFrame, low_limit: float) -> pd.DataFrame:
    # we assume a measurement <= low_limit is wrong
    return df[df["Valore"] > low_limit]

def import_database_inquinamento(base_path: Path, data_file_base_name: str, inquinante: str) -> pd.DataFrame:
    # Import xlsx files into a single dataframe
    dataframes = []
    for year in range(1900, 2200):
        file_path = base_path / f'{data_file_base_name}_{year}.xlsx'
        if file_path.is_file():
            df = pd.read_excel(file_path)
            dataframes.append(df)

    # Concatenate all dataframes
    df = pd.concat(dataframes)

    new_header = df.iloc[1]
    assert new_header.iloc[1] == "Codice stazione"
    df = df[2:]
    df.columns = new_header
    df = df[["Codice stazione", "Indirizzo", "Inquinante", "Valore", "Valido", "Data inizio", "Data fine"]]
    df = df[df["Inquinante"] == inquinante]

    # Convert the date column to datetime if necessary
    df['Data inizio'] = pd.to_datetime(df['Data inizio'])
    df['Data fine'] = pd.to_datetime(df['Data fine'])
    assert check_hourly_diff(df['Data inizio'], df['Data fine'])

    df = df[["Indirizzo", "Inquinante", "Valore", "Data fine"]]
    return df

def read_wind_data(file_path, encoding='ISO-8859-1'):
    """
    Reads and parses a wind data file into a pandas DataFrame with specified encoding.
    Skips malformed lines with bad data.
    """
    valid_values = ['SÏ', 'Sì', 'Si', 'S', 'si', 'sì', 'Sì\n']

    data = []  # To store valid rows

    with open(file_path, 'r', encoding=encoding) as f:
        header = None
        for line in f.readlines():
            tokens = line.strip().split(',')

            # Skip malformed lines that don't have exactly 5 columns
            if len(tokens) != 5:
                continue

            # Set header based on "Inizio rilevazione"
            if tokens[0].startswith('Inizio rilevazione'):
                header = tokens
                assert header == ["Inizio rilevazione","Fine rilevazione","Valore","Dataset","Valido"]
                continue

            try:
                # Parse start and end times
                inizio = datetime.strptime(tokens[0], '%d/%m/%Y %H:%M')  # Convert 'Inizio rilevazione'
                fine = datetime.strptime(tokens[1], '%d/%m/%Y %H:%M')  # Convert 'Fine rilevazione'
                valore = float(tokens[2])  # Parse the wind direction value
                dataset = tokens[3].strip()  # Dataset info
                assert tokens[4].strip() in valid_values

                assert check_hourly_diff(pd.Series(inizio), pd.Series(fine))

                # Append the parsed data as a row
                data.append([fine, valore])

            except ValueError as e:
                raise ValueError(f"Malformed line: {line} (Error: {e})")

        # Create DataFrame from the valid data
        df = pd.DataFrame(data, columns=["Data fine",HEADER_GRADI_VENTO])

        return df

def read_multiple_wind_files(directory_path, glob_expr="vento*.txt"):
    """
    Reads multiple wind data files from a given directory, concatenates them into a single DataFrame.
    """
    # Use Path to find all CSV files in the directory
    directory = Path(directory_path)
    assert directory.is_dir()
    all_files = list(directory.glob(glob_expr))  # Find all CSV files in the directory

    # Read and concatenate all files into one DataFrame
    df_list = [read_wind_data(file) for file in all_files]
    combined_df = pd.concat(df_list, ignore_index=True)

    return combined_df
def combina_inquinamento_e_vento(df_inquinanti, df_vento) -> pd.DataFrame:
    """
    Combine the inquinamento (pollution) and vento (wind) data frames on the 'Data fine' column.
    The resulting DataFrame will contain the pollution data and the wind direction data at the same time.
    """
    # Merge the pollution (df_inquinanti) and wind (df_vento) data on 'Data fine'
    df_unico = pd.merge(df_inquinanti, df_vento, on="Data fine", how="inner")

    return df_unico

def importa_tutto(base_name_inquinanti: str, base_path: Path = Path('/Users/mattia/Documents/qualita_aria/')):
    inquinante = "BIOSSIDO AZOTO"
    low_limit = 1
    unificazione_via_bari_e_san_francesco_da_paola = True

    wind_db = read_multiple_wind_files(base_path / "vento")

    df = import_database_inquinamento(base_path, base_name_inquinanti, inquinante)
    df = remove_wrong_measurements(df, low_limit=low_limit)
    if unificazione_via_bari_e_san_francesco_da_paola:
        df = unifica_via_bari_e_san_francesco_da_paola(df, nuovo_nome="S. Teodoro")

    df_unico = combina_inquinamento_e_vento(df_inquinanti=df, df_vento=wind_db)
    return df_unico


if __name__ == "__main__":
    db = importa_tutto(base_name_inquinanti="dati_genova")
    addresses = pd.unique(db["Indirizzo"])
    db2 = db[db["Indirizzo"]=="C.SO BUENOS AIRES -P.ZZA PAOLO DA NOVI"]
    print(db)

