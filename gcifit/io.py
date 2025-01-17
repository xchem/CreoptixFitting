
import itertools
from pathlib import Path
import pandas as pd
import mrich
from mrich import print

def parse_creoptix_files(
    data_file: str | Path,
    schema_file: str | Path,
    schema_index_column: int = 0,
    schema_type_column: int = 1,
    schema_sample_column: int = 5,
    debug: bool = True,
):

    # read the data file
    mrich.reading(data_file)
    data = pd.read_csv(data_file, sep="\t")

    # get the column names
    x_columns = list(c for c in data.columns if c.endswith("X"))
    y_columns = list(c for c in data.columns if c.endswith("Y"))
    assert len(x_columns) == len(y_columns)

    # read the schema and prepare the dataframe
    data.rename(columns={x_columns[0]: "time"}, inplace=True)
    data.set_index("time", inplace=True)
    data.drop(columns=x_columns, inplace=True, errors="ignore")

    # read the schema and prepare the dataframe
    mrich.reading(schema_file)
    schema = pd.read_csv(schema_file, sep="\t", header=None)
    schema.set_index(schema_index_column, inplace=True)

    # rename the columns
    multi_index_tuples = []
    for y_col in y_columns:

        # parse the column name
        fc_str, y_str = y_col.split(" - ")
        fc_str = fc_str.removeprefix("Fc=")
        cycle_index = int(y_str.removesuffix("_Y"))

        # get the related schema row
        schema_row = schema.loc[cycle_index]

        # get the related schema values
        type_str = schema_row[schema_type_column]
        sample_str = schema_row[schema_sample_column]

        if debug:
            mrich.debug(type_str, sample_str, cycle_index, fc_str)

        # construct the multi-index tuple
        multi_index_tuples.append((type_str, sample_str, cycle_index, fc_str))

    # assign the multi-index to the data columns
    data.columns = pd.MultiIndex.from_tuples(
        multi_index_tuples, names=["type", "sample", "cycle", "channel"]
    )

    # print some information
    for type in data.columns.get_level_values('type').unique():
        mrich.var(f"#cycles {type}", len(set(a[1] for a in data[type].columns)))

    if "Sample" not in data:
        mrich.error("No sample data")
        raise ValueError("No sample data")
    if "DMSO Cal." not in data:
        mrich.error("No calibration data")
        raise ValueError("No calibration data")

    return data

def write_all_fitting_csvs(data: pd.DataFrame, out_dir: str):

    # output directory
    out_dir = Path(out_dir)
    mrich.writing(out_dir)
    out_dir.mkdir(exist_ok=True)
    
    # split data by type
    try:
        samples = data["Sample"]
        calibrations = data["DMSO Cal."]
    except KeyError:
        mrich.print(data.columns)
        raise
        
    # prep calibration variables
    calibration_names = set(a[0] for a in calibrations.columns)
    assert len(calibration_names) == 1, "Multiple calibration types"
    calibration_name = list(calibration_names)[0]
    calibration_cycles = sorted(list(set(a[1] for a in calibrations.columns)))

    # write calibration files
    for calibration_cycle in mrich.track(calibration_cycles, prefix="Writing calibration CSVs"):
        file_name = f"calibration_cycle{calibration_cycle:04}.csv"
        write_trace_csv(calibrations[calibration_name][calibration_cycle], out_dir / file_name)

    # prep sample variables
    sample_names = set(a[0] for a in samples.columns)
    sample_cycles = set(a[1] for a in samples.columns)

    sample_pairs = set((a[0],a[1]) for a in samples.columns)

    # write calibration files
    for sample_name, sample_cycle in mrich.track(sample_pairs, prefix="Writing sample CSVs"):
        sample_code = "_".join(parse_sample_name(sample_name))
        file_name = f"sample_cycle{sample_cycle:04}_{sample_code}.csv"
        write_trace_csv(samples[sample_name][sample_cycle], out_dir / file_name)

def write_trace_csv(df: pd.DataFrame, out_file: Path):

    out_data = { 
        "2-1_X":[], "2-1_Y":[], 
        "3-1_X":[], "3-1_Y":[], 
        "4-1_X":[], "4-1_Y":[], 
    }

    assert len(df.columns) == 3

    for time, row in df.iterrows():

        for k,v in row.to_dict().items():
            out_data[f"{k}_X"].append(time)
            out_data[f"{k}_Y"].append(v)

    out_df = pd.DataFrame(out_data)

    mrich.writing(out_file)
    out_df.to_csv(out_file, index=False)

def parse_sample_name(name):

    values = name.split(" - ")

    assert len(values) >= 3

    location = values[0]
    sample = values[1]
    concentration = values[2]

    concentration = concentration.replace(" ", "").replace("μM", "uM")

    return location, sample, concentration

def clear_csvs(directory):

    # Define the directory and pattern
    directory = Path(directory)
    pattern = '*.csv'  # Example pattern to match all .txt files

    # Iterate through files matching the pattern and delete them
    for file in directory.glob(pattern):
        if file.is_file():  # Ensure it's a file (not a directory)
            file.unlink()

def get_file_pairs(directory):

    directory = Path(directory)

    calibration_files = directory.glob("calibration_cycle????.csv")
    sample_files = directory.glob("sample_cycle????_*.csv")

    return itertools.product(sample_files, calibration_files)

def parse_summary_csv(file: str | Path):

    summary_df = pd.read_csv(file)
    summary_df.rename(columns={summary_df.columns[0]:"value"}, inplace=True)
    summary_df.set_index("value", inplace=True)

    return summary_df

def parse_all_summary_csvs(out_dir: str | Path):

    out_dir = Path(out_dir)

    all_data = {}

    for file in out_dir.glob("*/Summary.csv"):
        df = parse_summary_csv(file)

        subdir_name = file.parent.name

        sample_name = subdir_name.removeprefix("fit_sample_cycle")[5:-22]  # 0085_L-E4_ASAP-0029213-001_10uM
        calibration_cycle = int(subdir_name[-4:])

        location, sample, concentration = sample_name.split("_")

        for col in df.columns:
            series = df[col]        
            all_data[location, sample, concentration, calibration_cycle, col] = series.values

        # break

    all_df = pd.DataFrame(all_data)
    all_df.columns.names = ['Location','Sample', 'Concentration', "Calibration Cycle", "Column"]
    all_df.index = df.index

    return all_df.T
