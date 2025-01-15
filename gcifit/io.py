from pathlib import Path
import pandas as pd
import mrich


def parse_creoptix_files(
    data_file: str | Path,
    schema_file: str | Path,
    schema_index_column: int = 0,
    schema_type_column: int = 1,
    schema_sample_column: int = 5,
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
        y_index = int(y_str.removesuffix("_Y"))

    	# get the related schema row
        schema_row = schema.loc[y_index]

    	# get the related schema values
        type_str = schema_row[schema_type_column]
        sample_str = schema_row[schema_sample_column]

    	# construct the multi-index tuple
        multi_index_tuples.append((type_str, sample_str, fc_str, y_index))

	# assign the multi-index to the data columns
    data.columns = pd.MultiIndex.from_tuples(
        multi_index_tuples, names=["type", "sample", "fc_str", "index"]
    )

    # print some information
    for type in data.columns.get_level_values('type').unique():
    	mrich.var(f"#{type}", len(set(a[-1] for a in data[type].columns)))

    return data
