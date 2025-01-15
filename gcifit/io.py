
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
	
	data = pd.read_csv(data_file, sep='\t')

	x_columns = list(c for c in data.columns if c.endswith("X"))
	y_columns = list(c for c in data.columns if c.endswith("Y"))

	mrich.var("#X columns", len(x_columns))
	mrich.var("#Y columns", len(y_columns))

	data.rename(columns={x_columns[0]:"time"}, inplace=True)
	data.set_index("time", inplace=True)
	data.drop(columns=x_columns, inplace=True, errors="ignore")

	schema = pd.read_csv(schema_file, sep='\t', header=None)
	schema.set_index(schema_index_column, inplace=True)

	rename_map = {}

	for y_col in y_columns:

		fc_str, y_str = y_col.split(" - ")
		fc_str = fc_str.removeprefix("Fc=")
		y_index = int(y_str.removesuffix("_Y"))

		# mrich.print(y_col, fc_str, y_str, y_index)

		schema_row = schema.loc[y_index]

		type_str = schema_row[schema_type_column]
		sample_str = schema_row[schema_sample_column]

		# mrich.print(y_index, type_str, sample_str)

		new_col_name = f"[{type_str}] {sample_str} ({fc_str} {y_index})"

		rename_map[y_col] = new_col_name

	data.rename(columns=rename_map, inplace=True)

	return data
