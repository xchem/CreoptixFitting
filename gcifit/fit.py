import os
from typer import Typer
from .io import parse_creoptix_files, write_all_fitting_csvs, clear_csvs, get_file_pairs
import subprocess
from pathlib import Path
import json
import mrich

app = Typer()


@app.command()
def fit_all(data_file: str, schema_file: str, config_file: str, out_dir: str):

    out_dir = Path(out_dir)

    clear_csvs(out_dir)
    df = parse_creoptix_files(data_file, schema_file)
    write_all_fitting_csvs(df, out_dir)
    pairs = get_file_pairs(out_dir)

    for sample_file, calibration_file in pairs:
        mrich.bold(sample_file.name, calibration_file.name)
        fit(sample_file, calibration_file, config_file, out_dir)
        # break


def fit(sample_file, calibration_file, config_file, out_dir):

    sample_key = sample_file.name.removesuffix(".csv")
    calibration_key = calibration_file.name.removesuffix(".csv")

    out_dir = Path(out_dir) / f"fit_{sample_key}_{calibration_key}"

    assert sample_key.endswith("uM")

    analyte_concentration = float(sample_key.split("_")[-1].removesuffix("uM"))

    commands = [
        "sbatch",
        "--job-name",
        out_dir.name,
        "../slurm/run_python.sh",
        "scripts/run_fitting_GCI.py",
    ]
    commands.append("--analyte_concentration_uM")
    commands.append(f"{analyte_concentration:.1f}")
    commands.append("--out_dir")
    commands.append(str(out_dir.resolve()))
    commands.append("--analyte_file")
    commands.append(str(sample_file.resolve()))
    commands.append("--calibration_file")
    commands.append(str(calibration_file.resolve()))
    commands.append("--analyte_keys_included")
    commands.append("2-1_Y;3-1_Y;4-1_Y")
    commands.append("--calibration_keys_included")
    commands.append("2-1_Y;3-1_Y;4-1_Y")

    config = json.load(open(config_file, "rt"))

    for flag in config["flags"]:
        commands.append(f"--{flag}")
    for k, v in config["kwargs"].items():
        commands.append(f"--{k}")
        commands.append(str(v))

    with mrich.loading("running run_fitting_GCI.py"):
        x = subprocess.run(commands, shell=False, stdout=subprocess.PIPE)

    if x.returncode != 0:
        mrich.error("Could not submit job")
    else:
        mrich.success(x.stdout.decode().strip())


def main():
    app()


if __name__ == "__main__":
    main()

# python -m gcifit.fit example/input/sample_traces_2.txt example/input/sample_schema_2.txt config.json test_output2
