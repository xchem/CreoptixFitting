import os
from typer import Typer
from .io import parse_creoptix_files, write_all_fitting_csvs, clear_csvs, get_file_pairs
import subprocess
from pathlib import Path
import json
import mrich
import time

app = Typer()


@app.command()
def fit_all(
    data_file: str, schema_file: str, config_file: str, out_dir: str, sleep: float = 0.1
):

    mrich.h1("gcifit.fit")

    mrich.h2("Configuration")

    root_dir = Path(__file__).resolve().parent.parent
    out_dir = Path(out_dir).resolve()
    log_dir = root_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    mrich.h3("Directories")
    mrich.var("root_dir", root_dir)
    mrich.var("out_dir", out_dir)
    mrich.var("log_dir", log_dir)

    data_file = Path(data_file).resolve()
    schema_file = Path(schema_file).resolve()
    config_file = Path(config_file).resolve()
    slurm_file = root_dir / "slurm_python.sh"
    fit_script = root_dir / "gciscripts" / "run_fitting_GCI.py"

    mrich.h3("Input files")
    mrich.var("data_file", data_file)
    mrich.var("schema_file", schema_file)
    mrich.var("config_file", config_file)
    mrich.var("slurm_file", slurm_file)
    mrich.var("fit_script", fit_script)

    assert data_file.exists(), f"File not found: {data_file}"
    assert schema_file.exists(), f"File not found: {schema_file}"
    assert config_file.exists(), f"File not found: {config_file}"
    assert slurm_file.exists(), f"File not found: {slurm_file}"
    assert fit_script.exists(), f"File not found: {fit_script}"

    config = json.load(open(config_file, "rt"))

    mrich.h3("Config")
    mrich.print(config)

    mrich.h2("Creating input files")

    with mrich.loading("Clearing output directory..."):
        clear_csvs(out_dir)

    df = parse_creoptix_files(data_file, schema_file)
    write_all_fitting_csvs(df, out_dir)

    with mrich.loading("Getting sample-calibration pairs..."):
        pairs = list(get_file_pairs(out_dir))

    if not pairs:
        raise FileNotFoundError("No sample and calibration files found")

    mrich.h2("Job submission")

    error_count = 0
    job_ids = set()

    for sample_file, calibration_file in mrich.track(pairs, prefix="Submitting jobs"):

        job_id = fit(
            sample_file=sample_file,
            calibration_file=calibration_file,
            config=config,
            out_dir=out_dir,
            slurm_file=slurm_file,
            log_dir=log_dir,
            fit_script=fit_script,
        )

        if job_id:
            job_ids.add(job_id)
        else:
            error_count += 1

        mrich.set_progress_field("ok", len(job_ids))
        mrich.set_progress_field("error", error_count)

        if sleep:
            time.sleep(sleep)

        break

    if job_ids:
        mrich.var("Job ID's", " ".join(str(i) for i in job_ids))
        mrich.success("Submitted", len(job_ids), "jobs")

    if error_count:
        mrich.error("Error submitting", error_count, "jobs")


def fit(
    *, sample_file, calibration_file, config, out_dir, slurm_file, log_dir, fit_script
):

    sample_key = sample_file.name.removesuffix(".csv")
    calibration_key = calibration_file.name.removesuffix(".csv")

    out_dir = Path(out_dir) / f"fit_{sample_key}_{calibration_key}"

    assert sample_key.endswith("uM")

    analyte_concentration = float(sample_key.split("_")[-1].removesuffix("uM"))

    commands = [
        "sbatch",
        "--job-name",
        out_dir.name,
        "--output=" f"{log_dir.resolve()}/%j.log",
        "--error=" f"{log_dir.resolve()}/%j.log",
        "--partition=cs04r",
        str(slurm_file.resolve()),
        str(fit_script.resolve()),
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

    for flag in config["flags"]:
        commands.append(f"--{flag}")
    for k, v in config["kwargs"].items():
        commands.append(f"--{k}")
        commands.append(str(v))

    with mrich.loading("running run_fitting_GCI.py"):
        x = subprocess.run(
            commands, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

    if x.returncode != 0:
        mrich.error(x.stderr.decode().strip())
        return False

    else:
        job_id = int(x.stdout.decode().strip().split()[-1])
        return job_id


def main():
    app()


if __name__ == "__main__":
    main()

# python -m gcifit.fit example/input/sample_traces_2.txt example/input/sample_schema_2.txt config.json test_output2
