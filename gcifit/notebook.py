import re
import mrich
from pathlib import Path
import pandas as pd

from PIL import Image

from ipywidgets import (
    interactive,
    # BoundedIntText,
    Checkbox,
    Dropdown,
    interactive_output,
    # HBox,
    # GridBox,
    # Layout,
    VBox,
)

from .io import parse_summary_csv, parse_map_pickle


def show_outputs(
    out_dir: str | Path, pattern: str = "fit_sample_cycle????_*_calibration_cycle????"
):

    out_dir = Path(out_dir)
    subdirs = list(out_dir.glob(pattern))

    # mrich.print(subdirs)

    sample_cycles = set()
    calibration_cycles = set()
    lookup = {}
    for subdir in subdirs:
        s = subdir.name
        sample_cycle = re.search(r"sample_cycle[0-9]{4}", s)
        calibration_cycle = re.search(r"calibration_cycle[0-9]{4}", s)
        if not (sample_cycle or calibration_cycle):
            continue

        sample_cycle = int(sample_cycle.group(0)[-4:])
        calibration_cycle = int(calibration_cycle.group(0)[-4:])

        lookup[sample_cycle, calibration_cycle] = subdir
        sample_cycles.add(sample_cycle)
        calibration_cycles.add(calibration_cycle)

    # mrich.print(lookup)

    sample_cycles = sorted(sample_cycles)
    calibration_cycles = sorted(calibration_cycles)

    dropdown_sample = Dropdown(
        options=sample_cycles,
        value=sample_cycles[0],
        description="Sample:",
        disabled=False,
    )

    dropdown_calibration = Dropdown(
        options=calibration_cycles,
        value=calibration_cycles[0],
        description="Calibration:",
        disabled=False,
    )

    checkbox_map = Checkbox(description="MAP", value=True)
    checkbox_summary = Checkbox(description="MCMC statistics", value=False)
    checkbox_rty = Checkbox(description="Rt__Y.png", value=True)
    checkbox_drdty = Checkbox(description="dRdt__Y.png", value=False)
    checkbox_trace = Checkbox(description="Plot_trace.png", value=False)
    checkbox_autocorrelation = Checkbox(description="Autocorrelation.png", value=False)

    ui = VBox(
        [
            dropdown_sample,
            dropdown_calibration,
            checkbox_map,
            checkbox_summary,
            checkbox_rty,
            checkbox_trace,
            checkbox_drdty,
            checkbox_autocorrelation,
        ]
    )

    def widget(
        sample_cycle,
        calibration_cycle,
        show_map,
        show_summary,
        show_rty,
        show_drdty,
        show_trace,
        show_autocorrelation,
    ):

        subdir = lookup.get((sample_cycle, calibration_cycle))

        if subdir is None:
            mrich.error("No subdirectory found")
            return

        mrich.var("Subdirectory", subdir.name)

        if show_map:
            mrich.h3("MAP (map.pickle)")
            df = parse_map_pickle(subdir / "map.pickle")
            display(df)

        # summary CSV
        if show_summary:
            mrich.h3("MCMC statistics (Summary.csv)")
            summary_df = parse_summary_csv(subdir / "Summary.csv")
            display(summary_df)

        if show_rty:
            mrich.h3("Rt__Y.png")
            display(Image.open(subdir / "Rt__Y.png"))

        if show_drdty:
            mrich.h3("dRdt__Y.png")
            display(Image.open(subdir / "dRdt__Y.png"))

        if show_trace:
            mrich.h3("Plot_trace.png")
            display(Image.open(subdir / "Plot_trace.png"))

        if show_autocorrelation:
            mrich.h3("Autocorrelation.png")
            display(Image.open(subdir / "Autocorrelation.png"))

        # # PNG's
        # for png_file in subdir.glob("*.png"):
        #     mrich.h3(png_file.name)
        #     img = Image.open(str(png_file.resolve()))
        #     display(img)

    out = interactive_output(
        widget,
        {
            "sample_cycle": dropdown_sample,
            "calibration_cycle": dropdown_calibration,
            "show_map": checkbox_map,
            "show_summary": checkbox_summary,
            "show_rty": checkbox_rty,
            "show_trace": checkbox_trace,
            "show_drdty": checkbox_drdty,
            "show_autocorrelation": checkbox_autocorrelation,
        },
    )

    display(ui, out)
