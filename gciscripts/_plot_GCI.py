import os
import matplotlib
import matplotlib.pyplot as plt

import scipy
import numpy as np


def _Rt_model(cL_ts, ts, ka, kd, Rmax, rate_decay):
    """
    Return the GCI sensogram given the parameters
    """
    Rmax_adj = np.exp(np.log(Rmax) + np.log(1.0 - rate_decay) * ts / 60)
    _cLt = lambda t: np.interp(t, ts, cL_ts)
    _Rmaxt = lambda t: np.interp(t, ts, Rmax_adj)
    f_dRdt = lambda R, t: ka * _cLt(t) * (_Rmaxt(t) - R) - kd * R
    Rt = scipy.integrate.odeint(f_dRdt, 0.0, ts)
    return np.concatenate(Rt)


def plot_Rt_dRdt(experiment, params, t_cutoff=None, fig_name=None, OUTFILE=None):
    """
    Plotting the fitted and the observed data for both derivatives and sensorgrams
    """
    [ka, kd, Rmax] = [params["ka"], params["kd"], params["Rmax"]]
    [rate_decay, y_offset] = [params["rate_decay"], params["y_offset"]]
    ts = experiment["ts"]
    cLt = experiment["cL_scale"] * experiment["analyte_concentration"]

    Rt = experiment["Rt"]
    dRdt = experiment["dRdt"]

    Rt_hat = _Rt_model(cLt, ts, ka, kd, Rmax, rate_decay) + y_offset
    dRdt_hat = ka * cLt * (Rmax - Rt_hat) - kd * Rt_hat

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    ax1.plot(ts, dRdt, "-", label="Data")
    handles, labels = ax1.get_legend_handles_labels()
    if "ts_peak" in experiment.keys():
        ts_peak = experiment["ts_peak"]
        for _ts_peak in ts_peak:
            ts_plot = np.around(np.arange(_ts_peak[0], _ts_peak[1], 0.1), 1)
            fit_plot = np.array([t in ts_plot for t in experiment["ts"]])
            ax1.plot(
                ts[fit_plot],
                dRdt_hat[fit_plot],
                "-",
                label="Fitting",
                color="orange",
                linewidth=3.0,
            )
            handles, labels = ax1.get_legend_handles_labels()
    else:
        ax1.plot(ts, dRdt_hat, "-", label="Fitting", color="orange", linewidth=3.0)
        handles, labels = ax1.get_legend_handles_labels()
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("dRdt")

    if not t_cutoff is None:
        ax1.set_xlim(left=0.0, right=t_cutoff)

    ax2.plot(ts, Rt, "-", label="Data")
    handles, labels = ax2.get_legend_handles_labels()
    ax2.plot(ts, Rt_hat, "-", label="Fitting", color="orange", linewidth=3.0)
    handles, labels = ax2.get_legend_handles_labels()
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Surface Mass (pg/mm$^2$)")

    if not t_cutoff is None:
        ax2.set_xlim(left=0.0, right=t_cutoff)

    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.1, 0.65))

    if fig_name is not None:
        fig.suptitle(fig_name)
    plt.tight_layout()
    plt.show()

    if OUTFILE is not None:
        fig.savefig(OUTFILE, bbox_inches="tight")


def plot_Creoptix_Bayesian(
    experiment,
    MAP,
    params_mean_std,
    ax=None,
    xlim=None,
    ylim=None,
    fig_name=None,
    OUTFILE=None,
):
    """
    Plotting the fitted and the observed sensorgrams
    """
    [ka, kd, Rmax] = [MAP["ka"], MAP["kd"], MAP["Rmax"]]
    [rate_decay, y_offset] = [MAP["rate_decay"], MAP["y_offset"]]

    ts = experiment["ts"]
    if "adjusted_analyte_concentration" in experiment.keys():
        cLt = experiment["cL_scale"] * experiment["adjusted_analyte_concentration"]
    else:
        cLt = experiment["cL_scale"] * experiment["analyte_concentration"]

    Rt = experiment["Rt"]
    Rt_hat = _Rt_model(cLt, ts, ka, kd, Rmax, rate_decay) + y_offset
    to_fit = experiment["to_fit"]

    if ax is None:
        fig = plt.figure(figsize=(6, 4))
        ax = plt.axes()

    if "ts_peak" in experiment.keys():
        ts_peak = experiment["ts_peak"]
        for _ts_peak in ts_peak:
            ts_plot = np.around(np.arange(_ts_peak[0], _ts_peak[1], 0.1), 1)
            fit_plot = np.array([t in ts_plot for t in experiment["ts"]])
            ax.plot(ts[fit_plot], Rt[fit_plot], "-", label="Data", color="r", alpha=0.4)
            handles, labels = ax.get_legend_handles_labels()
    else:
        fit_plot = experiment["to_fit"]
        ax.plot(ts[fit_plot], Rt[fit_plot], ".", label="Data", color="r", alpha=0.4)
        handles, labels = ax.get_legend_handles_labels()

    ax.plot(ts, Rt_hat, "-", label="Fitting", color="k", linewidth=2.0, alpha=0.5)
    handles, labels = ax.get_legend_handles_labels()

    if params_mean_std is not None:
        [ka_hat, kd_hat, Kd_hat, Rmax_hat, y_offset_hat] = [
            params_mean_std["ka"],
            params_mean_std["kd"],
            params_mean_std["Kd"],
            params_mean_std["Rmax"],
            params_mean_std["y_offset"],
        ]
        [ka, kd, Kd, Rmax, y_offset] = [
            ka_hat.n,
            kd_hat.n,
            Kd_hat.n * 1e6,
            Rmax_hat.n,
            y_offset_hat.n,
        ]
        [ka_std, kd_std, Kd_std, Rmax_std, y_offset_std] = [
            ka_hat.s,
            kd_hat.s,
            Kd_hat.s * 1e6,
            Rmax_hat.s,
            y_offset_hat.s,
        ]
        text = (
            r"$R_{max}$: "
            + str("%2.2f" % Rmax)
            + r" $\pm$ "
            + str("%2.2E" % Rmax_std)
            + str(" (pg/mm$^2$)")
        )
        text += (
            "\n$k_a$: "
            + str("%2.2E" % ka)
            + r" $\pm$ "
            + str("%2.2E" % ka_std)
            + str(" $(M^{-1}s^{-1})$")
        )
        text += (
            "\n$k_d$: "
            + str("%2.2E" % kd)
            + r" $\pm$ "
            + str("%2.2E" % kd_std)
            + str(" (s$^{-1}$)")
        )
        text += (
            "\n$K_d$: "
            + str("%2.2f" % Kd)
            + r" $\pm$ "
            + str("%2.2f" % Kd_std)
            + str(" $\mu$M")
        )
        if y_offset != 0:
            text += (
                "\n$y_{offset}$: "
                + str("%2.2f" % y_offset)
                + r" $\pm$ "
                + str("%2.2f" % y_offset_std)
            )
    else:
        text = r"$R_{max}$: " + str("%2.2f" % Rmax) + str(" (pg/mm$^2$)")
        text += +"\n$k_a$: " + str("%2.2E" % ka) + str(" $(M^{-1}s^{-1})$")
        text += "\n$k_d$: " + str("%2.2E" % kd) + str(" (s$^{-1}$)")
        text += "\n$K_d$: " + str("%2.2f" % Kd) + str(" $\mu$M")
        if y_offset != 0:
            text += "\n$y_{offset}$: " + str("%2.2f" % y_offset)
    ax.text(0.6, 0.6, text, fontsize=11, transform=ax.transAxes, color="k")

    if "injection" in experiment.keys():
        for inj in experiment["injection"]:
            ax.vlines(
                inj[0],
                ymin=min(Rt[to_fit]),
                ymax=max(Rt_hat),
                ls="--",
                color="k",
                alpha=0.5,
            )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Surface Mass (pg/mm$^2$)")

    if not xlim is None:
        ax.set_xlim(*xlim)
    if not ylim is None:
        ax.set_ylim(*ylim)

    if fig_name is not None:
        ax.set_title(fig_name)

    plt.tight_layout()
    plt.show()

    if OUTFILE is not None:
        fig.savefig(OUTFILE, bbox_inches="tight")

    return ax


def _Rt_model_specific(
    cLt, ts, ka_P, kd_P, Rmax_P, ka_C, kd_C, Rmax_C, rate_decay, alpha, epsilon, R_CL
):
    """
    Return the GCI sensogram given the parameters
    """
    Rmax_P_adj = np.exp(np.log(Rmax_P) + np.log(1.0 - rate_decay) * ts / 60)

    _cLt = lambda t: np.interp(t, ts, cLt)
    _Rmaxt = lambda t: np.interp(t, ts, Rmax_P_adj)
    dR_PLdt = (
        lambda R_PL, t: ka_P * _cLt(t) * (_Rmaxt(t) - R_PL) - kd_P * R_PL + epsilon
    )
    # dR_CLdt  = lambda R_CL, t: ka_C*_cLt(t)*(alpha*Rmax_C - R_CL) - kd_C*R_CL

    R_PL = scipy.integrate.odeint(dR_PLdt, 0.0, ts)
    # R_CL = scipy.integrate.odeint(dR_CLdt, 0.0, ts)

    return np.concatenate(R_PL) + alpha * R_CL


def plot_Rt_dRdt_complex(
    experiment, params, t_cutoff=None, fig_name=None, no_subtraction=False, OUTFILE=None
):
    """
    Plotting the fitted and the observed data for both derivatives and sensorgrams of non-specific model
    """
    # plt.plot(experiment['Rt'] - experiment['Rt_FC1'])
    # plt.savefig(f'Raw_{fig_name}')

    if experiment["binding_type"] == "non_specific":
        [ka_C, kd_C, Rmax_C, y_offset] = [
            params["ka_C"],
            params["kd_C"],
            params["Rmax_C"],
            params["y_offset"],
        ]
        _params_NSB = {"ka": ka_C, "kd": kd_C, "Rmax": Rmax_C, "y_offset": y_offset}
        plot_Rt_dRdt(
            experiment=experiment,
            params=_params_NSB,
            t_cutoff=t_cutoff,
            fig_name=fig_name,
            OUTFILE=OUTFILE,
        )
    else:
        [ka_P, kd_P, Rmax_P, ka_C, kd_C, Rmax_C] = [
            params["ka_P"],
            params["kd_P"],
            params["Rmax_P"],
            params["ka_C"],
            params["kd_C"],
            params["Rmax_C"],
        ]
        [rate_decay, alpha, epsilon, y_offset] = [
            params["rate_decay"],
            params["alpha"],
            params["epsilon"],
            params["y_offset"],
        ]

        ts = experiment["ts"]
        if "adjusted_analyte_concentration" in experiment.keys():
            cLt = experiment["cL_scale"] * experiment["adjusted_analyte_concentration"]
        else:
            cLt = experiment["cL_scale"] * experiment["analyte_concentration"]

        if no_subtraction:
            Rt = experiment["Rt"] + experiment["Rt_FC1"]
            Rt_CL = experiment["Rt_FC1"]
            Rt_hat = (
                _Rt_model_specific(
                    cLt,
                    ts,
                    ka_P,
                    kd_P,
                    Rmax_P,
                    ka_C,
                    kd_C,
                    Rmax_C,
                    rate_decay,
                    alpha,
                    epsilon,
                    Rt_CL,
                )
                + y_offset
                + Rt_CL
            )
            dRdt = np.concatenate(([0], np.diff(Rt) / np.diff(ts)))
            dRdt_hat = np.concatenate(([0], np.diff(Rt_hat) / np.diff(ts)))
        else:
            Rt = experiment["Rt"]
            Rt_CL = experiment["Rt_FC1"]
            Rt_hat = (
                _Rt_model_specific(
                    cLt,
                    ts,
                    ka_P,
                    kd_P,
                    Rmax_P,
                    ka_C,
                    kd_C,
                    Rmax_C,
                    rate_decay,
                    alpha,
                    epsilon,
                    Rt_CL,
                )
                + y_offset
            )
            dRdt = experiment["dRdt"]
            dRdt_hat = np.concatenate(([0], np.diff(Rt_hat) / np.diff(ts)))

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
        ax1.plot(ts, dRdt, "-", label="Data")
        handles, labels = ax1.get_legend_handles_labels()
        if "ts_peak" in experiment.keys():
            ts_peak = experiment["ts_peak"]
            for _ts_peak in ts_peak:
                ts_plot = np.around(np.arange(_ts_peak[0], _ts_peak[1], 0.1), 1)
                fit_plot = np.array([t in ts_plot for t in experiment["ts"]])
                ax1.plot(
                    ts[fit_plot],
                    dRdt_hat[fit_plot],
                    "-",
                    label="Fitting",
                    color="orange",
                    linewidth=3.0,
                )
                handles, labels = ax1.get_legend_handles_labels()
        else:
            ax1.plot(ts, dRdt_hat, "-", label="Fitting", color="orange", linewidth=3.0)
            handles, labels = ax1.get_legend_handles_labels()
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("dRdt")

        if not t_cutoff is None:
            ax1.set_xlim(left=0.0, right=t_cutoff)

        ax2.plot(ts, Rt, "-", label="Data")
        handles, labels = ax2.get_legend_handles_labels()
        ax2.plot(ts, Rt_hat, "-", label="Fitting", color="orange", linewidth=3.0)
        handles, labels = ax2.get_legend_handles_labels()
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Surface Mass (pg/mm$^2$)")

        if not t_cutoff is None:
            ax2.set_xlim(left=0.0, right=t_cutoff)

        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.1, 0.65))

        if fig_name is not None:
            fig.suptitle(fig_name)
        plt.tight_layout()
        plt.show()

        if OUTFILE is not None:
            fig.savefig(OUTFILE, bbox_inches="tight")


def plot_Creoptix_Bayesian_complex(
    experiment,
    MAP,
    params_mean_std,
    ax=None,
    xlim=None,
    ylim=None,
    fig_name=None,
    no_subtraction=False,
    OUTFILE=None,
):
    """
    Plotting the fitted and the observed sensorgrams for non-specific model
    """
    if experiment["binding_type"] == "non_specific":
        [ka_C, kd_C, Rmax_C, y_offset] = [
            MAP["ka_C"],
            MAP["kd_C"],
            MAP["Rmax_C"],
            MAP["y_offset"],
        ]
        _MAP_NSB = {"ka": ka_C, "kd": kd_C, "Rmax": Rmax_C, "y_offset": y_offset}
        _param_mean_std_NSB = {
            "ka": params_mean_std["ka_C"],
            "kd": params_mean_std["kd_C"],
            "Kd": params_mean_std["Kd_C"],
            "Rmax": params_mean_std["Rmax_C"],
            "y_offset": params_mean_std["y_offset"],
        }
        plot_Creoptix_Bayesian(
            experiment=experiment,
            MAP=_MAP_NSB,
            params_mean_std=_param_mean_std_NSB,
            ax=ax,
            xlim=xlim,
            ylim=ylim,
            fig_name=fig_name,
            OUTFILE=OUTFILE,
        )
    else:
        [ka_P, kd_P, Rmax_P, ka_C, kd_C, Rmax_C] = [
            MAP["ka_P"],
            MAP["kd_P"],
            MAP["Rmax_P"],
            MAP["ka_C"],
            MAP["kd_C"],
            MAP["Rmax_C"],
        ]
        [rate_decay, alpha, epsilon, y_offset] = [
            MAP["rate_decay"],
            MAP["alpha"],
            MAP["epsilon"],
            MAP["y_offset"],
        ]

        ts = experiment["ts"]
        if "adjusted_analyte_concentration" in experiment.keys():
            cLt = experiment["cL_scale"] * experiment["adjusted_analyte_concentration"]
        else:
            cLt = experiment["cL_scale"] * experiment["analyte_concentration"]

        if no_subtraction:
            Rt = experiment["Rt"] + experiment["Rt_FC1"]
            Rt_CL = experiment["Rt_FC1"]
            Rt_hat = (
                _Rt_model_specific(
                    cLt,
                    ts,
                    ka_P,
                    kd_P,
                    Rmax_P,
                    ka_C,
                    kd_C,
                    Rmax_C,
                    rate_decay,
                    alpha,
                    epsilon,
                    Rt_CL,
                )
                + y_offset
                + Rt_CL
            )
        else:
            Rt = experiment["Rt"]
            Rt_CL = experiment["Rt_FC1"]
            Rt_hat = (
                _Rt_model_specific(
                    cLt,
                    ts,
                    ka_P,
                    kd_P,
                    Rmax_P,
                    ka_C,
                    kd_C,
                    Rmax_C,
                    rate_decay,
                    alpha,
                    epsilon,
                    Rt_CL,
                )
                + y_offset
            )

        to_fit = experiment["to_fit"]

        if ax is None:
            fig = plt.figure(figsize=(6, 4))
            ax = plt.axes()

        if "ts_peak" in experiment.keys():
            ts_peak = experiment["ts_peak"]
            for _ts_peak in ts_peak:
                ts_plot = np.around(np.arange(_ts_peak[0], _ts_peak[1], 0.1), 1)
                fit_plot = np.array([t in ts_plot for t in experiment["ts"]])
                ax.plot(
                    ts[fit_plot], Rt[fit_plot], "-", label="Data", color="r", alpha=0.4
                )
                handles, labels = ax.get_legend_handles_labels()
        else:
            fit_plot = experiment["to_fit"]
            ax.plot(ts[fit_plot], Rt[fit_plot], ".", label="Data", color="r", alpha=0.4)
            handles, labels = ax.get_legend_handles_labels()

        ax.plot(ts, Rt_hat, "-", label="Fitting", color="k", linewidth=2.0, alpha=0.5)
        handles, labels = ax.get_legend_handles_labels()

        if params_mean_std is not None:
            [ka_hat, kd_hat, Kd_hat, Rmax_hat, y_offset_hat] = [
                params_mean_std["ka_P"],
                params_mean_std["kd_P"],
                params_mean_std["Kd_P"],
                params_mean_std["Rmax_P"],
                params_mean_std["y_offset"],
            ]
            [ka, kd, Kd, Rmax, y_offset] = [
                ka_hat.n,
                kd_hat.n,
                Kd_hat.n * 1e6,
                Rmax_hat.n,
                y_offset_hat.n,
            ]
            [ka_std, kd_std, Kd_std, Rmax_std, y_offset_std] = [
                ka_hat.s,
                kd_hat.s,
                Kd_hat.s * 1e6,
                Rmax_hat.s,
                y_offset_hat.s,
            ]
            text = (
                r"$R_{max}$: "
                + str("%2.2f" % Rmax)
                + r" $\pm$ "
                + str("%2.2E" % Rmax_std)
                + str(" (pg/mm$^2$)")
            )
            text += (
                "\n$k_a$: "
                + str("%2.2E" % ka)
                + r" $\pm$ "
                + str("%2.2E" % ka_std)
                + str(" $(M^{-1}s^{-1})$")
            )
            text += (
                "\n$k_d$: "
                + str("%2.2E" % kd)
                + r" $\pm$ "
                + str("%2.2E" % kd_std)
                + str(" (s$^{-1}$)")
            )
            text += (
                "\n$K_d$: "
                + str("%2.2f" % Kd)
                + r" $\pm$ "
                + str("%2.2f" % Kd_std)
                + str(" $\mu$M")
            )
            if y_offset != 0:
                text += (
                    "\n$y_{offset}$: "
                    + str("%2.2f" % y_offset)
                    + r" $\pm$ "
                    + str("%2.2f" % y_offset_std)
                )
        else:
            text = r"$R_{max}$: " + str("%2.2f" % Rmax) + str(" (pg/mm$^2$)")
            text += +"\n$k_a$: " + str("%2.2E" % ka) + str(" $(M^{-1}s^{-1})$")
            text += "\n$k_d$: " + str("%2.2E" % kd) + str(" (s$^{-1}$)")
            text += "\n$K_d$: " + str("%2.2f" % Kd) + str(" $\mu$M")
            if y_offset != 0:
                text += "\n$y_{offset}$: " + str("%2.2f" % y_offset)
        ax.text(0.6, 0.6, text, fontsize=11, transform=ax.transAxes, color="k")

        if "injection" in experiment.keys():
            for inj in experiment["injection"]:
                ax.vlines(
                    inj[0],
                    ymin=min(Rt[to_fit]),
                    ymax=max(Rt_hat),
                    ls="--",
                    color="k",
                    alpha=0.5,
                )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Surface Mass (pg/mm$^2$)")

        if not xlim is None:
            ax.set_xlim(*xlim)
        if not ylim is None:
            ax.set_ylim(*ylim)

        if fig_name is not None:
            ax.set_title(fig_name)

        plt.tight_layout()
        plt.show()

        if OUTFILE is not None:
            fig.savefig(OUTFILE, bbox_inches="tight")

        return ax
