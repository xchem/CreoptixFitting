import jax.numpy as jnp

from uncertainties import ufloat
from uncertainties.umath import *


def _extract_params_by_idx(full_params, idx):
    """
    Return parameters by index for complex model
    """
    params_names   = ['ka', 'kd', 'Kd', f'Rmax:{idx:02d}', 'ka_P', 'kd_P', 'Kd_P', f'Rmax_P:{idx:02d}', 
                      'ka_C', 'kd_C', 'Kd_C', 'Rmax_C', f'rate_decay:{idx:02d}', f'alpha:{idx:02d}', f'epsilon:{idx:02d}', f'y_offset:{idx:02d}']
    params_renames = ['ka', 'kd', 'Kd', 'Rmax', 'ka_P', 'kd_P', 'Kd_P', 'Rmax_P', 
                      'ka_C', 'kd_C', 'Kd_C', 'Rmax_C', 'rate_decay', 'alpha', 'epsilon','y_offset']
    params = {}
    for key, new_key in zip(params_names, params_renames): 
        if key in full_params.keys():
            params[new_key] = full_params[key]
    return params


def _report_params(trace, experiments, map_index, return_conc=False, return_y_offset=False):
    """
    Parameters:
    ----------
    trace           : dict, mcmc trace
    experiment      : dict, information of Creoptix dataset
    idx             : index of experiment
    return_conc     : boolean, if True, treating the analyte concentrationn as parameter
    return_y_offset : boolean, if True, estimating y_offset
    ----------
    Return variables contain MAP, mean and std of paramaters
    """
    params = {}
    params_hat = {}

    params['ka'] = jnp.exp(trace['logka'][map_index])

    if 'logKeq' in trace.keys():
        Kd = 1/jnp.exp(trace['logKeq'][map_index])
        kd = jnp.exp(trace['logka'][map_index] - trace['logKeq'][map_index])
    elif 'logKd' in trace.keys():
        Kd = jnp.exp(trace['logKd'][map_index])
        kd = jnp.exp(trace['logka'][map_index] + trace['logKd'][map_index])
    else:
        kd = jnp.exp(trace['logkd'][map_index])
    params['kd'] = kd
    params['Kd'] = Kd

    for idx in range(len(experiments)):
        for key in ['Rmax', 'rate_decay', 'alpha', 'epsilon', 'y_offset']:
            if f'{key}:{idx:02d}' in trace.keys():
                params[f'{key}:{idx:02d}'] = trace[f'{key}:{idx:02d}'][map_index]
            else:
                if key in ['rate_decay', 'y_offset']:
                    params[f'{key}:{idx:02d}'] = 0.

    if 'conc' in trace.keys():
        conc = trace['conc'][map_index]*1E-6
        # experiments[idx]['adjusted_analyte_concentration'] = conc
        print('Adjusted analyte concentration: %2.2f uM' %(conc*1E6))
        params['conc'] = conc

    params_hat['ka'] = ufloat(jnp.mean(jnp.exp(trace['logka'])), jnp.std(jnp.exp(trace['logka'])))

    if 'logKeq' in trace.keys():
        Keq_hat = ufloat(jnp.mean(jnp.exp(trace['logKeq'])), jnp.std(jnp.exp(trace['logKeq'])))
        trace_kd = jnp.exp(trace['logka']-trace['logKeq'])
        trace_Kd = jnp.exp(1/trace['logKeq'])
        Kd_hat = ufloat(jnp.mean(trace_Kd), jnp.std(trace_Kd))
    elif 'logKd' in trace.keys():
        Kd_hat = ufloat(jnp.mean(jnp.exp(trace['logKd'])), jnp.std(jnp.exp(trace['logKd'])))
        Kd_hat = Kd_hat
        trace_kd = jnp.exp(trace['logka']+trace['logKd'])
    kd_hat = ufloat(jnp.mean(trace_kd), jnp.std(trace_kd))
    params_hat['kd'] = kd_hat
    params_hat['Kd'] = Kd_hat

    for idx in range(len(experiments)):
        params_hat[f'Rmax:{idx:02d}'] = ufloat(jnp.mean(trace[f'Rmax:{idx:02d}']), jnp.std(trace[f'Rmax:{idx:02d}']))

        if f'y_offset:{idx:02d}' in trace.keys():
            y_offset_hat = ufloat(jnp.mean(trace[f'y_offset:{idx:02d}']), jnp.std(trace[f'y_offset:{idx:02d}']))
        else:
            y_offset_hat = ufloat(0., 0.)
        params_hat[f'y_offset:{idx:02d}'] = y_offset_hat

    return params, params_hat


def _report_params_complex(trace, experiments, map_index, return_conc=False, return_y_offset=False):
    """
    Parameters:
    ----------
    trace           : dict, mcmc trace
    experiment      : dict, information of Creoptix dataset
    idx             : index of experiment
    return_conc     : boolean, if True, treating the analyte concentrationn as parameter
    return_y_offset : boolean, if True, estimating y_offset
    ----------
    Return variables contain MAP, mean and std of paramaters
    """
    params = {}
    params_hat = {}

    params['ka_P'] = jnp.exp(trace['logka_P'][map_index])
    params['ka_C'] = jnp.exp(trace['logka_C'][map_index])

    if 'logKeq_P' in trace.keys():
        Kd_P = 1/jnp.exp(trace['logKeq_P'][map_index])
        kd_P = jnp.exp(trace['logka_P'][map_index] - trace['logKeq_P'][map_index])
    elif 'logKd_P' in trace.keys():
        Kd_P = jnp.exp(trace['logKd_P'][map_index])
        kd_P = jnp.exp(trace['logka_P'][map_index] + trace['logKd_P'][map_index])
    else:
        kd_P = jnp.exp(trace['logkd_P'][map_index])
    params['kd_P'] = kd_P
    params['Kd_P'] = Kd_P

    if 'logKeq_C' in trace.keys():
        Kd_C = 1/jnp.exp(trace['logKeq_C'][map_index])
        kd_C = jnp.exp(trace['logka_C'][map_index] - trace['logKeq_C'][map_index])
    elif 'logKd_C' in trace.keys():
        Kd_C = jnp.exp(trace['logKd_C'][map_index])
        kd_C = jnp.exp(trace['logka_C'][map_index] + trace['logKd_C'][map_index])
    else:
        kd_C = jnp.exp(trace['logkd_C'][map_index])
    params['kd_C'] = kd_C
    params['Kd_C'] = Kd_C

    params['Rmax_C'] = trace['Rmax_C'][map_index]

    for idx in range(len(experiments)):
        for key in ['Rmax_P', 'rate_decay', 'alpha', 'epsilon', 'y_offset']:
            if f'{key}:{idx:02d}' in trace.keys():
                params[f'{key}:{idx:02d}'] = trace[f'{key}:{idx:02d}'][map_index]
            else:
                if key == 'alpha':
                    params[f'{key}:{idx:02d}'] = 0.
                if key in ['rate_decay', 'epsilon', 'y_offset']:
                    params[f'{key}:{idx:02d}'] = 0.

    if 'conc' in trace.keys():
        conc = trace['conc'][map_index]*1E-6
        # experiments[idx]['adjusted_analyte_concentration'] = conc
        print('Adjusted analyte concentration: %2.2f uM' %(conc*1E6))
        params['conc'] = conc

    params_hat['ka_P'] = ufloat(jnp.mean(jnp.exp(trace['logka_P'])), jnp.std(jnp.exp(trace['logka_P'])))

    if 'logKeq_P' in trace.keys():
        Keq_hat = ufloat(jnp.mean(jnp.exp(trace['logKeq_P'])), jnp.std(jnp.exp(trace['logKeq_P'])))
        trace_kd = jnp.exp(trace['logka_P']-trace['logKeq_P'])
        trace_Kd = jnp.exp(1/trace['logKeq_P'])
        Kd_hat = ufloat(jnp.mean(trace_Kd), jnp.std(trace_Kd))
    elif 'logKd_P' in trace.keys():
        Kd_hat = ufloat(jnp.mean(jnp.exp(trace['logKd_P'])), jnp.std(jnp.exp(trace['logKd_P'])))
        Kd_hat = Kd_hat
        trace_kd = jnp.exp(trace['logka_P']+trace['logKd_P'])
    kd_hat = ufloat(jnp.mean(trace_kd), jnp.std(trace_kd))
    params_hat['kd_P'] = kd_hat
    params_hat['Kd_P'] = Kd_hat

    params_hat['ka_C'] = ufloat(jnp.mean(jnp.exp(trace['logka_C'])), jnp.std(jnp.exp(trace['logka_C'])))

    if 'logKeq_C' in trace.keys():
        Keq_hat = ufloat(jnp.mean(jnp.exp(trace['logKeq_C'])), jnp.std(jnp.exp(trace['logKeq_C'])))
        trace_kd = jnp.exp(trace['logka_C']-trace['logKeq_C'])
        trace_Kd = jnp.exp(1/trace['logKeq_C'])
        Kd_hat = ufloat(jnp.mean(trace_Kd), jnp.std(trace_Kd))
    elif 'logKd_C' in trace.keys():
        Kd_hat = ufloat(jnp.mean(jnp.exp(trace['logKd_C'])), jnp.std(jnp.exp(trace['logKd_C'])))
        Kd_hat = Kd_hat
        trace_kd = jnp.exp(trace['logka_C']+trace['logKd_C'])
    kd_hat = ufloat(jnp.mean(trace_kd), jnp.std(trace_kd))
    params_hat['kd_C'] = kd_hat
    params_hat['Kd_C'] = Kd_hat

    params_hat['Rmax_C'] = ufloat(jnp.mean(trace['Rmax_C']), jnp.std(trace['Rmax_C']))

    for idx in range(len(experiments)):
        for key in ['Rmax_P', 'rate_decay', 'alpha', 'epsilon', 'y_offset']:
            if f'{key}:{idx:02d}' in trace.keys():
                params_hat[f'{key}:{idx:02d}'] = ufloat(jnp.mean(trace[f'{key}:{idx:02d}']), jnp.std(trace[f'{key}:{idx:02d}']))

        if not f'y_offset:{idx:02d}' in params_hat.keys():
            params_hat[f'y_offset:{idx:02d}'] = ufloat(0., 0.)

    return params, params_hat