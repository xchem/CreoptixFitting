import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_GratingCoupledInterferometrySensogram(file_name, skiprows=0, keys_included=[], keys_excluded=[], tlim=None,
                                               plotting=True):
    """ 
    Load Creoptix Wave sensograms
    """
    sensograms = pd.read_csv(file_name, sep=',', skiprows=skiprows)

    keys_loaded = []
    keys_avoided = []
    ts = []
    rs = []
    for (key_x, key_y) in zip(sensograms.keys()[::2], sensograms.keys()[1::2]):
        if (key_y in keys_excluded):
            keys_avoided.append(key_y)
            continue
        if (key_y not in keys_included) and (keys_included != []):
            keys_avoided.append(key_y)
            continue
        keys_loaded.append(key_y)
        ts.append(sensograms[key_x].values)
        rs.append(sensograms[key_y].values)

        # print('Keys loaded : ', ','.join(keys_loaded))
        # print('Keys avoided: ', ','.join(keys_avoided))

        # Check that the x axis have all the same values
        for n in range(len(keys_loaded)):
            assert((ts[n]==ts[0]).all())
        ts = np.array(ts[0])
        rs = np.array(rs).T

        mean_rs = np.mean(rs,1)
        std_rs = np.std(rs,1)

        if plotting:
            plt.figure()
            plt.plot(ts,rs)
            plt.xlabel('Time (s)')
            plt.ylabel('Response')
            if tlim is not None:
                plt.xlim(tlim)
            plt.legend(keys_loaded)

        if tlim is not None:
            to_keep = np.logical_and(ts>tlim[0], ts<tlim[1])
            ts = ts[to_keep]
            mean_rs = mean_rs[to_keep]
            std_rs = std_rs[to_keep]

        return (ts, mean_rs, std_rs)


def _extract_GCI(init_data, args, plotting=False):
    """
    Based on given input, returning the list of dictionary, each contains experimental information
    Return: 
        Rt          : numpy.array, response of analyte sample
        dRdt        : numpy.array, derivatives of response of analyte sample
        ts          : numpy.array, time series
        cL_scale    : numpy.array, normalized response of calibration sample, used as concentration function over the time
        to_fit      : numpy.array of boolean values indicating association (False) or dissociation (True) segments
    """
    experiments = init_data.copy()
    
    for experiment in experiments:
        
        if experiment['type'] == 'GratingCoupledInterferometrySensogram':
            
            end_dissociation = experiment['end_dissociation']

            if args.fitting_complex or (not args.fitting_subtract):
                # FC1
                print("Loading", experiment['keys_included_FC1'], "from", experiment['file_name'])
                if 'keys_included_FC1' in experiment.keys():
                    ts, mean_rs, std_rs = load_GratingCoupledInterferometrySensogram(experiment['file_name'], 0, keys_included=experiment['keys_included_FC1'], 
                                                                                     tlim=experiment['tlim'], plotting=plotting)
                    experiment['Rt_FC1'] = mean_rs
                
                if 'calibration_keys_FC1' in experiment.keys():
                    ts, mean_rs, std_rs = load_GratingCoupledInterferometrySensogram(experiment['calibration_file_name'], 0, keys_included=experiment['calibration_keys_FC1'], 
                                                                                     tlim=experiment['tlim'], plotting=plotting)
                    mean_rs = [max(0, x) for x in mean_rs]
                    experiment['cL_scale_FC1'] = mean_rs/np.max(mean_rs)

            if args.fitting_subtract:
                print('Loading sample response from', experiment['keys_included'], "from", experiment['file_name'])
                ts, mean_rs, std_rs = load_GratingCoupledInterferometrySensogram(experiment['file_name'], 0, keys_included=experiment['keys_included'],
                                                                                 tlim=experiment['tlim'], plotting=plotting)
                experiment['ts'] = ts
                experiment['Rt'] = mean_rs
            else:
                # FC2
                print('Loading sample response from', experiment['keys_included'], "from", experiment['file_name'])
                ts, mean_rs, std_rs = load_GratingCoupledInterferometrySensogram(experiment['file_name'], 0, keys_included=experiment['keys_included'],
                                                                                 tlim=experiment['tlim'], plotting=plotting)
                experiment['ts'] = ts
                experiment['Rt'] = mean_rs - experiment['Rt_FC1']
                
            # c(t)
            print('Loading analyte concentration from', experiment['calibration_keys_included'], "from", experiment['calibration_file_name'])
            ts, mean_rs, std_rs = load_GratingCoupledInterferometrySensogram(experiment['calibration_file_name'], 0, keys_included=experiment['calibration_keys_included'],
                                                                             tlim=experiment['tlim'], plotting=plotting)
            mean_rs = [max(0, x) for x in mean_rs]
            experiment['cL_scale'] = mean_rs/np.max(mean_rs)

            # Derivative
            experiment['dRdt'] = np.concatenate(([0], np.diff(experiment['Rt'])/np.diff(experiment['ts'])))

            if end_dissociation is None:
                experiment['to_fit'] = experiment['cL_scale']<0.1
            else:
                experiment['to_fit'] = (experiment['cL_scale']<0.1)*(experiment['ts']<=end_dissociation)
        
        else:
            raise ValueError(f"{experiment['type']} not supported")

    return experiments


def _load_gci_infor_from_argument(analyte_keys, calibration_keys, args):
    """
    Parameters:
    ----------
    analyte_keys        : list of keys for analyte samples
    calibration_keys    : list of keys for calibration samples
    args                : class comprises other model arguments
    ----------
    Return              : dictionary contains experimental information
    """
    expt = {'type'                      : 'GratingCoupledInterferometrySensogram',
            'file_name'                 : args.analyte_file,
            'keys_included'             : analyte_keys,
            'analyte_concentration'     : args.analyte_concentration_uM*1E-6, #M
            'calibration_file_name'     : args.calibration_file,
            'calibration_keys_included' : calibration_keys,
            'end_dissociation'          : args.end_dissociation,
            'tlim'                      : (0, args.end_dissociation),
            }
    
    if args.fitting_complex or (not args.fitting_subtract):
        expt['keys_included_FC1']    = args.analyte_keys_included_FC1
        expt['calibration_keys_FC1'] = f'Fc=1-{calibration_keys[5:]}'

    if args.fitting_complex:
        if analyte_keys.startswith('Fc=1'):
            expt['binding_type'] = 'non_specific'
        else:
            expt['binding_type'] = 'specific'   
    else:
        expt['binding_type']     = None
    
    return expt