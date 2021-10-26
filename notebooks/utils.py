from collections import OrderedDict
from pathlib import Path
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.cosmology import WMAP9 as wmap9
from matplotlib.ticker import MultipleLocator
from scipy.stats.stats import pearsonr
from sndata.csp import DR3
from sndata.sdss import Sako18Spec
from sndata.utils import convert_to_jd

sako_18_spec = Sako18Spec()
sdss_master_table = sako_18_spec.load_table('master').to_pandas(index='CID')

dr3 = DR3()
csp_table_3 = dr3.load_table(3).to_pandas(index='SN')

proposed_cutoff = 7
feature_alias = {
    'pW1': 'Ca ii H&K',
    'pW2': 'Si ii λ4130',
    'pW3': 'Mg ii, Fe ii',
    'pW4': 'Fe ii, Si ii',
    'pW5': 'S ii λ5449, λ5622',
    'pW6': 'Si ii λ5972',
    'pW7': 'Si ii λ6355',
    'pW8': 'Ca ii IR triplet'}


@np.vectorize
def get_csp_t0(obj_id):
    """Get the t0 value for CSP targets

    Args:
        obj_id (str): The object identifier

    Returns:
        The time of B-band maximum in units of
    """
    
    # Unknown object ID
    if obj_id not in csp_table_3.index:
        return np.nan

    t0_mjd = csp_table_3.loc[obj_id]['T(Bmax)']

    # Known object Id with unknown peak time
    if np.isnan(t0_mjd):
        return np.nan

    return convert_to_jd(t0_mjd)


@np.vectorize
def get_sdss_t0(obj_id):
    """Get the t0 value for CSP targets

    Args:
        obj_id (str): The object identifier

    Returns:
        The time of B-band maximum in units of
    """
    
    obj_id = str(obj_id)
    
    # Unknown object ID
    if obj_id not in sdss_master_table.index:
        return np.nan

    t0_mjd = sdss_master_table.loc[obj_id]['PeakMJDSALT2zspec']

    # Known object Id with unknown peak time
    if np.isnan(t0_mjd):
        return np.nan

    return convert_to_jd(t0_mjd)


def branch_classification(pipeline_data):
    """Return a series with the Branch classification for each objects
    
    Args:
        pipeline_data (DataFrame): Data that has been read from a pipeline output file
        
    Returns:
        A Pandas Series object
    """
    
    peak_vals = pipeline_data[pipeline_data.is_peak]
    df = peak_vals.loc['pW6'].join(
        peak_vals.loc['pW7'], 
        lsuffix='_pw6', 
        rsuffix='_pw7', 
        on='obj_id')

    classifications = np.full_like(df.index, 'unknown', dtype='U10')
    classifications[df.pew_pw6 > 30] = 'CL'
    classifications[(df.pew_pw7 > 105) & (df.pew_pw6 < 30)] = 'BL'
    classifications[df.pew_pw7 < 70] = 'SS'
    classifications[(70 <= df.pew_pw7) & (df.pew_pw7 <= 105) & (df.pew_pw6 <= 30)] = 'CN'
    return pd.Series(classifications, index=df.index, name='branch_type')


def read_in_pipeline_result(path, survey, drop_flagged=False):
    """Read pEW values from analysis pipeline file
    
    Adds columns for Branch classifications determined by the
    measured pEW values and spectral subtypes determined from 
    CSP DR1.
    
    Args:
        path          (str): Path of the file to read
        survey        (str): Read in data for either `csp` or `sdss`
        drop_flagged (bool): Optionally drop flagged measurements / spectra

    Returns:
        A pandas Dataframe indexed by feat_name and obj_id
    """
    
    df = pd.read_csv(path, index_col=['feat_name', 'obj_id'])

    # Add phases using CSP DR3 t0 values
    obj_id = df.index.get_level_values(1)
    
    if survey == 'csp':
        df['phase'] = df.time - get_csp_t0(obj_id)
        
        csp_table_2 = dr3.load_table(2)
        subtypes = pd.DataFrame({'spec_type': csp_table_2['Subtype1']}, index=csp_table_2['SN'])
        df = df.join(subtypes, on='obj_id')

    elif survey == 'sdss':
        df['phase'] = df.time - get_sdss_t0(obj_id)
        df['spec_type'] = 'unknown'
        
        sako_master = sako_18_spec.load_table('master').to_pandas()
        sako_master = sako_master.rename({'CID': 'obj_id'}, axis='columns')
        sako_master['obj_id'] = sako_master.obj_id.astype(int)
        sako_master = sako_master.set_index('obj_id').replace(-99, np.nan)
        
        sako_master['arcmin'] = sako_master.separationhost / 60
        sako_master['kpc'] = wmap9.kpc_comoving_per_arcmin(sako_master.arcmin)

        df = df.join(sako_master, how='inner')
        
    else:
        warn(f'Could not calculate phases for survey {survey}. Expected "csp" or "sdss".')
    
    if drop_flagged:
        df = df[(df.spec_flag != 1) & (df.feat_flag != 1)]
    
    # Label measurements that represent that were taken nearest peak brightness
    df['delta_t'] = df.phase.abs()
    df = df.sort_values('delta_t')
    df['is_peak'] = ~df.index.duplicated()
    
    df = df.join(branch_classification(df), on='obj_id')
    return df
