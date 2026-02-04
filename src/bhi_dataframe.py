"""
Load and organize BHI data into a pandas DataFrame.

This module provides functions to:
1. Load BHI data from .mat files (experiment and baseline)
2. Extract epochs corresponding to extreme events
3. Organize into a structured DataFrame with statistics
"""

import numpy as np
import pandas as pd
import scipy.io
from pathlib import Path

# Constants
EEG_BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma']
HRV_BANDS = ['lf', 'hf']


def load_bhi_file(subject_id, phase='experiment'):
    """
    Load BHI data from .mat file.
    
    Parameters
    ----------
    subject_id : str
        Subject ID (e.g., 'i01')
    phase : str
        'experiment' or 'baseline'
    
    Returns
    -------
    dict with keys: 'BtH', 'HtB', 'time_BtH', 'time_HtB', 'fs_bhi', 'channels'
    None if file not found or error occurs
    """
    file_path = Path(f'results/bhi/{phase}/{subject_id}_bhi.mat') if phase=='experiment' else Path(f'results/bhi/{phase}/{subject_id}_bhi_baseline.mat')
    
    if not file_path.exists():
        return None
    
    try:
        mat_data = scipy.io.loadmat(str(file_path))
        
        return {
            'BtH': mat_data['BHI']['BtH'][0][0],  # (5, 2, 19, n_samples)
            'HtB': mat_data['BHI']['HtB'][0][0],  # (2, 5, 19, n_samples)
            'time_BtH': mat_data['BHI']['time'][0][0][0][0][1][0],
            'time_HtB': mat_data['BHI']['time'][0][0][0][0][0][0],
            'fs_bhi': float(mat_data['BHI']['FS_bhi'][0][0]),
            'channels': mat_data['BHI']['channels'][0][0][0]
        }
    except Exception as e:
        print(f"Error loading {subject_id} {phase}: {str(e)}")
        return None


def extract_bhi_epoch(bhi_array, time_array, start_time, end_time, fs_bhi=2.0, padding_seconds=0.0):
    """
    Extract epoch from BHI data based on time window.
    
    Parameters
    ----------
    bhi_array : ndarray
        BHI data array (5, 2, 19, n_samples) for BtH or (2, 5, 19, n_samples) for HtB
    time_array : ndarray
        Time array corresponding to BHI data
    start_time : float
        Epoch start time (seconds)
    end_time : float
        Epoch end time (seconds)
    fs_bhi : float
        Sampling frequency of BHI data (default: 2.0 Hz)
    padding_seconds : float
        Extra time (in seconds) to add before start_time and after end_time (default: 0.0)
    
    Returns
    -------
    ndarray : Extracted epoch with same shape except time dimension
    """
    # Apply padding to time window
    padded_start = start_time - padding_seconds
    padded_end = end_time + padding_seconds
    
    # Find closest index to padded start time
    start_idx = np.argmin(np.abs(time_array - padded_start))
    
    # Calculate expected number of samples (including padding)
    duration = padded_end - padded_start
    n_samples = int(np.round(duration * fs_bhi))
    
    # Extract exactly n_samples starting from start_idx
    end_idx = start_idx + n_samples
    
    # Handle edge case where we don't have enough samples
    if end_idx > len(time_array):
        end_idx = len(time_array)
        start_idx = max(0, end_idx - n_samples)
    
    return bhi_array[..., start_idx:end_idx]


def create_bhi_dataframe(extreme_events, valid_subjects=None, padding_seconds=0.0):
    """
    Create a DataFrame with BHI epochs organized by dimensions.
    
    Parameters
    ----------
    extreme_events : dict
        Dictionary mapping subject_id to extreme events with structure:
        {
            'events_exp': [{'start': float, 'end': float, 'event_type': str, 'segment_label': str}, ...],
            'n_surge': int,
            'n_drop': int,
            'n_spike': int,
            'n_total': int
        }
    valid_subjects : dict, optional
        Dictionary mapping subject_id to SubjectData objects (not used currently)
    
    Returns
    -------
    pd.DataFrame with columns:
        - id: Subject ID
        - direction: 'BtH' or 'HtB'
        - eeg_band: delta/theta/alpha/beta/gamma
        - hrv_band: lf/hf
        - channel: Complete channel object
        - channel_label: Channel name
        - event_type: SURGE/DROP/SPIKE/BL
        - emotion: POS/NEG/NEUT (None for baseline)
        - data: Raw time series array
        - mean: Mean of data
        - median: Median of data
        - z_data: Z-score normalized data array
        - z_median: Median of z-normalized data
    """
    rows = []
    
    for subject_id in extreme_events.keys():
        print(f"Processing {subject_id}...")
        
        # Load experiment and baseline BHI data
        bhi_exp = load_bhi_file(subject_id, 'experiment')
        bhi_bl = load_bhi_file(subject_id, 'baseline')
        
        if bhi_exp is None:
            print(f"  ✗ Could not load experiment BHI for {subject_id}")
            continue
        
        # Get events for this subject
        events = extreme_events[subject_id].get('events_exp', [])
        
        # ====================================================================
        # PROCESS EXPERIMENT EVENTS
        # ====================================================================
        for event in events:
            start = event['start']
            end = event['end']
            event_type = event['event_type'].upper()
            emotion = event.get('segment_label', None)
            
            # ----------------------------------------------------------------
            # Extract BtH epochs
            # ----------------------------------------------------------------
            bth_epoch = extract_bhi_epoch(bhi_exp['BtH'], bhi_exp['time_BtH'], start, end, padding_seconds=padding_seconds)
            
            # Create rows for each (eeg_band, hrv_band, channel) combination
            for eeg_idx, eeg_band in enumerate(EEG_BANDS):
                for hrv_idx, hrv_band in enumerate(HRV_BANDS):
                    for ch_idx in range(bth_epoch.shape[2]):
                        # Extract time series for this combination
                        data = bth_epoch[eeg_idx, hrv_idx, ch_idx, :]
                        
                        # Compute statistics
                        mean_val = np.mean(data)
                        median_val = np.median(data)
                        
                        # Z-score normalize
                        z_data = (data - mean_val) / (np.std(data) + 1e-10)
                        z_median = np.median(z_data)
                        
                        # Get channel info
                        channel = bhi_exp['channels'][ch_idx]
                        channel_label = channel['labels'][0]
                        
                        rows.append({
                            'id': subject_id,
                            'direction': 'BtH',
                            'eeg_band': eeg_band,
                            'hrv_band': hrv_band,
                            'channel': channel,
                            'channel_label': channel_label,
                            'event_type': event_type,
                            'emotion': emotion,
                            'data': data,
                            'mean': mean_val,
                            'median': median_val,
                            'z_data': z_data,
                            'z_median': z_median
                        })
            
            # ----------------------------------------------------------------
            # Extract HtB epochs
            # ----------------------------------------------------------------
            htb_epoch = extract_bhi_epoch(bhi_exp['HtB'], bhi_exp['time_HtB'], start, end, padding_seconds=padding_seconds)
            
            # Create rows for each (hrv_band, eeg_band, channel) combination
            for hrv_idx, hrv_band in enumerate(HRV_BANDS):
                for eeg_idx, eeg_band in enumerate(EEG_BANDS):
                    for ch_idx in range(htb_epoch.shape[2]):
                        # Extract time series for this combination
                        data = htb_epoch[eeg_idx, hrv_idx, ch_idx, :]
                        
                        # Compute statistics
                        mean_val = np.mean(data)
                        median_val = np.median(data)
                        
                        # Z-score normalize
                        z_data = (data - mean_val) / (np.std(data) + 1e-10)
                        z_median = np.median(z_data)
                        
                        # Get channel info
                        channel = bhi_exp['channels'][ch_idx]
                        channel_label = channel['labels'][0]
                        
                        rows.append({
                            'id': subject_id,
                            'direction': 'HtB',
                            'eeg_band': eeg_band,
                            'hrv_band': hrv_band,
                            'channel': channel,
                            'channel_label': channel_label,
                            'event_type': event_type,
                            'emotion': emotion,
                            'data': data,
                            'mean': mean_val,
                            'median': median_val,
                            'z_data': z_data,
                            'z_median': z_median
                        })
        
        # ====================================================================
        # PROCESS BASELINE DATA (fixed window: 60-180s)
        # ====================================================================
        if bhi_bl is not None:
            # Extract baseline window [60, 180] seconds
            bl_start = 60.0
            bl_end = 180.0
            
            # ----------------------------------------------------------------
            # BtH baseline
            # ----------------------------------------------------------------
            bth_bl_epoch = extract_bhi_epoch(bhi_bl['BtH'], bhi_bl['time_BtH'], bl_start, bl_end)
            
            for eeg_idx, eeg_band in enumerate(EEG_BANDS):
                for hrv_idx, hrv_band in enumerate(HRV_BANDS):
                    for ch_idx in range(bth_bl_epoch.shape[2]):
                        # Extract baseline time series for this combination
                        data = bth_bl_epoch[eeg_idx, hrv_idx, ch_idx, :]
                        
                        # Compute statistics
                        mean_val = np.mean(data)
                        median_val = np.median(data)
                        
                        # Z-score normalize
                        z_data = (data - mean_val) / (np.std(data) + 1e-10)
                        z_median = np.median(z_data)
                        
                        # Get channel info
                        channel = bhi_bl['channels'][ch_idx]
                        channel_label = channel['labels'][0]
                        
                        rows.append({
                            'id': subject_id,
                            'direction': 'BtH',
                            'eeg_band': eeg_band,
                            'hrv_band': hrv_band,
                            'channel': channel,
                            'channel_label': channel_label,
                            'event_type': 'BL',
                            'emotion': None,
                            'data': data,
                            'mean': mean_val,
                            'median': median_val,
                            'z_data': z_data,
                            'z_median': z_median
                        })
            
            # ----------------------------------------------------------------
            # HtB baseline
            # ----------------------------------------------------------------
            htb_bl_epoch = extract_bhi_epoch(bhi_bl['HtB'], bhi_bl['time_HtB'], bl_start, bl_end)
            
            for hrv_idx, hrv_band in enumerate(HRV_BANDS):
                for eeg_idx, eeg_band in enumerate(EEG_BANDS):
                    for ch_idx in range(htb_bl_epoch.shape[2]):
                        # Extract baseline time series for this combination
                        data = htb_bl_epoch[eeg_idx, hrv_idx, ch_idx, :]
                        
                        # Compute statistics
                        mean_val = np.mean(data)
                        median_val = np.median(data)
                        
                        # Z-score normalize
                        z_data = (data - mean_val) / (np.std(data) + 1e-10)
                        z_median = np.median(z_data)
                        
                        # Get channel info
                        channel = bhi_bl['channels'][ch_idx]
                        channel_label = channel['labels'][0]
                        
                        rows.append({
                            'id': subject_id,
                            'direction': 'HtB',
                            'eeg_band': eeg_band,
                            'hrv_band': hrv_band,
                            'channel': channel,
                            'channel_label': channel_label,
                            'event_type': 'BL',
                            'emotion': None,
                            'data': data,
                            'mean': mean_val,
                            'median': median_val,
                            'z_data': z_data,
                            'z_median': z_median
                        })
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # print(f"\n✓ Created DataFrame with {len(df)} rows")
    # print(f"  Subjects: {df['id'].nunique()}")
    # print(f"  Event types: {df['event_type'].value_counts().to_dict()}")
    
    return df
