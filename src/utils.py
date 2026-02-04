"""
Utility functions for RR preprocessing and multi-subject HRV visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from hrvanalysis import remove_outliers, remove_ectopic_beats, interpolate_nan_values
from pointprocess import get_ks_coords

VID_COLORS = {'POS': 'green', 'NEUT': 'gray', 'NEG': 'red'}
VIDEO_ALPHAS = {'POS': 0, 'NEUT': 0, 'NEG': 0}

def stitch_rr_phases(baseline_rr, experiment_rr):
    """
    Stitch together RR timestamps from baseline and experiment phases.
    
    Since the experimental protocol has both phases as part of a single 
    continuous recording session, this function concatenates the RR timestamps
    by shifting the experiment phase to continue from where baseline ends.
    
    Parameters
    ----------
    subject : SubjectData
        Subject data object with RR timestamp data
    baseline_phase : int, default=0
        Phase index for baseline
    experiment_phase : int, default=1
        Phase index for experiment
    
    Returns
    -------
    rr_times_full : np.ndarray
        Stitched RR timestamps in seconds
    stitch_info : dict
        Information about the stitching process including:
        - baseline_duration: Duration of baseline phase (s)
        - baseline_n_events: Number of events in baseline
        - experiment_n_events: Number of events in experiment
        - total_n_events: Total number of events
    """
    
    # Calculate baseline duration (last timestamp in baseline)
    baseline_duration = baseline_rr[-1] if len(baseline_rr) > 0 else 0
    baseline_duration = 184.50 if baseline_duration > 200 else baseline_duration
    
    # Shift experiment timestamps to continue from baseline
    experiment_rr_shifted = experiment_rr + baseline_duration
    
    # Concatenate the two phases
    rr_times_full = np.concatenate([baseline_rr, experiment_rr_shifted])
    
    stitch_info = {
        'baseline_duration': float(baseline_duration),
        'baseline_n_events': int(len(baseline_rr)),
        'experiment_n_events': int(len(experiment_rr)),
        'total_n_events': int(len(rr_times_full))
    }
    
    return rr_times_full, stitch_info

def preprocess_rr(
    rr_times,
    *,
    low_rri=300,
    high_rri=2000,
    ectopic_method="malik",
    interpolation_method="linear",
):
    """
    Clean RR intervals using hrvanalysis and rebuild timestamps.
    
    Pipeline: RR intervals (ms) -> outlier removal -> interpolation 
              -> ectopic removal -> interpolation -> timestamps (s)
    
    Parameters
    ----------
    rr_intervals : array-like
        RR timestamps in seconds
    low_rri, high_rri : float
        Physiological bounds for RR intervals (ms)
    ectopic_method : str
        Method for ectopic beat detection ('malik', 'karlsson', 'kamath', 'acar')
    interpolation_method : str
        Interpolation method ('linear', 'cubic', 'nearest')
    
    Returns
    -------
    rr_times_clean : np.ndarray
        Cleaned R-peak timestamps in seconds
    clean_info : dict
        Cleaning statistics and diagnostics
    """

    t0 = rr_times[0]
    rr_intervals = np.diff(rr_times) * 1000  # Convert to ms

    # Remove outliers
    rr_wo = remove_outliers(rr_intervals=rr_intervals, low_rri=low_rri, high_rri=high_rri)
    rr_i = interpolate_nan_values(rr_intervals=rr_wo, interpolation_method=interpolation_method)

    # Remove ectopic beats
    nn = remove_ectopic_beats(rr_intervals=rr_i, method=ectopic_method)
    nn_i = interpolate_nan_values(rr_intervals=nn, interpolation_method=interpolation_method)

    nn = np.asarray(nn_i, dtype=float)

    # Rebuild timestamps from first event
    rr_times_clean = np.empty(nn.size + 1, dtype=float)
    rr_times_clean[0] = float(t0)
    rr_times_clean[1:] = rr_times_clean[0] + np.cumsum(nn) / 1000.0

    return rr_times_clean, {
        "n_input_events": int(np.asarray(rr_intervals).size),
        "n_output_events": int(rr_times_clean.size),
        "rr_ms_mean_before": float(np.mean(rr_intervals)),
        "rr_ms_mean_after": float(np.mean(np.diff(rr_times_clean) * 1000.0)),
    }


def plot_rr_and_hrv_grid(hrv_results, valid_subjects, title="RR Intervals & HRV First Moment - All Subjects", video_time_shift=None):
    """
    Plot RR intervals and HRV first moment (μ) together for all subjects in a grid layout.
    Similar to the single-subject visualization with discrete RR intervals as points and
    continuous μ as a line.
    
    Parameters
    ----------
    hrv_results : dict
        Dictionary mapping subject_id -> {'rr_times', 'hrv_dict', 'taus'}
    valid_subjects : list
        List of subject IDs to plot
    title : str
        Figure title
    video_time_shift : dict or float, optional
        Time shift to apply to video markers (in seconds). If dict, keys are subject_ids.
        If float, same shift applied to all subjects. Use baseline_duration when plotting
        stitched data. If None, no shift is applied.
    
    Returns
    -------
    fig, axes : matplotlib figure and axes
    """
    n_subjects = len(valid_subjects)
    n_cols = int(np.ceil(np.sqrt(n_subjects)))
    n_rows = int(np.ceil(n_subjects / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 3*n_rows), dpi=100)
    axes = axes.flatten() if n_subjects > 1 else [axes]
    
    for idx, subject_id in enumerate(valid_subjects):
        ax = axes[idx]
        
        if hrv_results[subject_id] is not None:
            # Get HRV first moment
            hrv_dict = hrv_results[subject_id]['hrv_dict']
            time_hrv = hrv_dict["Time"]
            mu = hrv_dict["Mu"] * 1000  # Convert to ms
            
            # Get RR intervals
            rr_times = hrv_results[subject_id]['rr_times']
            rr_intervals = np.diff(rr_times) * 1000  # Convert to ms
            
            # Plot first moment as continuous line
            ax.plot(time_hrv, mu, 'b-', linewidth=0.5, alpha=0.8, label='μ (first moment)')
            
            # Plot discrete RR intervals as points
            ax.plot(rr_times[1:], rr_intervals, 'r*', markersize=0.5, alpha=0.5, label='RR interval')
            
            ax.set_title(f'{subject_id}', fontsize=10, fontweight='bold')

            video_info = valid_subjects[subject_id].get_all_video_info()
            if video_info:
                # Determine time shift for this subject
                shift = 0
                if video_time_shift is not None:
                    if isinstance(video_time_shift, dict):
                        shift = video_time_shift.get(subject_id, 0)
                    else:
                        shift = float(video_time_shift)
                
                for video in video_info:
                    if video:
                        start = video['time_start'] + shift
                        end = video['time_end'] + shift
                        vtype = video['type']
                        color = VID_COLORS.get(vtype, 'gray')
                        alpha = VIDEO_ALPHAS.get(vtype, 0.1)
                        ax.axvspan(start, end, alpha=alpha, color=color)
                        ax.axvline(start, color=color, linestyle='--', linewidth=0.5, alpha=0.5)
                        ax.axvline(end, color=color, linestyle='--', linewidth=0.5, alpha=0.5)
            
            # Add legend only to first subplot
            if idx == 0:
                ax.legend(fontsize=7, loc='upper right')
        else:
            ax.text(0.5, 0.5, 'Error', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{subject_id} (error)', fontsize=10, fontweight='bold', color='red')
        
        ax.set_xlabel('Time [s]', fontsize=8)
        ax.set_ylabel('RR / μ [ms]', fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)
    
    # Hide unused subplots
    for idx in range(n_subjects, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    return fig, axes


def plot_lfhf_power_grid(hrv_results, valid_subjects, metric='HF', title=None, video_time_shift=None, extreme_masks=None):
    """
    Plot LF power, HF power, or LF/HF ratio for all subjects in a grid layout.
    
    Parameters
    ----------
    hrv_results : dict
        Dictionary mapping subject_id -> {'rr_times', 'hrv_dict', 'taus'}
    valid_subjects : list
        List of subject IDs to plot
    metric : str, default='HF'
        Which metric to plot: 'LF', 'HF', or 'LF/HF'
    title : str, optional
        Figure title. If None, auto-generated based on metric
    video_time_shift : dict or float, optional
        Time shift to apply to video markers (in seconds). If dict, keys are subject_ids.
        If float, same shift applied to all subjects. Use baseline_duration when plotting
        stitched data. If None, no shift is applied.
    extreme_masks : dict, optional
        Dictionary mapping subject_id -> mask or {'surge': mask, 'drop': mask}.
        If single mask (bool array), plots as red points.
        If dict with 'surge'/'drop', plots surge as red and drop as blue.
    
    Returns
    -------
    fig, axes : matplotlib figure and axes
    """
    # Validate metric
    metric = metric.upper()
    if metric not in ['VLF', 'LF', 'HF', 'LF/HF']:
        raise ValueError(f"metric must be 'LF', 'HF', or 'LF/HF', got '{metric}'")
    
    # Set default title if not provided
    if title is None:
        if metric == 'LF/HF':
            title = "LF/HF Ratio - All Subjects"
        else:
            title = f"{metric} Power - All Subjects"
    
    # Set color and ylabel based on metric
    plot_config = {
        'VLF': {'color': 'green', 'ylabel': 'VLF Power [ms²]'},
        'LF': {'color': 'blue', 'ylabel': 'LF Power [ms²]'},
        'HF': {'color': 'red', 'ylabel': 'HF Power [ms²]'},
        'LF/HF': {'color': 'purple', 'ylabel': 'LF/HF Ratio'}
    }
    config = plot_config[metric]
    
    n_subjects = len(valid_subjects)
    n_cols = int(np.ceil(np.sqrt(n_subjects)))
    n_rows = int(np.ceil(n_subjects / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 3*n_rows), dpi=100)
    axes = axes.flatten() if n_subjects > 1 else [axes]
    has_surge = False
    has_drop = False
    
    for idx, subject_id in enumerate(valid_subjects):
        ax = axes[idx]
        
        if hrv_results[subject_id] is not None:
            hrv_dict = hrv_results[subject_id]['hrv_dict']
            time_hrv = hrv_dict["Time"].flatten()
            vlf_power = hrv_dict["powVLF"]
            lf_power = hrv_dict["powLF"]
            hf_power = hrv_dict["powHF"]
            
            # Compute the metric to plot
            if metric == 'LF':
                y_data = lf_power
            elif metric == 'VLF':
                y_data = vlf_power
            elif metric == 'HF':
                y_data = hf_power
            else:  # LF/HF
                # Avoid division by zero
                y_data = np.divide(lf_power, hf_power, 
                                   out=np.zeros_like(lf_power), 
                                   where=hf_power!=0)
            
            ax.plot(time_hrv, y_data, linewidth=0.5, color=config['color'], alpha=0.7)
            
            # Plot extreme events if provided
            if extreme_masks is not None and subject_id in extreme_masks:
                event_data = extreme_masks[subject_id]
                
                # Get events by type
                events = event_data.get('events', [])
                surge_windows = [e for e in events if e['event_type'] == 'surge']
                drop_windows = [e for e in events if e['event_type'] == 'drop']
                
                # Plot surge windows
                for win in surge_windows:
                    ax.axvspan(win['start'], win['end'], alpha=0.2, color='red', zorder=3)
                    ax.axvline(win['start'], color='red', linestyle='--', linewidth=0.5, alpha=0.6)
                    ax.axvline(win['end'], color='red', linestyle='--', linewidth=0.5, alpha=0.6)
                
                # Plot drop windows
                for win in drop_windows:
                    ax.axvspan(win['start'], win['end'], alpha=0.2, color='blue', zorder=3)
                    ax.axvline(win['start'], color='blue', linestyle='--', linewidth=0.5, alpha=0.6)
                    ax.axvline(win['end'], color='blue', linestyle='--', linewidth=0.5, alpha=0.6)
                
                # Track which event types exist
                if surge_windows:
                    has_surge = True
                if drop_windows:
                    has_drop = True
            
            ax.set_title(f'{subject_id}', fontsize=10, fontweight='bold')

            video_info = valid_subjects[subject_id].get_all_video_info()
            if video_info:
                # Determine time shift for this subject
                shift = 0
                if video_time_shift is not None:
                    if isinstance(video_time_shift, dict):
                        shift = video_time_shift.get(subject_id, 0)
                    else:
                        shift = float(video_time_shift)
                
                for video in video_info:
                    if video:
                        start = video['time_start'] + shift
                        end = video['time_end'] + shift
                        vtype = video['type']
                        color = VID_COLORS.get(vtype, 'gray')
                        alpha = VIDEO_ALPHAS.get(vtype, 0.1)
                        ax.axvspan(start, end, alpha=alpha, color=color)
                        ax.axvline(start, color=color, linestyle='--', linewidth=2, alpha=0.5)
                        ax.axvline(end, color=color, linestyle='--', linewidth=2, alpha=0.5)
                        
                        # Track which video types exist
                        if vtype == 'POS':
                            has_pos = True
                        elif vtype == 'NEG':
                            has_neg = True
                        elif vtype == 'NEUT':
                            has_neut = True

        else:
            ax.text(0.5, 0.5, 'Error', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{subject_id} (error)', fontsize=10, fontweight='bold', color='red')
        
        ax.set_xlabel('Time [s]', fontsize=8)
        ax.set_ylabel(config['ylabel'], fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)
    
    # Hide unused subplots
    for idx in range(n_subjects, len(axes)):
        axes[idx].axis('off')
    
    # Create figure-level legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = []
    
    # Add event types
    if has_surge:
        legend_elements.append(Patch(facecolor='red', alpha=0.2, edgecolor='red', label='Surge'))
    if has_drop:
        legend_elements.append(Patch(facecolor='blue', alpha=0.2, edgecolor='blue', label='Drop'))
    
    # Add separator if we have both events and videos
    if legend_elements and (has_pos or has_neg or has_neut):
        legend_elements.append(Patch(facecolor='none', edgecolor='none', label=''))  # Empty space
    
    # Add video segments
    if has_pos:
        legend_elements.append(Line2D([0], [0], color='green', linewidth=2, linestyle='--', alpha=0.5, label='Positive'))
    if has_neg:
        legend_elements.append(Line2D([0], [0], color='red', linewidth=2, linestyle='--', alpha=0.5, label='Negative'))
    if has_neut:
        legend_elements.append(Line2D([0], [0], color='gray', linewidth=2, linestyle='--', alpha=0.5, label='Neutral'))
    
    # Add legend if we have any elements
    if legend_elements:
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.99), 
                  fontsize=10, framealpha=0.9, ncol=1)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 0.88, 0.99])  # Make room for legend
    
    return fig, axes


def plot_ks_goodness_of_fit_grid(hrv_results, valid_subjects, title="KS Goodness-of-Fit - All Subjects"):
    """
    Plot KS goodness-of-fit for all subjects in a grid layout.
    
    Parameters
    ----------
    hrv_results : dict
        Dictionary mapping subject_id -> {'rr_times', 'hrv_dict', 'taus'}
    valid_subjects : list
        List of subject IDs to plot
    title : str
        Figure title
    
    Returns
    -------
    fig, axes : matplotlib figure and axes
    """
    n_subjects = len(valid_subjects)
    n_cols = int(np.ceil(np.sqrt(n_subjects)))
    n_rows = int(np.ceil(n_subjects / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 3*n_rows), dpi=100)
    axes = axes.flatten() if n_subjects > 1 else [axes]
    
    for idx, subject_id in enumerate(valid_subjects):
        ax = axes[idx]
        
        if hrv_results[subject_id] is not None:
            coords = get_ks_coords(hrv_results[subject_id]['taus'])
            
            ax.plot(coords.z, coords.inner, "k", linewidth=0.8)
            ax.plot(coords.lower, coords.inner, "b", linewidth=0.5)
            ax.plot(coords.upper, coords.inner, "b", linewidth=0.5)
            ax.plot(coords.inner, coords.inner, "r", linewidth=0.5)
            
            ax.set_title(f'{subject_id}', fontsize=10, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Error', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{subject_id} (error)', fontsize=10, fontweight='bold', color='red')
        
        ax.set_xlabel('Theoretical', fontsize=8)
        ax.set_ylabel('Empirical', fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)
    
    # Hide unused subplots
    for idx in range(n_subjects, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    return fig, axes


def plot_all_hrv_summaries(hrv_results, valid_subjects, phase_name="Experiment Phase", show=True, video_time_shift=None, save_path=None):
    """
    Plot all HRV summary visualizations (RR+μ, HF power, KS) for all subjects.
    
    Parameters
    ----------
    hrv_results : dict
        Dictionary mapping subject_id -> {'rr_times', 'hrv_dict', 'taus'}
    valid_subjects : list
        List of subject IDs to plot
    phase_name : str
        Name of the phase for titles
    show : bool
        Whether to call plt.show() after each plot
    video_time_shift : dict or float, optional
        Time shift to apply to video markers (in seconds). If dict, keys are subject_ids.
        If float, same shift applied to all subjects. Use baseline_duration when plotting
        stitched data. If None, no shift is applied.
    save_path : str, optional
        If provided, saves all plots to a PDF file at this path instead of showing them
    
    Returns
    -------
    figures : dict
        Dictionary with keys 'rr_mu', 'vlf', 'lf', 'hf', 'lf_hf', 'ks' containing figure objects
    """
    from matplotlib.backends.backend_pdf import PdfPages
    
    figures = {}
    
    # If saving to PDF, open the file
    pdf = PdfPages(save_path) if save_path else None
    
    try:
        print("\n1. Plotting RR intervals & HRV first moment (μ)...")
        fig_rr_mu, _ = plot_rr_and_hrv_grid(
            hrv_results, valid_subjects, 
            title=f'RR Intervals & HRV First Moment (μ) - All Subjects ({phase_name})',
            video_time_shift=video_time_shift
        )
        figures['rr_mu'] = fig_rr_mu
        if pdf:
            pdf.savefig(fig_rr_mu)
            plt.close(fig_rr_mu)
        elif show:
            plt.show()

        print("2. Plotting VLF Power...")
        fig_vlf, _ = plot_lfhf_power_grid(
            hrv_results, valid_subjects,
            metric='VLF',
            title=f'VLF Power - All Subjects ({phase_name})',
            video_time_shift=video_time_shift
        )
        figures['vlf'] = fig_vlf
        if pdf:
            pdf.savefig(fig_vlf)
            plt.close(fig_vlf)
        elif show:
            plt.show()
        
        print("3. Plotting LF Power...")
        fig_lf, _ = plot_lfhf_power_grid(
            hrv_results, valid_subjects,
            metric='LF',
            title=f'LF Power - All Subjects ({phase_name})',
            video_time_shift=video_time_shift
        )
        figures['lf'] = fig_lf
        if pdf:
            pdf.savefig(fig_lf)
            plt.close(fig_lf)
        elif show:
            plt.show()

        print("4. Plotting HF Power...")
        fig_hf, _ = plot_lfhf_power_grid(
            hrv_results, valid_subjects,
            metric='HF',
            title=f'HF Power - All Subjects ({phase_name})',
            video_time_shift=video_time_shift
        )
        figures['hf'] = fig_hf
        if pdf:
            pdf.savefig(fig_hf)
            plt.close(fig_hf)
        elif show:
            plt.show()

        print("5. Plotting LF/HF Ratio...")
        fig_lf_hf, _ = plot_lfhf_power_grid(
            hrv_results, valid_subjects,
            metric='LF/HF',
            title=f'LF/HF Ratio - All Subjects ({phase_name})',
            video_time_shift=video_time_shift
        )
        figures['lf_hf'] = fig_lf_hf
        if pdf:
            pdf.savefig(fig_lf_hf)
            plt.close(fig_lf_hf)
        elif show:
            plt.show()
        
        print("6. Plotting KS goodness-of-fit...")
        fig_ks, _ = plot_ks_goodness_of_fit_grid(
            hrv_results, valid_subjects,
            title=f'KS Goodness-of-Fit - All Subjects ({phase_name})'
        )
        figures['ks'] = fig_ks
        if pdf:
            pdf.savefig(fig_ks)
            plt.close(fig_ks)
        elif show:
            plt.show()
        
        print("\n✓ All visualizations complete!")
        
    finally:
        # Close PDF if it was opened
        if pdf:
            pdf.close()
            print(f"✓ Saved plots to: {save_path}")
    
    # return figures


def detect_extreme_events_gradient_peaks(
    results, 
    sigma=3.0, 
    window_seconds=10.0,
    threshold=2.0,
    overlap_ratio=0.5,
    valid_subjects=None,
    buffer_seconds=0.0
):
    """
    Detect extreme events in LF/HF ratio using a windowing approach to capture sustained trends.
    
    This method:
    1. Smooths LF/HF ratio with a Gaussian filter
    2. Processes the signal in sliding windows
    3. For each window:
       - Finds the maximum and minimum values
       - Calculates |first_sample - max| and |first_sample - min|
       - Classifies as surge if |first-max| > threshold
       - Classifies as drop if |first-min| > threshold
       - Classifies as biphasic if both exceed threshold
    4. Checks if window contains video start/end boundaries
    5. Returns time windows for detected events with metadata
    
    Parameters
    ----------
    results : dict
        Dictionary mapping subject_id to HRV results containing:
        - 'hrv_dict': Dictionary with 'Time', 'powLF', 'powHF' arrays
        - 'rr_info': Dictionary with 'baseline_duration'
    sigma : float, default=3.0
        Standard deviation for Gaussian smoothing kernel
    window_seconds : float, default=10.0
        Duration of each analysis window (in seconds)
    threshold : float, default=2.0
        Minimum change in LF/HF ratio to classify as an extreme event
        Higher values = fewer, more extreme events
    overlap_ratio : float, default=0.5
        Overlap between consecutive windows (0.0 = no overlap, 0.5 = 50% overlap)
    valid_subjects : dict, optional
        Dictionary mapping subject_id to SubjectData objects with video timing info.
        If provided, adds video boundary proximity flags to output.
    buffer_seconds : float, default=0.0
        Time buffer (in seconds) to consider when resolving overlapping windows.
        Windows within this buffer are considered overlapping. The window with
        highest magnitude is kept. Set to 0.0 for exact overlap only.
    
    Returns
    -------
    extreme_event_summary : dict
        Dictionary mapping subject_id to:
        - 'events': List of event dicts with keys:
            - 'start': Window start time (full timeline)
            - 'end': Window end time (full timeline)
            - 'event_type': Type of event ('surge', 'drop', or 'spike')
            - 'segment_label': Video segment at peak ('POS', 'NEG', 'NEUT', or 'REST')
        - 'events_exp': Same events in experiment-only time coordinates
        - 'n_surge': Number of surge events
        - 'n_drop': Number of drop events
        - 'n_spike': Number of spike events
        - 'n_total': Total number of events
    
    Examples
    --------
    >>> extreme_events = detect_extreme_events_gradient_peaks(
    ...     results,
    ...     sigma=3.0,
    ...     window_seconds=15.0,  # Longer windows for more sustained trends
    ...     threshold=1.5,        # Higher threshold for stronger events
    ...     overlap_ratio=0.5,
    ...     valid_subjects=all_subjects  # Include video boundary detection
    ... )
    """
    from scipy.ndimage import gaussian_filter1d
    
    extreme_event_summary = {}
    delta = 0.005  # Time step from HRV analysis
    
    for subject_id, subject_result in results.items():
        if subject_result is None:
            continue
            
        d = subject_result['hrv_dict']
        time = d["Time"].flatten()
        lf_power = d["powLF"]
        hf_power = d["powHF"]
        lf_hf_ratio = hf_power
        
        # Get baseline info for splitting
        baseline_duration = subject_result['rr_info']['baseline_duration']
        baseline_mask = time <= baseline_duration
        
        # Split into baseline and experiment
        time_exp = time[~baseline_mask]
        lf_hf_exp = lf_hf_ratio[~baseline_mask]
        
        # Smooth with Gaussian filter to reduce noise
        lf_hf_smoothed = gaussian_filter1d(lf_hf_exp, sigma=sigma)

        # Normalize
        mean = np.mean(lf_hf_smoothed)
        std = np.std(lf_hf_smoothed)
        z_score = (lf_hf_smoothed - mean) / std
        lf_hf_smoothed = z_score
        
        # Calculate window parameters
        window_points = int(window_seconds / delta)
        step_points = int(window_points * (1 - overlap_ratio))
        
        # Initialize lists for detected windows
        surge_windows = []
        drop_windows = []
        
        # Slide through the signal
        for start_idx in range(0, len(lf_hf_smoothed) - window_points + 1, step_points):
            end_idx = start_idx + window_points
            window_data = lf_hf_smoothed[start_idx:end_idx]
            
            # Get first sample, max, and min in window
            first_val = window_data[0]
            last_val = window_data[-1]
            diff = abs(first_val - last_val)
            # max_val = np.max(window_data)
            # min_val = np.min(window_data)
            
            # Calculate changes from first sample
            # surge_magnitude = max_val - first_val
            # drop_magnitude = first_val - min_val
            
            # Classify window based on threshold
            is_extreme = diff > threshold
            
            # Categorize the event with magnitude tracking
            if is_extreme:
                # Find peak locations within window
                # max_idx_in_window = np.argmax(window_data)
                # min_idx_in_window = np.argmin(window_data)
                
                # Determine which peak is more extreme
                # if surge_magnitude > drop_magnitude:
                #     # Surge is stronger - center around max
                #     peak_idx = start_idx + max_idx_in_window
                # else:
                #     # Drop is stronger - center around min
                #     peak_idx = start_idx + min_idx_in_window
                
                # Create centered window around peak
                # half_window = window_points // 2
                # new_start_idx = peak_idx - half_window
                # new_end_idx = peak_idx + half_window
                
                # Skip if window would be clipped at boundaries
                # if new_start_idx < 0 or new_end_idx >= len(lf_hf_smoothed):
                #     continue
                
                # Now classify the centered window
                # centered_window = lf_hf_smoothed[new_start_idx:new_end_idx+1]
                # first_val_centered = centered_window[0]
                # last_val_centered = centered_window[-1]
                # max_val_centered = np.max(centered_window)
                # min_val_centered = np.min(centered_window)
                
                # surge_magnitude_centered = max_val_centered - first_val_centered
                # drop_magnitude_centered = first_val_centered - min_val_centered
                
                # Re-check threshold: centered window must still be extreme
                # is_still_extreme = surge_magnitude_centered > threshold or drop_magnitude_centered > threshold
                
                # if not is_still_extreme:
                #     continue  # Skip this window
                
                # diff = abs(first_val_centered - last_val_centered)
                
                # Check if diff is large enough - must be > std to be considered extreme
                if diff < threshold:
                    continue  # Not a sustained extreme event, skip it
                
                # Classify based on centered window (surge or drop only)
                if last_val > first_val:
                    event_type = 'surge'
                else:
                    event_type = 'drop'
                
                # Convert indices to time (in full timeline)
                pad_length = np.sum(baseline_mask)
                new_start_idx_full = start_idx + pad_length
                new_end_idx_full = end_idx + pad_length
                start_time = float(time[new_start_idx_full])
                end_time = float(time[new_end_idx_full])
                
                # Append to appropriate list
                window_dict = {
                    'start': start_time, 
                    'end': end_time,
                    'magnitude': diff,
                    'event_type': event_type
                }
                
                if event_type == 'surge':
                    surge_windows.append(window_dict)
                else:  # drop
                    drop_windows.append(window_dict)
        
        # Add video segment label based on peak location
        def add_segment_labels(window_list, baseline_duration):
            """Add segment label (POS/NEG/NEUT/REST) based on peak location."""
            # if valid_subjects is None or subject_id not in valid_subjects:
            #     # No video info - all events are in REST
            #     for win in window_list:
            #         win['segment_label'] = 'REST'
            #     return
            
            video_info = valid_subjects[subject_id].get_all_video_info()
            # if not video_info:
            #     for win in window_list:
            #         win['segment_label'] = 'REST'
            #     return
            
            # Check each window's peak (center point)
            for win in window_list:
                peak_time = (win['start'] + win['end']) / 2.0
                segment_label = 'REST'  # Default
                
                # Check which video segment contains the peak
                for video in video_info:
                    if video:
                        # Shift to full timeline (add baseline_duration)
                        start = video['time_start'] + baseline_duration
                        end = video['time_end'] + baseline_duration
                        
                        if start <= peak_time <= end:
                            segment_label = video['type']  # POS, NEG, or NEUT
                            break
                
                win['segment_label'] = segment_label
        
        add_segment_labels(surge_windows, baseline_duration)
        add_segment_labels(drop_windows, baseline_duration)
        
        # Filter out events with 'REST' label before overlap resolution
        surge_windows = [w for w in surge_windows if w.get('segment_label') != 'REST']
        drop_windows = [w for w in drop_windows if w.get('segment_label') != 'REST']
        
        # Resolve overlapping windows with buffer (keep highest magnitude)
        def resolve_overlaps(surge_list, drop_list, buffer_seconds=0.0):
            """
            Resolve overlapping windows between surge and drop events.
            
            Windows are considered overlapping if they are within buffer_seconds of each other.
            When overlap occurs, keep the window with the highest magnitude.
            
            Parameters
            ----------
            surge_list : list
                List of surge window dicts
            drop_list : list
                List of drop window dicts
            buffer_seconds : float, default=0.0
                Time buffer to consider as overlap (in seconds)
            
            Returns
            -------
            resolved_surge : list
                Resolved surge windows
            resolved_drop : list
                Resolved drop windows
            """
            # Combine all windows with their type
            all_windows = []
            for win in surge_list:
                all_windows.append({**win, 'type': 'surge'})
            for win in drop_list:
                all_windows.append({**win, 'type': 'drop'})
            
            if not all_windows:
                return [], []
            
            # Sort by start time
            all_windows = sorted(all_windows, key=lambda w: w['start'])
            
            resolved = []
            current = all_windows[0]
            
            for i in range(1, len(all_windows)):
                next_win = all_windows[i]
                
                # Check if current overlaps with next (considering buffer)
                if next_win['start'] - buffer_seconds < current['end']:
                    # Overlap detected - keep the one with higher magnitude
                    if next_win['magnitude'] > current['magnitude']:
                        current = next_win
                    # else keep current
                else:
                    # No overlap - save current and move to next
                    resolved.append(current)
                    current = next_win
            
            # Don't forget the last window
            resolved.append(current)
            
            # Separate back into type lists
            resolved_surge = []
            resolved_drop = []
            
            for win in resolved:
                win_copy = {k: v for k, v in win.items() if k != 'type'}
                if win['type'] == 'surge':
                    resolved_surge.append(win_copy)
                elif win['type'] == 'drop':
                    resolved_drop.append(win_copy)
            
            return resolved_surge, resolved_drop
        
        # Resolve overlaps with specified buffer
        surge_windows, drop_windows = resolve_overlaps(
            surge_windows, drop_windows, buffer_seconds=buffer_seconds
        )
        
        # Remove magnitude field (no longer needed)
        for win in surge_windows:
            win.pop('magnitude', None)
        for win in drop_windows:
            win.pop('magnitude', None)
        
        # Merge all event types into single list and sort by time
        all_events = surge_windows + drop_windows
        all_events = sorted(all_events, key=lambda w: w['start'])
        
        # Create experiment-only time windows
        all_events_exp = [
            {**win, 'start': win['start'] - baseline_duration, 'end': win['end'] - baseline_duration}
            for win in all_events
        ]
        
        extreme_event_summary[subject_id] = {
            'events': all_events,
            'events_exp': all_events_exp,
            'n_surge': len(surge_windows),
            'n_drop': len(drop_windows),
            'n_total': len(all_events)
        }
    
    return extreme_event_summary
