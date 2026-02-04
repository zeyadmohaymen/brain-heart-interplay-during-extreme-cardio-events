"""
Data loader class for brain-heart microstates emotional arousal study.
Handles loading and accessing EMO and PH1 .mat files.
"""

import scipy.io
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import mne


class SubjectData:
    """
    Data class for a single subject's physiological and EEG recordings.
    
    Handles both EMO (emotional) and PH1 (physiological baseline) conditions.
    EMO files contain video information, while PH1 files do not.
    
    Attributes:
        subject_id (str): Subject identifier (e.g., 'i01')
        condition (str): Condition type ('EMO' or 'PH1')
        eeg_ph (np.ndarray): EEGLAB-like struct containing EEG data
        physio_ph (np.ndarray): 8x1 cell array containing physiological signals:
            ECG1, ECG2, Resp, BP, TCD sx, TCD dx, Percutaneous flow1, Percutaneous flow2
            Each element contains: data (1x120000 double), samplerate (400 Hz), 
            name (e.g., 'Periflux Rosso'), time (1x120000 double)
        rr_ph (np.ndarray): HRV (Heart Rate Variability) series
        rri_ph (np.ndarray): HRV series resampled at 4Hz
        t_rr_ph (np.ndarray): Time of the R-peaks
        t_rri_ph (np.ndarray): Time stamps for RRi (resampled at 4Hz)
        videos (Optional[np.ndarray]): Video information (only for EMO condition)
            Each video contains: time_start (sec), time_end (sec), 
            time_type ('POS', 'NEUT', or 'NEG')
    """
    
    def __init__(self, subject_id: str, condition: str, data_root: str = 'data'):
        """
        Initialize and load subject data.
        
        Args:
            subject_id: Subject identifier (e.g., 'i01', 'i02', etc.)
            condition: Condition type ('EMO' or 'PH1')
            data_root: Root directory containing subject folders
        """
        self.subject_id = subject_id
        self.condition = condition.upper()
        self.data_root = Path(data_root)
        
        if self.condition not in ['EMO', 'PH1']:
            raise ValueError(f"Condition must be 'EMO' or 'PH1', got '{condition}'")
        
        # Construct file path
        self.file_path = self.data_root / subject_id / f"{subject_id}_{self.condition}.mat"
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.file_path}")
        
        # Load data
        self._load_data()
    
    def _load_data(self):
        """Load .mat file and extract data fields."""
        raw_data = scipy.io.loadmat(str(self.file_path))
        
        # Extract main data fields (excluding MATLAB metadata)
        self.eeg_ph = raw_data.get('EEG_ph')
        self.physio_ph = raw_data.get('Physio_ph')
        self.rr_ph = raw_data.get('RR_ph')
        self.rri_ph = raw_data.get('RRi_ph')
        self.t_rr_ph = raw_data.get('T_RR_ph')
        self.t_rri_ph = raw_data.get('T_RRi_ph')
        
        # Videos only present in EMO condition
        self.videos = raw_data.get('videos') if self.condition == 'EMO' else None
        
        # Store raw data for advanced access
        self._raw_data = raw_data
    
    def get_eeg_data(self, phase_index: int = 0, return_raw_struct: bool = False):
        """
        Get EEG data as MNE Raw object for a specific phase.
        
        Converts the EEGLAB-like struct to an MNE Raw object for easy analysis,
        filtering, visualization, and processing using MNE-Python.
        
        Args:
            phase_index: Phase index to retrieve (default: 0)
            return_raw_struct: If True, return the raw EEGLAB struct instead of MNE object (default: False)
            
        Returns:
            mne.io.RawArray: MNE Raw object with EEG data (default)
            OR np.ndarray: Raw EEGLAB struct if return_raw_struct=True
            
        Raises:
            ImportError: If MNE is not installed (unless return_raw_struct=True)
            ValueError: If EEG data is not available or malformed
            
        Example:
            >>> subject = load_subject('i01', 'EMO')
            >>> raw = subject.get_eeg_data(phase_index=0)
            >>> raw.plot()  # Opens MNE visualization
            >>> raw.filter(1, 40)  # Bandpass filter
        """
        if self.eeg_ph is None:
            raise ValueError("EEG data not available")
        
        # Get raw EEGLAB struct
        eeg_struct_wrapped = self.eeg_ph[0, phase_index]
        
        # If user wants raw struct, return it directly
        if return_raw_struct:
            return eeg_struct_wrapped
        
        # Extract from nested structure
        eeg_struct = eeg_struct_wrapped[0, 0] if eeg_struct_wrapped.shape == (1, 1) else eeg_struct_wrapped
        
        # Extract data (channels × timepoints)
        data = eeg_struct['data']
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise ValueError(f"Expected 2D data array, got shape: {data.shape if isinstance(data, np.ndarray) else type(data)}")
        
        # Extract sampling rate
        srate = float(eeg_struct['srate'][0, 0] if isinstance(eeg_struct['srate'], np.ndarray) else eeg_struct['srate'])
        
        # Extract time information
        try:
            xmin = float(eeg_struct['xmin'][0, 0] if isinstance(eeg_struct['xmin'], np.ndarray) else eeg_struct['xmin'])
        except:
            xmin = 0.0
        
        # Extract channel information
        chanlocs = eeg_struct['chanlocs']
        ch_names = []
        ch_pos_dict = {}
        
        if chanlocs.shape == (1, data.shape[0]):
            # Channel labels are in a (1, n_channels) array
            for i in range(data.shape[0]):
                ch = chanlocs[0, i]
                ch_name = str(ch['labels'][0])
                ch_names.append(ch_name)
                
                # Extract 3D positions if available
                try:
                    x = float(ch['X'][0, 0]) if hasattr(ch['X'], '__getitem__') else float(ch['X'])
                    y = float(ch['Y'][0, 0]) if hasattr(ch['Y'], '__getitem__') else float(ch['Y'])
                    z = float(ch['Z'][0, 0]) if hasattr(ch['Z'], '__getitem__') else float(ch['Z'])
                    # MNE expects positions in meters; EEGLAB coordinates are in cm, so divide by 10
                    ch_pos_dict[ch_name] = np.array([x/10, y/10, z/10])
                except:
                    pass
        elif len(chanlocs) == data.shape[0]:
            # Direct array of channel structs
            for ch in chanlocs:
                ch_name = str(ch['labels'])
                ch_names.append(ch_name)
                try:
                    x = float(ch['X'][0, 0]) if hasattr(ch['X'], '__getitem__') else float(ch['X'])
                    y = float(ch['Y'][0, 0]) if hasattr(ch['Y'], '__getitem__') else float(ch['Y'])
                    z = float(ch['Z'][0, 0]) if hasattr(ch['Z'], '__getitem__') else float(ch['Z'])
                    # MNE expects positions in meters; EEGLAB coordinates are in cm, so divide by 10
                    ch_pos_dict[ch_name] = np.array([x/10, y/10, z/10])
                except:
                    pass
        else:
            # Fallback: generate generic channel names
            ch_names = [f"EEG{i+1:03d}" for i in range(data.shape[0])]
        
        # Create MNE info structure
        info = mne.create_info(
            ch_names=ch_names,
            sfreq=srate,
            ch_types='eeg'
        )
        
        # Add metadata to info
        try:
            setname = str(eeg_struct['setname'][0]) if len(eeg_struct['setname']) > 0 else None
            if setname:
                info['description'] = setname
        except:
            pass
        
        # Create Raw object with correct start time
        raw = mne.io.RawArray(data, info, first_samp=int(xmin * srate), verbose=False)
        
        # Add channel positions from EEGLAB data
        montage_type = 'none'  # Track what type of montage was applied
        
        if ch_pos_dict:
            try:
                # Create a custom montage from the EEGLAB positions
                montage = mne.channels.make_dig_montage(
                    ch_pos=ch_pos_dict,
                    coord_frame='head'
                )
                raw.set_montage(montage, verbose=False)
                montage_type = 'custom_eeglab'
            except Exception:
                # If custom montage fails, try standard montage
                try:
                    montage = mne.channels.make_standard_montage('standard_1020')
                    raw.set_montage(montage, on_missing='ignore', verbose=False)
                    montage_type = 'standard_1020'
                except Exception:
                    montage_type = 'none'
        else:
            # No custom positions, use standard montage
            try:
                montage = mne.channels.make_standard_montage('standard_1020')
                raw.set_montage(montage, on_missing='ignore', verbose=False)
                montage_type = 'standard_1020'
            except Exception:
                montage_type = 'none'
        
        # Store montage type in info for user reference
        raw.info['description'] = (
            f"{raw.info.get('description', 'EEG data')} | "
            f"Montage: {montage_type} ({len(ch_pos_dict)} custom positions)" 
            if montage_type == 'custom_eeglab' 
            else f"{raw.info.get('description', 'EEG data')} | Montage: {montage_type}"
        )
        
        # Extract and add events/annotations
        try:
            events_struct = eeg_struct['event']
            if events_struct.size > 0 and hasattr(events_struct.dtype, 'names'):
                annotations_onset = []
                annotations_duration = []
                annotations_description = []
                
                for event in events_struct.flat:
                    try:
                        # Get event latency (sample number)
                        latency = event['latency'][0, 0] if hasattr(event['latency'], '__getitem__') else event['latency']
                        # Convert to time in seconds
                        onset = (float(latency) - 1) / srate + xmin  # EEGLAB uses 1-based indexing
                        
                        # Get event type/description
                        evt_type = event['type']
                        if hasattr(evt_type, '__getitem__'):
                            evt_type = str(evt_type[0]) if len(evt_type) > 0 else 'event'
                        else:
                            evt_type = str(evt_type) if evt_type else 'event'
                        
                        # Get duration if available
                        duration = 0.0
                        if 'duration' in event.dtype.names:
                            dur = event['duration']
                            if hasattr(dur, '__getitem__') and len(dur) > 0:
                                duration = float(dur[0, 0]) / srate
                            elif dur:
                                duration = float(dur) / srate
                        
                        annotations_onset.append(onset)
                        annotations_duration.append(duration)
                        annotations_description.append(evt_type)
                    except:
                        continue
                
                if annotations_onset:
                    annotations = mne.Annotations(
                        onset=annotations_onset,
                        duration=annotations_duration,
                        description=annotations_description
                    )
                    raw.set_annotations(annotations, verbose=False)
        except Exception:
            pass  # No events or error extracting them
        
        # Add ICA information if available
        try:
            icaweights = eeg_struct['icaweights']
            icawinv = eeg_struct['icawinv']
            
            if (isinstance(icaweights, np.ndarray) and icaweights.size > 1 and
                isinstance(icawinv, np.ndarray) and icawinv.size > 1):
                
                # Store ICA info in raw.info for later use
                raw.info['proc_history'] = [{
                    'ica_weights': icaweights,
                    'ica_winv': icawinv,
                    'source': 'eeglab'
                }]
        except Exception:
            pass
        
        return raw
    
    def get_physio_data(self, phase_index: int = 0) -> np.ndarray:
        """
        Get physiological signals struct for a specific phase.
        
        Returns an 8x1 cell array containing:
        - ECG1: First ECG lead
        - ECG2: Second ECG lead
        - Resp: Respiration signal
        - BP: Blood pressure
        - TCD sx: Transcranial Doppler (left/sinister)
        - TCD dx: Transcranial Doppler (right/dexter)
        - Percutaneous flow1: First percutaneous flow measurement
        - Percutaneous flow2: Second percutaneous flow measurement
        
        Each signal contains: data (1x120000 double), samplerate (400 Hz),
        name (e.g., 'Periflux Rosso'), time (1x120000 double)
        
        Args:
            phase_index: Phase index to retrieve (default: 0)
            
        Returns:
            8x1 struct array with physiological signals
        """
        if self.physio_ph is None:
            raise ValueError("Physiological data not available")
        return self.physio_ph[0, phase_index]
    
    def get_rr_intervals(self, phase_index: int = 0) -> np.ndarray:
        """
        Get RR intervals series for a specific phase.
        
        Args:
            phase_index: Phase index to retrieve (default: 0)
            
        Returns:
            RR intervals series (R-R intervals) in milliseconds
        """
        if self.rr_ph is None:
            raise ValueError("RR data not available")
        return self.rr_ph[0, phase_index] * 1000.0  # Convert from seconds to milliseconds
    
    def get_rri_intervals(self, phase_index: int = 0) -> np.ndarray:
        """
        Get RR intervals series resampled at 4Hz for a specific phase.
        
        Args:
            phase_index: Phase index to retrieve (default: 0)
            
        Returns:
            RR intervals series resampled at 4Hz in milliseconds
        """
        if self.rri_ph is None:
            raise ValueError("RRi data not available")
        return self.rri_ph[0, phase_index] * 1000.0  # Convert from seconds to milliseconds
    
    def get_rr_timestamps(self, phase_index: int = 0) -> np.ndarray:
        """
        Get time stamps for R-R intervals for a specific phase.
        
        Args:
            phase_index: Phase index to retrieve (default: 0)
            
        Returns:
            Time stamps array for the specified phase in seconds
        """
        if self.t_rr_ph is None:
            raise ValueError("RR timestamps not available")
        return self.t_rr_ph[0, phase_index]
    
    def get_rri_timestamps(self, phase_index: int = 0) -> np.ndarray:
        """
        Get time stamps for HRV series resampled at 4Hz for a specific phase.
        
        Args:
            phase_index: Phase index to retrieve (default: 0)
            
        Returns:
            Time stamps array for RRi (4Hz sampling) in seconds
        """
        if self.t_rri_ph is None:
            raise ValueError("RRi timestamps not available")
        return self.t_rri_ph[0, phase_index]
    
    def get_video_info(self, video_index: int = 0) -> Optional[Dict[str, Any]]:
        """
        Get video information (only available for EMO condition).
        
        Each video contains:
        - time_start: Start time in seconds
        - time_end: End time in seconds
        - time_type: Emotional valence ('POS', 'NEUT', or 'NEG')
        
        Args:
            video_index: Video index to retrieve (default: 0)
            
        Returns:
            Dictionary with video info (time_start, time_end, time_type) 
            or None if not EMO condition
        """
        if self.condition != 'EMO':
            return None
        if self.videos is None:
            raise ValueError("Video data not available")
        
        video_data = self.videos[0, video_index]
        
        # Try to extract structured video information
        try:
            # The video data might be a structured array
            video_info = {
                'time_start': video_data['time_start'][0, 0][0, 0] if hasattr(video_data, '__getitem__') else None,
                'time_end': video_data['time_end'][0, 0][0, 0] if hasattr(video_data, '__getitem__') else None,
                'type': str(video_data['type'][0, 0][0]) if hasattr(video_data, '__getitem__') else None,
            }
            return video_info
        except (KeyError, IndexError, TypeError):
            # If extraction fails, return raw data
            return video_data
    
    def get_physio_signal(self, signal_name: str, phase_index: int = 0) -> Optional[Dict[str, Any]]:
        """
        Get a specific physiological signal by name.
        
        Available signals: 'ECG1', 'ECG2', 'Resp', 'BP', 'TCD_sx', 'TCD_dx', 
                          'Percutaneous_flow1', 'Percutaneous_flow2'
        
        Args:
            signal_name: Name of the signal to retrieve
            phase_index: Phase index to retrieve (default: 0)
            
        Returns:
            Dictionary with 'data', 'samplerate', 'name', 'time' keys
            or None if signal not found
        """
        signal_map = {
            'ECG1': 0, 'ECG2': 1, 'Resp': 2, 'BP': 3,
            'TCD_sx': 4, 'TCD_dx': 5, 
            'Percutaneous_flow1': 6, 'Percutaneous_flow2': 7
        }
        
        if signal_name not in signal_map:
            raise ValueError(f"Unknown signal '{signal_name}'. Available: {list(signal_map.keys())}")
        
        physio_data = self.get_physio_data(phase_index)
        signal_idx = signal_map[signal_name]
        
        try:
            signal_struct = physio_data[signal_idx, 0]
            return {
                'data': signal_struct['data'][0, 0],
                'samplerate': signal_struct['samplerate'][0, 0][0, 0],
                'name': str(signal_struct['name'][0, 0][0]),
                'time': signal_struct['time'][0, 0]
            }
        except (KeyError, IndexError, TypeError):
            return None
    
    def get_montage_info(self, phase_index: int = 0) -> Dict[str, Any]:
        """
        Get information about the channel montage/positions for a specific phase.
        
        Args:
            phase_index: Phase index to check (default: 0)
            
        Returns:
            Dictionary with montage information:
                - 'type': 'custom_eeglab', 'standard_1020', or 'none'
                - 'has_custom_positions': bool
                - 'num_channels': int
                - 'channels_with_positions': list of channel names
                
        Example:
            >>> subject = load_subject('i01', 'EMO')
            >>> info = subject.get_montage_info()
            >>> if info['type'] == 'custom_eeglab':
            >>>     print("Using actual EEGLAB positions!")
        """
        if self.eeg_ph is None:
            raise ValueError("EEG data not available")
        
        eeg_struct_wrapped = self.eeg_ph[0, phase_index]
        eeg_struct = eeg_struct_wrapped[0, 0] if eeg_struct_wrapped.shape == (1, 1) else eeg_struct_wrapped
        
        # Check for custom positions
        chanlocs = eeg_struct['chanlocs']
        ch_pos_dict = {}
        
        if chanlocs.shape == (1, eeg_struct['data'].shape[0]):
            for i in range(eeg_struct['data'].shape[0]):
                ch = chanlocs[0, i]
                ch_name = str(ch['labels'][0])
                try:
                    x = float(ch['X'][0, 0]) if hasattr(ch['X'], '__getitem__') else float(ch['X'])
                    y = float(ch['Y'][0, 0]) if hasattr(ch['Y'], '__getitem__') else float(ch['Y'])
                    z = float(ch['Z'][0, 0]) if hasattr(ch['Z'], '__getitem__') else float(ch['Z'])
                    # MNE expects positions in meters; EEGLAB coordinates are in cm, so divide by 10
                    ch_pos_dict[ch_name] = np.array([x/10, y/10, z/10])
                except:
                    pass
        
        has_custom = len(ch_pos_dict) > 0
        montage_type = 'custom_eeglab' if has_custom else 'standard_1020'
        
        return {
            'type': montage_type,
            'has_custom_positions': has_custom,
            'num_channels': eeg_struct['data'].shape[0],
            'channels_with_positions': list(ch_pos_dict.keys()) if has_custom else [],
            'num_custom_positions': len(ch_pos_dict)
        }
    
    def get_num_phases(self) -> int:
        """
        Get the number of phases in the data.
        
        Returns:
            Number of phases
        """
        if self.eeg_ph is not None:
            return self.eeg_ph.shape[1]
        return 0
    
    def get_num_videos(self) -> int:
        """
        Get the number of videos (only for EMO condition).
        
        Returns:
            Number of videos or 0 if not EMO condition
        """
        if self.condition != 'EMO' or self.videos is None:
            return 0
        return self.videos.shape[1]
    
    def get_all_video_info(self) -> Optional[list]:
        """
        Get information for all videos (only for EMO condition).
        
        Returns:
            List of dictionaries containing video info (time_start, time_end, time_type)
            or None if not EMO condition
        """
        if self.condition != 'EMO':
            return None
        
        num_videos = self.get_num_videos()
        return [self.get_video_info(i) for i in range(num_videos)]
    
    def summary(self) -> Dict[str, Any]:
        """
        Get a summary of the loaded data.
        
        Returns:
            Dictionary containing data summary information
        """
        summary_dict = {
            'subject_id': self.subject_id,
            'condition': self.condition,
            'file_path': str(self.file_path),
            'num_phases': self.get_num_phases(),
            'data_fields': {
                'EEG_ph': self.eeg_ph.shape if self.eeg_ph is not None else None,
                'Physio_ph': self.physio_ph.shape if self.physio_ph is not None else None,
                'RR_ph': self.rr_ph.shape if self.rr_ph is not None else None,
                'RRi_ph': self.rri_ph.shape if self.rri_ph is not None else None,
                'T_RR_ph': self.t_rr_ph.shape if self.t_rr_ph is not None else None,
                'T_RRi_ph': self.t_rri_ph.shape if self.t_rri_ph is not None else None,
            }
        }
        
        if self.condition == 'EMO':
            summary_dict['num_videos'] = self.get_num_videos()
            summary_dict['data_fields']['videos'] = self.videos.shape if self.videos is not None else None
        
        return summary_dict
    
    def __repr__(self) -> str:
        """String representation of the SubjectData object."""
        return f"SubjectData(subject_id='{self.subject_id}', condition='{self.condition}', phases={self.get_num_phases()})"
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        videos_info = f", videos={self.get_num_videos()}" if self.condition == 'EMO' else ""
        return f"Subject {self.subject_id} - {self.condition} condition (phases: {self.get_num_phases()}{videos_info})"


def load_subject(subject_id: str, condition: str = 'EMO', data_root: str = 'data') -> SubjectData:
    """
    Convenience function to load subject data.
    
    Args:
        subject_id: Subject identifier (e.g., 'i01')
        condition: Condition type ('EMO' or 'PH1')
        data_root: Root directory containing subject folders
        
    Returns:
        SubjectData object with loaded data
        
    Example:
        >>> subject = load_subject('i01', 'EMO')
        >>> raw = subject.get_eeg_data(phase_index=0)  # Returns MNE Raw object
        >>> raw.plot()  # Visualize with MNE
    """
    return SubjectData(subject_id, condition, data_root)


def load_all_subjects(data_root: str = 'data', condition: str = 'EMO') -> Dict[str, SubjectData]:
    """
    Load all available subjects for a given condition.
    
    Args:
        data_root: Root directory containing subject folders
        condition: Condition type ('EMO' or 'PH1')
        
    Returns:
        Dictionary mapping subject IDs to SubjectData objects
        
    Example:
        >>> all_emo = load_all_subjects(condition='EMO')
        >>> print(f"Loaded {len(all_emo)} subjects")
    """
    data_path = Path(data_root)
    subjects = {}
    
    # Find all subject folders (i01, i02, etc.)
    for subject_dir in sorted(data_path.glob('i*')):
        if subject_dir.is_dir():
            subject_id = subject_dir.name
            try:
                subjects[subject_id] = SubjectData(subject_id, condition, data_root)
            except FileNotFoundError:
                # Skip if this subject doesn't have data for this condition
                continue
    
    return subjects
