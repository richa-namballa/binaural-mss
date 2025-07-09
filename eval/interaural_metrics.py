import os
from datetime import datetime
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import rfft, irfft
from sklearn.utils.extmath import weighted_mode
import soundfile as sf


# constants
SAMPLE_RATE = 44100
ENERGY_THRESHOLD = 5e-4
TMAX = int(1e-3 * SAMPLE_RATE)  # maximum lag in samples = +/- 1 ms
FRAME_LENGTH = 0.5  # seconds

STEMS = ["drums", "bass", "other", "vocals"]

# ITD/ILD Evaluation Functions
# Veluri, B., Itani, M., Chan, J., Yoshioka, T., & Gollakota, S. (2023).

def tdoa(x1, x2, interp=1, fs=44100, phat=True, t_max=None):
    """
    This function computes the time difference of arrival (TDOA)
    of the signal at the two ears or microphones. We recover tau
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)
    method.
    
    Knapp, C., & Carter, G. C. (1976). The generalized correlation method for
    estimation of time delay.
    
    Parameters
    ----------
    x1 : np.ndarray
        Signal of the first microphone.
    x2 : np.ndarray
        Signal of the second microphone.
    interp : int, optional (default 1)
        Interpolation value for the cross-correlation,
        it can improve the time resolution (and hence DOA resolution).
    fs : int, optional (default 44100 Hz)
        Sampling frequency of the input signals.
    phat : bool, optional (default True)
        Whether to use the phase transform normalization.
    t_max : int, optional (default None)
        Maximum value of tau (lag) to use.

    Return
    ------
    tdoa : float
        The delay between the two signals (in seconds).
    """
    # zero padded length for the FFT
    n = x1.shape[-1] + x2.shape[-1] - 1
    if n % 2 != 0:
        n += 1

    # Generalized Cross Correlation Phase Transform
    # used to find the delay between the two microphones
    X1 = rfft(np.array(x1, dtype=np.float32), n=n, axis=-1)
    X2 = rfft(np.array(x2, dtype=np.float32), n=n, axis=-1)

    if phat:
        X1 /= np.abs(X1) + 1e-15  # added epsilon to avoid division error
        X2 /= np.abs(X2) + 1e-15  # added epsilon to avoid division error

    cc = irfft(X1 * np.conj(X2), n=interp * n, axis=-1)

    # alternative: compute phase spectrum first and then normalize
    # R = X1 * np.conj(X2)
    # R_phat = R / (np.abs(R) + 1e-15)
    # cc = irfft(R_phat, n=interp * n, axis=-1)

    # maximum possible delay given distance between microphones
    if t_max is None:
        t_max = n // 2 + 1

    # reorder the cross-correlation coefficients
    cc = np.concatenate((cc[..., -t_max:], cc[..., :t_max]), axis=-1)

    # pick max cross correlation index as delay
    tau = np.argmax(np.abs(cc), axis=-1)
    tau -= t_max  # because zero time is at the center of the array

    return tau / (fs * interp)


def framewise_gccphat(x, frame_dur, sr, window='tukey'):
    """
    Compute the TDOA using the GCC-PHAT algorithm, in
    a frame-wise manner.

    Parameters
    ----------
    x : np.ndarray
        The 2-channel binaural signal.
    frame_dur : float
        Desired length of each frame (in seconds).
    sr : int
        Sample rate of the signal.
    window : str, optional (default Tukey)
        Type of window to apply to each frame
        (using scipy window functions).

    Return
    ------
    itd : float
        Interaural time difference (ITD), in seconds.
    """
    # get size of the frame in samples
    frame_width = int(frame_dur * sr)

    # total number of frames T
    T = (x.shape[-1]) // frame_width

    # drop samples to get a multiple of frame size
    if x.shape[-1] % T != 0:
        x = x[..., :(frame_width * T)]
    
    assert x.shape[-1] % T == 0

    # split into frames
    frames = np.array(np.split(x, T, axis=-1))

    # apply window
    window = signal.get_window(window, frame_width)
    frames = frames * window

    # consider only frames that have energy above some threshold (ignore silence)
    frame_energy = np.max(np.mean(frames**2, axis=-1)**0.5, axis=-1)
    mask = frame_energy > ENERGY_THRESHOLD
    frames = frames[mask]

    # compute TDOA by frame
    fw_gccphat = tdoa(frames[..., 0, :], frames[..., 1, :], fs=sr, t_max=TMAX)

    # apply weighted mode to get single ITD value
    itd = weighted_mode(fw_gccphat, frame_energy[mask], axis=-1)[0]

    return itd[0]


def fw_itd_diff(s_est, s_gt, sr, frame_duration=0.25):
    """
    Compute the ITD error between the estimated signal and the ground-truth signal
    using the frame-wise GCC-PHAT algorithm.

    Parameters
    ----------
    s_est : np.ndarray
        Estimated 2-channel signal.
    s_gt : np.ndarray
        Ground-truth 2-channel signal.
    sr : int
        Sample rate of the signal.
    frame_duration : float, optional
        Length of each frame in seconds. Default is 0.25.

    Returns
    -------
    itd : float
        Difference in interaural time difference (ITD) between the estimated
        and ground-truth signals, in microseconds.
    """
    itd_gt = framewise_gccphat(s_gt, frame_duration, sr) * 1e6
    itd_est = framewise_gccphat(s_est, frame_duration, sr) * 1e6
    return np.abs(itd_est - itd_gt)


def compute_ild(s_left, s_right):
    """
    Compute the interaural level difference (ILD) between the
    left and right channels.

    Parameters
    ----------
    s_left : np.ndarray
        Signal of the left ear (1D array).
    s_right : np.ndarray
        Signal of the right ear (1D array).

    Returns
    -------
    ild : float
        Interaural level difference (ILD) in decibels (dB).
    """
    sum_sq_left = np.sum(s_left ** 2, axis=-1)
    sum_sq_right = np.sum(s_right ** 2, axis=-1)
    
    return 10 * np.log10(sum_sq_left / sum_sq_right)


def ild_diff(s_est, s_gt):
    """
    Compute the ILD error between the signal estimated
    by the model and the ground truth signal.

    Parameters
    ----------
    s_est : np.ndarray
        The estimated 2-channel signal.
    s_gt : np.ndarray
        The ground-truth 2-channel signal.
    
    Return
    ------
    ild : float
        Difference in interaural level difference (ILD) between the estimated
        and ground-truth signals, in decibels (dB).
    """
    ild_est = compute_ild(s_est[..., 0, :], s_est[..., 1, :])
    ild_gt = compute_ild(s_gt[..., 0, :], s_gt[..., 1, :])
    return np.abs(ild_est - ild_gt)


def main():
    """
    Parse command-line arguments and compute SPAUQ metrics.
    """
    # parse arguments
    parser = argparse.ArgumentParser(description="Compute interaural metrics for a \
                                     given set of estimated and reference stems")

    parser.add_argument("ref_dir", help="Path to the directory which contains \
                        subdirectories for each song containing the reference stems")
    parser.add_argument("est_dir", help="Path to the directory which contains \
                        subdirectories for each song containing the estimated stems")
    parser.add_argument("-o", "--output_dir", metavar="", default='./metrics',
                    help="Path to the output directory to save the metrics CSV \
                          file to (default: ./metrics)")
    parser.add_argument("-n", "--name", metavar="",
                        default="interaural_metrics",
                        help="Name of output CSV file with computed SPAUQ metrics \
                        (default: interaural_metrics)")
    
    args = parser.parse_args()

    # create the output directory if it does not already exist
    print("Creating evaluation directory, if it does not already exist...")
    os.makedirs(args.output_dir, exist_ok=True)

    # set input directories
    REFERENCE_DIR = args.ref_dir
    ESTIMATE_DIR = args.est_dir

    # get all of the files in the input directory
    print("Loading list of files...")
    song_list = [f for f in os.listdir(REFERENCE_DIR)
                 if os.path.isdir(os.path.join(REFERENCE_DIR, f))]
    print(f"There are {len(song_list)} files in the reference directory.")

    title_list = []
    source_list = []
    itd_list = []
    ild_list = []

    # iterate through each source and compute SSR and SRR
    print("Beginning to evaluate stems...")
    for source in STEMS:
        print(f"\n>>>>{source} <<<<")
        for song in tqdm(song_list):
            ref_file = os.path.join(REFERENCE_DIR, song, f"{source}.wav")
            est_file = os.path.join(ESTIMATE_DIR, song, f"{source}.wav")

            # load reference and estimate stems
            y_ref, sr_ref = sf.read(ref_file)
            y_est, sr_est = sf.read(est_file)

            # check sample rates
            assert sr_ref == sr_est == SAMPLE_RATE

            # calculate Delta ITD
            delta_itd = fw_itd_diff(y_est.T, y_ref.T, SAMPLE_RATE, FRAME_LENGTH)

            # calculate ILD
            delta_ild = ild_diff(y_est.T, y_ref.T)

            title_list.append(song)
            source_list.append(source)
            itd_list.append(delta_itd)
            ild_list.append(delta_ild)

    results_df = pd.DataFrame({"title": title_list, "source": source_list,
                           "diff_ITD": itd_list, "diff_ILD": ild_list})
    print("Evaluation complete!")

    # interaural metrics
    results_df.sort_values(by=['title', 'source'], inplace=True, ignore_index=True)

    save_path = os.path.join(args.output_dir, f'{args.name}.csv')
    results_df.to_csv(save_path, index=False)


if __name__ == '__main__':
    main()
