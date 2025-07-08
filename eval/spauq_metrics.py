import os
from datetime import datetime
import argparse
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from spauq.core.metrics import spauq_eval


RETURN_FRAMEWISE = False
RETURN_COST = True
RETURN_SHIFT = True
RETURN_SCALE = True

STEMS = ["drums", "bass", "other", "vocals"]


def evaluate(ref_path, est_path):
    """
    Compute SPAUQ metrics for predicted WAV file.

    Parameters
    ----------
    ref_path : str
               Path to ground-truth stem WAV file.
    est_path : str
               Path to estimated stem WAV file.

    Returns
    -------
    dict
        A dictionary of the following keys:
        - SSR
        - SRR
        - cost
        - shift
        - scale
    """
    # load reference and estimate stems
    y_ref, sr_ref = sf.read(ref_path)
    y_est, sr_est = sf.read(est_path)

    n_ref, c_ref = y_ref.shape
    n_est, c_est = y_est.shape
    
    # check sample rates
    assert sr_ref == sr_est
    
    # check number of channels
    assert c_ref == c_est == 2

    # make sure that the number of samples is the same
    # if not, trim the longer signal
    if n_ref != n_est:
        min_n = min(n_ref, n_est)
        y_ref = y_ref[:min_n, :]
        y_est = y_est[:min_n, :]
        
    # compute SPAUQ metrics
    transpose array so that the dimensions are (channels, samples)
    eval_out = spauq_eval(reference=y_ref.T, estimate=y_est.T, fs=sr_ref,
                          return_framewise=RETURN_FRAMEWISE, return_cost=RETURN_COST,
                          return_shift=RETURN_SHIFT, return_scale=RETURN_SCALE,
                          forgive_mode=None, verbose=False)
    return eval_out


def main():
    """
    Parse command-line arguments and compute SPAUQ metrics.
    """
    # parse arguments
    parser = argparse.ArgumentParser(description="Compute SPAUQ metrics for a \
                                     given set of estimated and reference stems")

    parser.add_argument("ref_dir", help="Path to the directory which contains \
                        subdirectories for each song containing the reference stems")
    parser.add_argument("est_dir", help="Path to the directory which contains \
                        subdirectories for each song containing the estimated stems")
    parser.add_argument("-o", "--output_dir", metavar='', default='./metrics',
                    help="Path to the output directory to save the metrics CSV \
                          file to (default: ./metrics)")
    parser.add_argument("-n", "--name", metavar='',
                        default="spauq_metrics",
                        help="Name of output CSV file with computed SPAUQ metrics \
                        (default: spauq_metrics)")
    
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
    ssr_list = []
    srr_list = []
    cost_list = []
    shift_list = []
    scale_list = []

    # iterate through each source and compute SSR and SRR
    print("Beginning to evaluate stems...")
    for source in STEMS:
        print(f"\n>>>>{source} <<<<")
        for song in tqdm(song_list):
            # get .wav file paths
            ref_file = os.path.join(REFERENCE_DIR, song, f"{source}.wav")
            est_file = os.path.join(ESTIMATE_DIR, song, f"{source}.wav")

            eval_out = evaluate(ref_file, est_file)

            title_list.append(song)
            source_list.append(source)
            ssr_list.append(eval_out["SSR"])
            srr_list.append(eval_out["SRR"])
            cost_list.append(eval_out["cost"])
            shift_list.append(eval_out["shift"])
            scale_list.append(eval_out["scale"])

    results_df = pd.DataFrame({"title": title_list, "source": source_list,
                                 "SSR": ssr_list, "SRR": srr_list,
                                 "cost": cost_list, "shift": shift_list,
                                 "scale": scale_list})
    print("Evaluation complete!")

    # spauq metrics by frame
    results_df.sort_values(by=['title', 'source'], inplace=True, ignore_index=True)

    save_path = os.path.join(args.output_dir, f'{args.name}.csv')
    results_df.to_csv(save_path, index=False)


if __name__ == '__main__':
    main()
