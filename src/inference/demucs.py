import os
import argparse
import torch
import numpy as np
import soundfile as sf
from demucs import pretrained
from demucs.apply import apply_model


# constants
STEMS = ["drums", "bass", "other", "vocals"]
SAMPLE_RATE = 44100


def separate(in_file, model, gpu=None):
    """
    Run Demucs source separation model on WAV files to isolate stems.

    :param in_file: (str) path of mixture WAV file to separate
    :param model: (demucs.BagOfModels) demucs model object
    :param gpu: (torch.device) if a gpu is available for use, pass in the device
    """
    # only process wav files
    if in_file.endswith(".wav"):
        # read the soundfile
        y, sr = sf.read(in_file)

        # check if audio is in mono
        if len(y.shape) == 1:
            # if the audio is in mono,
            # duplicate channels to create a stereo track
            # demucs network expects two channels of audio
            y = np.vstack([y, y])

        # get dimensions
        num_samples, num_channels = y.shape

        # convert to 3 dimensional tensor (1, num_channels, num_samples)
        x = torch.from_numpy(y.T.reshape(1, num_channels, num_samples).astype(np.float32))

        # output is [1, S, C, T] where S is the number of sources
        if gpu:
            # use current gpu device
            out = apply_model(model, x, progress=True, device=gpu)
        else:
            # use cpu
            out = apply_model(model, x, progress=True)

        # drums.wav, bass.wav, other.wav, vocals.wav
        out_stems = []
        for i in range(4):
            est_stem = out[0][i]
            out_stems.append(np.array(est_stem).T)

    else:
        out_stems = None
        print("Invalid input file type. Please use WAV files only.")

    return out_stems


def main():
    """
    Separate every mixture in the input directory.
    """
    # parse arguments
    parser = argparse.ArgumentParser(description="Run Demucs on a set of input mixtures.")

    parser.add_argument("input_dir", help="Path to the input directory \
                        which contains subdirectories for each song containing the mixtures")
    parser.add_argument("-o", "--output_dir", default='./demucs_out',
                        help="Path to the output directory to save the separated \
                        stems by song to (default: ./demucs_out)")
    parser.add_argument("-n", "--name", required=False,
                        choices=["htdemucs", "htdemucs_ft", "hdemucs_mmi"],
                        default="htdemucs_ft",
                        help="Name of pre-trained Demucs model to use (default: htdemucs_ft)")
    
    args = parser.parse_args()

    # load the model
    print("Loading pretrained model...")
    MODEL = pretrained.get_model('htdemucs_ft')
    print("Model loaded successfully.")

    # get all of the files in the input directory
    print("Loading list of files...")
    try:
        file_list = [f for f in os.listdir(args.input_dir)
                     if os.path.isdir(os.path.join(args.input_dir, f))]
        if len(file_list) == 0:
            raise ValueError(f"The directory '{args.input_dir}' is empty.")

        print(f"There are {len(file_list)} files in the input directory.")

        # create the output directory if it does not already exist
        print("Creating output directory, if it does not already exist...")
        os.makedirs(args.output_dir, exist_ok=True)

        # iterate through each file
        print("Beginning to process files...")
        for file in file_list:
            song_dir = os.path.join(args.output_dir, file)
            os.makedirs(song_dir, exist_ok=True)
            out = separate(os.path.join(args.input_dir, file, 'mixture.wav'), MODEL, DEVICE)
            if out:
                for i in range(len(STEMS)):
                    out_path = os.path.join(song_dir, STEMS[i] + '.wav')
                    sf.write(out_path, out[i], SAMPLE_RATE)
        print("Processing complete!")
    
    except FileNotFoundError as e:
        print(f"[Error] The directory '{args.input_dir}' was not found.")
    
    except Exception as e:
        print(f"[Error] An unexpected error occurred: {e}")


if __name__ == '__main__':

    # gpu check
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = None
        print("GPU not found. Defaulting to CPU.")

    main()



