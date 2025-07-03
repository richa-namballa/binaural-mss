import os
import time
import keras
import tensorflow as tf
from tqdm import tqdm
import numpy as np
from spleeter.audio import Codec, STFTBackend
from spleeter.audio.adapter import AudioAdapter
from spleeter.separator import Separator


# constants
STEMS = ["drums", "bass", "other", "vocals"]
SAMPLE_RATE = 44100
BITRATE = "128k"
CODEC = Codec.WAV
adapter_type = "spleeter.audio.ffmpeg.FFMPEGProcessAudioAdapter"
ADAPTER = AudioAdapter.get(adapter_type)


def separate(in_file, model):
    """
    Run Spleeter source separation model on WAV files to isolate stems.

    Parameters
    ----------
    in_file : str
              Path to the mixture WAV file to separate.
    model : spleeter.separator.Separator
            Spleeter model object used for source separation.

    Returns
    -------
    list of np.ndarray
        List of separated stems, each as a NumPy array.
    """
    # only process wav files
    if in_file.endswith(".wav"):
        # read the soundfile
        waveform, _ = ADAPTER.load(in_file, sample_rate=SAMPLE_RATE)
    
        out = model.separate(waveform)
        out_stems = []
        for s in STEMS:
            est_stem = out[s]
            out_stems.append(est_stem)

    else:
        out_stems = None
        print("Invalid input file type. Please use WAV files only")

    return out_stems


def main():
    """
    Parse command-line arguments and separate every mixture in the input directory.
    """
    # parse arguments
    parser = argparse.ArgumentParser(description="Run Spleeter on a set of input mixtures.")

    parser.add_argument("input_dir", help="Path to the input directory \
                        which contains subdirectories for each song containing the mixtures")
    parser.add_argument("-o", "--output_dir", default='./spleeter_out',
                        help="Path to the output directory to save the separated \
                        stems by song to (default: ./spleeter_out)")

    args = parser.parse_args()

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
        for file in tqdm(file_list):
            # load the model in each loop because of a bug in spleeter 2.3.2
            # see the following GitHub issues:
            # https://github.com/deezer/spleeter/issues/815
            # https://github.com/deezer/spleeter/issues/809
            sess = tf.compat.v1.InteractiveSession()
            MODEL = Separator('spleeter:4stems', MWF=False,
                              stft_backend=STFTBackend.AUTO,
                              multiprocess=False)
            song_dir = os.path.join(args.output_dir, file)
            os.makedirs(song_dir, exist_ok=True)
            out = separate(os.path.join(args.input_dir, file, 'mixture.wav'), MODEL)
            if out:
                for i in range(len(STEMS)):
                    out_path = os.path.join(song_dir, STEMS[i] + '.wav')
                    ADAPTER.save(out_path, out[i], SAMPLE_RATE, CODEC, BITRATE)
            sess.close()
            time.sleep(5)  # addresses race condition issue
        print("Processing complete!")

    except FileNotFoundError as e:
        print(f"[Error] The directory '{args.input_dir}' was not found.")

    except Exception as e:
        print(f"[Error] An unexpected error occurred: {e}")


if __name__ == '__main__':

    main()



