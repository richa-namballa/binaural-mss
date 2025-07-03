import os
import time
import torch
from tqdm import tqdm
import openunmix
import numpy as np
import soundfile as sf

# constants
STEMS = ["drums", "bass", "other", "vocals"]
SAMPLE_RATE = 44100


def separate(in_file, model, gpu=True):
    """
    Run Open-Unmix source separation model on WAV files to isolate stems.

    Parameters
    ----------
    in_file : str
              Path to the mixture WAV file to separate.
    model : Separator
            Open-Unmix model object model object used for source separation.
    gpu : bool
          Whether a GPU is available to be used for processing.

    Returns
    -------
    list of np.ndarray
        List of separated stems, each as a NumPy array.
    """
    # only process wav files
    if in_file.endswith(".wav"):
        # read the soundfile
        y, sr = sf.read(in_file)

        num_samples, num_channels = y.shape
      
        if gpu:
            # send model to gpu
            model.to('cuda')
            
            # convert to 3 dimensional tensor (1, num_channels, num_samples)
            x = torch.from_numpy(y.T.reshape(1, num_channels, num_samples).astype(np.float32)).to('cuda')
            with torch.no_grad():
                out = model(x)

            out_stems = []
            for i in range(len(STEMS)):
                est_stem = out[0][i].cpu().detach().numpy()  # convert to numpy array
                est_stem[np.isnan(est_stem)] = 0  # convert nan to 0
                out_stems.append(est_stem)
        else:
            print("Please enable CUDA for inference.")

    else:
        out_stems = None
        print("Invalid input file type. Please use WAV files only.")

    return out_stems


def main():
    """
    Parse command-line arguments and separate every mixture in the input directory.
    """
    # parse arguments
    parser = argparse.ArgumentParser(description="Run Open-Unmix on a set of input mixtures.")

    parser.add_argument("input_dir", help="Path to the input directory \
                        which contains subdirectories for each song containing the mixtures")
    parser.add_argument("-o", "--output_dir", default='./umx_out',
                        help="Path to the output directory to save the separated \
                        stems by song to (default: ./umx_out)")
    
    args = parser.parse_args()

    # load the model
    print("Loading pretrained model...")
    MODEL = openunmix.umxhq()
    assert MODEL.sample_rate == SAMPLE_RATE, "Sample rates do not match!"
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
        for file in tqdm(file_list):
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
    else:
        raise RuntimeError("GPU not found. CUDA device required for inference.")

    main()



