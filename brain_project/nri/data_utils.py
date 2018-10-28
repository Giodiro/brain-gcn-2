import os
import scipy.io as sio
import json
import numpy as np
import warnings
import time

from sklearn.model_selection import train_test_split

from util import mkdir_p, time_str


all_phases = {"REM_phasic": 1, "REM_tonic": 2,
              "S2_Kcomplex": 3, "S2_plain": 4,
              "S2_spindle": 5, "SWS_deep": 6
             }


def split_within_subj(subject_list, subj_data, seed=1922318, test_size=0.1):
    """
    subj_data : dictionary with key: int
    """
    # Restrict to valid subjects
    nsubj_data = {k: v for k, v in subj_data.items() if v["subj"] in subject_list}

    # random split into 2 subsets
    keys = list(nsubj_data.keys())
    strata = [subj_data[k]["phase"] for k in keys]
    keys_train, keys_test = train_test_split(keys, test_size=test_size,
        random_state=seed, shuffle=True, stratify=strata)

    return keys_train, keys_test




def prepare_timeseries_data(data_folder, subsample, sample_size, out_folder, save_batch_size):
    base_folder = os.path.join(out_folder, f"subsample{subsample}_size{sample_size}_batch{save_batch_size}")
    y_folder = os.path.join(base_folder, "Y")
    x_folder = os.path.join(base_folder, "X")
    mkdir_p(y_folder)
    mkdir_p(x_folder)
    print(f"{time_str()} Created output folders {x_folder} and {y_folder}.")

    t1 = time.time()
    X = []; Y = []
    i = 0; save_i = 0
    subj_data = {}

    for subj in os.listdir(data_folder):
        path_subj = os.path.join(data_folder, subj)
        if subj not in ["S01", "S12"]:
            continue
        if not os.path.isdir(path_subj):
            continue
        print(f"Loading subject {subj}...")
        for phase in os.listdir(path_subj):
            path_phase = os.path.join(path_subj, phase)
            if phase not in all_phases:
                continue
            if not os.path.isdir(path_phase):
                continue
            for file in os.listdir(path_phase):
                # if re.search(r'average', file) is not None:
                #     continue
                path_file = os.path.join(path_phase, file)
                mat_data = sio.loadmat(path_file)["Value"]
                np_data = np.asarray(mat_data).T # [tot_tpoints, num_nodes]
                ## Subsample and split
                if subsample != 1:
                    np_data = np_data[::subsample,:]
                if sample_size > np_data.shape[0]:
                    warnings.warn("Desired sample size larger than the actual number of samples.")
                    sample_size = np_data.shape[0]
                if sample_size < np_data.shape[0]:
                    split_idxs = [sample_size * i for i in
                        range(1, np_data.shape[0] // sample_size + 1)]
                    np_data = [s for s in np.split(np_data, split_idxs, axis=0)
                                if s.shape[0] == sample_size]
                else:
                    np_data = [np_data]

                ## Save to file
                X.extend(np_data)
                Y.extend([all_phases[phase]] * len(np_data))
                for k in range(i, i + len(np_data)):
                    subj_data[k] = {"subj": subj, "phase": all_phases[phase]}
                i += len(np_data)

                while len(X) >= save_batch_size:
                    save_fname = f"{subj}_{save_i}.npy"
                    for j, k in enumerate(range(save_i * save_batch_size, (save_i + 1) * save_batch_size)):
                        subj_data[k]["file"] = save_fname
                        subj_data[k]["index_in_file"] = j
                    np.save(os.path.join(x_folder, save_fname), X[:save_batch_size])
                    np.save(os.path.join(y_folder, save_fname), Y[:save_batch_size])

                    X = X[save_batch_size:]
                    Y = Y[save_batch_size:]
                    save_i += 1

    # Final save. Only the last file can be smaller than the rest, to make
    # reading data back easier.
    if len(X) >= 0:
        save_fname = f"{subj}_{save_i}.npy"
        for j, k in enumerate(range(save_i * save_batch_size,
                                    save_i * save_batch_size + len(X))):
            subj_data[k]["file"] = save_fname
            subj_data[k]["index_in_file"] = j
        np.save(os.path.join(x_folder, save_fname), X[:save_batch_size])
        np.save(os.path.join(y_folder, save_fname), Y[:save_batch_size])
        save_i += 1

    # Save metadata
    metadata_file = os.path.join(base_folder, "subj_data.json")
    with open(metadata_file, "w") as fh:
        json.dump(subj_data, metadata_file)
    print(f"{time_str()} Saved subject metadata at {metadata_file}.")

    t2 = time.time()
    print(f"{time_str()} Loaded {i} observations batched into {save_i} files "
          f"in {t2 - t1:.2f}s.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--type", "-t", type=str, choices=["timeseries", "imcoh"], default="timeseries")
    parser.add_argument("--data-folder", "-d", type=str, required=True)
    parser.add_argument("--out-folder", "-o", type=str, required=True)

    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--subsample", "-s", type=int, default=1)
    parser.add_argument("--batch-size", "-b", type=int, default=32)

    args = parser.parse_args()


    if args.type == "imcoh":
        raise NotImplementedError("Still not implemented")

    elif args.type == "timeseries":
        print(f"{time_str()} Parsing time-series data, subsampled by "
              f"{args.subsample}, aggregated at {args.sample_size}. "
              f"Saving to {args.out_folder}.")

        prepare_timeseries_data(args.data_folder,
                                args.subsample,
                                args.sample_size,
                                args.out_folder,
                                args.batch_size)

        print(f"{time_str()} Finished processing.")



