# brain-gcn-2

Few different projects in one repo.

 - Melanie code in the `brain_project` directory;
 - An experiment with clustering and VAEs in the `VAdE` directory;
 - Current graph inference experiments in `brain_project/nri` directory.
 
## Running NRI training

There are two different training loops:
 1. Training on the real data (`train.py`). Parameters are set within the file directly.
    Data needs to be generated first (using the `data_utils.py` file) and placed where the trainer can see it (see the `data_folder` parameter).
 2. Training on synthetic data (`train_synthetic.py`).
    Parameters are also set within the file, but this should run without much hassle.
    
    
## Dependencies

 - PyTorch (version 0.4)
 - tensorboardX (`pip install tensorboardX`)
 - The usual standard python packages `numpy`, `matplotlib`, `scipy`, etc.
 
