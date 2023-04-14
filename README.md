# eecs_6322_project


# Swin UNETR BTCV

In this notebook, SwinUNETR is trained on BTCV dataset

The notebook contains 3 main sections

#### **1 Dataloader** 
    # Visualizing data loading from BTCV
#### **2 Training**
    # In the root directory I have provided the configuration file (config.ini)
    # Load the parameters 
    # Load SwinUnetR
        # SwinUnetR* and its submodules are in SwinUnetR folder
            # Swin_UnetR.py
            # Swin_Tansfomerblock
            # Patch_merging
            # comon
                # window_partition
                # window_reverse
                # get_window_size
                # compute_mask
        *The modules that are taken from other paper/resources are explicily mentioned in each file
    # the tesnsorboard graph of training could be seen here:

View results on the tensorboard  [Tensorboard](https://tensorboard.dev/experiment/iYsNW9B3TGehd1299SaPvw/#scalars)
        
#### **3 Testing**
    # Load model trained in Section 2
    # Run testing


