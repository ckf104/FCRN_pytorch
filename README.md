1. Install dependencies:

    For PyTorch dependency, see [pytorch.org](https://pytorch.org/) for more details.

    For custom dependencies:
    ```Shell
    pip install matplotlib pillow tensorboardX
    ```

2. Configure your dataset path.

3. Reproduce our experiment results on Make3D dataset

    First, we do an offline data augmentation.
    ```shell
    python -m dataloaders.make3d_dataloader <your-make3d-data-path>
    ```
    This command will generate a train folder in your make3d dataset path, which contains augmented train dataset.

    Then run the following command to train a network model.

    ```shell
    python main.py --batch-size=16 --epochs=31 --lr_patience=5 --dataset=make3d --upper-limit=70.0 --dataset-dir=<your-make3d-dataset-path>
    ```

4. Reproduce our experiment results on KITTI dataset

    The KITTI dataset folder should be organized with four subdirectories
    * train: the downloaded train set of depth map should be decompressed into this folder
    * val: the downloaded test set of depth map should be decompressed into this folder
    * rgb_train: the downloaded train set of rgb images should be decompressed into this folder
    * rgb_val: the downloaded test set of rgb images should be decompressed into this folder

    Then run the following command to train a network model.
    ```shell
    python main.py --batch-size=16 --average-lr --epochs=30 --lr_patience=5 --dataset=kitti --dataset-dir=<your-kitti-dataset-path>
    ``` 




