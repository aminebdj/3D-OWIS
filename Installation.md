## Code structure
Our code is built on top of [Mask3D](https://github.com/JonasSchult/Mask3D) and follows the structure of [Mix3D](https://github.com/JonasSchult/Mix3D).

```
├── mix3d
│   ├── main_instance_segmentation.py <- the main file
│   ├── conf                          <- hydra configuration files
│   ├── datasets
│   │   ├── preprocessing             <- folder with preprocessing scripts
│   │   ├── semseg.py                 <- indoor dataset
│   │   └── utils.py        
│   ├── models                        <- Mask3D modules
│   ├── trainer
│   │   ├── __init__.py
│   │   └── trainer.py                <- train loop
│   └── utils
├── data
│   ├── processed                     <- folder for preprocessed datasets
│   └── raw                           <- folder for raw datasets
├── scripts                           <- train scripts
├── docs
├── README.md
└── saved                             <- folder that stores models and logs
```

### Dependencies :memo:
The main dependencies of the project are the following:
```yaml
python: 3.10.6
cuda: 11.6
```
You can set up a conda environment as follows
```
conda create --name=3d_owis python=3.10.6
conda activate 3d_owis

conda update -n base -c defaults conda
conda install openblas-devel -c anaconda

pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu116.html

pip install ninja==1.10.2.3
pip install pytorch-lightning==1.7.2 fire imageio tqdm wandb python-dotenv pyviz3d scipy plyfile scikit-learn trimesh loguru albumentations volumentations

pip install antlr4-python3-runtime==4.8
pip install black==21.4b2
pip install omegaconf==2.0.6 hydra-core==1.0.5 --no-deps
pip install 'git+https://github.com/facebookresearch/detectron2.git@710e7795d0eeadf9def0e7ef957eea13532e34cf' --no-deps

cd third_party/pointnet2 && python setup.py install

pip install fvcore
pip install reliability
pip install shortuuid
pip install pycocotools==2.0.4
pip install seaborn 
pip install cloudpickle==2.1.0

git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas

```

### Data preprocessing :hammer:
After installing the dependencies, we preprocess the datasets.

#### ScanNet200
First, we apply Felzenswalb and Huttenlocher's Graph Based Image Segmentation algorithm to the test scenes using the default parameters.
Please refer to the [original repository](https://github.com/ScanNet/ScanNet/tree/master/Segmentator) for details.
Put the resulting segmentations in `./data/raw/scannet_test_segments`.
```
python datasets/preprocessing/scannet_preprocessing.py preprocess \
--data_dir="PATH_TO_RAW_SCANNET_DATASET" \
--save_dir="../../data/processed/scannet200" \
--git_repo="PATH_TO_SCANNET_GIT_REPO" \
--scannet200=true
```
Adding the open world label database to scannet200 folder
```
cp datasets/scannet200/OW_label_database.yaml ./data/scannet200
```

Adding the examplars used for examplar replay 
```
cp -r datasets/scannet200/exemplars ./data/scannet200
```
