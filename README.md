Below text provides the entire step by step format of running the individual project.
## StackGAN 

### 1. Install Requirements

```
pip install -r requirements.txt
```
### 2. Check for Dataset Folder
For the data we have provided link in below path into CUB_200_2011 folder for CUB image dataset, and into text_embeddings for captions.

data/
```
├── CUB_200_2011/
|   └── images/ 
└── cub_text_embeddings/
    └── filenames.pickle
```
### 3. Use of Caption

You can run generate_caption_pickle.py for converting pickle to txt format. This will save file as filenames.pickle.

### 4. Train StackGAN

**Stage I Training**
```
python train_stage1.py
```
**Stage II Training**
```
python train_stage2.py
```
### 5. View Outputs

Generated outputs will be saved in:

output/ → Stage I images(It will be 16 images in 4x4 format per epoch with individual image size of 64x64)

output22/ → Stage II refined images(It will be 16 images in 4x4 format per epoch with individual image size of 256x256)

**Notes**

The training uses pre-extracted 1024-dim text embeddings.

Make sure your dataset paths match those in data_loader.py.

We used two nodes of BigRed's GPU(its mentioned 2 in our code)

## GAN-INT-CLS

The file is ```gan_int_cls.py``` for this, its kaggle file and has used [This](https://www.kaggle.com/datasets/wenewone/cub2002011) dataset and was run as a notebook.

You can directly run this on kaggle with in-built GPU.

## Stable Diffusion

This code was incomplete and was not able to provide us any results. However mentioned here for effort and progress purpose.
