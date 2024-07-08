# AnomalGRN

This is the PyTorch code implementation of **AnomalGRN: Deciphering Single Cell Gene Regulation Network with Graph Anomaly Detection**.

## Dataset

First, you need to download the dataset from [Google Drive](https://drive.google.com/open?id=1DJW-Y27qpjw_XhQHrtfxdG3tI88ztYPe&usp=drive_fs) or [Zenodo](https://zenodo.org/records/12176604).

## Requirements

Configure the following Python requirements:

- torch 1.13.1
- dgl 1.1.2
- scipy 1.13.0
- numpy 1.23.0
- pandas 2.2.2
- scikit-learn 1.4.2
- torch_geometric 2.5.2

## Running the Model

You can enter the following code on the command line to run the model:

```bash
python main.py --dataset_type STRING --dataset_name hESC --top 500
