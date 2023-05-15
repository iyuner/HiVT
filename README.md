HiVT
---

## Setup

### Dataset

1. Download [Argoverse Motion Forecasting Dataset v1.1](https://www.argoverse.org/av1.html). After downloading and extracting the tar.gz files, the dataset directory should be organized as follows:
  ```
  /path/to/dataset_root/
  ├── train/
  |   └── data/
  |       ├── 1.csv
  |       ├── 2.csv
  |       ├── ...
  └── val/
      └── data/
          ├── 1.csv
          ├── 2.csv
          ├── ...
  ```

2. Setup Docker
  ```
  docker build -t zhangkin/hivt .
  # or directly from dockerhub
  docker pull zhangkin/hivt
  ```

3. Run Container
  ```
  docker run --gpus all -it -v /home/kin/DATA_HDD/yy:/root/data -p 6006:6006 zhangkin/hivt:full /bin/zsh
  ```

## Training

### From Scarth
To train HiVT-64:
```
python train.py --root /root/data --embed_dim 64
```

To train HiVT-128:
```
python train.py --root /root/data --embed_dim 128
```

### Continuing checkpoint

```
python train.py --root /root/data --embed_dim 128 --ckpt_path /root/HiVT/checkpoints/HiVT-128/checkpoints/epoch=63-step=411903.ckpt
```

### Monitor
**Note**: When running the training script for the first time, it will take several hours to preprocess the data (~3.5 hours on my machine). Training on an RTX 2080 Ti GPU takes 35-40 minutes per epoch.

During training, the checkpoints will be saved in `lightning_logs/` automatically. To monitor the training process:
```
tensorboard --logdir lightning_logs/
```

## Evaluation

To evaluate the prediction performance:
```
python eval.py --root /path/to/dataset_root/ --batch_size 32 --ckpt_path /path/to/your_checkpoint.ckpt
```

## Pretrained Models

We provide the pretrained HiVT-64 and HiVT-128 in [checkpoints/](checkpoints). You can evaluate the pretrained models using the aforementioned evaluation command, or have a look at the training process via TensorBoard:
```
tensorboard --logdir checkpoints/
```

## Results

### Quantitative Results

For this repository, the expected performance on Argoverse 1.1 validation set is:

| Models | minADE | minFDE | MR |
| :--- | :---: | :---: | :---: |
| HiVT-64 | 0.69 | 1.03 | 0.10 |
| HiVT-128 | 0.66 | 0.97 | 0.09 |

### Qualitative Results

![](assets/visualization.png)

## Citation

If you found this repository useful, please consider citing our work:

```
@inproceedings{zhou2022hivt,
  title={HiVT: Hierarchical Vector Transformer for Multi-Agent Motion Prediction},
  author={Zhou, Zikang and Ye, Luyao and Wang, Jianping and Wu, Kui and Lu, Kejie},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```

## License

This repository is licensed under [Apache 2.0](LICENSE).

