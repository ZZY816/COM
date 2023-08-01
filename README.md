# **CVPR2023 : Curricular Object Manipulation in LiDAR-based Object Detection**

This repository is the official PyTorch implementation of our COM. [[**arXiv**](https://arxiv.org/abs/2304.04248)]

## **Abstract**

This paper explores the potential of curriculum learning in LiDAR-based 3D object detection by proposing a curricular object manipulation (COM) framework. The framework embeds the curricular training strategy into both the loss design and the augmentation process. For the
loss design, we propose the COMLoss to dynamically predict object-level difficulties and emphasize objects of different difficulties based on training stages. On top of the widely-used augmentation technique called GT-Aug in LiDAR detection tasks, we propose a novel COMAug strategy
which first clusters objects in ground-truth database based on well-designed heuristics. Group-level difficulties rather than individual ones are then predicted and updated during training for stable results. Model performance and generalization capabilities can be improved by sampling and augmenting progressively more difficult objects into the training samples. Extensive experiments and ablation studies reveal the superior and generality of the proposed framework.

## Overview
- [Installation](docs/INSTALL.md)
- [Getting Started on Waymo](docs/GETTING_STARTED_WAYMO.md)

## TODO list
- [ ] Support the training on KITTI Dataset
- [ ] Support the training on NuScenes Dataset
- [ ] Support more state-of-the-art detectors

## Acknowledgement
Our code is based on the following repos.
- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)
- [CenterPoint](https://github.com/tianweiy/CenterPoint)
