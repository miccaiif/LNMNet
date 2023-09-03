# :pig2: LNMNet

This is a PyTorch/GPU implementation of our paper [Predicting Lymph Node Metastasis from Primary Cervical Squamous Cell Carcinoma Based on Deep Learning in Histopathological Images](https://www.sciencedirect.com/science/article/abs/pii/S0893395223002211).

This paper has been accepted by Modern Pathology in August 2023! 

<p align="center">
  <img src="https://github.com/miccaiif/LNMNet/blob/main/work_flow.jpg" width="720">
</p>

### For training
* Please refer to the [training code](https://github.com/miccaiif/LNMNet/blob/main/train_load_by_epoch.py) for model training.

<p align="center">
  <img src="https://github.com/miccaiif/LNMNet/blob/main/framework.jpg" width="720">
</p>

### For inference
* Please refer to the [testing code](https://github.com/miccaiif/LNMNet/blob/main/fun_load_by_epoch_main.py) for model inference.

<p align="center">
  <img src="https://github.com/miccaiif/LNMNet/blob/main/ROC_curve.jpg" width="720">
</p>

### For attention-based visualization
* Please refer to the [attention visualization](https://github.com/miccaiif/LNMNet/blob/main/features_found.py) for localization of key instances.

<p align="center">
  <img src="https://github.com/miccaiif/LNMNet/blob/main/attention.png" width="720">
</p>

### Citation
If this work is helpful to you, please cite it as:
```
@article{guo2023predicting,
  title={Predicting Lymph Node Metastasis from Primary Cervical Squamous Cell Carcinoma Based on Deep Learning in Histopathological Images},
  author={Guo, Qinhao and Qu, Linhao and Zhu, Jun and Li, Haiming and Wu, Yong and Wang, Simin and Yu, Min and Wu, Jiangchun and Wen, Hao and Ju, Xingzhu and others},
  journal={Modern Pathology},
  pages={100316},
  year={2023},
  publisher={Elsevier}
}
```
### Contact Information
If you have any question, please email to me [lhqu20@fudan.edu.cn](lhqu20@fudan.edu.cn).
