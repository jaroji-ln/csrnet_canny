# CSRNet-Canny: Crowd Counting with Edge-Aware Enhancement

This repository implements an enhanced version of [CSRNet](https://doi.org/10.1109/CVPR.2018.00120) for crowd counting by integrating Canny edge detection. The model is optimized for extremely dense crowds such as in the Hajj pilgrimage scene using the HaCrowd dataset.

## Features
- Based on CSRNet with VGG16 frontend.
- Fourth input channel added (RGB + Canny edge).
- Pretrained weights and evaluation scripts included.
- SHAP-based interpretability supported.
- Compatible with HaCrowd and ShanghaiTech datasets.

---

## Project Structure

attention/
csrnet_canny/
├── model.py
├── train.py
├── dataset.py
├── image.py
├── utils.py
├── earlystoping.py
├── train.ipynb
├── make_dataset.ipynb
└── evaluation.ipynb

---
## Download Pretrained Models

Due to GitHub's file size limit, the pretrained `.pth.tar` models are hosted externally. Use the following script to download them:

| Model                             | Link                                                                       |
| --------------------------------- | -------------------------------------------------------------------------- |
| canny-hacrowd-model\_best.pth.tar | [Google Drive](https://drive.google.com/file/d/1w5aryyir8ze23MEL9OYdG6Oj1KNht3Cp/view?usp=sharing) |
| canny-shta-model\_best.pth.tar    | [Google Drive](https://drive.google.com/file/d/1baR0KFwNEfi44WG6J6hvOqkVxmGRcRID/view?usp=sharing) |
| canny-shtb-model\_best.pth.tar    | [Google Drive](https://drive.google.com/file/d/1hwEYDgmpVsJlq7AGPZu2yPgg2OfvZHqJ/view?usp=sharing) |


## Download Pretrained Models
python train.py hacrowd_train_data.json hacrowd_val_data.json 0 canny-hacrowd-

## Result
| Dataset        | Model                 | MAE  | RMSE  | SSIM | PSNR  |
| -------------- | --------------------- | ---- | ----- | ---- | ----- |
| HaCrowd        | CSRNet + Canny (Ours) | 54.9 | 76.2  | 27.0 | 0.765 |
| ShanghaiTech A | CSRNet + Canny (Ours) | 76.1 | 115.0 | 23.8 | 0.649 |
| ShanghaiTech B | CSRNet + Canny (Ours) | 14.8 | 24.1  | 26.2 | 0.737 |

## Citation