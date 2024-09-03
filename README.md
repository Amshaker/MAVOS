# MAVOS
### **Efficient Video Object Segmentation via Modulated Cross-Attention Memory**

<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="Oryx Video-ChatGPT">
</p>

#### [Abdelrahman Shaker](https://amshaker.github.io), [Syed Talal Wasim](https://talalwasim.github.io/), [Martin Danelljan](https://martin-danelljan.github.io/), [Salman Khan](https://salman-h-khan.github.io/), [Ming-Hsuan Yang](https://scholar.google.com.pk/citations?user=p9-ohHsAAAAJ&hl=en) and [Fahad Khan](https://sites.google.com/view/fahadkhans/home)


#### **Mohamed bin Zayed University of AI, ETH Zurich, University of California - Merced, Yonsei University, Google Research, Linköping University**

<!-- [![Website](https://img.shields.io/badge/Project-Website-87CEEB)](site_url) -->
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/pdf/2403.17937.pdf)
[![video](https://img.shields.io/badge/Video-Presentation-F9D371)](https://youtu.be/wfIDLZ9aN3M)


## Latest Updates
- `2024/09/02`: MAVOS checkpoints, training, and evaluation code are now available.
- `2024/08/30`: MAVOS has been accepted at WACV 2025! 🎊
- `2024/03/27`: Our technical report on MAVOS has been published on arXiv.

<br>
<details>
  <summary>
  <font size="+1">Abstract</font>
  </summary>
Recently, transformer-based approaches have shown promising results for semi-supervised video object segmentation. However, these approaches typically struggle on long videos due to increased GPU memory demands, as they frequently expand the memory bank every few frames. We propose a transformer-based approach, named MAVOS, that introduces an optimized and dynamic long-term modulated cross-attention (MCA) memory to model temporal smoothness without requiring frequent memory expansion. The proposed MCA effectively encodes both local and global features at various levels of granularity while efficiently maintaining consistent speed regardless of the video length. Extensive experiments on multiple benchmarks, LVOS, Long-Time Video, and DAVIS 2017, demonstrate the effectiveness of our proposed contributions leading to real-time inference and markedly reduced memory demands without any degradation in segmentation accuracy on long videos. Compared to the best existing transformer-based approach, our MAVOS increases the speed by 7.6x, while significantly reducing the GPU memory by 87% with comparable segmentation performance on short and long video datasets. Notably on the LVOS dataset, our MAVOS achieves a J&F score of 63.3% while operating at 37 frames per second (FPS) on a single V100 GPU. Our code and models will be publicly released.
</details>

## Intro

- **MAVOS** is a transformer-based VOS method that achieves real-time FPS and reduced GPU memory for long videos.


<img src="source/MAVOS_overview.png" width="90%"/>

- **MAVOS**  increases the speed by 7.6x over the baseline DeAOT, while significantly reducing the GPU memory by 87% on long videos with comparable segmentation performance on short and long video datasets.

<img src="source/Intro_figure.png" width="90%"/>

## Examples

https://github.com/user-attachments/assets/ca2902ed-0b82-4129-89c3-3824c782818a

https://github.com/user-attachments/assets/d4d5b77c-5fa3-4fbb-a94c-df7f8ccb3413

<br>


## Requirements
   * Python3
   * pytorch >= 1.7.0 and torchvision
   * opencv-python
   * Pillow
   * Pytorch Correlation. Recommend to install from [source](https://github.com/ClementPinard/Pytorch-Correlation-extension) instead of using `pip`:
     ```bash
     git clone https://github.com/ClementPinard/Pytorch-Correlation-extension.git
     cd Pytorch-Correlation-extension
     python setup.py install
     cd -
     ```

## Model Zoo
Pre-trained models of our project can be found in [MODEL_ZOO.md](MODEL_ZOO.md).


## Getting Started
0. Prepare a valid environment follow the [requirements](#requirements).
1. We use the pre-trained weights of DeAOT-L on static images as baseline (Recommended). No need for pretraining. If you want to pre-train MAVOS from scratch, consider the following dataset preperation:
   1. Prepare datasets:

       Please follow the below instruction to prepare datasets in each corresponding folder.
       * **Static** 
    
           [datasets/Static](datasets/Static): pre-training dataset with static images. Guidance can be found in [AFB-URR](https://github.com/xmlyqing00/AFB-URR), which we referred to in the implementation of the pre-training.
       * **YouTube-VOS**

           A commonly-used large-scale VOS dataset.

           [datasets/YTB/2019](datasets/YTB/2019): version 2019, download [link](https://drive.google.com/drive/folders/1BWzrCWyPEmBEKm0lOHe5KLuBuQxUSwqz?usp=sharing). `train` is required for training.

       * **DAVIS**

           A commonly-used small-scale VOS dataset.

           [datasets/DAVIS](datasets/DAVIS): [TrainVal](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip) (480p) contains both the training and validation split. [Test-Dev](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-test-dev-480p.zip) (480p) contains the Test-dev split. The [full-resolution version](https://davischallenge.org/davis2017/code.html) is also supported for training and evaluation but not required.

   2. Prepare ImageNet pre-trained encoders

      Select and download below checkpoints into [pretrain_models](pretrain_models):
      - [MobileNet-V2](https://download.pytorch.org/models/mobilenet_v2-b0353104.pth) (default encoder)
      - [ResNet-50](https://download.pytorch.org/models/resnet50-0676ba61.pth)
      - [Swin-Base](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth)



3. Training: the [training script](train.sh) will fine-tune the pre-trained models using 4 GPUs on both `YouTube-VOS 2019 train` and `DAVIS-2017 train`, resulting in a model that can generalize to different domains.

    
4. Evaluation
: the [evaluation script](evaluate.sh) will evaluate the models on LVOS, DAVIS, and LTV. The results will be packed into Zip files. For calculating scores, please use official [LVOS toolkit](https://github.com/LingyiHongfd/lvos-evaluation) (for Val), [DAVIS toolkit](https://github.com/davisvideochallenge/davis-2017) (for Val). For the Long-Time Video dataset, use the same DAVIS toolkit and replace --davis_path to long_video videos with the corresponding annotations.

## Results on Long videos benchmarks
### LVOS val set
<img src="source/LVOS.png" width="90%"/>

### LTV 
<img src="source/LTV.png" width="90%"/>

## Acknowledgment
Our code base is based on the [AOT](https://github.com/yoxu515/aot-benchmark/tree/main) repository. We thank the authors for their open-source implementation.

The computations were enabled by resources provided by the National Academic Infrastructure for Supercomputing in Sweden (NAISS) at Alvis partially funded by the Swedish Research Council through grant agreement no. 2022-06725, the LUMI supercomputer hosted by CSC (Finland) and the LUMI consortium, and by the Berzelius resource provided by the Knut and Alice Wallenberg Foundation at the National Supercomputer Centre.

## Citations
Please consider citing our paper in your publications if it helps your research.
```

@article{Shaker2024MAVOS,
  title={Efficient Video Object Segmentation via Modulated Cross-Attention Memory},
  author={Shaker, Abdelrahman and Wasim, Syed and Danelljan, Martin and Khan, Salman and Yang, Ming-Hsuan and Khan, Fahad Shahbaz},
  journal={arXiv:2403.17937},
  year={2024}
}
```

## License
This project is released under the BSD-3-Clause license. See [LICENSE](LICENSE) for additional details.
