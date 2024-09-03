## Model Zoo

### Environment and Settings

- 4/1 NVIDIA V100 GPUs for training/evaluation.
- Auto-mixed precision was enabled in training but disabled in evaluation.
- Test-time augmentations were not used.
- The inference resolution was 480p as [DeAOT](https://github.com/yoxu515/aot-benchmark).
- Fully online inference. We passed all the modules frame by frame.


### Pre-trained Models
Stages:

- `PRE`: the pre-training stage with static images are the same as [DeAOT](https://github.com/yoxu515/aot-benchmark).

- `PRE_YTB_DAV`: the main-training stage with YouTube-VOS and DAVIS. 


| Model       | Param |                                             PRE                                              | PRE_YTB_DAV (LVOS eval checkpoints) | PRE_YTB_DAV (LTV eval checkpoints) | PRE_YTB_DAV (DAVIS eval checkpoints) |
|:------------|:-----:|:--------------------------------------------------------------------------------------------:|:-----------------------------------:|:-----------------------:|:------------------------------------:|
| MAVOS       |  34M  | [gdrive](https://drive.google.com/file/d/1_vBL4KJlmBy0oBE4YFDOvsYL1ZtpEL32/view?usp=sharing) |             [gdrive](https://drive.google.com/file/d/1KhD6e-590vlAUitIXWFuhOv1typfTB-N/view?usp=sharing)             |       [gdrive](https://drive.google.com/file/d/1wIgUZ36BVWzUL5nmOifct7U42I2nS8mT/view?usp=sharing)       |              [gdrive](https://drive.google.com/file/d/1jEh1qDacEZSnPyZTL913hW3GUHKhqaOn/view?usp=sharing)              |
| R50-MAAVOS  |  41M  | [gdrive](https://drive.google.com/file/d/1sTRQ1g0WCpqVCdavv7uJiZNkXunBt3-R/view?usp=sharing) |             [gdrive](https://drive.google.com/file/d/1Hp-j0hNWesSBle6HQxm6ghJE9b1THn1I/view?usp=sharing)             |       [gdrive](https://drive.google.com/file/d/1AdyEiwjc6l5OrK9PBViFwcJs00UdQe5U/view?usp=sharing)       |             [gdrive](https://drive.google.com/file/d/1pfKbm--DdHPYXJfZwE7LKaaIPzjRDn2K/view?usp=sharing)              |
| SwinB-MAVOS |  91M  | [gdrive](https://drive.google.com/file/d/16BZEE53no8CxT-pPLDC2q1d6Xlg8mWPU/view?usp=sharing) |             [gdrive](https://drive.google.com/file/d/1HbRHeChnQWK73OYaYov-bchfwbm4lRyf/view?usp=sharing)             |       [gdrive](https://drive.google.com/file/d/1QYxyHa_GFrW4IT44qBv8vqEHpDAj4gVv/view?usp=sharing)       |             [gdrive](https://drive.google.com/file/d/1KSbKmKZg66u6edSqLVZyKzlanyVOhAvt/view?usp=sharing)              |

To use our pre-trained models to infer, a simple way is to set `--model` and `--ckpt_path` to your downloaded checkpoint's model type and file path when running `eval.py`.
