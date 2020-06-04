# SOLOv2-detectron2
Unofficial implementation for [SOLOv2: Dynamic, Faster and Stronger](https://arxiv.org/abs/2003.10152) instance segmentation.
There are still a few bugs.

|config|bbox|mask|weight|
|-|:-:|-:|-:|
|MS_R_50_2x.yaml|37.486|35.953|[google drive](https://drive.google.com/file/d/1BFTtOOcheJBbxp7Bkk-hgs2SNh2upIIr/view?usp=sharing)|

## Install
The code is based on [detectron2](https://github.com/facebookresearch/detectron2). Please check [Install.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md) for installation instructions.


## Training 
Follows the same way as detectron2.

Single GPU:
```
python train_net.py --config-file configs/MS_R_50_2x.yaml
```
Multi GPU(for example 8):
```
python train_net.py --num-gpus 8 --config-file configs/MS_R_50_2x.yaml
```
Please adjust the IMS_PER_BATCH in the config file according to the GPU memory.


## Inference
First replace the original detectron2 installed postprocessing.py with the [file](https://github.com/gakkiri/SOLOv2-detectron2/blob/master/postprocessing.py).

Single GPU:
```
python train_net.py --config-file configs/MS_R_50_2x.yaml --eval-only MODEL.WEIGHTS /path/to/checkpoint_file
```
Multi GPU(for example 8):
```
python train_net.py --num-gpus 8 --config-file configs/MS_R_50_2x.yaml --eval-only MODEL.WEIGHTS /path/to/checkpoint_file
```

## Results 
![box](https://raw.githubusercontent.com/gakkiri/SOLOv2-detectron2/master/img/box50.png?x-oss-Process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzNDk3ODQ1,size_16,color_FFFFFF,t_70)
![seg](https://raw.githubusercontent.com/gakkiri/SOLOv2-detectron2/master/img/mask50.png?x-oss-Process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzNDk3ODQ1,size_16,color_FFFFFF,t_70)  
