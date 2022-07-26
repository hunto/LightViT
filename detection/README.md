# Applying LightViT to Object Detection  

## Usage  
## Preparations  
* Install mmdetection `v2.14.0` and mmcv-full  
    ```shell
    pip install mmdet==2.14.0
    ```
    For mmcv-full, please check [here](https://mmdetection.readthedocs.io/en/latest/get_started.html#install-mmdetection) and [compatibility](https://mmdetection.readthedocs.io/en/latest/compatibility.html).

* Put the COCO dataset into `./data` folder following [[this url]](https://mmdetection.readthedocs.io/en/latest/1_exist_data_model.html#prepare-datasets).  

**Note:** if you use a different mmdet version, please replace the `configs`, `train.py`, and `test.py` with the corresponding files of the version, then add `import lightvit` and `import lightvit_fpn` into `train.py` to register the model.

## Train Mask-RCNN with LightViT backbones  
|backbone|params|FLOPs|setting|box AP|mask AP|config|log|ckpt|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|LightViT-T|28|187|1x|37.8|35.9|[config](configs/mask_rcnn_lightvit_tiny_fpn_1x_coco.py)|[log](https://github.com/hunto/LightViT/releases/download/v0.0.1/mask_rcnn_lightvit_tiny_fpn_1x_coco.log.json)|[google drive](https://drive.google.com/file/d/15sSit0VpcsmL3Pr-fdjM6nRdmSZre0zh/view?usp=sharing)|
|LightViT-S|38|204|1x|40.0|37.4|[config](configs/mask_rcnn_lightvit_small_fpn_1x_coco.py)|[log](https://github.com/hunto/LightViT/releases/download/v0.0.1/mask_rcnn_lightvit_small_fpn_1x_coco.log.json)|[google drive](https://drive.google.com/file/d/146RWYNShe5IrltvmgWoelj27LaRQBcXE/view?usp=sharing)|
|LightViT-B|54|240|1x|41.7|38.8|[config](configs/mask_rcnn_lightvit_base_fpn_1x_coco.py)|[log](https://github.com/hunto/LightViT/releases/download/v0.0.1/mask_rcnn_lightvit_base_fpn_1x_coco.log.json)|[google drive](https://drive.google.com/file/d/19IHkVO_Qy__NHe0syrUpIKrhGNoS9VCd/view?usp=sharing)|
|LightViT-T|28|187|3x+ms|41.5|38.4|[config](configs/mask_rcnn_lightvit_tiny_fpn_3x_ms_coco.py)|[log](https://github.com/hunto/LightViT/releases/download/v0.0.1/mask_rcnn_lightvit_tiny_fpn_3x_ms_coco.log.json)|[google drive](https://drive.google.com/file/d/1EfhtuCVCoprbwb_6W-32z_po4hGvAzu8/view?usp=sharing)|
|LightViT-S|38|204|3x+ms|43.2|39.9|[config](configs/mask_rcnn_lightvit_small_fpn_3x_ms_coco.py)|[log](https://github.com/hunto/LightViT/releases/download/v0.0.1/mask_rcnn_lightvit_small_fpn_3x_ms_coco.log.json)|[google drive](https://drive.google.com/file/d/16l5lXYwz3Anx28ncqPfc3mir-nyZyRd6/view?usp=sharing)|
|LightViT-B|54|240|3x+ms|45.0|41.2|[config](configs/mask_rcnn_lightvit_base_fpn_3x_ms_coco.py)|[log](https://github.com/hunto/LightViT/releases/download/v0.0.1/mask_rcnn_lightvit_base_fpn_3x_ms_coco.log.json)|[google drive](https://drive.google.com/file/d/1hLKdemruEKW2DO0227ZNm1zSZCqMWVfi/view?usp=sharing)|


### Training script:  

```shell
sh dist_train.sh ${CONFIG} 8 work_dirs/${EXP_NAME}
```