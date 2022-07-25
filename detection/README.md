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
|backbone|params|FLOPs|Setting|box AP|mask AP|config|log|ckpt|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|LightViT-T|28|187|1x|37.8|35.9|[config](configs/mask_rcnn_lightvit_tiny_fpn_1x_coco.py)|||
|LightViT-S|38|204|1x|40.0|37.4||||
|LightViT-B|54|240|1x|41.7|38.8||||
|LightViT-T|28|187|3x+ms|41.5|38.4||||
|LightViT-S|38|204|3x+ms|43.2|39.9||||
|LightViT-B|54|240|3x+ms|45.0|41.2||||