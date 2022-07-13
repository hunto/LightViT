# LightViT
Official implementation for paper "[LightViT: Towards Light-Weight Convolution-Free Vision Transformers](https://arxiv.org/abs/2207.05557)".

By Tao Huang, Lang Huang, Shan You, Fei Wang, Chen Qian, Chang Xu.


## Updates  
Code will be released soon.

<p align='center'>
<img src='./assests/architecture.png' alt='mask' width='1000px'>
</p>

## Results on ImageNet-1K  

|model|resolution|acc@1|acc@5|#params|FLOPs|ckpt|log|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|LightViT-T|224x224|78.7|94.4|9.4M|0.7G|||
|LightViT-S|224x224|80.9|95.3|19.2M|1.7G|||
|LightViT-B|224x224|82.1|95.9|35.2M|3.9G|||

### Evaluation

### Train from scratch on ImageNet-1K

### Throughput

## License  
This project is released under the [Apache 2.0 license](LICENSE).

## Citation  
```
@article{huang2022lightvit,
  title = {LightViT: Towards Light-Weight Convolution-Free Vision Transformers},
  author = {Huang, Tao and Huang, Lang and You, Shan and Wang, Fei and Qian, Chen and Xu, Chang},
  journal = {arXiv preprint arXiv:2207.05557},
  year = {2022}
}
```