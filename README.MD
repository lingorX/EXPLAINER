## Installation

For installation and data preparation, please refer to the guidelines in [MMClassification v0.18.0](https://github.com/open-mmlab/mmclassification/tree/v0.18.0) and [MMSegmentation v0.20.2](https://github.com/open-mmlab/mmsegmentation/tree/v0.20.2).

All studies are conducted on ```CUDA 11.4``` and  ```pytorch 1.8.1``` 

```
pip install torchvision==0.9.1
pip install timm==0.3.2
pip install mmcv-full==1.4.1
pip install opencv-python==4.5.1.48
cd DNC_classification && pip install -e . --user
```

We also provide the conda environment file. You can directly activate it without complex installation via
```
export PATH="$(pwd)/cellp/bin:$PATH"
export LD_LIBRARY_PATH="$(pwd)/cellp/lib:$LD_LIBRARY_PATH"
```

## Training


For the training of EXPLAINER with ResNet-50 as the backbone, you can run it on Cifar-100 by simply applying the following command:
```
sh env_run.sh
```

For other setups, please replace the config file specified in env_run.sh with your customized file. For example, train on ImageNet-1K with swin-small as the backbone, you can apply
```
./tools/dist_train.sh configs/_pattern_/swin-small_8xb128_in1k.py 4
```

## Testing

For the testing of EXPLAINER, please specify the config and checkpoint file in the following command

```
python tools/test.py configs/_pattern_/resnet50_EXPLAINER_4xb32_cifar100.py model.pth --metrics accuracy
```