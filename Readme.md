# Pyramid-Net: Intra-layer Pyramid-scale Feature Aggregation for Retinal Vessel Segmentation

![1](./figures/0.pdf)



## Requirements

The  code requires

- Python 3.6 or higher
- PyTorch 0.4 or higher

and the requirements highlighted in [requirements.txt](./requirements.txt) (for Anaconda)

## Abstract
Retinal vessel segmentation plays an important role in the diagnosis of eye-related diseases and biomarkers discovery. Existing works perform multi-scale feature aggregation in an inter-layer manner, namely inter-layer feature aggregation. However, such an approach only fuses feature at either a lower scale or a higher scale, which may result with a low segmentation performance especially on thin vessels. This discovery motivates us to fuse multi-scale features in each layer, namely intra-layer feature aggregation, to mitigate the problem. Therefore, in this paper, we propose Pyramid-Net for accurate retinal vessel segmentation, which features intra-layer pyramid-scale aggregation blocks (IPABs). At each layer, IPABs generate two associated branches at a higher scale and a lower scale, respectively, and the two with the main brunch at the current scale operate in a pyramid-scale manner. Three further enhancements including pyramid inputs enhancement, deep pyramid supervision, and pyramid skip connections are proposed to boost the performance. We have evaluated Pyramid-Net on three public retinal fundus photography datasets (DRIVE, STARE, and CHASE-DB1). The experimental results show that Pyramid-Net can effectively improve the segmentation performance especially on thin vessels, and outperforms the current state-of-the-art methods on all the adopted three datasets. In addition, our method is more efficient than existing methods with a large reduction on computational cost.



## Architecture
![2](./figures/1.pdf)

## Training

To train the Pyramid-Net with default setting in the paper on DRIVE dataset, run this command:

```train
python train_pyramid.py 
```


## Evaluation

To evaluate my model on DRIVE dataset, run:

```eval
python eval_pyramid.py 
```

![5](./figures/2.pdf)








