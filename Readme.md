# Quantization-based Multi-scale Ensemble Network for Accurate Biomedical Image Segmentation

![1](./figures/1.png)





## Requirements

The  code requires

- Python 3.6 or higher
- PyTorch 0.4 or higher

and the requirements highlighted in [requirements.txt](./requirements.txt) (for Anaconda)



## Training

*The default setting of QME-Net is as follows. The ensemble size K is 4, and Q1: 4-bit, Q2: 5-bit and Q3: 6-bit are the quantization bit for base learners B1, B2, B3, while Q0: 16-bit for meta learner B0.*

To train the CE-Net based QME-Net with default setting in the paper on DRIVE dataset, run this command:

```train
python train_pyramid.py 
```

![2](./figures/2.png)

## Evaluation

To evaluate my model on DRIVE dataset, run:

```eval
python eval_pyramid.py 
```

![5](./figures/5.png)








