# HFC-MFFD
These files contain source codes we use in our paper for testing the forgery classification evaluation accuracy on ForgeryNIR dataset.
Considering that preprocessing might take a certain time, we provide feature extracted from these images by our trained feature extraction models, and provide ForgeryClassifier to obtain testing accuracy.

## Dependencies

* Anaconda3 (Python3.9, with Numpy etc.)
* Pytorch 1.12.0

## Datasets
[ForgeryNIR Dataset](https://github.com/AEP-WYK/forgerynir) contains 240,000 forgery NIR images:
- images generated via 4 different GAN techniques. 
- images added different number of perturbation.
- images generated by different epoch models of the same GAN.   

### Download dataset

| Dataset Name | Download                                                   | Images  |
| ------------ | ---------------------------------------------------------- | ------- |
| ForgeryNIR   | [ForgeryNIR](https://github.com/AEP-WYK/forgerynir)        | 240,000 |


## Evaluation
### Download feature and trained models obtained from the ForgeryNIR dataset for testing

| Feature and Model   | Download                                                     |
| ------- | ------------------------------------------------------------ |
| HFC-MFFD | [BaiduNetDisk(ia97)](https://pan.baidu.com/s/1Wgm8uSrkGdWeAYuhRSITcw) |


After downloading the feature and trained models, the feature should be put to `./ForgeryNIR/feature`, and the models should be put to `./ForgeryNIR/model`. Otherwise, you should change the default path we declare in `config.py`.

### Test the model

```
python -m simple_test --train_dir std_multi --test_dir mix_multi
```

## Download Models
