# RealisticTTA

### Unraveling Batch Normalization for Realistic Test-Time Adaptation <AAAI 2024> 
Zixian Su, Jingwei Guo, Kai Yao, Xi Yang, Qiufeng Wang,  Kaizhu Huang

**Abstract**

While recent test-time adaptations exhibit efficacy by adjusting batch normalization to narrow domain disparities, their effectiveness diminishes with realistic mini-batches due to inaccurate target estimation. As previous attempts merely introduce source statistics to mitigate this issue, the fundamental problem of inaccurate target estimation still persists, leaving the intrinsic test-time domain shifts unresolved. This paper delves into the problem of mini-batch degradation. By unraveling batch normalization, we discover that the inexact target statistics largely stem from the substantially reduced class diversity in batch. Drawing upon this insight, we introduce a straightforward tool, Test-time Exponential Moving Average (TEMA), to bridge the class diversity gap between training and testing batches. Importantly, our TEMA adaptively extends the scope of typical methods beyond the current batch to incorporate a diverse set of class information, which in turn boosts an accurate target estimation. Built upon this foundation, we further design a novel layer-wise rectification strategy to consistently promote test-time performance.
Our proposed method enjoys a unique advantage as it requires neither training nor tuning parameters, offering a truly hassle-free solution. It significantly enhances model robustness against shifted domains and maintains resilience in diverse real-world scenarios with various batch sizes, achieving state-of-the-art performance on several major benchmarks.

**Paper link**: [Arxiv](https://arxiv.org/abs/2312.09486) (Extended version with supplymentary materials)

### Dataset

* [CIFAR-10-C](https://zenodo.org/record/2535967#.ZBiI7NDMKUk)
* [CIFAR-100-C](https://zenodo.org/records/3555552#.ZBiJA9DMKUk)
* [IMAGENET-C](https://zenodo.org/records/2235448#.Yj2RO_co_mF)

### Pretrained Model
* CIFAR-10-C [WildResNet-28(from RobustBench)]
* CIFAR-100-C [ResNeXt-29(from RobustBench)]
* IMAGENET-C [ResNet-50(from torchvision)]

### Run Experiments
``python test_time.py --cfg cfgs/[cifar10_c/cifar100_c/imagenet_c]/norm_TTBN.yaml``

Please note that you need to specify the root folder for all datasets `_C.DATA_DIR = "./data" in the file conf.py` and  for all checkpoints `_C.CKPT_DIR = "./ckpt"` in the file `conf.py`. You also need to change the batch size and the corresponding momentum in ``cfgs/[cifar10_c/cifar100_c/imagenet_c]/norm_TTBN.yaml``.

### Detailed Momentum for Different Batch Size

| Dataset| 200 | 64 | 16 | 4 | 2 | 1 |
|-----|-----|-----|-----|-----|-----|-----|
|CIFAR-10-C|1.0|1.0|0.1|0.1|0.01|0.01|
|CIFAR-100-C|1.0|1.0| 0.1|0.1|0.01|0.01|
|IMAGENET-C| 1.0|0.1| 0.1|0.01|0.01|0.01 |

### Acknowledgement
Our codes are built upon [Online Test-Time Adaptation](https://github.com/mariodoebler/test-time-adaptation/tree/main). Thanks for their contribution to the community and the development of research!

You may follow this link to explore more settings and the compared methods.
