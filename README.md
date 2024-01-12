# RealisticTTA

### Unraveling Batch Normalization for Realistic Test-Time Adaptation <AAAI 2024> 
Zixian Su, Jingwei Guo, Kai Yao, Xi Yang, Qiufeng Wang,  Kaizhu Huang

**Abstract**

While recent test-time adaptations exhibit efficacy by adjusting batch normalization to narrow domain disparities, their effectiveness diminishes with realistic mini-batches due to inaccurate target estimation. As previous attempts merely introduce source statistics to mitigate this issue, the fundamental problem of inaccurate target estimation still persists, leaving the intrinsic test-time domain shifts unresolved. This paper delves into the problem of mini-batch degradation. By unraveling batch normalization, we discover that the inexact target statistics largely stem from the substantially reduced class diversity in batch. Drawing upon this insight, we introduce a straightforward tool, Test-time Exponential Moving Average (TEMA), to bridge the class diversity gap between training and testing batches. Importantly, our TEMA adaptively extends the scope of typical methods beyond the current batch to incorporate a diverse set of class information, which in turn boosts an accurate target estimation. Built upon this foundation, we further design a novel layer-wise rectification strategy to consistently promote test-time performance.
Our proposed method enjoys a unique advantage as it requires neither training nor tuning parameters, offering a truly hassle-free solution. It significantly enhances model robustness against shifted domains and maintains resilience in diverse real-world scenarios with various batch sizes, achieving state-of-the-art performance on several major benchmarks.

**Paper link**: [Arxiv](https://arxiv.org/abs/2312.09486) (Extended version with supplymentary materials)

**Dataset**

* [CIFAR-10-C](https://zenodo.org/record/2535967#.ZBiI7NDMKUk)
* [CIFAR-100-C](https://zenodo.org/records/3555552#.ZBiJA9DMKUk)
* [IMAGENET-C](https://zenodo.org/records/2235448#.Yj2RO_co_mF)

