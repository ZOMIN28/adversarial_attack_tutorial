# adversarial_attack_tutorial
A simple tutorial on adversarial attacks against deep neural networks.

[中文版本文档](./README_CN.md)

### 1. Introduction

This is a basic tutorial on adversarial attacks. Through this tutorial, you can learn: 1) the implementation of gradient-based and generation-based adversarial attacks; 2) adversarial attacks on classification models and generative models; 3) the robustness and transferability of adversarial attacks.

### 2. Start

#### 2.0 Preparing

Creating a conda environment：
```
conda create -n adversarial_attack_tutorial python=3.6
conda activate adversarial_attack_tutorial
```

Install the required dependency packages given in requirements.txt.

#### 2.1 Download the pre-trained model and datasets

You can get the pre-trained model from the following link, unzip it and place it at `adversarial_attack_tutorial/checkpoints/`：

[Google Drive](https://drive.google.com/file/d/1nyzCfxoG8I-zJe-2odJohdDwWCYgzVFQ/view?usp=sharing).

[Quark Drive](https://pan.quark.cn/s/450579236ae7) the key：MCEy

You need to download the [cifar-10](http://www.cs.toronto.edu/~kriz/cifar.html) and [celebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) datasets. In addition, we provide a celebA-256-mini dataset, which contains 30,000 face images sampled from the celebA dataset and their attribute information, which you can get at [celebA-256-mini](https://drive.google.com/file/d/1v4KazZb9DFr_DpOFCwL-qGZc0AdYhBXq/view?usp=sharing). This tutorial uses the celebA-256-mini dataset. For ease of operation, it is recommended to use this dataset.


#### 2.2 Running demo

Run gradient-based adversarial attacks on classifier models in grad_attack2resnet.ipynb; run gradient-based adversarial attacks on generative models in grad_attack2AE.ipynb; run generative-based adversarial attacks on generative models in gen_attck2AE.ipynb; learn about the robustness and transferability of adversarial attacks in robustness_transferability.ipynb.

#### 2.3 Use it more flexibly

（1）Gradient-based adversarial attacks on classifiers：
```python
    from utils.attack import LinfPGDAttack4Classifier

    attack = LinfPGDAttack4Classifier(model=model, epsilon=0.05, k=10, device=device)

    with torch.no_grad():
        x_real = images
        y = model(images.to(device))
    
    adv_images, eta = attack.perturb(images, y)
```

<img src="images\grad_res.png" alt="output" style="zoom:67%;" />

（2）Gradient-based Adversarial Attacks on Generative Models：
```python
    from utils.attack import LinfPGDAttack4Gen
    
    attack = LinfPGDAttack4Gen(model=model, epsilon=0.05, k=20, device=device)
    
    with torch.no_grad():
            x_real = images
            y = model(images.to(device))
        
    adv_images, eta = attack.perturb(images, y)
```

<img src="images\grad_ae.png" alt="output" style="zoom:67%;" />

（3）Generation-based Adversarial Attacks on Generative Models：
```python
    from net.advGenerator import ResnetGenerator
    
    advG = ResnetGenerator(input_nc=3).to(device)
    advG.load_state_dict(torch.load('checkpoints/adv_gen.pth'))
    advG.eval()
    
    perturbation = advG(images)
    perturbation = torch.clamp(perturbation, -epsilon, epsilon)
    adv_images = torch.clamp(images + perturbation, -1.0, 1.0)
```

<img src="images\gen_ae.png" alt="output" style="zoom:67%;" />
