# adversarial_attack_tutorial
针对深度神经网络的对抗攻击的简单教程。

[English Version](./README.md)

### 1. 介绍

这是一个关于对抗攻击的基础教程。通过该教程你可以了解：1）基于梯度和基于生成的对抗攻击的实现；2）针对分类模型和生成模型的对抗攻击；3）对抗攻击的鲁棒性和可迁移性。

### 2. 使用

#### 2.0 准备

创建`conda`环境：
```
conda create -n adversarial_attack_tutorial python=3.6
conda activate adversarial_attack_tutorial
```

安装requirements.txt文件中所提供的依赖项。

#### 2.1 下载预训练模型和训练数据

你可以从以下链接获取预训练模型，将其解压后安放在`adversarial_attack_tutorial/checkpoints/`目录下。

[Google Drive](https://drive.google.com/file/d/1nyzCfxoG8I-zJe-2odJohdDwWCYgzVFQ/view?usp=sharing).

[Quark Drive](https://pan.quark.cn/s/450579236ae7) 提取码：MCEy

你需要下载 [cifar-10](http://www.cs.toronto.edu/~kriz/cifar.html) 和 [celebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) 数据集。此外，我们提供了celebA-256-mini数据集，其包含30000张从celebA数据集中采样的人脸图像及其属性信息，可以从[celebA-256-mini](https://drive.google.com/file/d/1v4KazZb9DFr_DpOFCwL-qGZc0AdYhBXq/view?usp=sharing)中获取。本教程所使用的为celebA-256-mini数据集，为了方便运行，建议使用该数据集。


#### 2.2 运行demo

在grad_attack2resnet.ipynb中运行基于梯度的针对分类器模型的对抗攻击；在grad_attack2AE.ipynb中运行基于梯度的对生成模型的对抗攻击；在gen_attack2AE.ipynb中运行基于生成式的对生成模型的对抗攻击；在robustness_transferability.ipynb中了解对抗攻击的鲁棒性和可转移性。

#### 2.3 灵活使用

（1）基于梯度的对分类器的对抗攻击：
```python
    from utils.attack import LinfPGDAttack4Classifier

    attack = LinfPGDAttack4Classifier(model=model, epsilon=0.05, k=10, device=device)

    with torch.no_grad():
        x_real = images
        y = model(images.to(device))
    
    adv_images, eta = attack.perturb(images, y)
```

<img src="images\grad_res.png" alt="output" style="zoom:67%;" />

（2）基于梯度的对生成模型的对抗攻击：
```python
    from utils.attack import LinfPGDAttack4Gen
    
    attack = LinfPGDAttack4Gen(model=model, epsilon=0.05, k=20, device=device)
    
    with torch.no_grad():
            x_real = images
            y = model(images.to(device))
        
    adv_images, eta = attack.perturb(images, y)
```

<img src="images\grad_ae.png" alt="output" style="zoom:67%;" />

（3）基于生成的对生成模型的对抗攻击：
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
