# adversarial_attack_tutorial
A simple tutorial on adversarial attacks against deep neural networks（针对深度神经网络的对抗攻击的简单教程）

### 1. Introduction（介绍）

This is a basic tutorial on adversarial attacks. Through this tutorial, you can learn: 1) the implementation of gradient-based and generation-based adversarial attacks; 2) adversarial attacks on classification models and generative models; 3) the robustness and transferability of adversarial attacks.

这是一个关于对抗攻击的基础教程。通过该教程你可以了解：1）基于梯度和基于生成的对抗攻击的实现；2）针对分类模型和生成模型的对抗攻击；3）对抗攻击的鲁棒性和可迁移性。

### 2. Start（使用）

#### 2.0 Preparing（准备）

Creating a conda environment：
```
conda create -n adversarial_attack_tutorial python=3.6
conda activate adversarial_attack_tutorial
```

Install the required dependency packages given in requirements.txt.

#### 2.1 Download the pre-trained model and datasets（下载预训练模型和训练数据）

You can get the pre-trained model from the following link, unzip it and place it in `adversarial_attack_tutorial/checkpoints/`：

你可以从以下链接获取预训练模型，将其解压后安放在`adversarial_attack_tutorial/checkpoints/`目录下:

[Google Drive](https://drive.google.com/file/d/1nyzCfxoG8I-zJe-2odJohdDwWCYgzVFQ/view?usp=sharing).

[Quark Drive](https://pan.quark.cn/s/450579236ae7) 提取码：MCEy

You need to download the [cifar-10](http://www.cs.toronto.edu/~kriz/cifar.html) and [celebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) datasets


#### 2.2 Running demo（运行demo）

Run gradient-based adversarial attacks on classifier models in grad_attack2resnet.ipynb; run gradient-based adversarial attacks on generative models in grad_attack2AE.ipynb; run generative-based adversarial attacks on generative models in gen_attck2AE.ipynb; learn about the robustness and transferability of adversarial attacks in robustness_transferability.ipynb.

在grad_attack2resnet.ipynb中运行基于梯度的针对分类器模型的对抗攻击；在grad_attack2AE.ipynb中运行基于梯度的对生成模型的对抗攻击；在gen_attack2AE.ipynb中运行基于生成式的对生成模型的对抗攻击；在robustness_transferability.ipynb中了解对抗攻击的鲁棒性和可转移性。

#### 2.3 Use it more flexibly（灵活使用）

（1）Gradient-based adversarial attacks on classifiers（基于梯度的对分类器的对抗攻击）：
```python
    from utils.attack import LinfPGDAttack4Classifier

    attack = LinfPGDAttack4Classifier(model=model, epsilon=0.05, k=10, device=device)

    with torch.no_grad():
        x_real = images
        y = model(images.to(device))
    
    adv_images, eta = attack.perturb(images, y)
```

<img src="images\grad_res.png" alt="output" style="zoom:67%;" />

（2）Gradient-based Adversarial Attacks on Generative Models（基于梯度的对生成模型的对抗攻击）：
```python
    from utils.attack import LinfPGDAttack4Gen
    
    attack = LinfPGDAttack4Gen(model=model, epsilon=0.05, k=20, device=device)
    
    with torch.no_grad():
            x_real = images
            y = model(images.to(device))
        
    adv_images, eta = attack.perturb(images, y)
```

<img src="images\grad_ae.png" alt="output" style="zoom:67%;" />

（3）Generation-based Adversarial Attacks on Generative Models（基于生成的对生成模型的对抗攻击）：
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
