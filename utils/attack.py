import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class LinfPGDAttack4Classifier(object):
    def __init__(self, model=None, device=None, epsilon=0.05, k=10, a=0.01):
        """
        Linf PGD attack for classifier models (e.g., ResNet)
        epsilon: magnitude of attack (maximum perturbation)
        k: number of iterations
        a: step size
        """
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.loss_fn = nn.CrossEntropyLoss().to(device)  # For classification
        self.device = device

        # Random start for PGD
        self.rand = True

    def perturb(self, X_nat, y):
        """
        X_nat: Original input data (e.g., images)
        y: True labels for classification
        """
        if self.rand:
            X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, X_nat.shape).astype('float32')).to(self.device)
        else:
            X = X_nat.clone().detach_()

        # Perform PGD attack
        for i in range(self.k):
            X.requires_grad = True
            output = self.model(X)
            self.model.zero_grad()

            # Calculate loss (Cross-Entropy Loss for classification)
            loss = self.loss_fn(output, y)
            loss.backward()

            # Calculate the gradient of loss w.r.t. input
            grad = X.grad

            # Generate adversarial example by updating X in the direction of the gradient
            X_adv = X + self.a * grad.sign()

            # Clip the adversarial example to ensure it stays within the epsilon ball of X_nat
            eta = torch.clamp(X_adv - X_nat, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(X_nat + eta, min=-1, max=1).detach_()

        self.model.zero_grad()
        return X, eta




class LinfPGDAttack4Gen(object):
    def __init__(self, model=None, device=None, epsilon=0.05, k=10, a=0.01):
        """
        FGSM, I-FGSM and PGD attacks
        epsilon: magnitude of attack
        k: iterations
        a: step size
        """
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.loss_fn = nn.MSELoss().to(device)
        self.device = device

        # PGD or I-FGSM?
        self.rand = True

    def perturb(self, X_nat, y):
        """
        X_net is the output of network.
        y: True output for generator
        """
        if self.rand:
            X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, X_nat.shape).astype('float32')).to(self.device)
        else:
            X = X_nat.clone().detach_()
            
        grad = 0.0
        for i in range(self.k):
            X.requires_grad = True
            output = self.model(X)
            self.model.zero_grad()

            # 损失函数，修改这里实现不同破坏目标
            # Minus in the loss means "towards" and plus means "away from"
            loss = self.loss_fn(output, y)
            loss.backward()

            # Calculate the gradient of loss w.r.t. input
            grad = X.grad

            # Generate adversarial example by updating X in the direction of the gradient
            X_adv = X + self.a * grad.sign()

            # Clip the adversarial example to ensure it stays within the epsilon ball of X_nat
            eta = torch.clamp(X_adv - X_nat, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(X_nat + eta, min=-1, max=1).detach_()

        self.model.zero_grad()
        return X, eta