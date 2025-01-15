import torch
import numpy as np
import random
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

from tqdm import tqdm
import torch.nn.functional as F

class Gen_Attack:
    def __init__(self, device, epsilon=0.05, target_model=None, advGen=None):

        self.device = device
        self.epsilon = epsilon
        self.lr = 0.0001
        self.beta1 = 0.5
        self.beta2 = 0.999

        self.savelist = []
        self.count = 0

        self.advG = advGen
        self.optimizer_G = torch.optim.Adam(self.advG.parameters(), self.lr, [self.beta1, self.beta2])

        self.target_model = target_model


    def train_batch(self, x_real):
        self.advG.train()
        perturbation = self.advG(x_real)
        perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
        x_adv = torch.clamp(x_real + perturbation, -1.0, 1.0)

        output_real = self.target_model(x_real)

        output_adv = self.target_model(x_adv)

        loss_G = -F.mse_loss(output_adv, output_real) 
 
        self.optimizer_G.zero_grad()
        loss_G.backward()
        self.optimizer_G.step()

        return -loss_G.item()



    def train(self, train_dataloader, epochs = 2):
        for epoch in range(1, epochs+1): 
            loss_G = 0.
            with tqdm(total=len(train_dataloader), desc=f'Epoch {epoch}/{epochs}', unit='batch') as pbar:
                for i,(img, _) in enumerate(train_dataloader):
                    img = img.to(self.device)
                    
                    loss_g = self.train_batch(img)
                    with torch.no_grad(): 
                        loss_G += loss_g
                        
                    pbar.set_postfix(loss_G = loss_G / (i+1))
                    pbar.update()

            if epoch % 1 == 0:
                torch.save(self.netG.state_dict(), "checkpoints/adv_gen.pth")

                