import copy
import numpy as np
from collections import Iterable
from scipy.stats import truncnorm

import torch
import torch.nn as nn
import piq

class TVLoss():
    def __init__(self):
        self.loss = piq.TVLoss()

    def __call__(self, pred, ref):
        pred = (pred+1)/2
        pred = pred.clamp(0, 1)
        return self.loss(pred)

class NegTVLoss():
    def __init__(self):
        self.loss = piq.TVLoss()

    def __call__(self, pred, ref):
        pred = (pred+1)/2
        pred = pred.clamp(0, 1)
        return -self.loss(pred)

class BRISQUELoss():
    def __init__(self):
        self.loss = piq.BRISQUELoss()

    def __call__(self, pred, ref):
        pred = (pred+1)/2
        pred = pred.clamp(0, 1)
        return self.loss(pred)

class NEGBRISQUELoss():
    def __init__(self):
        self.loss = piq.BRISQUELoss()

    def __call__(self, pred, ref):
        pred = (pred+1)/2
        pred = pred.clamp(0, 1)
        return -self.loss(pred)

class DSSIMLoss():
    def __init__(self):
        self.loss = piq.SSIMLoss()

    def __call__(self, pred, ref):
        pred = (pred+1)/2
        pred = pred.clamp(0, 1)
        ref = (ref+1)/2
        ref = ref.clamp(0, 1)
        return -(1-self.loss(pred, ref))/2

class LinfPGDAttack(object):
    def __init__(self, model=None, device=None, epsilon=0.05, k=10, a=0.01, attack_loss='mse'):
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

        if attack_loss == 'mse':
            self.loss_fn = nn.MSELoss().to(device)
        elif attack_loss == 'tv':
            self.loss_fn = TVLoss()
        elif attack_loss == 'negtv':
            self.loss_fn = NegTVLoss()
        elif attack_loss == 'brisque':
            self.loss_fn = BRISQUELoss()
        elif attack_loss == 'negbrisque':
            self.loss_fn = NEGBRISQUELoss()
        elif attack_loss == 'dssim':
            self.loss_fn = DSSIMLoss()

        self.device = device

        # PGD or I-FGSM?
        self.rand = True

    def perturb(self, X_nat, y, c_trg):
        """
        Vanilla Attack.
        """
        if self.rand:
            X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, X_nat.shape).astype('float32')).to(self.device)
        else:
            X = X_nat.clone().detach_()
            # use the following if FGSM or I-FGSM and random seeds are fixed
            # X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-0.001, 0.001, X_nat.shape).astype('float32')).cuda()    

        for i in range(self.k):
            X.requires_grad = True
            output_att, output_img = self.model(X, c_trg)

            out = imFromAttReg(output_att, output_img, X)

            self.model.zero_grad()

            # Attention attack
            # loss = self.loss_fn(output_att, y)

            # Output attack
            # Minus in the loss means "towards" and plus means "away from"
            loss = self.loss_fn(out, y)
            loss.backward()
            grad = X.grad

            X_adv = X + self.a * grad.sign()

            eta = torch.clamp(X_adv - X_nat, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(X_nat + eta, min=-1, max=1).detach_()

            # Debug
            # X_adv, loss, grad, output_att, output_img = None, None, None, None, None

        return X, eta

    def perturb_iter_class(self, X_nat, y, c_trg):
        """
        Iterative Class Conditional Attack
        """
        X = X_nat.clone().detach_()

        j = 0
        J = c_trg.size(0)

        for i in range(self.k):
            X.requires_grad = True
            output_att, output_img = self.model(X, c_trg[j,:].unsqueeze(0))

            out = imFromAttReg(output_att, output_img, X)

            self.model.zero_grad()

            # loss = self.loss_fn(output_att, y)
            loss = self.loss_fn(out, y)
            loss.backward()
            grad = X.grad

            X_adv = X + self.a * grad.sign()

            eta = torch.clamp(X_adv - X_nat, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(X_nat + eta, min=-1, max=1).detach_()

            j += 1
            if j == J:
                j = 0

        return X, eta

    def perturb_joint_class(self, X_nat, y, c_trg):
        """
        Joint Class Conditional Attack
        """
        X = X_nat.clone().detach_()

        J = c_trg.size(0)
        
        for i in range(self.k):
            full_loss = 0.0
            X.requires_grad = True
            self.model.zero_grad()

            for j in range(J):
                output_att, output_img = self.model(X, c_trg[j,:].unsqueeze(0))

                out = imFromAttReg(output_att, output_img, X)

                # loss = self.loss_fn(output_att, y)
                loss = self.loss_fn(out, y)
                full_loss += loss

            full_loss.backward()
            grad = X.grad

            X_adv = X + self.a * grad.sign()

            eta = torch.clamp(X_adv - X_nat, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(X_nat + eta, min=-1, max=1).detach_()

        return X, eta

def clip_tensor(X, Y, Z):
    # Clip X with Y min and Z max
    X_np = X.data.cpu().numpy()
    Y_np = Y.data.cpu().numpy()
    Z_np = Z.data.cpu().numpy()
    X_clipped = np.clip(X_np, Y_np, Z_np)
    X_res = torch.FloatTensor(X_clipped)
    return X_res

def imFromAttReg(att, reg, x_real):
    """Mixes attention, color and real images"""
    return (1-att)*reg + att*x_real