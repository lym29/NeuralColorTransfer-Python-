import numpy as np
import torch
from torch import optim
from sklearn.neighbors import NearestNeighbors

class WLSFilter:
    def __init__(self, S, device, alpha=1.2, epsilon = 0.0001):
        self.device = device
        self.alpha = alpha
        self.epsilon = epsilon
        self.paramA = torch.zeros(S.shape).to(device)
        self.paramB = torch.zeros(S.shape).to(device)

        self.S = S

    def rgb_to_l(self, img_tensor):
        r = img_tensor[:, :, 0]
        g = img_tensor[:, :, 1]
        b = img_tensor[:, :, 2]
        y = (r + r + g + g + g + b) / 6
        y = torch.unsqueeze(y, 2)
        return y

    def make_four_direction_copies(self, S):
        up_S = down_S = left_S = right_S = torch.zeros(S.shape)

        up_S[-1, :, :] = S[-1, :, :]
        down_S[0, :, :] = S[0, :, :]
        right_S[:, -1, :] = S[:, -1, :]
        left_S[:, 0, :] = S[:, 0, :]

        up_S[:-1, :, :] = S[1:, :, :]
        down_S[1:, :, :] = S[:-1, :, :]
        right_S[:, :-1, :] = S[:, 1:, :]
        left_S[:, 1:, :] = S[:, :-1, :]
        return up_S, down_S, right_S, left_S

    def compute_WLS_loss_for_each_copy(self, lumi, A, b, dir_A, dir_b, dir_l):
        normA = A.norm(2, keepdim=True)
        normb = b.norm(2, keepdim=True)
        dir_loss = pow((A - dir_A)/normA, 2).sum(2) + pow((b - dir_b)/normb, 2).sum(2)
        dir_weight = pow((lumi - dir_l).abs(), self.alpha).sum(2) + self.epsilon
        dir_loss = dir_loss / dir_weight
        return dir_loss

    def loss(self, A, b):
        lumi = self.rgb_to_l(self.S)
        up_l, down_l, right_l, left_l = self.make_four_direction_copies(lumi)
        up_A, down_A, right_A, left_A = self.make_four_direction_copies(A)
        up_b, down_b, right_b, left_b = self.make_four_direction_copies(b)
        up_loss = self.compute_WLS_loss_for_each_copy(lumi, A, b, up_A, up_b, up_l)
        down_loss = self.compute_WLS_loss_for_each_copy(lumi, A, b, down_A, down_b, down_l)
        right_loss = self.compute_WLS_loss_for_each_copy(lumi, A, b, right_A, right_b, right_l)
        left_loss = self.compute_WLS_loss_for_each_copy(lumi, A, b, left_A, left_b, left_l)
        return torch.mean(up_loss + down_loss + right_loss + left_loss)

    def smooth_loss(self):
        return self.loss(self.paramA, self.paramB)

    def upsample_constraint(self, A, b):
        loss = pow(self.paramA - A, 2).sum(2) + pow(self.paramB - b, 2).sum(2)
        return loss.sum(1).sum(0)

    def upsample_train(self, A, b):
        A = torch.from_numpy(A).float()
        b = torch.from_numpy(b).float()
        self.paramA = A
        self.paramB = b
        self.paramA.requires_grad_()
        self.paramB.requires_grad_()

        optimizer = optim.Adam([self.paramA, self.paramB], lr=0.99, weight_decay=0)

        total_iter = 50
        for iter in range(total_iter):
            optimizer.zero_grad()
            smooth_loss = self.smooth_loss( )
            constraint = self.upsample_constraint(A, b)
            loss_upsample = 0.24 * smooth_loss + constraint
            loss_upsample.backward()
            optimizer.step()
            if (iter + 1) % 10 == 0:
                print("Iteration:", str(iter + 1) + "/" + str(total_iter), "Loss: {0:.4f}".format(loss_upsample.data))
