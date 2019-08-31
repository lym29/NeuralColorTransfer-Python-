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


class LocalColorTransfer:
    def __init__(self, s, g, L, featS_norm, featG_norm, kmeans_labels, device, kmeans_ratio=1, patch_size=3):
        self.source = torch.from_numpy(s).float().to(device)
        self.guide = torch.from_numpy(g).float().to(device)
        self.L = L
        self.featS_norm = torch.from_numpy(featS_norm).float().to(device)
        self.featG_norm = torch.from_numpy(featG_norm).float().to(device)
        self.height = s.shape[0]
        self.width = s.shape[1]
        self.channel = s.shape[2]
        self.patch_size = patch_size
        self.device = device

        self.paramA = torch.zeros(s.shape).to(device)
        self.paramB = torch.zeros(s.shape).to(device)

        self.sub = torch.ones(*s.shape[:2], 1).to(device)

        self.kmeans_labels = np.zeros(s.shape[:2]).astype(np.int32)
        self.kmeans_ratio = kmeans_ratio

        self.init_params(kmeans_labels)

    def init_params(self, kmeans_labels):
        """
            Initialize a and b from source and guidance using mean and std
        """
        eps = 0.002
        for y in range(self.height):
            for x in range(self.width):
                dy0 = dx0 = self.patch_size // 2
                dy1 = dx1 = self.patch_size // 2 + 1
                dy0 = min(y, dy0)
                dy1 = min(self.height - y, dy1)
                dx0 = min(x, dx0)
                dx1 = min(self.width - x, dx1)

                patchS = self.source[y - dy0:y + dy1, x - dx0:x + dx1].reshape(-1, self.channel)
                patchG = self.guide[y - dy0:y + dy1, x - dx0:x + dx1].reshape(-1, self.channel)
                self.paramA[y, x] = patchG.std(0) / (patchS.std(0) + eps)
                self.paramB[y, x] = patchG.mean(0) - self.paramA[y, x] * patchS.mean(0)
                self.sub[y, x, 0] += self.patch_size ** 2 - (dy0 + dy1) * (dx0 + dx1)

                y_adj = min(y // self.kmeans_ratio, kmeans_labels.shape[0] - 1)
                x_adj = min(x // self.kmeans_ratio, kmeans_labels.shape[1] - 1)
                self.kmeans_labels[y, x] = kmeans_labels[y_adj, x_adj]
        self.paramA.requires_grad_()
        self.paramB.requires_grad_()

    def visualize(self):
        transfered = self.paramA * self.source + self.paramB
        # imshow(transfered.data.cpu().numpy().astype(np.float64))
        # imshow(color.lab2rgb(transfered.data.cpu().numpy().astype(np.float64)))

    def loss_d(self):
        matching_error = torch.pow(self.featS_norm - self.featG_norm, 2).sum(2)
        confidence = 1 - matching_error
        confidence[confidence < 1e-6] = 1e-6
        transfered = self.paramA * self.source + self.paramB
        loss_d = pow(4, self.L-1) * confidence * torch.pow(transfered - self.guide, 2).sum(2)

        return loss_d.mean()

    def loss_l(self):
        wls_filter = WLSFilter(self.source, self.device)
        return wls_filter.loss(self.paramA, self.paramB)

    def loss_nl(self):
        patchS_stack = list()
        patchA_stack = list()
        patchB_stack = list()
        mixedS = list()
        mixedA = list()
        mixedB = list()

        index_map = np.zeros((2, self.height, self.width)).astype(np.int32)
        index_map[0] = np.arange(self.height)[:, np.newaxis] + np.zeros(self.width).astype(np.int32)
        index_map[1] = np.zeros(self.height).astype(np.int32)[:, np.newaxis] + np.arange(self.width)

        for i in range(10):
            index_map_cluster = index_map[:, self.kmeans_labels == i]
            source_cluster = self.source[index_map_cluster[0], index_map_cluster[1]]
            paramA_cluster = self.paramA[index_map_cluster[0], index_map_cluster[1]]
            paramB_cluster = self.paramB[index_map_cluster[0], index_map_cluster[1]]

            nbrs = NearestNeighbors(n_neighbors=9, n_jobs=1).fit(source_cluster)
            indices = nbrs.kneighbors(source_cluster, return_distance=False)

            patchS_stack.append(source_cluster[indices[:, 1:].reshape(-1)].reshape(-1, 8, self.channel))
            patchA_stack.append(paramA_cluster[indices[:, 1:].reshape(-1)].reshape(-1, 8, self.channel))
            patchB_stack.append(paramB_cluster[indices[:, 1:].reshape(-1)].reshape(-1, 8, self.channel))
            mixedS.append(source_cluster.unsqueeze(1))
            mixedA.append(paramA_cluster.unsqueeze(1))
            mixedB.append(paramB_cluster.unsqueeze(1))

        patchS_stack = torch.cat(patchS_stack)
        patchA_stack = torch.cat(patchA_stack)
        patchB_stack = torch.cat(patchB_stack)
        mixedS = torch.cat(mixedS)
        mixedA = torch.cat(mixedA)
        mixedB = torch.cat(mixedB)

        mixedT = mixedA * mixedS + mixedB
        patchT_stack = patchA_stack * patchS_stack + patchB_stack
        patchSD = torch.norm(mixedS - patchS_stack, 2, 2).exp()
        wgt = patchSD / patchSD.sum(1, keepdim=True)
        term1 = torch.pow(mixedT - patchT_stack, 2).sum(2)
        loss_nl = torch.sum(wgt * term1, 1).mean()

        return loss_nl

    def train(self, total_iter=1000):
        optimizer = optim.Adam([self.paramA, self.paramB], lr=0.1, weight_decay=0)
        hyper_l = 0.125
        hyper_nl = 2.0
        for iter in range(total_iter):
            optimizer.zero_grad()

            loss_d = self.loss_d()
            loss_l = self.loss_l()
            loss_nl = 0 #self.loss_nl()
            loss = loss_d + hyper_l * loss_l + hyper_nl * loss_nl

            #print(loss_d)
            #print("Loss_d: {0:.4f}, Loss_l: {1:.4f}, loss_nl: {2:.4f}".format(loss_d.data, loss_l.data, loss_nl.data))
            if (iter + 1) % 10 == 0:
                print("Iteration:", str(iter + 1) + "/" + str(total_iter), "Loss: {0:.4f}".format(loss.data))
                print("Loss_d: {0:.4f}, Loss_l: {1:.4f}".format(loss_d.data, loss_l.data))
            loss.backward()
            optimizer.step()
