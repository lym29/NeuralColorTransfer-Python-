import argparse
import os

import numpy as np
import torch
import torchvision.models
from torchvision import transforms, utils
from guided_filter_pytorch.guided_filter import FastGuidedFilter
from PIL import Image
from skimage import color
from sklearn.cluster import KMeans
import cv2
from lct_filter import LocalColorTransferFilter

from models import LocalColorTransfer
from models import WLSFilter
from torchvision.models.vgg import model_urls
from utils import *


FEATURE_IDS = [1, 6, 11, 20, 29]
LEFT_SHIFT = (1, 2, 0)
RIGHT_SHIFT = (2, 0, 1)
model_urls['vgg19'] = model_urls['vgg19'].replace('https://', 'http://')
maxL = 5


def image_loader(img_path, device):
    pil_img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    tensor_img = transform(pil_img).unsqueeze(0).to(device)
    opencv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)/255
    opencv_img = cv2.cvtColor(opencv_img.astype(np.float32), cv2.COLOR_BGR2LAB)

    return tensor_img, opencv_img


def resize_img(img, size):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    img_tensor = transform(img)

    return img_tensor


def get_feature(vgg19, img_tensor, feature_id):
    feature_tensor = vgg19.features[:feature_id](img_tensor)
    feature = feature_tensor.data.squeeze().cpu().numpy().transpose(LEFT_SHIFT)

    return feature


def normalize(feature):
    return feature / np.linalg.norm(feature, ord=2, axis=2, keepdims=True)


def main(config):
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)
    if not os.path.exists(config.processing_dir):
        os.makedirs(config.processing_dir)

    device = torch.device(('cuda:' + str(config.gpu)) if config.cuda else 'cpu')

    vgg19 = torchvision.models.vgg19(pretrained=True)
    vgg19.to(device)

    S_tensor, S_opencv = image_loader(config.source, device)
    R_tensor, R_opencv = image_loader(config.reference, device)
    imgS = S_opencv

    orig_S_height, orig_S_width = S_tensor.shape[2:4]

    #currentS_shape = origS_shape = np.asarray(imgS.shape, dtype=int)[2:4]


    for L in range(maxL, 0, -1):
        current_feature_S = get_feature(vgg19, S_tensor.float(), FEATURE_IDS[L-1])
        norm_feature_S = normalize(current_feature_S)
        current_feature_R = get_feature(vgg19, R_tensor.float(), FEATURE_IDS[L-1])
        norm_feature_R = normalize(current_feature_R)
        if L == maxL:
            kmeans = KMeans(n_clusters=10, n_jobs=1).fit(current_feature_S.reshape(-1, current_feature_S.shape[2]))
            kmeans_labels = kmeans.labels_.reshape(current_feature_S.shape[:2])

        mapSR = PatchMatch(norm_feature_S, norm_feature_R)
        mapRS = PatchMatch(norm_feature_R, norm_feature_S)
        mapSR.solve()
        mapRS.solve()

        S_height, S_width = current_feature_S.shape[:2]
        R_height, R_width = current_feature_R.shape[:2]
        imgS_resized = cv2.resize(imgS, (S_width, S_height), interpolation=cv2.INTER_LINEAR)
        imgR_resized = cv2.resize(R_opencv, (R_width, R_height), interpolation=cv2.INTER_LINEAR)

        imgG = bds_vote(np.asarray(imgR_resized), mapSR.nnf, mapRS.nnf)
        imgG_show = cv2.cvtColor(imgG.astype(np.float32), cv2.COLOR_LAB2BGR)*255
        cv2.imwrite(config.result_dir + 'img' + str(L) + 'G.png', imgG_show)

        feature_G = bds_vote(current_feature_R, mapSR.nnf, mapRS.nnf)

        lct = LocalColorTransferFilter(imgS_resized, imgG, norm_feature_S, normalize(feature_G))
        A, b = lct.solve(L)
        transfered_S = A * imgS_resized + b
        cv2.imwrite(config.result_dir + 'img' + str(L) + 'transS.png', cv2.cvtColor(transfered_S.astype(np.float32), cv2.COLOR_LAB2BGR)*255)

        A = cv2.resize(A, (orig_S_width, orig_S_height), interpolation=cv2.INTER_LINEAR)
        b = cv2.resize(b, (orig_S_width, orig_S_height), interpolation=cv2.INTER_LINEAR)
        imgS = A * imgS + b

        # FastGuidedFilter
        cv2.imwrite(config.result_dir + 'img' + str(L) + 'S.png', cv2.cvtColor(imgS.astype(np.float32), cv2.COLOR_LAB2BGR)*255)
        S_tensor = cv2.cvtColor(imgS.astype(np.float32), cv2.COLOR_LAB2RGB) * 255
        S_tensor = torch.from_numpy(S_tensor.transpose(RIGHT_SHIFT))
        S_tensor.unsqueeze_(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Neural Color Transfer between Images PyTorch")

    parser.add_argument('--source', type=str, default='./image/3_Source1.jpg', help="Source Image that has Content")
    parser.add_argument('--reference', type=str, default='./image/3_Reference.jpg', help="Reference Image to Get Style")
    parser.add_argument('--result_dir', type=str, default='./results/')
    parser.add_argument('--processing_dir', type=str, default='./processImage')
    parser.add_argument('--cuda', dest='feature', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    parser.set_defaults(cuda=False)
    # need more arguments?

    args = parser.parse_args()
    print(args)
    main(args)
