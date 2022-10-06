# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
from timm.data import Mixup
import torch.nn.functional as F
import time

def rand_bbox(img_shape, lam, margin=0., count=None):
    """ Standard CutMix bounding-box
    Generates a random square bbox based on lambda value. This impl includes
    support for enforcing a border margin as percent of bbox dimensions.

    Args:
        img_shape (tuple): Image shape as tuple
        lam (float): Cutmix lambda value
        margin (float): Percentage of bbox dimension to enforce as margin (reduce amount of box outside image)
        count (int): Number of bbox to generate
    """
    ratio = np.sqrt(1 - lam)
    img_h, img_w = img_shape[-2:]
    cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
    margin_y, margin_x = int(margin * cut_h), int(margin * cut_w)
    cy = np.random.randint(0 + margin_y, img_h - margin_y, size=count)
    cx = np.random.randint(0 + margin_x, img_w - margin_x, size=count)
    yl = np.clip(cy - cut_h // 2, 0, img_h)
    yh = np.clip(cy + cut_h // 2, 0, img_h)
    xl = np.clip(cx - cut_w // 2, 0, img_w)
    xh = np.clip(cx + cut_w // 2, 0, img_w)
    return yl, yh, xl, xh

def datamixing(x, lam):
    if lam is None:
        lam = np.random.beta(1., 1., size=[x.shape[0],1,1,1]).astype(np.float32)
        lam = torch.from_numpy(lam).cuda()
    # if np.random.randn()< 0.5:
        # use mixup
    new_lams = lam * 0.8 + 0.1
    x_flip = x.flip(0).clone()
    for i in range(x.shape[0]):
        x[i:i+1,:,:,:],x_flip[i:i+1,:,:,:] = colorful_spectrum_mix(x[i:i+1,:,:,:],x_flip[i:i+1,:,:,:],1,1)

    #for i in range(x.shape[0]):
    #    yl, yh, xl, xh = rand_bbox(x.shape, new_lams[i].cpu())
    #    bbox_area = (yh - yl) * (xh - xl)
    #    new_lams[i] = torch.tensor(1. - bbox_area / float(x.shape[-2] * x.shape[-1])).cuda()
    #    x[i:i+1, :, yl:yh, xl:xh] = F.interpolate(x_flip[i:i+1], (yh - yl, xh - xl), mode="bilinear")
        
    # else:
    #     #use cutmix
    #     yl, yh, xl, xh = rand_bbox(x.shape, lam)
    #     bbox_area = (yh - yl) * (xh - xl)
    #     new_lam = 1. - bbox_area / float(x.shape[-2] * x.shape[-1])
    #     x[:, :, yl:yh, xl:xh] = x.flip(0)[:, :, yl:yh, xl:xh] 
    #get low and high frequency


    # else:
    #     #use frequency mixup


    # else:
    #     #use cut frequency mix

    # else:
    #     #use rezie frequency mix
    return x, lam, new_lams


def colorful_spectrum_mix(img1, img2, alpha, ratio=1.0):
    """Input image size: ndarray of [C, H, W]"""
    lam = np.random.uniform(0, alpha)
    assert img1.shape == img2.shape
    b, c, h, w  = img1.shape
    ratio = torch.tensor(ratio, dtype=torch.int8)
    h_crop = int(h * torch.sqrt(ratio))
    w_crop = int(w * torch.sqrt(ratio))
    h_start = h // 2 - h_crop // 2
    w_start = w // 2 - w_crop // 2

    img1_fft = torch.fft.fft2(img1, dim=(2,3))
    img2_fft = torch.fft.fft2(img2, dim=(2,3))
    img1_abs, img1_pha = torch.abs(img1_fft), torch.angle(img1_fft)
    img2_abs, img2_pha = torch.abs(img2_fft), torch.angle(img2_fft)

    img1_abs = torch.fft.fftshift(img1_abs, dim=(2, 3))
    img2_abs = torch.fft.fftshift(img2_abs, dim=(2, 3))

    img1_abs_ = torch.clone(img1_abs)
    img2_abs_ = torch.clone(img2_abs)
    ## phase1 + lam * amp2 + (1-lam) * amp1
    img1_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img2_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img1_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]
    ## phase2 + lam * amp1 + (1-lam) * amp2
    img2_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img1_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img2_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]
    img1_abs = torch.fft.ifftshift(img1_abs, dim=(2, 3))
    img2_abs = torch.fft.ifftshift(img2_abs, dim=(2, 3))
    img21 = img1_abs * (np.e ** (1j * img1_pha)) 
    img12 = img2_abs * (np.e ** (1j * img2_pha))
    img21 = torch.real(torch.fft.ifft2(img21,dim=(2, 3)))
    img12 = torch.real(torch.fft.ifft2(img12,dim=(2, 3)))
    img21 = torch.clip(img21[0], 0, 255).int()
    img12 = torch.clip(img12[0], 0, 255).int()

    return img21, img12


def colorful_spectrum_mix_cpu(img1, img2, alpha, ratio=1.0):
    """Input image size: ndarray of [C, H, W ]"""
    lam = np.random.uniform(0, alpha)
    assert img1.shape == img2.shape
    c, h, w = img1.shape
    #ratio = torch.tensor(ratio, dtype=torch.int8)
    h_crop = int(h * np.sqrt(ratio))
    w_crop = int(w * np.sqrt(ratio))
    h_start = h // 2 - h_crop // 2
    w_start = w // 2 - w_crop // 2

    img1_fft = np.fft.fft2(img1, axes=(1, 2))
    img2_fft = np.fft.fft2(img2, axes=(1, 2))
    img1_abs, img1_pha = np.abs(img1_fft), np.angle(img1_fft)
    img2_abs, img2_pha = np.abs(img2_fft), np.angle(img2_fft)

    img1_abs = np.fft.fftshift(img1_abs, axes=(1, 2))
    img2_abs = np.fft.fftshift(img2_abs, axes=(1, 2))

    img1_abs_ = np.copy(img1_abs)
    img2_abs_ = np.copy(img2_abs)
    ## phase1 + lam * amp2 + (1-lam) * amp1
    img1_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img2_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img1_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]
    ## phase2 + lam * amp1 + (1-lam) * amp2
    img2_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img1_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img2_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]
    img1_abs = np.fft.ifftshift(img1_abs, axes=(1, 2))
    img2_abs = np.fft.ifftshift(img2_abs, axes=(1, 2))
    img21 = img1_abs * (np.e ** (1j * img1_pha)) 
    img12 = img2_abs * (np.e ** (1j * img2_pha))
    img21 = np.real(np.fft.ifft2(img21,axes=(1, 2)))
    img12 = np.real(np.fft.ifft2(img12,axes=(1, 2)))
    img21 = np.uint8(np.clip(img21, 0, 255))
    img12 = np.uint8(np.clip(img12, 0, 255))


    return torch.from_numpy(img21), torch.from_numpy(img12), torch.tensor(lam)    


    

class MoCo(nn.Module):
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(MoCo, self).__init__()

        self.T = T

        # build encoders
        self.base_encoder = base_encoder(num_classes=mlp_dim)
        self.momentum_encoder = base_encoder(num_classes=mlp_dim)

        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def contrastive_loss(self, q, k, redunction, flip):
        # normalize
        bz = q.shape[0]
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # gather all targets
        k = concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        if flip:
            rank = torch.distributed.get_rank()
            labels = torch.arange(k.size()[0], dtype=torch.long).flip(0)[rank*bz:bz*(rank+1)].cuda()
        else:
            rank = torch.distributed.get_rank()
            labels = torch.arange(k.size()[0], dtype=torch.long)[rank*bz:bz*(rank+1)].cuda()
        return nn.CrossEntropyLoss(reduction=redunction)(logits, labels) * (2 * self.T)

   # def forward(self, x1, x2, m):
   # def forward(self, x1, x2, x1_mix, x2_mix, lam, m):
    def forward(self, images1, images2, lam, m):
    # images1 16*1*2*3*224*224    
    #def forward(self, x1, x2, x1_mix, x2_mix, lam, m):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Outpiut:
            li
            
            oss
        """
        # compute features
        #bz, _,_,_ = x1.shape
        #rank = torch.distributed.get_rank()
        #lam = lam.view(lam.shape[0],1,1,1)
        #x1, x2, x1_mix, x2_mix, lam = concat_all_gather(x1), concat_all_gather(x2), concat_all_gather(x1_mix), concat_all_gather(x2_mix), concat_all_gather(lam)
        #x1, x2 = concat_all_gather(x1), concat_all_gather(x2)
        start = time.time()
        _,bz,_,_,_ = images1.shape #2*128*3*224*224
        rank = torch.distributed.get_rank()
        #lam = lam.view(lam.shape[0],1,1,1)
        #x1, x2, x1_mix, x2_mix, lam = concat_all_gather(x1), concat_all_gather(x2), concat_all_gather(x1_mix), concat_all_gather(x2_mix), concat_all_gather(lam)
        x1, x2, x1_mix, x2_mix = images1[0,:,:,:], images2[0,:,:,:], images1[1,:,:,:,:], images2[1,:,:,:,:]
        x1, x2, x1_mix, x2_mix, lam = x1.contiguous(), x2.contiguous(), x1_mix.contiguous(), x2_mix.contiguous(), lam.contiguous()
        x1, x2, x1_mix, x2_mix = concat_all_gather(x1), concat_all_gather(x2), concat_all_gather(x1_mix), concat_all_gather(x2_mix)
        lam = concat_all_gather(lam)
        #x1, x2, x1_mix, x2_mix, lam = x1.cuda(), x2.cuda(), x1_mix.cuda(), x2_mix.cuda(), lam.cuda()
        '''swap = torch.rand([1]).cuda()
        swap = concat_all_gather(swap)
        if swap[0]>0.5:
            x1, x2 = x2, x1'''
        x1, x2, x1_mix, x2_mix = x1.cuda(), x2.cuda(), x1_mix.cuda(), x2_mix.cuda()
        end_data_concat = time.time()
        print("data transform time", end_data_concat-start)    
       # x1_mix = x1_mix.half()
       # x2_mix = x2_mix.half()
        #x1_mix, lam, new_lam = datamixing(x1.clone(), None)
        #x2_mix, lam, new_lam = datamixing(x2.clone(), lam)
        #x1_mix = x1_mix.half()
        #x2_mix = x2_mix.half()
        #x1_mix, lam, new_lam = datamixing(x1.clone(), None)
        #x2_mix, lam, new_lam = datamixing(x2.clone(), lam)
        q1 = self.predictor(self.base_encoder(x1[bz*rank:bz*(rank+1)]))
        q1_end_encoder = time.time()
        print("q1 encoder time", q1_end_encoder-end_data_concat) 
        q2_mix = self.predictor(self.base_encoder(x2_mix[bz*rank:bz*(rank+1)]))
        q2mix_end_encoder = time.time()
        print("q2mix encoder time", q2mix_end_encoder-q1_end_encoder) 
        # part 1
        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m)  # update the momentum encoder
            # compute momentum features as targets
            k1 = self.momentum_encoder(x1[bz*rank:bz*(rank+1)])
            k1_mix = self.momentum_encoder(x1_mix[bz*rank:bz*(rank+1)])
            k2 = self.momentum_encoder(x2[bz*rank:bz*(rank+1)])
        end_k = time.time()
        print("k encoder time", end_k-q2mix_end_encoder)      
        source_loss = self.contrastive_loss(q1, k2, "mean", False)
        
        #if new_lam is not None:
        #    lam = new_lam
        lam = lam.squeeze()
        mixloss_source = self.contrastive_loss(q2_mix, k1, "none", False).mul(lam[bz*rank:bz*(rank+1)]).mean() + \
                         self.contrastive_loss(q2_mix, k1, "none", True).mul(1.-lam[bz*rank:bz*(rank+1)]).mean()

        common = np.minimum(lam[bz*rank:bz*(rank+1)].cpu(), 1-lam[bz*rank:bz*(rank+1)].clone().flip(0).cpu()).cuda()+np.minimum(1-lam[bz*rank:bz*(rank+1)].cpu(), lam[bz*rank:bz*(rank+1)].clone().flip(0).cpu()).cuda()
        mixloss_mix = self.contrastive_loss(q2_mix, k1_mix, "none", False).mul(1/(1+common)).mean() + \
                      self.contrastive_loss(q2_mix, k1_mix, "none", True).mul(common/(1+common)).mean()
        end_loss_compute =  time.time()
        print("loss compute time", end_loss_compute-end_k)             
        return source_loss, mixloss_source/2, mixloss_mix/2



class MoCo_ResNet(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.fc.weight.shape[1]
        del self.base_encoder.fc, self.momentum_encoder.fc # remove original fc layer

        # projectors
        self.base_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)


class MoCo_ViT(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.head.weight.shape[1]
        del self.base_encoder.head, self.momentum_encoder.head # remove original fc layer

        # projectors
        self.base_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim)


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
