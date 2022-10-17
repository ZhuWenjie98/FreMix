import torch
import torch.nn as nn
import numpy as np
from timm.data import Mixup
import torch.nn.functional as F
import time


def amp_spectrum_mixup(img1, img2, lam, ratio=1.0):
    """Input image size: ndarray of [1, C, H, W]"""
    #lam = np.random.uniform(0, alpha)

    assert img1.shape == img2.shape
 
    b, c, h, w  = img1.shape
    ratio = torch.tensor(ratio, dtype=torch.float).cuda()
    h_crop = int(h * torch.sqrt(ratio))
    w_crop = int(w * torch.sqrt(ratio))
    h_start = h // 2 - h_crop // 2
    w_start = w // 2 - w_crop // 2

    
    img1_fft = torch.fft.fft2(img1, dim=(2,3))
    img2_fft = torch.fft.fft2(img2, dim=(2,3))
    img1_abs, img1_pha = torch.abs(img1_fft).cuda(), torch.angle(img1_fft).cuda()
    img2_abs, img2_pha = torch.abs(img2_fft).cuda(), torch.angle(img2_fft).cuda()

    img1_abs = torch.fft.fftshift(img1_abs.cuda(), dim=(2, 3)).cuda()
    img2_abs = torch.fft.fftshift(img2_abs.cuda(), dim=(2, 3)).cuda()

    img1_abs_ = torch.clone(img1_abs).cuda()
    img2_abs_ = torch.clone(img2_abs).cuda()
    ## phase1 + lam * amp2 + (1-lam) * amp1

    img1_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img1_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img2_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]
    ## phase2 + lam * amp1 + (1-lam) * amp2
    img2_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img2_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img1_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]
                                                                                        
    img1_abs = torch.fft.ifftshift(img1_abs.cuda(), dim=(2, 3))
    img2_abs = torch.fft.ifftshift(img2_abs.cuda(), dim=(2, 3))
    
    
    img21 = img1_abs * (np.e ** (1j * img1_pha))
    img12 = img2_abs * (np.e ** (1j * img2_pha))
    img21 = torch.real(torch.fft.ifft2(img21,dim=(2, 3)))
    img12 = torch.real(torch.fft.ifft2(img12,dim=(2, 3)))
    img21 = torch.clip(img21[0], 0, 255).int()
    img12 = torch.clip(img12[0], 0, 255).int()
    
    return img21, img12

def amp_spectrum_cutmix(img1, img2, lam, ratio=1.0):
    """Input image size: ndarray of [1, C, H, W]"""
    #cutmix must pass ratio = 1-lam ratio decide the spectrum cut area size

    assert img1.shape == img2.shape
 
    b, c, h, w  = img1.shape
    ratio = torch.tensor(ratio, dtype=torch.float).cuda()
    h_crop = int(h * torch.sqrt(ratio))
    w_crop = int(w * torch.sqrt(ratio))
    h_start = h // 2 - h_crop // 2
    w_start = w // 2 - w_crop // 2

    
    img1_fft = torch.fft.fft2(img1, dim=(2,3))
    img2_fft = torch.fft.fft2(img2, dim=(2,3))
    img1_abs, img1_pha = torch.abs(img1_fft).cuda(), torch.angle(img1_fft).cuda()
    img2_abs, img2_pha = torch.abs(img2_fft).cuda(), torch.angle(img2_fft).cuda()

    img1_abs = torch.fft.fftshift(img1_abs.cuda(), dim=(2, 3)).cuda()
    img2_abs = torch.fft.fftshift(img2_abs.cuda(), dim=(2, 3)).cuda()

    img1_abs_ = torch.clone(img1_abs).cuda()
    img2_abs_ = torch.clone(img2_abs).cuda()
    ## phase1 + lam * amp2 + (1-lam) * amp1

    img1_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = img2_abs_[h_start:h_start + h_crop,
                                                                            w_start:w_start + w_crop]
    ## phase2 + lam * amp1 + (1-lam) * amp2
    img2_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = img1_abs_[h_start:h_start + h_crop,
                                                                            w_start:w_start + w_crop]
                                                                                        
    img1_abs = torch.fft.ifftshift(img1_abs.cuda(), dim=(2, 3))
    img2_abs = torch.fft.ifftshift(img2_abs.cuda(), dim=(2, 3))
    
    
    img21 = img1_abs * (np.e ** (1j * img1_pha))
    img12 = img2_abs * (np.e ** (1j * img2_pha))
    img21 = torch.real(torch.fft.ifft2(img21,dim=(2, 3)))
    img12 = torch.real(torch.fft.ifft2(img12,dim=(2, 3)))
    img21 = torch.clip(img21[0], 0, 255).int()
    img12 = torch.clip(img12[0], 0, 255).int()
    
    
    return img21, img12    

def amp_spectrum_cut_mixup(img1, img2, lam, ratio=1.0):
    """Input image size: ndarray of [1, C, H, W]"""
    #lam = np.random.uniform(0, alpha)

    assert img1.shape == img2.shape
 
    b, c, h, w  = img1.shape
    ratio = torch.tensor(ratio, dtype=torch.float).cuda()
    h_crop = int(h * torch.sqrt(ratio))
    w_crop = int(w * torch.sqrt(ratio))
    h_start = h // 2 - h_crop // 2
    w_start = w // 2 - w_crop // 2

    
    img1_fft = torch.fft.fft2(img1, dim=(2,3))
    img2_fft = torch.fft.fft2(img2, dim=(2,3))
    img1_abs, img1_pha = torch.abs(img1_fft).cuda(), torch.angle(img1_fft).cuda()
    img2_abs, img2_pha = torch.abs(img2_fft).cuda(), torch.angle(img2_fft).cuda()

    img1_abs = torch.fft.fftshift(img1_abs.cuda(), dim=(2, 3)).cuda()
    img2_abs = torch.fft.fftshift(img2_abs.cuda(), dim=(2, 3)).cuda()

    img1_abs_ = torch.clone(img1_abs).cuda()
    img2_abs_ = torch.clone(img2_abs).cuda()
    ## phase1 + lam * amp2 + (1-lam) * amp1

    img1_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img1_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img2_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]
    ## phase2 + lam * amp1 + (1-lam) * amp2
    img2_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img2_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img1_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]
                                                                                        
    img1_abs = torch.fft.ifftshift(img1_abs.cuda(), dim=(2, 3))
    img2_abs = torch.fft.ifftshift(img2_abs.cuda(), dim=(2, 3))
    
    
    img21 = img1_abs * (np.e ** (1j * img1_pha))
    img12 = img2_abs * (np.e ** (1j * img2_pha))
    img21 = torch.real(torch.fft.ifft2(img21,dim=(2, 3)))
    img12 = torch.real(torch.fft.ifft2(img12,dim=(2, 3)))
    img21 = torch.clip(img21[0], 0, 255).int()
    img12 = torch.clip(img12[0], 0, 255).int()
    
    
    return img21, img12      

def phase_spectrum_mixup(img1, img2, lam, ratio=1.0):
    """Input image size: ndarray of [C, H, W]"""

    assert img1.shape == img2.shape
    b, c, h, w  = img1.shape
    ratio = torch.tensor(ratio, dtype=torch.float).cuda()
    h_crop = int(h * torch.sqrt(ratio))
    w_crop = int(w * torch.sqrt(ratio))
    h_start = h // 2 - h_crop // 2
    w_start = w // 2 - w_crop // 2

    img1_fft = torch.fft.fft2(img1, dim=(2,3))
    img2_fft = torch.fft.fft2(img2, dim=(2,3))
    img1_abs, img1_pha = torch.abs(img1_fft), torch.angle(img1_fft)
    img2_abs, img2_pha = torch.abs(img2_fft), torch.angle(img2_fft)

    img1_pha = torch.fft.fftshift(img1_pha, dim=(2, 3))
    img2_pha = torch.fft.fftshift(img2_pha, dim=(2, 3))

    img1_pha_ = torch.clone(img1_pha)
    img2_pha_ = torch.clone(img2_pha)
    ## amp1 + lam * phase1 + (1-lam) * phase2
    img1_pha[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img1_pha_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img2_pha_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]
    ## amp2 + lam * phase2 + (1-lam) * phase1
    img2_pha[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img2_pha_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img1_pha_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]
    img1_pha = torch.fft.ifftshift(img1_pha, dim=(2, 3))
    img2_pha = torch.fft.ifftshift(img2_pha, dim=(2, 3))
    img21 = img1_abs * (np.e ** (1j * img1_pha)) 
    img12 = img2_abs * (np.e ** (1j * img2_pha))
    img21 = torch.real(torch.fft.ifft2(img21,dim=(2, 3)))
    img12 = torch.real(torch.fft.ifft2(img12,dim=(2, 3)))
    img21 = torch.clip(img21[0], 0, 255).int()
    img12 = torch.clip(img12[0], 0, 255).int()

    return img21, img12 

def phase_spectrum_cutmix(img1, img2, lam, ratio=1.0):
    """Input image size: ndarray of [C, H, W]"""

    assert img1.shape == img2.shape
    b, c, h, w  = img1.shape
    ratio = torch.tensor(ratio, dtype=torch.float).cuda()
    h_crop = int(h * torch.sqrt(ratio))
    w_crop = int(w * torch.sqrt(ratio))
    h_start = h // 2 - h_crop // 2
    w_start = w // 2 - w_crop // 2

    img1_fft = torch.fft.fft2(img1, dim=(2,3))
    img2_fft = torch.fft.fft2(img2, dim=(2,3))
    img1_abs, img1_pha = torch.abs(img1_fft), torch.angle(img1_fft)
    img2_abs, img2_pha = torch.abs(img2_fft), torch.angle(img2_fft)

    img1_pha = torch.fft.fftshift(img1_pha, dim=(2, 3))
    img2_pha = torch.fft.fftshift(img2_pha, dim=(2, 3))

    img1_pha_ = torch.clone(img1_pha)
    img2_pha_ = torch.clone(img2_pha)
    ## amp1 + lam * phase1 + (1-lam) * phase2
    img1_pha[h_start:h_start + h_crop, w_start:w_start + w_crop] = img2_pha_[h_start:h_start + h_crop,
                                                                            w_start:w_start + w_crop]
    ## amp2 + lam * phase2 + (1-lam) * phase1
    img2_pha[h_start:h_start + h_crop, w_start:w_start + w_crop] = img1_pha_[h_start:h_start + h_crop,
                                                                            w_start:w_start + w_crop]
    img1_pha = torch.fft.ifftshift(img1_pha, dim=(2, 3))
    img2_pha = torch.fft.ifftshift(img2_pha, dim=(2, 3))
    img21 = img1_abs * (np.e ** (1j * img1_pha)) 
    img12 = img2_abs * (np.e ** (1j * img2_pha))
    img21 = torch.real(torch.fft.ifft2(img21,dim=(2, 3)))
    img12 = torch.real(torch.fft.ifft2(img12,dim=(2, 3)))
    img21 = torch.clip(img21[0], 0, 255).int()
    img12 = torch.clip(img12[0], 0, 255).int()

    return img21, img12 

def phase_spectrum_cut_mixup(img1, img2, lam, ratio=1.0):
    """Input image size: ndarray of [C, H, W]"""

    assert img1.shape == img2.shape
    b, c, h, w  = img1.shape
    ratio = torch.tensor(ratio, dtype=torch.float).cuda()
    h_crop = int(h * torch.sqrt(ratio))
    w_crop = int(w * torch.sqrt(ratio))
    h_start = h // 2 - h_crop // 2
    w_start = w // 2 - w_crop // 2

    img1_fft = torch.fft.fft2(img1, dim=(2,3))
    img2_fft = torch.fft.fft2(img2, dim=(2,3))
    img1_abs, img1_pha = torch.abs(img1_fft), torch.angle(img1_fft)
    img2_abs, img2_pha = torch.abs(img2_fft), torch.angle(img2_fft)

    img1_pha = torch.fft.fftshift(img1_pha, dim=(2, 3))
    img2_pha = torch.fft.fftshift(img2_pha, dim=(2, 3))

    img1_pha_ = torch.clone(img1_pha)
    img2_pha_ = torch.clone(img2_pha)
    ## amp1 + lam * phase1 + (1-lam) * phase2
    img1_pha[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img1_pha_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img2_pha_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]
    ## amp2 + lam * phase2 + (1-lam) * phase1
    img2_pha[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img2_pha_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img1_pha_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]
    img1_pha = torch.fft.ifftshift(img1_pha, dim=(2, 3))
    img2_pha = torch.fft.ifftshift(img2_pha, dim=(2, 3))
    img21 = img1_abs * (np.e ** (1j * img1_pha)) 
    img12 = img2_abs * (np.e ** (1j * img2_pha))
    img21 = torch.real(torch.fft.ifft2(img21,dim=(2, 3)))
    img12 = torch.real(torch.fft.ifft2(img12,dim=(2, 3)))
    img21 = torch.clip(img21[0], 0, 255).int()
    img12 = torch.clip(img12[0], 0, 255).int()

    return img21, img12   



   