r""""Contains definitions of the methods used by the _BaseDataLoaderIter workers to
collate samples fetched from dataset into Tensor(s).

These **needs** to be in global scope since Py2 doesn't support serializing
static methods.
"""

import torch
import re
import numpy as np
from torch._six import container_abcs, string_classes, int_classes

np_str_obj_array_pattern = re.compile(r'[SaUO]')


def default_convert(data):
    r"""Converts each NumPy array data field into a tensor"""
    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        return data
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        # array of string classes and object
        if elem_type.__name__ == 'ndarray' \
                and np_str_obj_array_pattern.search(data.dtype.str) is not None:
            return data
        return torch.as_tensor(data)
    elif isinstance(data, container_abcs.Mapping):
        return {key: default_convert(data[key]) for key in data}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return elem_type(*(default_convert(d) for d in data))
    elif isinstance(data, container_abcs.Sequence) and not isinstance(data, string_classes):
        return [default_convert(d) for d in data]
    else:
        return data


default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


def default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    bz = len(batch)
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__!= 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))
             
            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        if isinstance(elem[0],list) and len(elem[0][0]) == 3: # elem[0]=3 means that the batch is 16*2*3*224*224
                result = [] 
                for samples in transposed:  #some samples are 16*2*image some are 16*int, we need judge
                  judge = samples[0]
                  if isinstance(judge, int):
                    result.append(default_collate(samples)) 
                  else:  
                    lam = np.random.uniform(0, 1, bz)
                    result.append(images_collate(samples, lam))
                return result
        return [default_collate(samples) for samples in transposed] 


    raise TypeError(default_collate_err_msg_format.format(elem_type))

def images_collate(batch, lam):
    # input 16*2*3*224*224
    """Puts each data field into a tensor with outer dimension batch size"""
    batch = list(batch)
    bz = len(batch)
    elem = batch[0][0]
    elem_type = type(elem)
    out = None
    if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
        
        numel = bz*len(batch[0])*3*224*224
        storage = elem.storage()._new_shared(numel)
        out = elem.new(storage)
        lams = torch.zeros(bz)
        for i in range(len(batch)):
            batch[i] = list(batch[i])
            batch[i] = torch.stack(batch[i],0)
        images_mix = torch.stack(batch, 0, out=out)
        for i, img1 in enumerate(batch):
            images_mix[i][0],images_mix[bz-i-1][0], lams[i] = colorful_spectrum_mix_cpu(img1[0], batch[bz-i-1][0], lam[i], 1)
            images_mix[i][1],images_mix[bz-i-1][1], lams[i] = colorful_spectrum_mix_cpu(img1[1], batch[bz-i-1][1], lam[i], 1)  
        batch_tensor = torch.stack(batch, 0, out=out)
        images_concat = torch.stack((batch_tensor, images_mix),2)   #concat origin and mix at 2-dimension  1-dimension is 2 of different views   
    return [images_concat, lams]
    #else:
    #   return images_collate(default_collate(batch), lam)


def colorful_spectrum_mix_cpu(img1, img2, alpha, ratio=1.0):
    """Input image size: ndarray of [C, H, W ]"""
    if alpha == None:
       lam = np.random.uniform(0, alpha)
    else:
       lam = alpha
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
