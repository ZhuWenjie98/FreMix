python -m torch.distributed.launch --nproc_per_node=8 eval_knn.py \
--pretrained_weights '/code/save/FreMix/pretrained/Phase_cutmix_mixup_ep300/checkpoint_0099.pth.tar' \
--checkpoint_key state_dict --data_path '/data/systemuser/ImageNet' 

python -m torch.distributed.launch --nproc_per_node=8 eval_knn.py \
--pretrained_weights '/code/save/FreMix/pretrained/Phase_cutmix_mixup_ep300/checkpoint_0129.pth.tar' \
--checkpoint_key state_dict --data_path '/data/systemuser/ImageNet' 
