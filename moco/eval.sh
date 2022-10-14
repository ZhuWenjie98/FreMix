python main_lincls.py \
	  -a vit_small --lr 5 \
	   --epochs 110 \
	    --dist-url 'tcp://localhost:32' \
	      --multiprocessing-distributed --world-size 1 --rank 0 \
	        --pretrained '/code/save/FreMix/pretrained/Phase_cutmix_mixup_ep300/checkpoint_0099.pth.tar' \
			 --resume '/code/save/FreMix/eval/Phase_cutmix_mixup_ep100/checkpoint_0086.pth.tar' \
			  --save-dir '/code/save/FreMix/eval/Phase_cutmix_mixup_ep100/' \
		 /data/systemuser/ImageNet 
