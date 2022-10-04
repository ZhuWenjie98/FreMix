python main_lincls.py \
	  -a vit_small --lr 0.1 \
	    --dist-url 'tcp://localhost:32' \
	      --multiprocessing-distributed --world-size 1 --rank 0 \
	        --pretrained '/code/save/FreMix/pretrained/AmpMix_gpu_80ep/checkpoint_0079.pth.tar' \
		 /data/systemuser/ImageNet 
