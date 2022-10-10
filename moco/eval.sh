python main_lincls.py \
	  -a vit_small --lr 5 \
	    --dist-url 'tcp://localhost:32' \
	      --multiprocessing-distributed --world-size 1 --rank 0 \
	        --pretrained '/code/save/FreMix/pretrained/AmpMix_300ep/checkpoint_0099.pth.tar' \
		 /data/systemuser/ImageNet 
