
python main_moco.py \
	  -a vit_small \
	   -b 1028 \
	    --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 \
	      --epochs=300 --warmup-epochs=40 \
	        --stop-grad-conv1 --moco-m-cos --moco-t=.2 \
		  --dist-url 'tcp://localhost:32' \
		   --resume '/code/save/FreMix/pretrained/AmpMix/checkpoint_0079.pth.tar' \
		    --multiprocessing-distributed --world-size 1 --rank 0 \
		      /data/systemuser/ImageNet
