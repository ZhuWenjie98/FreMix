
python main_moco.py \
	  -a vit_small \
	   -b 1024\
	   --workers 64 \
	    --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 \
	      --epochs=300 --warmup-epochs=40 \
	        --stop-grad-conv1 --moco-m-cos --moco-t=.2 \
		     --dist-url 'tcp://localhost:32' \
		      --multiprocessing-distributed --world-size 1 --rank 0 \
			   --mix-strategy 'amp_mixup' \
		      /data/systemuser/ImageNet
