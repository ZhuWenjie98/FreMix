
python main_moco.py \
	  -a vit_small \
	   -b 256 \
	   --workers 16 \
	    --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 \
	      --epochs=300 --warmup-epochs=40 \
	        --stop-grad-conv1 --moco-m-cos --moco-t=.2 \
		  --dist-url 'tcp://localhost:32' \
		    --multiprocessing-distributed --world-size 2 --rank 0 \
		      /data/systemuser/ImageNet