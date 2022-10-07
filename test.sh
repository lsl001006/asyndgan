python test.py --dataroot ./datasets/BRATS/splitedH5 \
--name bratsTest0929_test --model asyndgan --direction BtoA \
--gpu_ids 1 --batch_size 1 --dataset_mode brats_split --num_netD 2