python train.py --dataroot ./datasets/BRATS/splitedH5 \
--dataset_mode brats_split --gpu_ids 1 --batch_size 1 \
--name bratsTest0929_test --model asyndgan --direction BtoA --num_netD 2