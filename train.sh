python train.py --dataroot /home/lsl/Research/1sl/Datasets/BRATS/splitedH5 \
--dataset_mode brats_split --gpu_ids 0 --batch_size 2 \
--name bratsTest1012_test --model asyndgan --direction BtoA --num_netD 10