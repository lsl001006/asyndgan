python save_syn.py --dataroot /home/lsl/Research/1sl/Datasets/BRATS/splitedH5 \
--name bratsTest1012 --model asyndgan --netG unet_256 \
--direction AtoB --dataset_mode brats_split --epoch latest --results_dir results/brats_asyndgan_withoutL1