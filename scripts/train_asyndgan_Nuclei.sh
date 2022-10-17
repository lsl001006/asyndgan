set -ex

# python train.py --dataroot /data/repo/code/1sl/FullNet-varCE/original_data/MultiOrgan/AsynDGAN_Nuclei_dataset --name brats_gan_nuclei_withoutL1_asyndgan --model asyndgan --netG resnet_9blocks --direction AtoB --lambda_L1 0 --delta_perceptual 10 --dataset_mode nuclei_split --pool_size 0 --gpu_ids 0 --batch_size 4 --num_threads 0  --niter=200 --niter_decay=200 --num_netD 4

python save_syn_nuclei.py --dataroot /data/repo/code/1sl/FullNet-varCE/original_data/MultiOrgan/AsynDGAN_Nuclei_dataset --name brats_gan_nuclei_withoutL1_asyndgan --model pix2pix --netG resnet_9blocks --direction AtoB  --dataset_mode nuclei  --gpu_ids 0 --batch_size 4 --num_threads 0  --num_netD 4
