import torch
from torch import nn
import torch.nn.functional as F

import os
import skimage.morphology as morph
from skimage import measure, io

from models.UNet import setup_unet
from models.perception_loss import vgg16_feat, perceptual_loss
from .base_model import BaseModel
from . import networks
from parse_config import ConfigParser
import parse_config
from util.meters import AverageMeter
from util.metric_nuclei_seg import accuracy, dice, aji

import pdb

class FedGANModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        
        parser.add_argument('--netT', type=str, default='UNet_seg', help='model type of netS and netT')
        parser.add_argument('--T_input_nc', type=int, default=3, help='input channel of seg model')
        parser.add_argument('--T_output_nc', type=int, default=3, help='output channel of seg model')
        parser.add_argument('--teacher-ckpt', type=str, default='/data/repo/code/1sl/DFFKD_byx/nuclei_teachers', help='path to load teacher checkpoint')


        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--delta_perceptual', type=float, default=1.0, help='weight for perceptual loss')

            parser.add_argument('--lambda_G', type=float, default=0.1, help='weight for asyndgan G ')
            parser.add_argument('--lambda_D', type=float, default=0.05, help='weight for asyndgan D')

            parser.add_argument('--warm-up', type=int, default=100, help='warm-up epochs for netS')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.num_netD = opt.num_netD
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        # self.loss_names = ['G_GAN', 'G_L1', 'G_Seg', 'D_real', 'D_fake']
        self.loss_names = ['G_GAN_all', 'G_L1_all', 'G_perceptual_all', 'D_real_all', 'D_fake_all']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A_2', 'fake_B_2', 'real_B_2','real_A_7', 'fake_B_7', 'real_B_7']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D', 'S', 'T']
        else:  # during test time, only load G
            self.model_names = ['G', 'S']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netS = networks.define_T(opt.T_input_nc, opt.T_output_nc, opt.netT,
                                    opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = []
            # for i in range(10):
            for i in range(self.num_netD):
                self.netD.append(networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids))
            
            self.netT = []
            for i in range(self.num_netD):
                self.netT.append(networks.define_T(opt.T_input_nc, opt.T_output_nc, opt.netT,
                                                    opt.init_type, opt.init_gain, self.gpu_ids).eval())
            self.load_pretrainedT(opt)
            

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionSeg = nn.NLLLoss(reduction='none')
            self.criterionDistill = nn.KLDivLoss(reduction='none', log_target=True)

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizer_D = []
            for i in self.netD:
                opt_D = torch.optim.Adam(i.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_D.append(opt_D)
                self.optimizers.append(opt_D)
            self.optimizer_S = torch.optim.Adam(self.netS.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_S)

            self.vgg_model = vgg16_feat().cuda()
            self.criterion_perceptual = perceptual_loss()
            # self.unet = setup_unet().cuda()
        
        self.test_metric_ftns = [accuracy, dice, aji]
        self.best_seg_performance = [-1, -1, -1]

    def load_pretrainedT(self, opt):
        organs_names = ["breast", "kidney", "liver", "prostate"]
        for n in range(len(self.netT)):
            teacher = self.netT[n]
            teacher_ckpt_path = os.path.join(opt.teacher_ckpt, "{}.pth".format(organs_names[n]))
            checkpoint = torch.load(teacher_ckpt_path)
            teacher.module.load_state_dict(checkpoint)
            print(teacher_ckpt_path, "loaded")

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = []
        self.real_B = []
        self.image_paths = []
        self.label = []
        self.weight_map = []
        # for i in range(10):
        
        for i in range(self.num_netD):
            self.real_A.append(input['A_' + str(i)].to(self.device))
            self.real_B.append(input['B_' + str(i)].to(self.device))
            self.image_paths.append(input['A_paths_' + str(i)])
            self.label.append(input['label_ternary_' + str(i)].to(self.device))
            self.weight_map.append(input['weight_map_' + str(i)].to(self.device))

        # import pdb;pdb.set_trace()
        self.real_A_2 = self.real_A[0]
        self.real_B_2 = self.real_B[0]
        self.real_A_7 = self.real_A[1]
        self.real_B_7 = self.real_B[1]


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = []
        for i in range(len(self.real_A)):
            self.fake_B.append(self.netG(self.real_A[i])) # G(A)
        # self.fake_B = self.netG(self.real_A)  # G(A)
        self.fake_B_2 = self.fake_B[0]
        self.fake_B_7 = self.fake_B[1]

    def forward_teacher_outs(self, images):
        """the prediction logits by teachers
        Args:
            images (tensor): input images
            localN (int): number of local teachers. Defaults to None.

        Returns:
            tensor: the original teacher logits
        """
        def renormalize(images, m1=[0.5,0.5,0.5], s1=[0.5,0.5,0.5], m2=[0.7442, 0.5381, 0.6650], s2=[0.1580, 0.1969, 0.1504]):
            '''the normalize parameter is different'between gan and seg'''
            m1,s1 = torch.tensor(m1).to(self.device).view(-1,1,1), torch.tensor(s1).to(self.device).view(-1,1,1)
            m2,s2 = torch.tensor(m2).to(self.device).view(-1,1,1), torch.tensor(s2).to(self.device).view(-1,1,1)
            images = images*s1+m1
            images = (images-m2)/s2
            return images

        images = renormalize(images)
        total_logits = []
        for i in range(len(self.netT)):
            logits = self.netT[i](images)
            total_logits.append(logits)
        total_logits = torch.stack(total_logits)  # nlocal*batch*channel*H*W
            
        return total_logits

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""

        self.loss_D_fake = []
        self.loss_D_real = []
        for i in range(len(self.real_A)):
            # Fake; stop backprop to the generator by detaching fake_B
            fake_AB = torch.cat((self.real_A[i], self.fake_B[i]),1)  # we use conditional GANs; we need to feed both input and output to the discriminator
            pred_fake = self.netD[i](fake_AB.detach())
            self.loss_D_fake.append(self.criterionGAN(pred_fake, False))
            # Real
            real_AB = torch.cat((self.real_A[i], self.real_B[i]), 1)
            pred_real = self.netD[i](real_AB)
            self.loss_D_real.append(self.criterionGAN(pred_real, True))

        self.loss_D_fake_all = None
        self.loss_D_real_all = None
        for i in range(len(self.loss_D_fake)):
            if self.loss_D_fake_all is None:
                self.loss_D_fake_all = self.loss_D_fake[i]
                self.loss_D_real_all = self.loss_D_real[i]
            else:
                self.loss_D_fake_all += self.loss_D_fake[i]
                self.loss_D_real_all += self.loss_D_real[i]

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake_all + self.loss_D_real_all)*self.opt.lambda_D #0.05
        self.loss_D.backward()


        # # Fake; stop backprop to the generator by detaching fake_B
        # fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        # pred_fake = self.netD(fake_AB.detach())
        # self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # # Real
        # real_AB = torch.cat((self.real_A, self.real_B), 1)
        # pred_real = self.netD(real_AB)
        # self.loss_D_real = self.criterionGAN(pred_real, True)
        # # combine loss and calculate gradients
        # self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        # self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        self.loss_G_GAN = []
        self.loss_G_L1 = []
        self.loss_G_perceptual = []

        # for i in range(10):
        for i in range(self.num_netD):
            # First, G(A) should fake the discriminator
            fake_AB = torch.cat((self.real_A[i], self.fake_B[i]), 1)
            pred_fake = self.netD[i](fake_AB)
            self.loss_G_GAN.append(self.criterionGAN(pred_fake, True))
            self.loss_G_L1.append(self.criterionL1(self.fake_B[i], self.real_B[i]) * self.opt.lambda_L1)

            pred_feat = self.vgg_model(self.fake_B[i])
            target_feat = self.vgg_model(self.real_B[i])
            self.loss_G_perceptual.append(self.criterion_perceptual(pred_feat, target_feat) * self.opt.delta_perceptual)

        self.loss_G_GAN_all = None
        self.loss_G_L1_all = None
        self.loss_G_perceptual_all = None
        for i in range(len(self.loss_G_GAN)):
            if self.loss_G_GAN_all is None:
                self.loss_G_GAN_all = self.loss_G_GAN[i]
                self.loss_G_L1_all = self.loss_G_L1[i]
                self.loss_G_perceptual_all = self.loss_G_perceptual[i]
            else:
                self.loss_G_GAN_all += self.loss_G_GAN[i]
                self.loss_G_L1_all += self.loss_G_L1[i]
                self.loss_G_perceptual_all += self.loss_G_perceptual[i]

        self.loss_G = (self.loss_G_GAN_all + self.loss_G_L1_all + self.loss_G_perceptual_all)*self.opt.lambda_G #0.1
        self.loss_G.backward()

        # # First, G(A) should fake the discriminator
        # fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        # pred_fake = self.netD(fake_AB)
        # self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # # Second, G(A) = B
        # self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        #
        # pred_feat = self.vgg_model(self.fake_B)
        # target_feat = self.vgg_model(self.real_B)
        # self.loss_G_perceptual = self.criterion_perceptual(pred_feat, target_feat) * self.opt.delta_perceptual

        # pred_seg = self.unet(self.fake_B)
        #
        # pred_seg = torch.sigmoid(pred_seg)
        #
        # self.loss_G_Seg = self.criterionSeg(pred_seg, self.seg)

        # combine loss and calculate gradients
        # self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_Seg
        # self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_perceptual
        # self.loss_G.backward()

    def backward_S(self):
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN_all', 'G_L1_all', 'G_perceptual_all', 'D_real_all', 'D_fake_all', 'S_SUP', 'S_DISTILL', 'S_ALL']

        weight_map = torch.cat(self.weight_map, 0)
        input = torch.cat(self.fake_B, 0)
        label = torch.cat(self.label, 0)

        if weight_map.dim() == 4:
            weight_map = weight_map.squeeze(1)
        if label.dim() == 4:
            label = label.squeeze(1)
        
        # supervision loss
        output = self.netS(input)
        log_prob_maps = F.log_softmax(output, dim=1)
        loss_map = self.criterionSeg(log_prob_maps, label)
        loss_map *= weight_map
        self.loss_S_SUP = loss_map.mean()

        # distillation loss
        with torch.no_grad():
            output_teachers = self.forward_teacher_outs(input)
            ensemble_output_teachers = self.ensemble_locals(output_teachers)
        loss_distill = self.criterionDistill(F.log_softmax(output), F.log_softmax(ensemble_output_teachers))
        loss_distill = loss_distill.sum(1)*weight_map
        self.loss_S_DISTILL = loss_distill.mean()
        

        self.loss_S_ALL = self.loss_S_SUP + self.loss_S_DISTILL
        self.loss_S_ALL.backward()


    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        for opt in self.optimizer_D:
            opt.zero_grad()
        self.backward_D()
        for opt in self.optimizer_D:
            opt.step()
        # self.optimizer_D.zero_grad()     # set D's gradients to zero
        # self.backward_D()                # calculate gradients for D
        # self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

        if self.epoch>self.opt.warm_up:
            self.forward()
            self.optimizer_S.zero_grad()
            self.backward_S()
            self.optimizer_S.step()

    def ensemble_locals(self, locals, localweight=None):
        """
        locals: (nlocal, batch, ncls) or (nlocal, batch/ncls) or (nlocal)
        * self.local_weight[localid]
        """
        if localweight==None:
            localweight = torch.ones(len(locals)).to(self.device)

        if len(locals.shape) == 5:  # [n_locals, B, C, H, W]
            localweight = localweight.view(-1,1,1,1,1)
            ensembled = (locals * localweight).sum(dim=0)
        elif len(locals.shape) == 1: # gan_loss
            ensembled = (locals * localweight).sum()  # 1
        else:
            pdb.set_trace()

        return ensembled

    def split_forward(self, model, input, size, overlap, outchannel=3):
        '''
        split the input image for forward process
        '''

        b, c, h0, w0 = input.size()

        # zero pad for border patches
        pad_h = 0
        if h0 - size > 0:
            pad_h = (size - overlap) - (h0 - size) % (size - overlap)
            tmp = torch.zeros((b, c, pad_h, w0)).to(self.device)
            input = torch.cat((input, tmp), dim=2)

        if w0 - size > 0:
            pad_w = (size - overlap) - (w0 - size) % (size - overlap)
            tmp = torch.zeros((b, c, h0 + pad_h, pad_w)).to(self.device)
            input = torch.cat((input, tmp), dim=3)

        _, c, h, w = input.size()

        output = torch.zeros((input.size(0), outchannel, h, w)).to(self.device)
        for i in range(0, h - overlap, size - overlap):
            r_end = i + size if i + size < h else h
            ind1_s = i + overlap // 2 if i > 0 else 0
            ind1_e = i + size - overlap // 2 if i + size < h else h
            for j in range(0, w - overlap, size - overlap):
                c_end = j + size if j + size < w else w

                input_patch = input[:, :, i:r_end, j:c_end]
                input_var = input_patch
                with torch.no_grad():
                    output_patch = model(input_var)

                ind2_s = j + overlap // 2 if j > 0 else 0
                ind2_e = j + size - overlap // 2 if j + size < w else w
                output[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = output_patch[:, :, ind1_s - i:ind1_e - i,
                                                             ind2_s - j:ind2_e - j]

        output = output[:, :, :h0, :w0]

        return output

    def validate(self, valid_dataset):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        if self.epoch<=self.opt.warm_up:
            return
        else:
            model = self.netS
            model.eval()

            acc = AverageMeter('acc@1', ':6.2f')
            dice = AverageMeter('dice', ':6.2f')
            aji = AverageMeter('aji', ':6.2f')
            meters = [acc, dice, aji]
            self.isbest = False
            
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(valid_dataset):
                    data, weight_map, target, instance_label = batch_data['B'].to(self.device), batch_data['weight_map'].to(self.device), batch_data['label_ternary'].to(self.device), batch_data['instance_label'].to(self.device)
                    
                    if weight_map.dim() == 4:
                        weight_map = weight_map.squeeze(1)
                    if target.dim() == 4:
                        target = target.squeeze(1)
                    if target.max() == 255:
                        target /= 255

                    # output = self.model(data)
                    output = self.split_forward(model, data, 256, 80)
                    log_prob_maps = F.log_softmax(output, dim=1)
                    
                    loss_map = self.criterionSeg(log_prob_maps, target)
                    loss_map *= weight_map
                    loss = loss_map.mean()

                    pred = torch.argmax(output, dim=1).detach().cpu().numpy()
                    pred_inside = pred == 1
                    instance_label = instance_label.detach().cpu().numpy()
                    for k in range(data.size(0)):
                        pred_inside[k] = morph.remove_small_objects(pred_inside[k], 20)  # remove small object
                        pred[k] = measure.label(pred_inside[k])  # connected component labeling
                        pred[k] = morph.dilation(pred[k], selem=morph.disk(2))
                        instance_label[k] = measure.label(instance_label[k])

                    for i, met in enumerate(self.test_metric_ftns):
                        meters[i].update(met(pred, instance_label, istrain=False), output.size(0))
            
            print(
            ' [Eval] Epoch={current_epoch} Acc@1={acc.avg:.4f} dice={dice.avg:.4f} aji={aji.avg:.4f}'
            .format(current_epoch=self.epoch, acc=acc, dice=dice, aji=aji))
            
            if aji.avg > self.best_seg_performance[-1]:
                self.isbest = True
                self.best_seg_performance = [acc.avg, dice.avg, aji.avg]
            print(
            ' [Eval] Best: Acc@1={:.4f} dice={:.4f} aji={:.4f}'
            .format(*self.best_seg_performance))

