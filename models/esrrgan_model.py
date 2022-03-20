import torch
import numpy as np
import copy
from collections import OrderedDict

from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel
from basicsr.models.srgan_model import SRGANModel
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img


@MODEL_REGISTRY.register()
class ESRRGAN_Model(SRModel):
    """ESRGAN model for single image super-resolution."""
    def init_training_settings(self):
        train_opt = self.opt['train']
        self.batch_size_per_gpu = self.opt['datasets']['train'].get('batch_size_per_gpu',16)
        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define network net_d
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_d', 'params')
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True), param_key)

        self.net_g.train()
        self.net_d.train()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)
#             print("*******************/n")
#             print("self.cri_gan is",self.cri_gan)
#             print("/n*******************")
        else:
            self.cri_gan = None
        # if train_opt.get('frequency_opt'):
        #     self.cri_ff = build_loss(train_opt['frequency_opt']).to(self.device)

        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)
        self.net_g_iters = train_opt.get('net_g_iters', 1)
        
        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, self.net_g.parameters(), **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
        # optimizer d
        optim_type = train_opt['optim_d'].pop('type')
        self.optimizer_d = self.get_optimizer(optim_type, self.net_d.parameters(), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)

    def optimize_parameters(self, current_iter):
        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, self.gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix
            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, self.gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style
            # gan loss (CategoricalLoss of RGAN with relativistic gan)
            Triplet_Loss = copy.deepcopy(self.cri_gan)
            num_outcomes = self.opt['network_d'].get('num_outcomes')
            # define anchors (you can experiment different shapes)

            # e.g. skewed normals
            # skew = skewnorm.rvs(-5, size=1000)
            # count, bins = np.histogram(skew, param.num_outcomes)
            # anchor0 = count / sum(count)

            # skew = skewnorm.rvs(5, size=1000)
            # count, bins = np.histogram(skew, param.num_outcomes)
            # anchor1 = count / sum(count)

            # e.g. normal and uniform
            gauss = np.random.normal(0, 0.1, 1000)
            count, bins = np.histogram(gauss, num_outcomes)
            anchor0 = count / sum(count)

            unif = np.random.uniform(-1, 1, 1000)
            count, bins = np.histogram(unif, num_outcomes)
            anchor1 = count / sum(count)

            anchor_real = torch.zeros((self.batch_size_per_gpu, num_outcomes), dtype=torch.float).to(self.device) + torch.tensor(anchor1, dtype=torch.float).to(self.device)
            anchor_fake = torch.zeros((self.batch_size_per_gpu, num_outcomes), dtype=torch.float).to(self.device) + torch.tensor(anchor0, dtype=torch.float).to(self.device)

            # real images
            feat_real = self.net_d(self.gt).log_softmax(1).exp()

            # fake images
            feat_fake = self.net_d(self.output).log_softmax(1).exp()
            gan_opt = self.opt['train']['gan_opt']
            # compute loss
            if gan_opt.get('relativisticG'):
                l_g_gan = -Triplet_Loss(anchor_fake, feat_fake, skewness=gan_opt.get('negative_skew'), is_disc=False) + Triplet_Loss(feat_real, feat_fake, skewness=0.0, is_disc=False)
            else:
                l_g_gan = -Triplet_Loss(anchor_fake, feat_fake, skewness=gan_opt.get('negative_skew'), is_disc=False) + Triplet_Loss(anchor_real, feat_fake, skewness=gan_opt.get('positive_skew'), is_disc=False)


            # original gan
            # real_d_pred = self.net_d(self.gt).detach()
            # fake_g_pred = self.net_d(self.output)
            # l_g_real = self.cri_gan(real_d_pred - torch.mean(fake_g_pred), False, is_disc=False)
            # l_g_fake = self.cri_gan(fake_g_pred - torch.mean(real_d_pred), True, is_disc=False)
            # l_g_gan = (l_g_real + l_g_fake) / 2
            #

            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan


            l_g_total.backward()
            self.optimizer_g.step()
#02实验的改动，使得d更新的速度下降
        # optimize net_d
        if (current_iter % self.net_g_iters == 0):
            for p in self.net_d.parameters():
                p.requires_grad = True

            self.optimizer_d.zero_grad()
            # gan loss (relativistic gan)

            anchor_real = torch.zeros((self.batch_size_per_gpu, num_outcomes), dtype=torch.float).to(self.device) + torch.tensor(anchor1, dtype=torch.float).to(self.device)
            anchor_fake = torch.zeros((self.batch_size_per_gpu, num_outcomes), dtype=torch.float).to(self.device) + torch.tensor(anchor0, dtype=torch.float).to(self.device)

            # real images
            feat_real = self.net_d(self.gt).log_softmax(1).exp()

            # fake images
            feat_fake = self.net_d(self.output.detach()).log_softmax(1).exp()

            l_d_real = Triplet_Loss(anchor_real, feat_real, skewness=Triplet_Loss.v_max, is_disc=True)
            l_d_real.backward()

            l_d_fake = Triplet_Loss(anchor_fake, feat_fake, skewness=Triplet_Loss.v_min, is_disc=True) 
#             l_d_fake = Triplet_Loss(feat_real, feat_fake, skewness=0.0, is_disc=True)  #实验02在判别器的更改

            l_d_fake.backward()

            lossD = l_d_real + l_d_fake

            # # In order to avoid the error in distributed training:
            # # "Error detected in CudnnBatchNormBackward: RuntimeError: one of
            # # the variables needed for gradient computation has been modified by
            # # an inplace operation",
            # # we separate the backwards for real and fake, and also detach the
            # # tensor for calculating mean.
            #  original gan
            # # real
            # fake_d_pred = self.net_d(self.output).detach()
            # real_d_pred = self.net_d(self.gt)
            # l_d_real = self.cri_gan(real_d_pred - torch.mean(fake_d_pred), True, is_disc=True) * 0.5
            # l_d_real.backward()
            # # fake
            # fake_d_pred = self.net_d(self.output.detach())
            # l_d_fake = self.cri_gan(fake_d_pred - torch.mean(real_d_pred.detach()), False, is_disc=True) * 0.5
            # l_d_fake.backward()
            self.optimizer_d.step()

            loss_dict['l_d_real'] = l_d_real
            loss_dict['l_d_fake'] = l_d_fake

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)