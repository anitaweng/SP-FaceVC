#from model_vc_repar import Generator
from model_vc_gan import Generator
import torch
import torch.nn.functional as F
import time
import datetime
import os
from tensorboardX import SummaryWriter 
from pathlib import Path 
import pytorch_ssim
from model_bl import D_VECTOR
from collections import OrderedDict
from collections import namedtuple
import random
#from augment import *
from pytorch_metric_learning.losses import NTXentLoss, CrossBatchMemory
from Dis_noncond import MultiDiscriminator
from torch.autograd import Variable
import io
import matplotlib.pyplot as plt
import tensorflow as tf
import librosa.display
import matplotlib
matplotlib.use('Agg') 
import numpy as np

log_dir = 'log_gan'
ckpt_dir = 'checkpoint'

class Solver(object):

    def __init__(self, vcc_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.vcc_loader = vcc_loader

        # Model configurations.
        self.lambda_cd = config.lambda_cd
        self.dim_neck = config.dim_neck
        self.dim_emb = config.dim_emb
        self.dim_pre = config.dim_pre
        self.freq = config.freq

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.model_id = config.model_id
        self.autoVC_path = config.autoVC_path
        self.no_repar = config.no_repar
        self.no_attn = config.no_attn
        self.no_dis = config.no_dis
        self.mel = config.mel

        # Miscellaneous.
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.use_cuda else 'cpu')
        self.log_step = config.log_step
        self.save_step = config.save_step
        self.writer = SummaryWriter(os.path.join(log_dir, self.model_id))
        print('Save log in: '+os.path.join(log_dir, self.model_id))

        self.ssim_loss = pytorch_ssim.SSIM().cuda() 
        self.ssim_loss_psnt = pytorch_ssim.SSIM().cuda() 
        self.ssim_loss_cont = pytorch_ssim.SSIM().cuda() 
        # Build the model and tensorboard.
        self.build_model()
        #self.build_buffer()
        #self.build_loss()
        self.maxpool = torch.nn.MaxPool2d((1,3), stride=1, padding=(0,1))

    def build_loss(self):
        memory_size = 1024
        inner_loss = NTXentLoss(temperature=0.50) 
        self.sloss = CrossBatchMemory(loss=inner_loss, embedding_size=256, memory_size=memory_size)


    def build_buffer(self):
        dic = [None for i in range(len(self.vcc_loader)*self.batch_size)]
        print(len(self.vcc_loader))

        for i in range(len(self.vcc_loader)):
            #print(i)
            try:
                _, emb, _, index, _ = next(data_iter)
                #emb = emb.to(self.device)
                for j, ind in enumerate(index):
                    dic[ind]= np.array(emb[j].data)
                    #print(ind)
            except:
                data_iter = iter(self.vcc_loader)
                _, emb, _, index, _ = next(data_iter)
                #emb = emb.to(self.device)
                for j, ind in enumerate(index):
                    dic[ind]= np.array(emb[j].data)
                    #print(ind)
            #print(dic[ind])
        #print(dic)
        

        self.buffer = torch.Tensor(dic) 
        self.buffer2 = self.buffer.clone() 
        self.buffer = self.buffer.to(self.device) 
        self.buffer2 = self.buffer2.to(self.device)  
        #print(self.buffer.shape)
        self.labels = torch.arange(0, self.buffer.shape[0]).to(self.device) #[i for i in range(self.buffer.shape[0])]

        
            
    def build_model(self):
        
        self.G = Generator(self.dim_neck, self.dim_emb, self.dim_pre, self.freq, self.no_attn, self.no_repar)  
        #dir = 'gan'
        #g_checkpoint = torch.load('/disk/autovc/checkpoint/'+dir+'/G.ckpt')
        #self.G.load_state_dict(g_checkpoint) 
        '''g_checkpoint = torch.load(self.autoVC_path)     
        self.G.load_state_dict(g_checkpoint)
        for p in self.G.parameters():
            p.requires_grad = False
        for p in self.G.reparam.parameters():
            p.requires_grad = True
        for p in self.G.decoder.parameters():
       	    p.requires_grad = True'''
        for p in self.G.parameters():
            p.requires_grad = True

        self.G.to(self.device)

        '''self.C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).cuda()
        c_checkpoint = torch.load('3000000-BL.ckpt')
        new_state_dict = OrderedDict()
        for key, val in c_checkpoint['model_b'].items():
            new_key = key[7:]
            new_state_dict[new_key] = val
        self.C.load_state_dict(new_state_dict)
        for p in self.C.parameters():
            p.requires_grad = False'''
        
        self.g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), 0.0001)
        #g_op = torch.load('/disk/autovc/checkpoint/'+dir+'/op.ckpt')
        #self.g_optimizer.load_state_dict(g_op) 
        if not self.no_dis:
            self.D = MultiDiscriminator(80, seglen=128, kernel_size=3, num_blocks=4, num_dis=3).to(self.device)
            #d_checkpoint = torch.load('/disk/autovc/checkpoint/'+dir+'/D.ckpt')
            #self.D.load_state_dict(d_checkpoint) 
            self.optimizerD = torch.optim.Adam(self.D.parameters(), lr = 0.0004, betas=(0.9, 0.999))
            #d_op = torch.load('/disk/autovc/checkpoint/'+dir+'/opD.ckpt')
            #self.optimizerD.load_state_dict(d_op) 
            self.avgpool = torch.nn.AvgPool1d(3)
    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
    def reset_gradD(self):
        if not self.no_dis:
            self.optimizerD.zero_grad()
      
    
    #=====================================================================================================================================#
    
    
                
    def train(self):
        # Set data loader.
        data_loader = self.vcc_loader
        real_label, fake_label = 0.9, 0
        label = Variable(torch.FloatTensor(self.batch_size)).to(self.device)
    
        
        # Print logs in specified order
        if self.no_dis:
            keys = ['G/loss_id','G/loss_id_psnt','G/loss_cd','G/g_loss']#,'G/sloss' ,
        else:
            keys = ['G/loss_id_psnt','G/loss_cd','G/errG','G/g_loss','G/floss','D/errD','D/errD_fake','D/errD_real']#,'G/sloss' 'G/loss_id',
            
        # Start training.
        print('Start training...')
        start_time = time.time()
        start = 0#67610#0#546000#1068000
        for i in range(start, start+self.num_iters):
            #print(i)
            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch data.
            try:
                x_real, emb_org, emb_tgt, index, index_b, sp, _ = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, emb_org, emb_tgt, index, index_b, sp, _ = next(data_iter)
            
            
            x_real = x_real.to(self.device) 
            emb_org = emb_org.to(self.device) 
            emb_tgt = emb_tgt.to(self.device) 
            index = index.to(self.device) 
            index_b = index_b.to(self.device) 
            sp = sp.float().to(self.device) 
            if not self.no_dis:
                x_pool = self.avgpool(x_real.transpose(1,2))
                x_pool_ = self.avgpool(x_pool) 
            #a = random.uniform(0, 1)
            #emb_tgt = a*emb_tgt + (1-a) * emb_org           
       
            # =================================================================================== #
            #                               2. Train the generator                                #
            # =================================================================================== #
            self.G = self.G.train()
                        
            # Identity mapping loss
            if self.mel:
                x_identic, x_identic_psnt, code_real = self.G(x_real, emb_org, emb_org)
                x_identicb, x_identic_psntb, code_realb = self.G(x_real, emb_org, emb_tgt)
            else:
                x_identic, x_identic_psnt, code_real = self.G(sp, emb_org, emb_org)
                x_identicb, x_identic_psntb, code_realb = self.G(sp, emb_org, emb_tgt)

            
            if self.no_dis:
                g_loss_id = F.l1_loss(x_real, x_identic.squeeze())
                g_loss_id_psnt = F.l1_loss(x_real, x_identic_psnt.squeeze())   
                # Code semantic loss.
                code_reconst = self.G(x_identic_psnt, emb_org, None)
                g_loss_cd = F.l1_loss(code_real, code_reconst)
                g_loss = g_loss_id + g_loss_id_psnt + self.lambda_cd * g_loss_cd  
                g_loss.backward()
                self.g_optimizer.step()
                self.reset_grad()
                #self.buffer2[index_b, :] = emb_anchor.detach()
                #self.buffer[index_b, :] = emb_anchor.detach()
                
                # Logging.
                loss = {}
                loss['G/loss_id'] = g_loss_id.item()
                loss['G/loss_id_psnt'] = g_loss_id_psnt.item()
                loss['G/loss_cd'] = g_loss_cd.item()
                loss['G/g_loss'] = g_loss.item()

                self.writer.add_scalar('G/loss_id', g_loss_id.item(), i+1)
                self.writer.add_scalar('G/loss_id_psnt', g_loss_id_psnt.item(), i+1)
                self.writer.add_scalar('G/loss_cd', g_loss_cd.item(), i+1)
                self.writer.add_scalar('G/g_loss', g_loss.item(), i+1)
            else:
                for k in range(1):
                    self.reset_gradD()
                    label.data.resize_(x_real.size(0)).fill_(real_label)
                    routs, rfeatures = self.D([x_real.transpose(1,2), x_pool, x_pool_])
                    errD_real = 0
                    for out in routs:
                        errD_real += 0.5 * torch.mean((out-label)**2)
                    errD_real.backward()

                    # with fake data
                    label.data.fill_(fake_label)
                    fake = x_identic.squeeze().transpose(1,2).detach()
                    fake_pool = self.avgpool(fake)
                    fake_pool_ = self.avgpool(fake_pool)
                    fouts, features = self.D([fake, fake_pool, fake_pool_])
                    errD_fake = 0
                    for out in fouts:
                        errD_fake += 0.5 * torch.mean((out-label)**2)
                    errD_fake.backward()

                    errD = errD_fake + errD_real
                    self.optimizerD.step()

                #train G
                self.reset_grad()
                label.data.fill_(real_label)
                dec_pool = self.avgpool(x_identic.squeeze().transpose(1,2))
                dec_pool_ = self.avgpool(dec_pool)
                routs, rfeatures = self.D([x_real.transpose(1,2), x_pool, x_pool_])
                fouts, features = self.D([x_identic.squeeze().transpose(1,2), dec_pool, dec_pool_]) # Forward propagation of generated image, this should result in '1'
                errG = 0
                for out in fouts:
                    errG += 0.5 * torch.mean((out - label)**2) # criterion(output, label)

                floss = 0
                for rf,ff in zip(rfeatures, features):
                    for j in range(len(rf)):
                        if j == 0:
                            #tmp = nn.functional.l1_loss(ff[i].view(ff[i].shape[0], -1), rf[i].view(rf[i].shape[0], -1), reduction = 'sum')
                            tmp = torch.nn.functional.l1_loss(ff[j].view(ff[j].shape[0], -1), rf[j].view(rf[j].shape[0], -1))
                        else:
                            #tmp += nn.functional.l1_loss(ff[i].view(ff[i].shape[0], -1), rf[i].view(rf[i].shape[0], -1), reduction = 'sum')
                            tmp += torch.nn.functional.l1_loss(ff[j].view(ff[j].shape[0], -1), rf[j].view(rf[j].shape[0], -1))
                    floss += tmp.mean()

                
                g_loss_id_psnt = F.l1_loss(x_real, x_identic_psnt.squeeze())   
                #g_loss_id_psnt = -self.ssim_loss_psnt(x_real.unsqueeze(1), x_identic_psnt.squeeze().unsqueeze(1))
                # Code semantic loss.
                code_reconst = self.G(x_identic_psnt, emb_org, None)
                g_loss_cd = F.l1_loss(code_real, code_reconst)
                #g_loss_cd = self.ssim_loss_cont(code_real, code_reconst)

                #x_convert, x_convert_psnt, code_real_convert = self.G(x_real, emb_org, emb_tgt)

                #emb_anchor = self.C(x_convert_psnt.squeeze())
                #emb = torch.cat((self.buffer , self.buffer2, emb_anchor))
                '''if i%400 == 0:
                    emb = torch.cat((self.buffer , emb_anchor))
                    labels = torch.cat((self.labels , index_b))
                else:
                    emb = emb_anchor
                    labels = index_b'''
                #emb_pos =  self.buffer[index_b]
                #print(self.index_lst[:index_b]+self.index_lst[index_b:])
                #emb_neg =  torch.index_select(self.buffer, 0, self.index_lst[:index_b]+self.index_lst[index_b:])
                #labels = torch.cat((self.labels , self.labels ,index_b))
                

                #print(emb.shape)
                #print(labels.shape)
                #sloss = self.sloss(emb, labels, enqueue_idx=None)
                #sloss = self.sloss(emb, labels)
                #s_loss_cd = self.ssim_loss_cont(code_real.unsqueeze(0).unsqueeze(1), code_real_convert.unsqueeze(0).unsqueeze(1))
                #print(sloss.item())
                #time_mask(freq_mask(time_warp(spectro.transpose(-1, -2)), num_masks=2), num_masks=2)
                #assert 0

                # Backward and optimize.
                g_loss = 100*g_loss_id_psnt + self.lambda_cd * g_loss_cd + errG + 0.1*floss #+ 1000*errG_rb#+ sloss #+ s_loss_cd g_loss_id + 
                g_loss.backward()
                self.g_optimizer.step()
                self.reset_grad()
                #self.buffer2[index_b, :] = emb_anchor.detach()
                #self.buffer[index_b, :] = emb_anchor.detach()
                
                # Logging.
                loss = {}
                #loss['G/loss_id'] = g_loss_id.item()
                loss['G/loss_id_psnt'] = g_loss_id_psnt.item()
                loss['G/loss_cd'] = g_loss_cd.item()
                loss['G/errG'] = errG.item()
                loss['G/g_loss'] = g_loss.item()
                #loss['G/errG_b'] = errG_b.item()
                loss['G/floss'] = floss.item()
                #loss['G/errG_rb'] = errG_rb.item()

                loss['D/errD'] = errD.item()
                loss['D/errD_fake'] = errD_fake.item()
                loss['D/errD_real'] = errD_real.item()
                self.writer.add_scalar('G/loss_id_psnt', g_loss_id_psnt.item(), i+1)
                self.writer.add_scalar('G/loss_cd', g_loss_cd.item(), i+1)
                self.writer.add_scalar('G/errG', errG.item(), i+1)
                self.writer.add_scalar('G/floss', floss.item(), i+1)
                self.writer.add_scalar('G/g_loss', g_loss.item(), i+1)
                self.writer.add_scalar('G/errD', errD.item(), i+1)
                self.writer.add_scalar('G/errD_fake', errD_fake.item(), i+1)
                self.writer.add_scalar('G/errD_real', errD_real.item(), i+1)

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if i == 0:
                if not os.path.exists(ckpt_dir):
                    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)  
                if not os.path.exists(os.path.join(ckpt_dir, self.model_id)):
                    os.mkdir(os.path.join(ckpt_dir, self.model_id))
                    
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag in keys:
                    log += ", {}: {:.4f}".format(tag, loss[tag])
                print(log)

            if (i+1) % self.save_step == 0:
                torch.save(self.G.state_dict(), os.path.join(ckpt_dir, self.model_id, 'G.ckpt'))
                torch.save(self.g_optimizer.state_dict(), os.path.join(ckpt_dir, self.model_id, 'op.ckpt'))
                if not self.no_dis:
                    torch.save(self.D.state_dict(), os.path.join(ckpt_dir, self.model_id, 'D.ckpt'))
                    torch.save(self.optimizerD.state_dict(), os.path.join(ckpt_dir, self.model_id, 'opD.ckpt'))
                print('Save ckpt in: '+os.path.join(ckpt_dir, self.model_id, 'G.ckpt'))

                fig = plt.figure()
                plt.title('origin')
                librosa.display.specshow(np.transpose(x_real[0].detach().squeeze().cpu().numpy(), (-1,-2)), x_axis='time', y_axis='mel', sr=22050)
                plt.colorbar(format='%f')
                #plt.savefig(os.path.join(output_dir,'convert_'+ name + '.png'))
                plt.close()
                self.writer.add_figure('origin', fig, i+1)

                fig = plt.figure()
                plt.title('reconstruct')
                librosa.display.specshow(np.transpose(x_identic_psnt[0].detach().squeeze().cpu().numpy(), (-1,-2)), x_axis='time', y_axis='mel', sr=22050)
                plt.colorbar(format='%f')
                #plt.savefig(os.path.join(output_dir,'convert_'+ name + '.png'))
                plt.close()
                self.writer.add_figure('reconstruct', fig, i+1)

                figb = plt.figure()
                plt.title('convert')
                librosa.display.specshow(np.transpose(x_identic_psntb[0].detach().squeeze().cpu().numpy(), (-1,-2)), x_axis='time', y_axis='mel', sr=22050)
                plt.colorbar(format='%f')
                #plt.savefig(os.path.join(output_dir,'convert_'+ name + '.png'))
                plt.close()
                self.writer.add_figure('convert', figb, i+1)
            
            if (i+1) % 100000 == 0:
                torch.save(self.G.state_dict(), os.path.join(ckpt_dir, self.model_id, 'G_'+str(i+1)+'.ckpt'))
                torch.save(self.g_optimizer.state_dict(), os.path.join(ckpt_dir, self.model_id, 'op.ckpt'))
                if not self.no_dis:
                    torch.save(self.D.state_dict(), os.path.join(ckpt_dir, self.model_id, 'D_'+str(i+1)+'.ckpt'))
                
                print('Save ckpt in: '+os.path.join(ckpt_dir, self.model_id, 'G.ckpt'))
                

    
    

    