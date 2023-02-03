import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from distribute import *

n_mel =  80   

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self, dim_emb_in, dim_emb_out):
        super(Self_Attn,self).__init__()
        
        self.query = LinearNorm(dim_emb_in, dim_emb_out)
        self.key = LinearNorm(dim_emb_in, dim_emb_out)
        self.value = LinearNorm(dim_emb_in, dim_emb_out)

        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( dim_emb )
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        x = x.unsqueeze(1)
        proj_query  = self.query(x)
        proj_key =  self.key(x)
        energy =  torch.matmul(proj_query.permute(0,2,1), proj_key)
        ################################ Mask ###############################
        attn_mask = torch.cuda.FloatTensor(energy.size()).uniform_() > 0.9
        attn_mask = attn_mask.cuda()
        energy.masked_fill_(attn_mask, -1e9)
        ####################################################################
        attention = self.softmax(energy)
        proj_value = self.value(x)

        out = torch.matmul(proj_value, attention.permute(0,2,1))
        out = out.squeeze(1)
        
        return out, attention

class Repara(nn.Module):

    def __init__(self, dim_emb, hidden_sizes, latent_size):
        super(Repara,self).__init__()

        self.linearelu = nn.Sequential(
                nn.Linear(dim_emb, hidden_sizes),
                nn.ReLU(),
                nn.Linear(hidden_sizes, latent_size),
                nn.ReLU())

        self.linearelu_mu = nn.Linear(latent_size, latent_size)
        self.linearelu_logvar = nn.Linear(latent_size, latent_size)


    def gaussian_param_projection(self, x):
        return self.linearelu_mu(x), self.linearelu_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        x = self.linearelu(x)
        mu, logvar = self.gaussian_param_projection(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class Encoder(nn.Module):
    """Encoder module:
    """
    def __init__(self, dim_neck, dim_emb, freq):
        super(Encoder, self).__init__()
        self.dim_neck = dim_neck
        self.freq = freq
        
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(n_mel if i==0 else 512,
                         512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(512))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        
        self.lstm = nn.LSTM(512, dim_neck, 2, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = x.squeeze(1).transpose(2,1)
        
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
        
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        out_forward = outputs[:, :, :self.dim_neck]
        out_backward = outputs[:, :, self.dim_neck:]
        
        codes = []
        for i in range(0, outputs.size(1), self.freq):
            codes.append(torch.cat((out_forward[:,i+self.freq-1,:],out_backward[:,i,:]), dim=-1))

        return codes
      
        
class Decoder(nn.Module):
    """Decoder module:
    """
    def __init__(self, dim_neck, dim_emb, dim_pre):
        super(Decoder, self).__init__()
        
        self.lstm1 = nn.LSTM(dim_neck*2+dim_emb, dim_pre, 1, batch_first=True)
        
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(dim_pre,
                         dim_pre,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(dim_pre))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        
        self.lstm2 = nn.LSTM(dim_pre, 1024, 2, batch_first=True)
        
        self.linear_projection = LinearNorm(1024, n_mel)#80

    def forward(self, x):
        
        #self.lstm1.flatten_parameters()
        x, _ = self.lstm1(x)
        x = x.transpose(1, 2)
        
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
        
        outputs, _ = self.lstm2(x)
        
        decoder_output = self.linear_projection(outputs)

        return decoder_output   
    
    
class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(n_mel, 512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(512))
        )#80

        for i in range(1, 5 - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(512,
                             512,
                             kernel_size=5, stride=1,
                             padding=2,
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(512))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(512, n_mel,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(n_mel))
            )#160

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = torch.tanh(self.convolutions[i](x))

        x = self.convolutions[-1](x)

        return x    
    

class Generator(nn.Module):
    """Generator network."""
    def __init__(self, dim_neck, dim_emb, dim_pre, freq, no_attn=False, no_repar=False):
        super(Generator, self).__init__()
        
        self.encoder = Encoder(dim_neck, dim_emb, freq)
        self.decoder = Decoder(dim_neck, dim_emb, dim_pre)
        if not no_attn:
            self.attn = Self_Attn(dim_neck*2, dim_neck*2)
        else:
            self.attn = None
        self.postnet = Postnet()
        if not no_repar:
            self.repar = Repara(dim_emb, dim_emb//2, dim_emb)
        else:
            self.repar = None


    def forward(self, x, c_org, c_trg, eps=None):
        
        codes = self.encoder(x)
        if c_trg is None:
            return torch.cat(codes, dim=-1)
        
        tmp = []
        for code in codes:
            if self.attn is not None:
                code, _ = self.attn(code)
            tmp.append(code.unsqueeze(1).expand(-1,int(x.size(1)/len(codes)),-1))
        code_exp = torch.cat(tmp, dim=1)
        
        if self.repar is not None:
            if c_org is not None:
                c_org, _, _ = self.repar(c_org)#
            c_trg, _, _ = self.repar(c_trg)
        encoder_outputs = torch.cat((code_exp, c_trg.unsqueeze(1).expand(-1,x.size(1),-1)), dim=-1)
        
        mel_outputs = self.decoder(encoder_outputs)
                
        mel_outputs_postnet = self.postnet(mel_outputs.transpose(2,1))
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(2,1)
        
        mel_outputs = mel_outputs.unsqueeze(1)
        mel_outputs_postnet = mel_outputs_postnet.unsqueeze(1)
        
        return mel_outputs, mel_outputs_postnet, torch.cat(codes, dim=-1)

    def inter(self, x, a, b, alpha=0, eps=None):
        
        codes = self.encoder(x)
        
        tmp = []
        for code in codes:
            if self.attn is not None:
                code, _ = self.attn(code)
            tmp.append(code.unsqueeze(1).expand(-1,int(x.size(1)/len(codes)),-1))
        code_exp = torch.cat(tmp, dim=1)
        
        c_trg = a + (b-a)*alpha
        c_trg, _, _ = self.repar(c_trg)
        encoder_outputs = torch.cat((code_exp, c_trg.unsqueeze(1).expand(-1,x.size(1),-1)), dim=-1)
        mel_outputs = self.decoder(encoder_outputs)
                
        mel_outputs_postnet = self.postnet(mel_outputs.transpose(2,1))
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(2,1)
        
        mel_outputs = mel_outputs.unsqueeze(1)
        mel_outputs_postnet = mel_outputs_postnet.unsqueeze(1)
        
        return mel_outputs, mel_outputs_postnet, torch.cat(codes, dim=-1)

    
