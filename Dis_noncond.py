import torch
import torch.nn as nn
from torch.autograd import Variable

class Dblock(nn.Module):
	def __init__(self, in_channel, seglen, kernel_size=3):
		super(Dblock, self).__init__()
		self.spec_conv1 = nn.utils.spectral_norm(nn.Conv1d(in_channel, 2*in_channel, kernel_size, padding=1))
		self.spec_conv2 = nn.utils.spectral_norm(nn.Conv1d(2*in_channel, 2*in_channel, kernel_size, padding=1))

		self.lrelu = nn.LeakyReLU(0.2, True)

	def forward(self, mel):
		# mel B*N*T
		# cond B*256*1

		spec_sc1 = self.spec_conv1(mel) # B*2N*T
		spec_cat = spec_sc1#+cond_cc1 # B*2N*T
		spec_cat = self.lrelu(spec_cat)  # B*2N*T
		spec_sc2 = self.spec_conv2(spec_cat)  # B*2N*T
		feature = spec_sc1+spec_sc2 # B*2N*T
		

		return feature

class Discriminator(nn.Module):
	def __init__(self, in_channel, seglen=128, kernel_size=3, num_blocks=4):
		super(Discriminator, self).__init__()

		self.dblocks = nn.ModuleList()

		for i in range(num_blocks):
			self.dblocks.add_module('dblock_{}'.format(i), Dblock(in_channel, seglen, kernel_size))
			in_channel = 2*in_channel

		self.dblocks.add_module('projection', nn.utils.spectral_norm(nn.Linear(in_channel*seglen, 1)))

	def forward(self, mel):
		
		features = []
		for i, module in enumerate(self.dblocks):
			if i != len(self.dblocks)-1:
				mel= module(mel)
				features.append(mel.clone())
			else:
				mel = module(mel.view(mel.shape[0], -1))

		return mel, features #B*1 , [B*2N*T, B*4N*T, B*8N*T, B*16N*T...]

class MultiDiscriminator(nn.Module):
	def __init__(self, in_channel, seglen=128, kernel_size=3, num_blocks=4, num_dis=3):
		super(MultiDiscriminator, self).__init__()

		self.multi_dis = nn.ModuleList()
		for i in range(num_dis):
			self.multi_dis.add_module('Dis_{}'.format(i), Discriminator(in_channel, seglen//(kernel_size**i), kernel_size, num_blocks))

	def forward(self, mels):
		outs = []
		features = []
		for module, mel in zip(self.multi_dis, mels):
			noise = torch.rand(mel.shape[0],mel.shape[1],mel.shape[2]).cuda()
			out, feature = module(mel+noise)
			outs.append(out.clone())
			features.append(feature)

		return outs, features


if __name__ == "__main__":
    net = MultiDiscriminator(80, seglen=128, kernel_size=3, num_blocks=4, num_dis=3)
    x = Variable(torch.randn(10,80,128))
    p = nn.AvgPool1d(3)
    x_pool = p(x)
    x_pool_ = p(x_pool)
    outs, features = net([x, x_pool, x_pool_])
    for out in outs:
    	print(out.shape)
    for f in features:
    	for i in range(len(f)):
    		print(f[i].shape)
