#!/usr/bin/env python

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import wandb
import os
from tqdm import tqdm
from data import DataLoader, prepare_tinystories
from model import make_model
from utils import *
from matplotlib import pyplot as plt

np.random.seed(1)
torch.manual_seed(0)

args = parse_arguments()
vars(args)['total_steps'] = 5000

setup_directories()
prepare_tinystories()

wandb.init(project='transformer-gpu', config=vars(args))

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

model = make_model(args).to(device)
print(f"# model parameters: {round(sum(p.numel() for p in model.parameters())/1e6,2)}M")


wandb.finish()



def setup(rank, world_size):
	os.environ['MASTER_ADDR'] = 'localhost'
	os.environ['MASTER_PORT'] = '12355'
	dist.init_process_group('gloo', rank=rank, world_size=world_size)

def cleanup():
	dist.destroy_process_group()

def train(rank, world_size):
	setup(rank, world_size)

	model = make_model(args).to(rank)
	ddp_model = DDP(model, device_ids=[rank])

	optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=args.lr_max, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
	scheduler = CosineLRWithWarmup(optimizer, args.lr_max, args.lr_min, args.warmup_steps, args.total_steps)
	criterion = torch.nn.CrossEntropyLoss()
	trainloader = DataLoader(args.train_datafile, args.batch_size, args.context_length, random_seed=rank)

	TQDM!!

	ddp_model.train()
	for i in tqdm(range(1,args.total_steps+1)):

		X,Y = trainloader.get_batch()
		X,Y = X.to(rank), Y.to(rank)

		optimizer.zero_grad()
		output = ddp_model(X)
		loss = criterion(output.view(-1, args.vocab_size), Y.view(-1))
		loss.backward()
		dist.reduce(loss,dst=0,op=dist.ReduceOp.SUM)
		if args.gradient_clip is not None:
			torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), args.gradient_clip)

		if rank == 0:
			param_norm = torch.sqrt(sum([torch.norm(p)**2 for p in ddp_model.parameters()])).item()
			grad_norm = torch.sqrt(sum([torch.norm(p.grad)**2 for p in ddp_model.parameters()])).item()
			wandb.log({'loss': loss.item(), 'param_norm': param_norm, 'grad_norm': grad_norm, 'stepsize': optimizer.param_groups[0]['lr']})

			# if i % args.eval_frequency == 0 or i == args.total_steps:
			# 	perplexity = calculate_perplexity(model, args)
			# 	wandb.log({'test_perplexity': perplexity}, commit=False)
			
			# if i % args.checkpoint_frequency == 0 or i == args.total_steps:
			# 	outfile = args.checkpoint_path + f'checkpoint{i}.pt'
			# 	save_checkpoint(model, outfile)

		optimizer.step()
		scheduler.step()



# if __name__ == '__main__':
# 	pass




