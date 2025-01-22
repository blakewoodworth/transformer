import torch
import numpy as np
import wandb
import time
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

device = get_device()
model = make_model(args).to(device)
print(f"# model parameters: {round(sum(p.numel() for p in model.parameters())/1e6,2)}M")

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_max, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
scheduler = CosineLRWithWarmup(optimizer, args.lr_max, args.lr_min, args.warmup_steps, args.total_steps)
criterion = torch.nn.CrossEntropyLoss()
trainloader = DataLoader(args.train_datafile, args.batch_size, args.context_length)

wandb.init(project='transformer-v100', config=vars(args))

t0 = time.time()
model.train()
for i in tqdm(range(1,args.total_steps+1)):

	X,Y = trainloader.get_batch()
	X,Y = X.to(device), Y.to(device)

	model.zero_grad()
	output = model(X)
	loss = criterion(output.view(-1, args.vocab_size), Y.view(-1))
	loss.backward()

	time_elapsed = time.time()-t0
	wandb_dict = {'loss': loss.item(), 'time': time_elapsed, 'avg_time_per_step': time_elapsed/i, 'stepsize': optimizer.param_groups[0]['lr']}
	wandb.log(wandb_dict)

	optimizer.step()
	scheduler.step()

	if args.gradient_clip is not None:
		torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)

	if i % args.eval_frequency == 0 or i == args.total_steps:
		eval_t0 = time.time()
		perplexity = calculate_perplexity(model, args)
		wandb.log({'test_perplexity': perplexity}, commit=False)
		eval_time = time.time() - eval_t0
		t0 += eval_time

wandb.finish()






