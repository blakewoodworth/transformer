import torch
import numpy as np
import wandb
from tqdm import tqdm
from data import DataLoader, prepare_tinystories
from model import TransformerLM
from utils import *

args = parse_arguments()
wandb.init(project='transformer-gpu', config=vars(args))
setup_directories()
prepare_tinystories()

if torch.cuda.is_available():
	device = torch.device('cuda:0')
else:
	device = torch.device('cpu')
print(f'Using device {device}...')

model_hyperparameters = {
	'd_model': args.d_model,
	'num_heads': args.num_heads,
	'd_ff': args.d_ff,
	'vocab_size': args.vocab_size,
	'context_length': args.context_length,
	'num_layers': args.num_layers,
	'embed_pdrop': args.embed_pdrop,
	'attn_pdrop': args.attn_pdrop,
	'residual_pdrop': args.residual_pdrop
}
# model = torch.compile(TransformerLM(**model_hyperparameters)).to(device)
model = TransformerLM(**model_hyperparameters).to(device)
print(f"# model parameters: {sum(p.numel() for p in model.parameters())}")

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_min, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
scheduler = CosineLRWithWarmup(optimizer, args.lr_max, args.lr_min, args.warmup_steps, args.total_steps)
criterion = torch.nn.CrossEntropyLoss()
trainloader = DataLoader(args.train_datafile, args.batch_size, args.context_length)

model.train()
for i in tqdm(range(1,args.total_steps+1)):
	X,Y = trainloader.get_batch()
	X,Y = X.to(device), Y.to(device)

	model.zero_grad()
	output = model(X)

	loss = criterion(output.view(-1, args.vocab_size), Y.view(-1))
	wandb.log({'loss': loss.item()})
	loss.backward()

	if args.gradient_clip is not None:
		torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)

	optimizer.step()
	scheduler.step()
	
	if i % args.checkpoint_frequency == 0:
		outfile = checkpoint_path + f'checkpoint{i}.pt'
		save_checkpoint(model, optimizer, scheduler, outfile)

	# if i < 5:
	# 	if device.type == 'cuda':
	# 	    print(torch.cuda.get_device_name(0))
	# 	    print('Memory Usage:')
	# 	    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
	# 	    print('Available:', round(torch.cuda.get_device_properties(0).total_memory/1024**3,1), 'GB')

wandb.finish()






