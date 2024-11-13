import torch
import numpy as np
import wandb
from tqdm import tqdm
from data import DataLoader, prepare_tinystories
from model import TransformerLM
from utils import *
from matplotlib import pyplot as plt

np.random.seed(1)
torch.manual_seed(0)

args = parse_arguments()
setup_directories()
prepare_tinystories()

vars(args)['total_steps'] = 3000

wandb.init(project='transformer-gpu', config=vars(args))

if torch.cuda.is_available():
	device = torch.device('cuda:0')
else:
	device = torch.device('cpu')

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

if int(torch.__version__[0]) >= 2:
	model = torch.compile(TransformerLM(**model_hyperparameters)).to(device)
else:
	model = TransformerLM(**model_hyperparameters).to(device)
print(f"# model parameters: {round(sum(p.numel() for p in model.parameters())/1e6,2)}M")

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_max, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
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
	loss.backward()

	param_norm = torch.sqrt(sum([torch.norm(p)**2 for p in model.parameters()])).item()
	grad_norm = torch.sqrt(sum([torch.norm(p.grad)**2 for p in model.parameters()])).item()
	wandb_dict = {'loss': loss.item(), 'param_norm': param_norm, 'grad_norm': grad_norm, 'stepsize': optimizer.param_groups[0]['lr']}
	wandb.log(wandb_dict)

	optimizer.step()
	scheduler.step()

	if args.gradient_clip is not None:
		torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)

	if i % args.eval_frequency == 0:
		perplexity = calculate_perplexity(model, args)
		wandb.log({'test_perplexity': perplexity}, commit=False)
	
	if i % args.checkpoint_frequency == 0 or i == args.total_steps:
		outfile = args.checkpoint_path + f'checkpoint{i}.pt'
		save_checkpoint(model, outfile)

wandb.log({'test_perplexity': calculate_perplexity(model, args)}, commit=False)
wandb.finish()
