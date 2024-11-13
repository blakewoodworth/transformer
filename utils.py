import torch
import math
import argparse
import os
import wandb
from tqdm import tqdm
from data import DataLoader

def setup_directories():
	for dirname in ['./checkpoints', './data']:
		if not os.path.isdir(dirname):
			os.mkdir(dirname)

def parse_arguments():
	argparser = argparse.ArgumentParser(description='My transformer language model')

	# general
	argparser.add_argument('--train_datafile', type=str, default='data/tinystories-train-tokenized.npy', help='Training data location')
	argparser.add_argument('--test_datafile', type=str, default='data/tinystories-test-tokenized.npy', help='Testing data location')
	argparser.add_argument('--vocab_size', type=int, default=50257, help='Tokenizer vocabulary size')
	argparser.add_argument('--context_length', type=int, default=256, help='Text context length')
	argparser.add_argument('--total_tokens_processed', type=int, default=327680000, help='Total tokens processed')
	argparser.add_argument('--eval_frequency', type=int, default=1000, help='Number of steps between evaluations')
	argparser.add_argument('--checkpoint_frequency', type=int, default=10000, help='Number of steps between checkpoints')
	argparser.add_argument('--checkpoint_path', type=str, default='checkpoints/', help='Path to checkpoint directory')

	# model hyperparameters
	argparser.add_argument('--d_model', type=int, default=512, help='Model feature dimension')
	argparser.add_argument('--d_ff', type=int, default=2048, help='Model feedforward layer dimension')
	argparser.add_argument('--num_layers', type=int, default=4, help='Model number of transformer layers')
	argparser.add_argument('--num_heads', type=int, default=16, help='Model number of attention heads')
	argparser.add_argument('--embed_pdrop', type=float, default=0.0, help='Dropout probability for embedding layer')
	argparser.add_argument('--attn_pdrop', type=float, default=0.0, help='Dropout probability for attention layer')
	argparser.add_argument('--residual_pdrop', type=float, default=0.0, help='Dropout probability for residual connections')

	# optimizer hyperparameters
	argparser.add_argument('--batch_size', type=int, default=24, help='Batch size')
	argparser.add_argument('--lr_min', type=float, default=1e-7, help='Minimum stepsize')
	argparser.add_argument('--lr_max', type=float, default=6e-4, help='Maximum stepsize')
	argparser.add_argument('--warmup_steps', type=int, default=100, help='Number of warmup steps')
	argparser.add_argument('--beta1', type=float, default=0.9, help='AdamW beta1 parameter')
	argparser.add_argument('--beta2', type=float, default=0.95, help='AdamW beta2 parameter')
	argparser.add_argument('--weight_decay', type=float, default=0.0, help='AdamW weight decay parameter')
	argparser.add_argument('--gradient_clip', type=float, default=1000., help='Gradient clipping norm')

	args = argparser.parse_args()
	vars(args)['total_steps'] = args.total_tokens_processed // (args.batch_size * args.context_length)
	return args

def save_checkpoint(model:torch.nn.Module, outfile:str):
	print(f'Saving checkpoint to file {outfile}...')
	model.eval()
	torch.save(model.state_dict(), outfile)
	model.train()

def load_checkpoint(model:torch.nn.Module, infile:str):
	model.eval()
	model.load_state_dict(torch.load(infile))
	model.train()

class CosineLRWithWarmup(torch.optim.lr_scheduler._LRScheduler):
	def __init__(self, optimizer:torch.optim.Optimizer, lr_max:float, lr_min:float, warmup_steps:int, total_steps:int):
		self.lr_max = lr_max
		self.lr_min	= lr_min
		self.warmup_steps = warmup_steps
		self.total_steps = total_steps
		self.warmup_factor = (self.lr_max-self.lr_min) / self.warmup_steps
		self.cos_factor = math.pi / (total_steps - warmup_steps)
		verbose = 'deprecated' if int(torch.__version__[0]) == 2 else False
		super().__init__(optimizer, -1, verbose)

	def get_lr(self):
		if self._step_count < self.warmup_steps:
			lr = self.lr_min + self._step_count * self.warmup_factor
		elif self._step_count < self.total_steps:
			lr = self.lr_min + 0.5*(self.lr_max - self.lr_min)*(1 + math.cos((self._step_count - self.warmup_steps) * self.cos_factor))
		else:
			lr = self.lr_min
		return [lr for group in self.optimizer.param_groups]

class SineLR(torch.optim.lr_scheduler._LRScheduler):
	def __init__(self, optimizer:torch.optim.Optimizer, lr_max:float, lr_min:float, periods:float, total_steps:int):
		self.lr_max = lr_max
		self.lr_min	= lr_min
		self.sine_factor = 2*periods*math.pi/total_steps
		self.total_steps = total_steps
		verbose = 'deprecated' if int(torch.__version__[0]) == 2 else False
		super().__init__(optimizer, -1, verbose)

	def get_lr(self):
		lr = self.lr_min + 0.5*(self.lr_max - self.lr_min)*(1 + math.sin(self.sine_factor*self._step_count - math.pi/2))
		return [lr for group in self.optimizer.param_groups]

def calculate_perplexity(model, args):
	testloader = DataLoader(args.test_datafile, args.batch_size, args.context_length, is_test=True)
	criterion = torch.nn.CrossEntropyLoss(reduction='none')
	model.eval()
	perplexity = 0.
	# n_batches = testloader.n // testloader.batch_size
	n_batches = 1000
	device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
	with torch.no_grad():
		for _ in tqdm(range(n_batches), leave=False):
			X,Y = testloader.get_batch()
			X, Y = X.to(device), Y.to(device)
			output = model(X)
			loss = criterion(output.view(-1, args.vocab_size), Y.view(-1)).view(args.batch_size,args.context_length)
			perplexity += torch.mean(torch.exp(torch.mean(loss,dim=-1))).item()
	model.train()
	return perplexity / n_batches








