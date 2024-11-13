import torch
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import os.path

def prepare_tinystories():
	if os.path.isfile('data/tinystories-train-tokenized.npy'):
		return
	print('Loading/preprocessing TinyStories dataset...')
	tokenizer = AutoTokenizer.from_pretrained('gpt2') 
	ts = load_dataset("roneneldan/TinyStories")
	for split in ['train', 'validation']: 
		ts[split] = ts[split].map(lambda d: {'token_ids': tokenizer.encode(d['text'])})
	train = np.array([token_id for line in ts['train']['token_ids'] for token_id in line], dtype='uint16')
	np.save('data/tinystories-train-tokenized.npy', train)
	test = np.array([token_id for line in ts['validation']['token_ids'] for token_id in line], dtype='uint16')
	np.save('data/tinystories-test-tokenized.npy', test)

class DataLoader():
	def __init__(self, datafile:str, batch_size:int, context_length:int, random_seed:int=1):
		self.batch_size = batch_size
		self.context_length = context_length
		self._batches_gotten = 0
		self.dataset = np.load(datafile, mmap_mode='r')
		self.n = self.dataset.shape[0]
		self.random_state = np.random.RandomState(random_seed)

	def get_batch(self):
		self._batches_gotten += 1
		idxs = self.random_state.choice(self.n-self.context_length-1, self.batch_size)
		inputs = torch.IntTensor(np.array([self.dataset[i:i+self.context_length] for i in idxs], dtype='int32'))
		outputs = torch.LongTensor(np.array([self.dataset[i+1:i+self.context_length+1] for i in idxs], dtype='int32'))
		return inputs, outputs