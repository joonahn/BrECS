import random
from collections import Iterator, defaultdict
from torch.utils.data import DataLoader
from utils.util import timeit
from datasets import DATASET
from utils.metrics import FIDCalculator, SingleIoUCalculator
from models.classifier_model import MinkowskiFCNN
import torch
from utils.phase import Phase
from copy import deepcopy
import MinkowskiEngine as ME
from tqdm import tqdm

# =====================
# Base Classes and ABCs
# =====================


class DataScheduler(Iterator):
	def __init__(self, config):
		self.config = config
		self.dataset = DATASET[self.config['dataset']](config, mode='train')
		self.eval_datasets = [
			DATASET[x[0]](config, mode=x[1])
			for x in self.config['eval_datasets']
		]

		if self.config.get('test_datasets') is not None:
			self.test_datasets = [
				DATASET[x[0]](config, mode=x[1])
				for x in self.config['test_datasets']
			]
		self.total_epoch = self.config['epoch']
		self.step_cnt = 0
		self.epoch_cnt = 0
		self._remainder = len(self.dataset)
		self.data_loader = DataLoader(
			self.dataset,
			batch_size=self.config['batch_size'],
			num_workers=self.config['num_workers'],
			collate_fn=self.dataset.collate_fn,
			shuffle=True
		)
		self.iter = iter(self.data_loader)
		self._check_vis = {}

		self.use_buffer = config['model'] in [
			'gca',
			'lego_transition',
			'cgca_transition',
			'cgca_transition_condition',
			'cgca_transition_connection',
		]
		if self.use_buffer:
			self.data_buffer = DataBuffer(config)

		if self.config.get('fid_n_data') is not None and self.config.get('fid_model') is not None:
			model = MinkowskiFCNN(in_channel=1, out_channel=40, embedding_channel=1024).to("cuda")
			state_dict = torch.load(self.config['fid_model'])
			model.load_state_dict(state_dict['model'])
			self.fid_calc = FIDCalculator(self.data_loader, model, self.config['fid_n_data'])
		else:
			print("[WARN] cannot read 'fid_n_data' and 'fid_model' from config")
			self.fid_calc = None

	@timeit
	def __next__(self):
		'''
		:return:
			data: dict of corresponding data
		'''
		if self.data_loader is None:
			raise StopIteration

		if self.use_buffer:
			while self.data_buffer.is_full() is False:
				try:
					data = next(self.iter)
				except StopIteration:
					self.iter = iter(self.data_loader)
					data = next(self.iter)
				self.data_buffer.push(data)
				self.update_epoch_cnt()
			data = self.data_buffer.sample(self.config['batch_size'])
		else:
			# used for training patch_autoencoder
			try:
				data = next(self.iter)
			except StopIteration:
				self.iter = iter(self.data_loader)
				data = next(self.iter)
			self.update_epoch_cnt()
		self.step_cnt += 1
		return data, self.epoch_cnt

	def __len__(self):
		return len(self.sampler)

	def check_eval_step(self, step):
		if (step + 1) < self.config['min_eval_step']:
			return False
		return ((step + 1) % self.config['eval_step'] == 0) \
			   or self.config['debug_eval']

	def check_test_step(self, step):
		if (step + 1) < self.config['min_test_step']:
			return False

		return (step + 1) % self.config['test_step'] == 0 \
			if self.config.get('test_step') is not None else False

	def check_vis_step(self, step):
		if (step + 1) < self.config['min_vis_step']:
			return False

		vis = False
		vis_config = self.config['vis']
		for (k, v) in vis_config.items():
			# check if valid visualization config
			if not isinstance(v, dict):
				continue
			if ((step + 1) % v['step'] == 0) or (self.config['debug_vis']):
				self._check_vis[k] = True
				vis = True
			else:
				self._check_vis[k] = False
		return vis

	def check_summary_step(self, step):
		return (step + 1) % self.config['summary_step'] == 0

	def check_empty_cache_step(self, step):
		if self.config.get('empty_cache_step') is None:
			return False
		return (step + 1) % self.config['empty_cache_step'] == 0

	def evaluate(self, model, writer, step):
		for eval_dataset in self.eval_datasets:
			eval_dataset.evaluate(model, writer, step)

	def test(self, model, writer, step):
		print('Testing...')
		if self.test_datasets is not None:
			for test_dataset in self.test_datasets:
				test_dataset.test(model, writer, step)
		self.measure_fid(model, writer, step)

	def generate_assembly(self, model, n_data):
		data_loader = DataLoader(
			self.dataset,
			batch_size=self.config['batch_size'],
			num_workers=self.config['num_workers'],
			collate_fn=self.dataset.collate_fn,
			shuffle=True
		)
		data_iter = iter(data_loader)
		acc_n_data = 0
		sparse_results = []

		with tqdm(total=n_data) as pbar:
			while n_data > acc_n_data:
				try:
					data = data_iter.next()
				except StopIteration:
					data_iter = iter(data_loader)
					data = data_iter.next()
				
				batch_size = len(data['state_feat'])
				s = ME.SparseTensor(
					features=torch.cat(data['state_feat']),
					coordinates=ME.utils.batched_coordinates(data['state_coord']),
					device="cuda"
				)
				pos = deepcopy(data['state0_pos'])
				phase = Phase(
					self.config['max_phase'],
					self.config['equilibrium_max_phase']
				)
				# do transition
				max_phase = self.config['max_eval_phase']
				max_ml_phase = self.config['test_mode_seeking_phase']
				for phase_cnt in range(max_phase + max_ml_phase):
					if self.config['model'] in ('gca', 'lego_transition'):
						sigma = None
					else:
						sigma = model.sigma_scheduler.sample([phase])[0] \
							if phase_cnt < max_phase else None
					with torch.no_grad():
						s_next, pos_next = model.transition(s, pos, sigma)
					s = s_next
					pos = pos_next
					phase += 1

				sparse_results.append(s)
				acc_n_data += batch_size
				pbar.update(batch_size)
			
		return sparse_results

	def measure_fid(self, model, writer, step):
		if self.fid_calc is None:
			return

		sparse_results = self.generate_assembly(model, self.config['fid_n_data'])
		
		model.scalar_summaries['fid'] = self.fid_calc.calculate_fid(sparse_results)

	def measure_single_iou(self, model, n_data):
		iou_calculator = SingleIoUCalculator(self.data_loader)
		sparse_results = self.generate_assembly(model, n_data)

		iou = iou_calculator.calculate_iou(sparse_results)
		print(f"[INFO] single_iou: {iou}")

	def visualize_test(self, model, writer, step):
		self.test_dataset.visualize_test(model, writer, step)

	def visualize(self, model, writer, step):

		# find options to visualize in this step
		options = []
		for (k, v) in self._check_vis.items():
			if not v:
				continue
			else:
				options.append(k)

		if isinstance(self.config['overfit_one_ex'], int):
			self.dataset.visualize(model, options, step)
		else:
			self.dataset.visualize(model, options, step)  # train dataset
			for eval_dataset in self.eval_datasets:  # eval dataset
				eval_dataset.visualize(model, options, step)
		# reset _check_vis
		self._check_vis = {}

	def update_epoch_cnt(self):
		self._remainder -= self.config['batch_size']
		if self._remainder < self.config['batch_size']:
			self._remainder += len(self.dataset)
			self.epoch_cnt += 1


class DataBuffer:
	def __init__(self, config):
		self.config = config
		self.buffer_size = config['buffer_size']
		self.buffer = []
		self.device = config['device']
		self.max_batch_points = config['mean_vox_points'] * config['batch_size']
		self.buffer_removal_cnt = 0

	def push(self, data):
		phase = data['phase']
		for batch_idx in range(len(phase)):
			if phase[batch_idx].finished:
				continue
			coord_cnt = data['state_coord'][batch_idx].shape[0]
			max_coord_cnt = self.config.get('voxel_overflow') \
				if self.config.get('voxel_overflow') is not None else 1000000000000
			if (coord_cnt == 0) or (coord_cnt > max_coord_cnt):
				self.buffer_removal_cnt += 1
				continue
			self.buffer.append({
				k: v[batch_idx]
				for k, v in data.items()
			})

	def sample(self, batch_size):
		data = defaultdict(list)
		cum_batch_points = 0
		for batch_idx in range(batch_size):
			idx = random.randint(0, len(self.buffer) - 1)
			# assure that the # coords in batch is not too big
			if ((cum_batch_points + self.buffer[idx]['state_coord'].shape[0]) > self.max_batch_points) \
					and (batch_idx > 0):
				break
			pop_data = self.buffer.pop(idx)
			for k, v in pop_data.items():
				data[k].append(pop_data[k])
			cum_batch_points += pop_data['state_coord'].shape[0]
		# convert back to dict so that no accidents occur
		return dict(data)

	def is_full(self):
		return len(self.buffer) >= self.buffer_size


