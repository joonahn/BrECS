import math
from typing import List
import torch
import os
import random
import numpy as np
import MinkowskiEngine as ME
import shutil
import glob
from baselines.config import get_inputs_field
from tqdm import tqdm
from datasets.base_dataset import BaseDataset
from collections import defaultdict
from lego.brick_generator import BrickGenerator
from models.transition_model import TransitionModel
from MinkowskiEngine import SparseTensor, MinkowskiInterpolationFunction
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.metrics import FIDCalculator
from utils.util import downsample, quantize
from utils.phase import LegoPhase, Phase
from utils.visualization import (
	sparse_tensors2tensor_imgs, save_tensor_img, tensors2tensor_imgs
)
from lego import binvox_rw
from scipy.ndimage import zoom, convolve
from copy import deepcopy
from lego.utils import brick_position_to_ldr, brick_position_to_pickle, is_pos_the_same

def change_feat(f):
	def wrapper(*args):
		data = f(*args)
		if args[0].config['model'] == 'gca':
			data['state_feat'] = torch.ones(data['state_feat'].shape[0], 1)
		elif args[0].config['model'] == 'cgca_transition_condition':
			data['state_feat'] = torch.zeros(data['state_feat'].shape[0], args[0].config['z_dim'])
		return data
	return wrapper

class LegoTransitionDataset(BaseDataset):
	def __init__(self, config: dict, mode: str):
		BaseDataset.__init__(self, config, mode)
		self.z_dim = config['z_dim']
		self.voxel_size = config['voxel_size']
		self.data_root = None
		self.data_list = []
		self.test_using_gt = config.get('lego_test_using_gt', False)
		self.brick_generator = BrickGenerator(config, parallel=False)
		if self.test_using_gt:
			print("[warn] test results are based on gt voxel!")


	def cache(self, model, data, rel_file_names, step):
		batch_size = len(data['state_feat'])
		pos_asm = []
		for _ in range(batch_size):
			pos_asm.append(list())
		s_init = SparseTensor(
			features=torch.cat(data['state_feat']),
			coordinates=ME.utils.batched_coordinates(data['state_coord']),
			device=self.device
		)
		vis_dir = os.path.join(
			self.config['log_dir'], 'test_save',
			'step-{}'.format(step), 'vis'
		)
		os.makedirs(vis_dir, exist_ok=True)
		pos_dir = os.path.join(
			self.config['log_dir'], 'test_save',
			'step-{}'.format(step), 'pos'
		)
		os.makedirs(pos_dir, exist_ok=True)
		pos_assemble_dir = os.path.join(
			self.config['log_dir'], 'test_save',
			'step-{}'.format(step), 'pos_asm'
		)
		os.makedirs(pos_assemble_dir, exist_ok=True)
		ldr_dir = os.path.join(
			self.config['log_dir'], 'test_save',
			'step-{}'.format(step), 'ldr'
		)
		os.makedirs(ldr_dir, exist_ok=True)
		input_imgs = tensors2tensor_imgs(
			data['state_coord'], self.config['data_dim'],
			self.config['vis']['vis_collated_imgs']['vis_3d'], batch_size
		)
		for batch_idx in range(batch_size):
			save_img_path = os.path.join(vis_dir, rel_file_names[batch_idx] + '-input.png')
			os.makedirs(os.path.dirname(save_img_path), exist_ok=True)
			save_tensor_img(input_imgs[batch_idx], save_img_path)

		if self.config['model'] == 'cgca_transition_condition':
			model.register_s0(data['input_pc'])

		for trial in range(self.config['test_trials']):
			s = s_init
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

				for batch_idx in range(batch_size):
					pos_asm[batch_idx].append(deepcopy(pos[batch_idx]))
				with torch.no_grad():
					if self.test_using_gt:
						s_next, pos_next = model.transition_with_gt(s, pos, data, sigma)
					else:
						s_next, pos_next = model.transition(s, pos, sigma)

				# generation is done!
				if is_pos_the_same(pos, pos_next):
					print(f"[debug] generation done! trial={phase_cnt}")
					break

				s = s_next
				pos = pos_next
				phase += 1

			out_imgs = sparse_tensors2tensor_imgs(
				s, self.config['data_dim'],
				self.config['vis']['vis_collated_imgs']['vis_3d'], batch_size
			)
			for batch_idx in range(batch_size):
				save_tensor_img(
					out_imgs[batch_idx],
					os.path.join(
						vis_dir,
						rel_file_names[batch_idx] + '-trial{}.png'.format(trial)
					)
				)
				np.save(
					os.path.join(
						pos_dir,
						rel_file_names[batch_idx] + '-trial{}.npy'.format(trial)
					),
					pos[batch_idx]
				)
				brick_position_to_ldr(
					os.path.join(
						ldr_dir,
						rel_file_names[batch_idx] + '-trial{}.ldr'.format(trial)
					),
					pos[batch_idx]
				)
				brick_position_to_pickle(
					os.path.join(
						pos_assemble_dir,
						rel_file_names[batch_idx] + '-trial{}.pkl'.format(trial)
					),
					pos_asm[batch_idx]
				)

			# cache dicts
			cache_dir = os.path.join(
				self.config['log_dir'], 'test_save',
				'step-{}'.format(step), 'cache'
			)
			cache_dicts = model.sparsetensors2cache_dicts(s)
			for cache_dict, file_name in zip(cache_dicts, data['file_name']):
				save_path = os.path.join(cache_dir, file_name + '-trial={}.pt'.format(trial))
				save_dir = os.path.dirname(save_path)
				os.makedirs(save_dir, exist_ok=True)
				torch.save(cache_dict, save_path)

	def __len__(self):
		return len(self.data_list)

class LegoModelnetDataset(LegoTransitionDataset):
	name = 'lego_modelnet'

	def __init__(self, config: dict, mode: str):
		LegoTransitionDataset.__init__(self, config, mode)
		self.obj_class = config['obj_class']
		self.max_sphere_centers = self.config['max_sphere_centers']
		self.sphere_radius = self.config['sphere_radius']
		self.surface_cnt = config['surface_cnt']
		self.data_voxel_size = config['data_voxel_size']

		if mode == 'train':
			self.data_root = os.path.join(
				config['data_root'], 'train'
			)
		elif mode == 'val' or mode == 'test':
			self.data_root = os.path.join(
				config['data_root'], 'test'
			)
		else:
			raise ValueError()

		
		self.data_list = sorted([
			os.path.basename(x) for x in glob.glob(f"{self.data_root}/{self.obj_class}_*_{self.data_voxel_size}.binvox")
		])

		if (mode == 'val') and (config['eval_size'] is not None):
			# fix vis_indices
			eval_size = config['eval_size']
			if isinstance(eval_size, int):
				val_indices = torch.linspace(0, len(self.data_list) - 1, eval_size).int().tolist()
				self.data_list = [self.data_list[i] for i in val_indices]

		if (mode == 'train') and (config.get('train_size', None) is not None):
			# fix vis_indices
			train_size = config['train_size']
			train_data_offset = config.get('train_data_offset', 0)
			if isinstance(train_size, int):
				val_indices = torch.linspace(train_data_offset, len(self.data_list) - 1, train_size).int().tolist()
				self.data_list = [self.data_list[i] for i in val_indices]


	@change_feat
	def __getitem__(self, idx):
		if self.config['overfit_one_ex'] is not None:
			idx = self.config['overfit_one_ex']
		data_name = self.data_list[idx]
		data_path = os.path.join(self.data_root, data_name)

		with open(data_path, "rb") as voxel_file:
			voxel_data = binvox_rw.read_as_3d_array(voxel_file).data
			assert voxel_data.shape == (self.data_voxel_size,) * 3

		# preprocess voxel data
		if self.config['data_conv_stride'] > 1:
			conv_filter = np.ones((self.config['data_conv_stride'],) * 3)
			voxel_data = convolve(voxel_data.astype(bool), conv_filter, mode='constant', cval=0.0)
		if self.config['data_scale'] != 1.0:
			voxel_data = zoom(voxel_data.astype(bool), self.config['data_scale'], order=0)
		voxel_data = voxel_data.astype(bool)

		n_pad_l = int((self.data_voxel_size - voxel_data.shape[0]) // 2)
		n_pad_r = math.ceil((self.data_voxel_size - voxel_data.shape[0]) / 2)
		voxel_data = np.pad(voxel_data.astype(np.int32), ((n_pad_l, n_pad_r), (n_pad_l, n_pad_r), (n_pad_l, n_pad_r)), mode='constant', constant_values=0)

		assert voxel_data.shape == (self.data_voxel_size,) * 3

		# obtain embeddings
		embedding_coord = np.concatenate([x[..., None] for x in np.where(voxel_data != 0)], axis=-1) - np.array([self.data_voxel_size // 2 - 1] * 3)
		assert embedding_coord.shape[1:] == (3,)
		coord_idx = np.argmin(np.sum(np.abs(embedding_coord), axis=1))
		if self.config.get("lego_assemble_from_bottom", False):
			coord_idx = np.argmin(np.sum(embedding_coord, axis=1))
		embedding_coord_center = embedding_coord[coord_idx]

		# embedding_coord_center = np.min(embedding_coord, axis=0)
		embedding_coord = torch.tensor(embedding_coord) # (N, 3)
		embedding_feat = torch.ones(embedding_coord.shape[0], 1) # (N, 1)

		# load brick list
		brick_list = self.config.get('lego_brick_list', [(2,4), (4,2)])
		brick_list = [tuple(brick_size) for brick_size in brick_list]

		# obtain initial state
		if (2, 4) in brick_list:
			# (2x4 brick)
			brick_data = np.array([[-1,0,-2], [-1,0,-1], [-1,0,0], [-1,0,1], [0,0,-2], [0,0,-1], [0,0,0], [0,0,1]])
			initial_brick_shape = (2,4)
		elif (2, 2) in brick_list:
			# (2x2 brick)
			brick_data = np.array([[-1,0,-1], [-1,0,0], [0,0,-1], [0,0,0]])
			initial_brick_shape = (2,2)
		else:
			raise ValueError("brick list does not contain either (2,4) or (2,2)")

		state_coord = brick_data + embedding_coord_center
		state_coord = torch.tensor(state_coord)
		state_feat = torch.ones(state_coord.shape[0], 1)
		state_pos = [(initial_brick_shape, embedding_coord_center)]

		if self.config.get('lego_loss') == "bce_supervised" and self.mode == "train":
			phase = LegoPhase(
				self.config,
				self.brick_generator,
				self.config['max_phase'],
				self.config['equilibrium_max_phase'],
				state_coord.clone(),
				deepcopy(state_pos),
				embedding_coord,
			)
		else:
			phase = Phase(
				self.config['max_phase'],
				self.config['equilibrium_max_phase'],
			)

		return {
			'input_pc': None,  # used for condition model
			'state0_coord': None,  # used for condition model
			'state0_feat': None,  # used for condition model
			'state_coord': state_coord,
			'state_feat': state_feat,
			'embedding_coord': embedding_coord,
			'embedding_feat': embedding_feat,
			'file_name': data_name,
			'phase': phase,
			"state0_pos": deepcopy(state_pos),
			"state_pos" : deepcopy(state_pos),
		}

	def test(self, model: TransitionModel, writer: SummaryWriter, step:int):
		training = model.training
		model.eval()

		# collect testset
		test_sample_nums = self.config['test_sample_nums']  # list

		data_loader = DataLoader(
			self,
			batch_size=self.config['test_batch_size'],
			num_workers=self.config['num_workers'],
			collate_fn=self.collate_fn,
			drop_last=False,
			shuffle=True
		)

		for test_step, data in tqdm(enumerate(data_loader)):
			batch_size = len(data['state_feat'])
			self.cache(model, data, data['file_name'], step)  # cache sparse voxel embedding
			torch.cuda.empty_cache()  # saves memory
			cache_dicts = model.load_cache(step, data['file_name'])
			for trial, cache_dicts_single_trial in enumerate(cache_dicts):
				s = model.cache_dicts2sparse_tensor(cache_dicts_single_trial)
				_, mesh_dict = model.get_pointcloud(s, test_sample_nums, return_mesh=True)

				mesh_save_dir = os.path.join(
					self.config['log_dir'], 'test_save',
					'step-{}'.format(step), 'mesh'
				)
				for k, meshes in mesh_dict.items():
					for batch_idx, mesh in enumerate(meshes):
						file_name = data['file_name'][batch_idx]
						os.makedirs(os.path.join(mesh_save_dir, k), exist_ok=True)
						mesh.export(os.path.join(mesh_save_dir, k, '{}_{}.obj'.format(file_name, trial)))

			# convert trials x batch_size -> batch_size x trials
			torch.cuda.empty_cache()

			cache_dir = os.path.join(
				self.config['log_dir'], 'test_save',
				'step-{}'.format(step), 'cache'
			)
			shutil.rmtree(cache_dir, ignore_errors=True)
			break

		model.write_dict_summaries(step)
		model.train(training)
