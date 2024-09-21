from collections import defaultdict
import numpy as np
import torch
import MinkowskiEngine as ME
from MinkowskiEngine import SparseTensor
from torch.utils.tensorboard import SummaryWriter
from models.transition_model import TransitionModel
from utils.pad import unpack, get_gt_values
from utils.phase import LegoPhase, Phase
from utils.util import timeit
from utils.scheduler import InfusionScheduler
from typing import List, Tuple
from lego.brick_generator import BrickGenerator
from utils.marching_cube import marching_cubes_sparse_voxel

class LegoTransitionModel(TransitionModel):
    name = 'lego_transition'

    def __init__(self, config, writer: SummaryWriter):
        TransitionModel.__init__(self, config, writer)
        self.infusion_scheduler = InfusionScheduler(config)
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.brick_generator = BrickGenerator(config)
        self.sigmoid = torch.nn.Sigmoid()

    @timeit
    def forward(self, x):
        out_packed = self.backbone(x)
        out_unpacked = unpack(out_packed, self.shifts[:, 1:], self.out_dim)
        return out_unpacked

    @timeit
    def learn_sup(
        self, data:dict, step:float, mode:str='train'
    ):
        s_coords = ME.utils.batched_coordinates(data['state_coord'])
        s_feats = torch.ones(s_coords.shape[0], 1)
        s = SparseTensor(
            features=s_feats,
            coordinates=s_coords,
            device=self.device,
        )

        y_coords = ME.utils.batched_coordinates(data['embedding_coord'])
        y_feats = torch.ones(y_coords.shape[0], 1)
        y = SparseTensor(
            features=y_feats,
            coordinates=y_coords,
            device=self.device,
        )
        # forward pass
        s_hat = self.forward(s)

        # variables
        losses = []
        phases:List[LegoPhase] = data['phase']
        batch_size = len(phases)

        # retrieve s_next from phases
        s_next_coords = []
        for batch_idx, phase in enumerate(phases):
            s_next_coords.append(phase.target_voxel.type(torch.IntTensor))
            
        s_next_coord = ME.utils.batched_coordinates(s_next_coords)
        s_next_feat = torch.ones(s_next_coord.shape[0], 1)
        s_next = ME.SparseTensor(
            s_next_feat, s_next_coord,
            device=self.device
        )

        # get gt
        one_hot_gt, y_pad_coords = get_gt_values(s_hat, s_next)

        # loss and phase calculation
        for batch_idx, phase in enumerate(phases):
            # compute loss
            idx = s_hat.C[:, 0] == batch_idx
            s_hat_feat = s_hat.F[idx, :]
            losses.append(self.bce_loss(s_hat_feat.squeeze(1), one_hot_gt[batch_idx].float()))

            # update_phases
            phases[batch_idx] = phase + 1
            if phase.saturated:
                if not phase.equilibrium_mode:
                    phase.set_complete()
                    self.list_summaries['completion_phase/{}'.format(mode)] += [phase.phase]
            elif (phase.phase > self.config['max_phase']) and (mode == 'train'):
                incomplete_key = 'phase/incomplete_cnt'
                self.scalar_summaries[incomplete_key] = [self.scalar_summaries[incomplete_key][0] + 1] if \
                    len(self.scalar_summaries[incomplete_key]) != 0 else [1]

        loss = torch.stack(losses).mean()
        data['state_coord'] = [phase.current_voxel.type(torch.IntTensor) for phase in phases]

        # write summaries
        self.scalar_summaries['loss/{}/total'.format(mode)] += [loss.item()]
        self.list_summaries['loss/{}/total_histogram'.format(mode)] += torch.stack(losses).cpu().tolist()
        self.scalar_summaries['num_points/input'] += [(s.C[:, 0] == i).sum().item() for i in range(batch_size)]
        self.scalar_summaries['num_points/output'] += [one_hot_gt[i].shape[0] for i in range(batch_size)]

        if mode != 'train':
            return loss.detach().cpu().item(), data

        # take gradient descent
        self.zero_grad()
        loss.backward()
        self.clip_grad()
        self.optimizer.step()
        self.lr_scheduler.step()

        return loss.detach().cpu().item(), data

    @timeit
    def learn(
            self, data: dict,
            step: float, mode: str = 'train'
    ) -> Tuple[dict, float]:
        """
        :param data: dict containing key, value pairs of
            - state_coord: Tensor containing coordinates of input voxels
            - state_feat: Tensor containing features of input voxels
            - query_point: Tensor of B x N x data_dim
            - dist: Tensor of B x N x data_dim
            - phase: List of phases for each data
        :param step: training step
        :param mode: mode of training
        :return:
            - next_step: dict containing same keys and values as parameter data
            - loss: float of the current step's lose
        """

        s_coords = ME.utils.batched_coordinates(data['state_coord'])
        s_feats = torch.ones(s_coords.shape[0], 1)
        s = SparseTensor(
            features=s_feats,
            coordinates=s_coords,
            device=self.device,
        )

        y_coords = ME.utils.batched_coordinates(data['embedding_coord'])
        y_feats = torch.ones(y_coords.shape[0], 1)
        y = SparseTensor(
            features=y_feats,
            coordinates=y_coords,
            device=self.device,
        )
        # forward pass
        s_hat = self.forward(s)

        # compute loss
        losses = []
        one_hot_gt, y_pad_coords = get_gt_values(s_hat, y)
        if self.config.get('lego_complete', False):
            one_hot_lego_gt, _ = get_gt_values(s, y)

        phases = data['phase']
        batch_size = len(phases)
        if self.config.get('lego_sample_feat', True):
            feats = self.sample_feat(s_hat.F)
        else:
            feats = s_hat.F[..., 0]
            if self.config.get('lego_model_sigmoid', False):
                feats = self.sigmoid(feats)
        infusion_rates = self.infusion_scheduler.sample(phases)

        s_next_feats = []
        s_next_coords = []
        A_sampled_list:List[ME.SparseTensor] = []
        A_gt_list:List[ME.SparseTensor] = []
        phases_list:List[int] = list(phase.phase for phase in phases)
        completion_rates = []

        for batch_idx, (infusion_rate, phase) in enumerate(zip(infusion_rates, phases)):
            # compute loss
            idx = s_hat.C[:, 0] == batch_idx
            s_hat_feat = s_hat.F[idx, :]
            if self.config.get('lego_loss') is None:
                losses.append(self.bce_loss(s_hat_feat.squeeze(1), one_hot_gt[batch_idx].float()))

            # infusion training
            feat = feats[idx]
            coord = s_hat.C[idx, :]

            A_sampled = ME.SparseTensor(features=feat[..., None].type(torch.FloatTensor), coordinates=ME.utils.batched_coordinates([coord[:, 1:].type(torch.IntTensor)]), device="cuda")
            A_gt = ME.SparseTensor(features=one_hot_gt[batch_idx][..., None].type(torch.FloatTensor), coordinates=ME.utils.batched_coordinates([coord[:, 1:].type(torch.IntTensor)]), device="cuda")

            if 'lego_test' in self.config and self.config['lego_test']:
                A_sampled_list.append(A_gt)
                A_gt_list.append(A_gt)
            else:
                A_sampled_list.append(A_sampled)
                A_gt_list.append(A_gt)

            # update_phases
            phases[batch_idx] = phase + 1
            if self.config.get('lego_complete', False):
                completion_rate = one_hot_lego_gt[batch_idx].sum().item() \
                              / float((y.C[:, 0] == batch_idx).sum().item())
            else:
                completion_rate = one_hot_gt[batch_idx].sum().item() \
                              / float((y.C[:, 0] == batch_idx).sum().item())
            completion_rates.append(completion_rate)
            if completion_rate >= self.config['completion_rate']:
                if not phase.equilibrium_mode:
                    phase.set_complete()
                    self.list_summaries['completion_phase/{}'.format(mode)] += [phase.phase]
            elif (phase.phase > self.config['max_phase']) and (mode == 'train'):
                incomplete_key = 'phase/incomplete_cnt'
                self.scalar_summaries[incomplete_key] = [self.scalar_summaries[incomplete_key][0] + 1] if \
                    len(self.scalar_summaries[incomplete_key]) != 0 else [1]

        if self.config.get('lego_loss') is None: 
            s_next_coords = self.brick_generator.generate_next_state_parallel(A_sampled_list, A_gt_list, data['state0_pos'], infusion_rates, phases_list)
        else:
            s_next_coords, data['state_pos'], losses = self.brick_generator.generate_next_state_parallel_with_loss(A_sampled_list, A_gt_list, data['state_pos'], infusion_rates)

        loss = torch.stack(losses).mean() if len(losses) > 0 else torch.tensor(0.0, dtype=torch.float32, device="cuda", requires_grad=True)
        # print("[debug] max phase:", np.mean([phase.phase for phase in phases]))
        data['state_coord'] = s_next_coords

        self.scalar_summaries['completion_rate'] = np.mean(completion_rates)

        # write summaries
        if len(losses) > 0:
            self.scalar_summaries['loss/{}/total'.format(mode)] += [loss.item()]
            self.list_summaries['loss/{}/total_histogram'.format(mode)] += torch.stack(losses).cpu().tolist()
            self.scalar_summaries['num_points/input'] += [(s.C[:, 0] == i).sum().item() for i in range(batch_size)]
            self.scalar_summaries['num_points/output'] += [one_hot_gt[i].shape[0] for i in range(batch_size)]
            self.list_summaries['scheduler/infusion_rates'] += infusion_rates

        if mode != 'train':
            return loss.detach().cpu().item(), data

        # take gradient descent
        self.zero_grad()
        loss.backward()
        self.clip_grad()
        self.optimizer.step()
        self.lr_scheduler.step()

        return loss.detach().cpu().item(), data

    def transition(self,
                   s: SparseTensor,
                   brick_positions:List[List[np.ndarray]],
                   remained_budget:List[List[int]]
                   ) -> Tuple[SparseTensor, List[List[np.ndarray]]]:
        y_hat = self.forward(s)
        if self.config.get('lego_sample_feat', True):
            feat_sample = self.sample_feat(y_hat.F)
        else:
            feat_sample = y_hat.F[..., 0]
            if self.config.get('lego_model_sigmoid', False):
                feat_sample = self.sigmoid(feat_sample)
                
        s_next_coord = y_hat.C[feat_sample.bool(), :]
        s_next_feat = feat_sample[feat_sample.bool()]

        s_next_coords = []
        batch_size = s.C[:, 0].max().item() + 1
        A_sampled_list:List[ME.SparseTensor] = []
        for batch_idx in range(batch_size):
            idx = s_next_coord[:, 0] == batch_idx
            coord = s_next_coord[idx]
            feat = s_next_feat[idx]

            A_sampled = ME.SparseTensor(features=feat[:, None].type(torch.FloatTensor), coordinates=ME.utils.batched_coordinates([coord[:, 1:].type(torch.IntTensor)]), device="cuda")
            A_sampled_list.append(A_sampled)

        no_pivot = self.config.get("lego_no_pivot", False)
        if no_pivot:
            B_next_list, next_brick_positions = self.brick_generator.transition_parallel_no_pivot(A_sampled_list, brick_positions, remained_budget)
        else:
            B_next_list, next_brick_positions = self.brick_generator.transition_parallel(A_sampled_list, brick_positions, remained_budget)
        
        for batch_idx in range(batch_size):
            if B_next_list[batch_idx].shape[0] == 0:
                idx = s.C[:, 0] == batch_idx
                s_next_coords.append(s.C[idx, 1:].type(torch.IntTensor))
            else:
                s_next_coords.append(B_next_list[batch_idx].C[:, 1:].type(torch.IntTensor))

        s_next_coord = ME.utils.batched_coordinates(s_next_coords)
        s_next_feat = torch.ones(s_next_coord.shape[0], 1)
        try:
            s_next = SparseTensor(
                s_next_feat, s_next_coord,
                device=self.device
            )
        except RuntimeError:
            breakpoint()
        return s_next, next_brick_positions

    def transition_with_gt(self,
                           s: SparseTensor,
                           brick_positions:List[List[np.ndarray]],
                           data,
                           remained_budget:List[List[int]]=None):
        y_coords = ME.utils.batched_coordinates(data['embedding_coord'])
        y_feats = torch.ones(y_coords.shape[0], 1)
        y = SparseTensor(
            features=y_feats,
            coordinates=y_coords,
            device=self.device,
        )

        feat_sample = y.F[..., 0]
                
        s_next_coord = y.C[feat_sample.bool(), :]
        s_next_feat = feat_sample[feat_sample.bool()]

        s_next_coords = []
        batch_size = s.C[:, 0].max().item() + 1
        A_sampled_list:List[ME.SparseTensor] = []
        for batch_idx in range(batch_size):
            idx = s_next_coord[:, 0] == batch_idx
            coord = s_next_coord[idx]
            feat = s_next_feat[idx]

            A_sampled = ME.SparseTensor(features=feat[:, None].type(torch.FloatTensor), coordinates=ME.utils.batched_coordinates([coord[:, 1:].type(torch.IntTensor)]), device="cuda")
            A_sampled_list.append(A_sampled)

        B_next_list, next_brick_positions = self.brick_generator.transition_parallel(A_sampled_list, brick_positions, remained_budget)
        
        for batch_idx in range(batch_size):
            if B_next_list[batch_idx].shape[0] == 0:
                idx = s.C[:, 0] == batch_idx
                s_next_coords.append(s.C[idx, 1:].type(torch.IntTensor))
            else:
                s_next_coords.append(B_next_list[batch_idx].C[:, 1:].type(torch.IntTensor))

        s_next_coord = ME.utils.batched_coordinates(s_next_coords)
        s_next_feat = torch.ones(s_next_coord.shape[0], 1)
        try:
            s_next = SparseTensor(
                s_next_feat, s_next_coord,
                device=self.device
            )
        except RuntimeError:
            breakpoint()
        return s_next, next_brick_positions

    def evaluate(self, data, step, dataset_mode) -> float:
        max_eval_phase = self.config['max_eval_phase']
        losses = []
        for mode in ['eval_infusion']:
            data_next = data
            for p in range(max_eval_phase):
                loss, data_next = self.learn(data_next, step, mode=mode)
                losses.append(loss)
        return sum(losses) / float(len(losses))

    def get_pointcloud(self, s: SparseTensor, sample_nums: List, return_mesh=True):
        ret = defaultdict(list)
        meshes = defaultdict(list)
        for batch_idx in range(s.C[:, 0].max().item() + 1):
            idx = s.C[:, 0] == batch_idx
            coord = s.C[idx, 1:]
            mesh = marching_cubes_sparse_voxel(coord, voxel_size=self.voxel_size)
            meshes['initial_mesh'] += [mesh]
            
        if return_mesh:
            return ret, meshes
        return ret
