import numpy as np
import torch
import MinkowskiEngine as ME
from MinkowskiEngine import SparseTensor
from torch.utils.tensorboard import SummaryWriter
from models.transition_model import TransitionModel
from utils.pad import unpack, get_gt_values
from utils.util import timeit
from utils.scheduler import InfusionScheduler
from typing import List, Tuple
from lego.brick_generator import generate_next_state_brick, generate_next_state_brick_parallel, sparseTensorToDense

class LegoTransitionModel(TransitionModel):
    name = 'lego_transition'

    def __init__(self, config, writer: SummaryWriter):
        TransitionModel.__init__(self, config, writer)
        self.infusion_scheduler = InfusionScheduler(config)
        self.bce_loss = torch.nn.BCEWithLogitsLoss()

    @timeit
    def forward(self, x):
        out_packed = self.backbone(x)
        out_unpacked = unpack(out_packed, self.shifts[:, 1:], self.out_dim)
        return out_unpacked

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

        phases = data['phase']
        batch_size = len(phases)
        feats = self.sample_feat(s_hat.F)
        infusion_rates = self.infusion_scheduler.sample(phases)

        s_next_feats = []
        s_next_coords = []
        A_sampled_list:List[np.ndarray] = []
        A_gt_list:List[np.ndarray] = []
        phases_list:List[int] = list(phase.phase for phase in phases)

        for batch_idx, (infusion_rate, phase) in enumerate(zip(infusion_rates, phases)):
            # compute loss
            idx = s_hat.C[:, 0] == batch_idx
            s_hat_feat = s_hat.F[idx, :]
            losses.append(self.bce_loss(s_hat_feat.squeeze(1), one_hot_gt[batch_idx].float()))

            # infusion training
            feat = feats[idx]
            coord = s_hat.C[idx, :]

            A_sampled = ME.SparseTensor(features=feat[..., None].type(torch.FloatTensor), coordinates=ME.utils.batched_coordinates([coord[:, 1:].type(torch.IntTensor)]), device="cuda")
            A_gt = ME.SparseTensor(features=one_hot_gt[batch_idx][..., None].type(torch.FloatTensor), coordinates=ME.utils.batched_coordinates([coord[:, 1:].type(torch.IntTensor)]), device="cuda")

            # B_next = generate_next_state_brick(A_sampled, A_gt, infusion_rate, phases[batch_idx].phase)

            # s_next_coords.append(B_next.C[..., 1:].cpu())

            # infusion_idx = (torch.rand(feat.shape[0]) < infusion_rate)
            # s_next_feat = generate_next_state_brick()
            # s_next_feat = torch.where(
            #     infusion_idx, one_hot_gt[batch_idx].float().cpu(), feat.cpu()
            # )
            # s_next_coords.append(coord[s_next_feat.bool(), 1:].cpu())


            # s_next_feats.append(torch.ones(s_next_coords[batch_idx].shape[0], 1).cpu())


            A_sampled_list.append(sparseTensorToDense(A_sampled).detach().cpu().numpy())
            A_gt_list.append(sparseTensorToDense(A_gt).cpu().numpy())

            # update_phases
            phases[batch_idx] = phase + 1
            completion_rate = one_hot_gt[batch_idx].sum().item() \
                              / float((y.C[:, 0] == batch_idx).sum().item())
            if completion_rate >= self.config['completion_rate']:
                if not phase.equilibrium_mode:
                    phase.set_complete()
                    self.list_summaries['completion_phase/{}'.format(mode)] += [phase.phase]
            elif (phase.phase > self.config['max_phase']) and (mode == 'train'):
                incomplete_key = 'phase/incomplete_cnt'
                self.scalar_summaries[incomplete_key] = [self.scalar_summaries[incomplete_key][0] + 1] if \
                    len(self.scalar_summaries[incomplete_key]) != 0 else [1]

        s_next_coords = generate_next_state_brick_parallel(A_sampled_list, A_gt_list, infusion_rates, phases_list)

        loss = torch.stack(losses).mean()
        data['state_coord'] = s_next_coords

        # write summaries
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

    def transition(self, s: SparseTensor, sigma=None) -> SparseTensor:
        y_hat = self.forward(s)
        feat_sample = self.sample_feat(y_hat.F)
        s_next_coord = y_hat.C[feat_sample.bool(), :]

        # if the sampled output contains no coords
        batch_size = s.C[:, 0].max().item() + 1
        for batch_idx in range(batch_size):
            if (s_next_coord[:, 0] == batch_idx).shape[0] == 0:
                if s_next_coord[:, 0].shape[0] == 0:
                    s_next_coord = torch.zeros(1, 4).int().to(s_next_coord.device)
                else:
                    s_next_coord = torch.stack([
                        s_next_coord,
                        torch.tensor([[batch_idx] + [0, ] * self.config['data_dim']]).int().to(s_next_coord.device)
                    ], dim=0)

        s_next_feat = torch.ones(s_next_coord.shape[0], 1)
        try:
            s_next = SparseTensor(
                s_next_feat, s_next_coord,
                device=self.device
            )
        except RuntimeError:
            breakpoint()
        return s_next

    def evaluate(self, data, step, dataset_mode) -> float:
        max_eval_phase = self.config['max_eval_phase']
        losses = []
        for mode in ['eval_infusion']:
            data_next = data
            for p in range(max_eval_phase):
                loss, data_next = self.learn(data_next, step, mode=mode)
                losses.append(loss)
        return sum(losses) / float(len(losses))