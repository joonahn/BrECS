import torch
import numpy as np
import os
import warnings
from utils.chamfer_distance import ChamferDistance
from typing import List
from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy
from scipy import linalg
import MinkowskiEngine as ME
from lego.brick_generator import sparseTensorToDense

chamfer_dist = ChamferDistance()


def compute_chamfer_dist(c1, c2, no_sqrt=True):
    with torch.no_grad():
        dist1, dist2 = chamfer_dist(c1, c2)

    if no_sqrt:
        return dist1.mean().item() + dist2.mean().item()
    else:
        return (dist1.sqrt().mean().item() + dist2.sqrt().mean().item()) / 2


class MMDCalculator:
    INF = 9999999.

    def __init__(self, testset):
        self.testset = testset.cpu()
        self.test_size = testset.shape[0]
        self.min_dists = torch.ones(self.test_size).float() * MMDCalculator.INF

    def add_generated_set(self, preds: torch.tensor):
        '''
        Args:
            preds: torch tensor of {test_trials} x {test_pred_downsample} x 3
            testset: torch tensor of {self.test_size} x {test_pred_downsample} x 3
        '''
        for pred in preds:
            # reshape tensor to {test_size} x {test_pred_downsample} x 3
            pred = pred.unsqueeze(0).expand(self.test_size, -1, -1)
            with torch.no_grad():
                dist1, dist2 = chamfer_dist(pred, self.testset.to(pred.device))
            dists = dist1.mean(dim=1).cpu() + dist2.mean(dim=1).cpu()  # {test_size}
            self.min_dists = torch.where(dists < self.min_dists, dists, self.min_dists)
        self.testset = self.testset.cpu()

    def calculate_mmd(self) -> float:
        return self.min_dists.mean().item()

    @property
    def dists(self) -> List[float]:
        return self.min_dists.tolist()

    def reset(self):
        self.min_dists = torch.ones(self.test_size).float() * MMDCalculator.INF


def compute_chamfer_l1(pred_set, gt):
    '''
    :param point_cloud1: torch.tensor of modality x M x 3 tensor
    :param point_cloud2: torch.tensor M x 3
    :return: list of chamfer l1 distances
    '''
    return [
        compute_chamfer_dist(
            pred.unsqueeze(0).float(),
            gt.unsqueeze(0).float(),
            no_sqrt=True
        ) for pred in pred_set
    ]


def mutual_difference(inferred_set):
    '''
    inferred_set : modality x M x 3 tensor (modality = 10, M = 2048)
    '''
    inferred_set = inferred_set.view(inferred_set.shape[0], 1, inferred_set.shape[1], -1)

    md = 0
    for j in range(inferred_set.shape[0]):
        for l in range(j + 1, inferred_set.shape[0], 1):
            md += compute_chamfer_dist(inferred_set[j], inferred_set[l], no_sqrt=True)

    return 2 * md / (inferred_set.shape[0] - 1)


def directed_hausdorff(point_cloud1, point_cloud2, reduce_mean=False):
    """
    point_cloud1: (B, N, 3) torch tensor
    point_cloud2: (B, M, 3) torch tensor
    return: directed hausdorff distance, pc1 -> pc2
    """
    n_pts1 = point_cloud1.shape[1]
    n_pts2 = point_cloud2.shape[1]
    pc1 = torch.transpose(point_cloud1, 1, 2)  # (B, 3, N)
    pc2 = torch.transpose(point_cloud2, 1, 2)  # (B, 3, M)
    pc1 = pc1.unsqueeze(3)
    pc1 = pc1.repeat((1, 1, 1, n_pts2))
    pc2 = pc2.unsqueeze(2)
    pc2 = pc2.repeat((1, 1, n_pts1, 1))

    l2_dist = torch.sqrt(torch.sum((pc1 - pc2) ** 2, dim=1))
    shortest_dist, _ = torch.min(l2_dist, dim=2)
    hausdorff_dist, _ = torch.max(shortest_dist, dim=1)

    if reduce_mean:
        hausdorff_dist = torch.mean(hausdorff_dist)

    return hausdorff_dist.item()


def directed_hausdorff_chamfer(point_cloud1, point_cloud2, reduce_mean=False):
    """
    point_cloud1: (B, N, 3) torch tensor
    point_cloud2: (B, M, 3) torch tensor
    return: directed hausdorff distance, pc1 -> pc2
    """
    dist1, dist2 = chamfer_dist(point_cloud1, point_cloud2)
    return dist1.max().sqrt().item()


def unidirected_hausdorff_distance(partial_input, pred_set, use_chamfer=True):
    '''
    Args:
        partial_input: torch.tensor of shape {test_input_downsample} x 3
        pred_set: torch.tensor of {trials} x {test_pred_downsample} x 3

    Returns:
        float output of the calculated score
    '''
    partial_input = partial_input.view(1, partial_input.shape[0], -1)
    pred_set = pred_set.view(pred_set.shape[0], 1, pred_set.shape[1], -1)
    uhd = 0
    for i in range(partial_input.shape[0]):
        if use_chamfer:
            uhd += directed_hausdorff_chamfer(partial_input, pred_set[i])
        else:
            uhd += directed_hausdorff(partial_input, pred_set[i])

    return uhd / pred_set.shape[0]


class FIDCalculator:
    def __init__(self, dataloader, eval_model, fid_n_data):
        self.eval_model = eval_model
        self.fid_n_data = fid_n_data
        with torch.no_grad():
            self.activations = []
            for batch in dataloader:
                s_coords = ME.utils.batched_coordinates(batch['embedding_coord'])
                s_feats = torch.ones(s_coords.shape[0], 1)
                s = ME.TensorField(
                    features=s_feats,
                    coordinates=s_coords,
                    device="cuda",
                )
                embeddings, logit = eval_model(s)
                for embedding in embeddings:
                    self.activations.append(embedding.detach().cpu().numpy())
        
        self.mu, self.cov = self.__calculate_dataset_stats(self.activations)

    def __calculate_dataset_stats(self, activations):
        mu = np.mean(activations, axis = 0)
        cov = np.cov(activations, rowvar = False)

        return mu, cov

    def calculate_fid(self, sparse_results) -> float:
        activations = []
        with torch.no_grad():
            for s_batch in sparse_results:
                s = ME.TensorField(
                    features=s_batch.F,
                    coordinates=s_batch.C,
                    device="cuda",
                )
                embeddings, logit = self.eval_model(s)
                for embedding in embeddings:
                    if len(activations) < self.fid_n_data:
                        activations.append(embedding.detach().cpu().numpy())

        assert len(activations) == self.fid_n_data

        mu_generated, cov_generated = self.__calculate_dataset_stats(activations)
        return self.compute_fid(self.mu, mu_generated, self.cov, cov_generated)

    def compute_fid(self, mu1, mu2, cov1, cov2, eps = 1e-6) -> float:
        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert cov1.shape == cov2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2
        # Product might be almost singular
        covmean, _ = linalg.sqrtm(cov1.dot(cov2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(cov1.shape[0]) * eps
            covmean = linalg.sqrtm((cov1 + offset).dot(cov2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                #raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(cov1) +
                np.trace(cov2) - 2 * tr_covmean)

class SingleIoUCalculator:
    def __init__(self, dataloader):
        self.target_voxel = None
        with torch.no_grad():
            for batch in dataloader:
                s_coords = ME.utils.batched_coordinates(batch['embedding_coord'])
                s_feats = torch.ones(s_coords.shape[0], 1)
                s = ME.TensorField(
                    features=s_feats,
                    coordinates=s_coords,
                    device="cuda",
                )
                self.target_voxel = sparseTensorToDense(s)[0, 0].detach().cpu().numpy()
                break
        assert self.target_voxel.shape == (64, 64, 64)
    
    def calculate_iou(self, sparse_results):
        iou_list = []
        with torch.no_grad():
            for s_batch in sparse_results:
                s = ME.TensorField(
                    features=s_batch.F,
                    coordinates=s_batch.C,
                    device="cuda",
                )
                s_dense_batch = sparseTensorToDense(s)[:, 0].detach().cpu().numpy()
                for brick_voxel in s_dense_batch:
                    brick_voxel.shape == (64, 64, 64)
                    iou = np.sum(np.logical_and(brick_voxel, self.target_voxel)) / np.sum(np.logical_or(brick_voxel, self.target_voxel))
                    iou_list.append(iou)
        
        return np.mean(iou_list)