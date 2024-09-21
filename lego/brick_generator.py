from copy import deepcopy
import math
import random
import time
from turtle import position
from typing import Dict, List, Tuple, Union
from lego.LegoLogic import LegoMnistBoard, MultiBrickTypeLegoBoard, SimpleLegoBoard
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import MinkowskiEngine as ME
from torch.multiprocessing import Pool, set_start_method
# try:
#      set_start_method('spawn')
# except RuntimeError:
#     pass
import contextlib

class DotDict(dict):
    """dot.notation access to dictionary attributes"""

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError from e

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def sparseTensorToDense(sparse_tensor:ME.SparseTensor):
    min_coord = torch.tensor([[-31, -31, -31]], dtype=torch.int32, device="cuda")
    max_coord = torch.tensor([[32, 32, 32]], dtype=torch.int32, device="cuda")
    coord = sparse_tensor.C
    in_bound_idx = torch.sum((min_coord <= coord[..., 1:]) & (coord[..., 1:] <= max_coord), dim=1) == 3
    feat = sparse_tensor.F
    merged = ME.SparseTensor(feat[in_bound_idx], coord[in_bound_idx] + torch.tensor([0,31,31,31], dtype=torch.int32, device="cuda"))
    return merged.dense(shape=torch.Size([1,1,64,64,64]))[0]

def denseArrayToSparse(dense_tensor:np.ndarray):
    coords = np.concatenate([x[..., None] for x in np.where(dense_tensor != 0.0)], axis=-1) - np.array([31, 31, 31])
    coords = [coords]
    coords = ME.utils.batched_coordinates(coords)
    feats = torch.ones(coords.shape[0], 1)
    return ME.SparseTensor(
        features = feats,
        coordinates = coords,
        device="cuda"
    )

def clip(value, min_value, max_value):
    return max(min(value, max_value), min_value)

def reposition(coordinate):
    prev_x, y, prev_z, prev_dir = coordinate

    # rad = 3/2 * math.pi
    rad = 1/2 * math.pi

    new_x = math.cos(rad) * prev_x - math.sin(rad) * prev_z
    new_z = math.sin(rad) * prev_x + math.cos(rad) * prev_z

    new_dir = (prev_dir + 1) % 2

    return np.array([round(new_x), y, round(new_z), new_dir], dtype = 'int')

def normalize(array):
    sum_all = sum(array)
    return [x / sum_all for x in array]

def get_pivot_scores_and_indices_differentiable(
        w_b:int,
        h_b:int,
        C:Union[torch.Tensor, np.ndarray],
        X:np.ndarray,
        arranging_coordinate:np.ndarray,
        filter_nonzero:bool=False,
        reduction:str='sum') -> Tuple[List[torch.Tensor], List[int]]:
    """
    Calculates pivot scores and returns scores and corresponding indices

    Args:
        C (torch.Tensor): shape of (D, D, D)
        X (np.ndarray): shape of (N, 4)
    """
    assert C.shape[0] == C.shape[1] == C.shape[2]

    is_numpy = (type(C) == np.ndarray)

    pivot_scores = []
    pivot_idxs = []
    for pivot in range(len(X)):
        pivot_size, pivot_coordinate = X[pivot]

        arranged_x, arranged_y, arranged_z = np.multiply(np.array(pivot_coordinate, dtype = 'int'), np.array([1,1,1])) + \
                                                        np.array(arranging_coordinate, dtype = 'int')

        # get pivot size
        # if dir == 0:
        #     w_p, h_p = 2, 4
        # elif dir == 1:
        #     w_p, h_p = 4, 2
        w_p, h_p = pivot_size

        # set pivot brick range
        voxel_size = C.shape[0]
        offset = 7
        pivot_x_min = clip(arranged_x - (((w_b + w_p) // 2) - 1), offset, voxel_size - offset)
        pivot_x_max = clip(arranged_x + (((w_b + w_p) // 2) - 1), offset, voxel_size - offset)
        pivot_y_lower = arranged_y - 1 if (offset <= (arranged_y - 1) <= voxel_size - offset) else None
        pivot_y_upper = arranged_y + 1 if (offset <= (arranged_y + 1) <= voxel_size - offset) else None
        pivot_z_min = clip(arranged_z - (((h_b + h_p) // 2) - 1), offset, voxel_size - offset)
        pivot_z_max = clip(arranged_z + (((h_b + h_p) // 2) - 1), offset, voxel_size - offset)

        # filter out clipped range
        if pivot_x_min == pivot_x_max or pivot_z_min == pivot_z_max:
            continue

        reduce_fcn = None
        if is_numpy:
            if reduction == 'sum':
                reduce_fcn = np.sum
            elif reduction == 'max':
                reduce_fcn = np.max
            elif reduction == 'mean':
                reduce_fcn = np.mean
        else:
            if reduction == 'sum':
                reduce_fcn = torch.sum
            elif reduction == 'max':
                reduce_fcn = lambda x:torch.max(x)[0]
            elif reduction == 'mean':
                reduce_fcn = torch.mean

        # aggregate pivot scores
        pivot_score = 0
        pivot_score += reduce_fcn(C[pivot_x_min:pivot_x_max+1,
                pivot_y_lower,
                pivot_z_min:pivot_z_max+1]) if pivot_y_lower is not None else 0
        pivot_score += reduce_fcn(C[pivot_x_min:pivot_x_max+1,
                pivot_y_upper,
                pivot_z_min:pivot_z_max+1]) if pivot_y_upper is not None else 0

        if not filter_nonzero or pivot_score > 0:
            pivot_scores.append(pivot_score)
            pivot_idxs.append(pivot)

    return pivot_scores, pivot_idxs

def get_pivot_scores_and_indices(w_b:int, h_b:int, C:Union[torch.Tensor, np.ndarray] , X:np.ndarray, arranging_coordinate:np.ndarray, reduction:str='sum') -> Tuple[List[float], List[int]]:
    """
    Calculates pivot scores and returns scores and corresponding indices

    Args:
        C (torch.Tensor): shape of (D, D, D)
        X (np.ndarray): shape of (N, 4)
    """
    is_numpy = (type(C) == np.ndarray)
    pivot_scores = []
    pivot_idxs = []

    pivot_scores_from, pivot_idxs_from = get_pivot_scores_and_indices_differentiable(w_b, h_b, C, X, arranging_coordinate, filter_nonzero=True, reduction=reduction)

    for pivot_score, pivot_idx in zip(pivot_scores_from, pivot_idxs_from):
        pivot_scores.append(pivot_score.item())
        pivot_idxs.append(pivot_idx)

    return pivot_scores, pivot_idxs

def get_connection_scores_and_indices_differentiable(w_b:int, h_b:int, C:torch.Tensor, pivot_idx:int, X:np.ndarray, rules, arranging_coordinate, no_bottom=False, filter_nonzero=False):
    """
    Calculates connection scores and returns scores and corresponding connection indices

    Args:
        C (torch.Tensor): shape of (D, D, D)
        X (np.ndarray): shape of (N, 4)
    """
    move_scores = []
    move_list = []
    pivot_coordinate = X[pivot_idx]
    pivot_size, pivot_position = pivot_coordinate
    w_p, h_p = pivot_size

    scratch_pad = np.zeros((20,2,20), dtype=np.int32)

    height_diff_list = [0] if no_bottom else [0, 1]

    for height_diff in height_diff_list:
        for x_diff in range(-(((w_b + w_p) // 2) - 1), (((w_b + w_p) // 2) - 1) + 1):
            for z_diff in range(-(((h_b + h_p) // 2) - 1), (((h_b + h_p) // 2) - 1) + 1):
                positional_change = np.array([x_diff, 1 - 2 * height_diff, z_diff])
                new_brick_coordinate = pivot_position + positional_change

                arranged_x, arranged_y, arranged_z = np.array(new_brick_coordinate, dtype = 'int') + \
                                                                np.array(arranging_coordinate, dtype = 'int')[:3]
                
                offset = 3
                voxel_size = C.shape[0]

                if not all(offset <= value < voxel_size - offset for value in [arranged_x - math.ceil(w_b / 2), arranged_x + (w_b // 2), arranged_y, arranged_z - math.ceil(h_b / 2), arranged_z + (h_b // 2)]):
                    continue
                
                move_score = C[arranged_x, arranged_y, arranged_z]

                scratch_pad[positional_change[0]+10,height_diff,positional_change[2]+10] = 1

                if not filter_nonzero or move_score != 0:
                    move_scores.append(move_score)
                    move_list.append(((w_b, h_b), (positional_change[0], positional_change[1], positional_change[2])))
    
    return move_scores, move_list

def get_connection_scores_and_indices(w_b:int, h_b:int, C:torch.Tensor, pivot_idx:int, X:np.ndarray, rules, arranging_coordinate, no_bottom=False):
    """
    Calculates connection scores and returns scores and corresponding connection indices

    Args:
        C (torch.Tensor): shape of (D, D, D)
        X (np.ndarray): shape of (N, 4)
    """
    move_scores = []
    move_idxs = []

    move_scores_from, move_idxs_from = get_connection_scores_and_indices_differentiable(w_b, h_b, C, pivot_idx, X, rules, arranging_coordinate, no_bottom, filter_nonzero=True)
    
    for move_score, move_idx in zip(move_scores_from, move_idxs_from):
        move_scores.append(move_score.item())
        move_idxs.append(move_idx)

    return move_scores, move_idxs

def select_pivot_and_connection(args:Tuple[int, List[Tuple[int]], Dict[Tuple[int], np.ndarray], Dict[Tuple[int], np.ndarray], List[np.ndarray], np.ndarray, List, bool, bool, float]):
    batch_idx, brick_list, C, C_rand, brick_info_list, translation, rules, no_random, only_argmax, epsilon, budget, budget_scaling = args

    eps = 0.00
    pivot_info = dict()
    for brick_size in brick_list:
        budget_scaler = 1.0
        if budget is not None:
            if budget_scaling:
                budget_scaler = brick_size[0] * brick_size[1] * budget[brick_size] + eps
            else:
                budget_scaler = float(budget[brick_size] > 0)
        pivot_scores, pivot_indices = get_pivot_scores_and_indices(*brick_size, C[brick_size], brick_info_list, translation)
        for pivot_score, pivot_index in zip(pivot_scores, pivot_indices):
            pivot_info[pivot_index] = (pivot_info.get(pivot_index, 0.0) + (pivot_score * budget_scaler))

    pivot_indices = list(pivot_info.keys())
    pivot_scores = [pivot_info[pivot] for pivot in pivot_indices]
    # negative filtering
    pivot_scores = [(x if x >= 0 else 0) for x in pivot_scores]

    if sum(pivot_scores) == 0:
        return batch_idx, None, None, None

    # sampling pivot
    if only_argmax:
        pivot:int = pivot_indices[np.argmax(normalize(pivot_scores))]
    else:
        is_random = random.random() < epsilon
        if is_random:
            pivot:int = np.random.choice(pivot_indices)
        else:    
            pivot:int = np.random.choice(pivot_indices, p=normalize(pivot_scores))

    rule_score_list = []
    rule_merged_list = []
    rule_brick_map = []
    for brick_size in brick_list:
        budget_scaler = 1.0
        if budget is not None:
            budget_scaler = brick_size[0] * brick_size[1] * budget[brick_size] + eps
        rule_scores, rule_list = get_connection_scores_and_indices(*brick_size, C[brick_size], pivot, brick_info_list, rules, translation)
        rule_scores = [x * budget_scaler for x in rule_scores]
        rule_score_list.extend(rule_scores)
        rule_merged_list.extend(rule_list)
        rule_brick_map.extend([brick_size] * len(rule_list))

    rule_score_list = [(x if x >= 0 else 0) for x in rule_score_list]

    # sampling rule
    rule_idx = np.argmax(rule_score_list)
    rule:int = rule_merged_list[rule_idx]
    selected_brick_size = rule_brick_map[rule_idx]

    return batch_idx, pivot, rule, selected_brick_size

# quantize_C_gt(C_gt_batch[brick_size], brick_position, brick_translation)
def quantize_C_gt(C:torch.Tensor, score:torch.Tensor, brick_size:Tuple, brick_info_list:List[Tuple], brick_translation:np.ndarray, fill_neg=False, quantize_thresh=None):
    # Shape of C: (64,64,64)
    C = C.detach().clone()
    C = C.reshape((-1,))
    w_b, h_b = brick_size

    # thresholding
    if quantize_thresh is not None:
        C[(C < quantize_thresh * w_b * h_b) & (C > 0)] = 0.0
        C[C > quantize_thresh * w_b * h_b] = 1.0
    else:
        C[C > 0] = 1.0

    C = C.reshape((64, 64, 64))
    
    if fill_neg:
        C[C <= 0] = -1.0
        return C

    for brick_info in brick_info_list:
        _, brick_pos = brick_info
        arranged_x, arranged_y, arranged_z = np.multiply(np.array(brick_pos, dtype = 'int'), np.array([1,1,1])) + \
                                                        np.array(brick_translation, dtype = 'int')
        if score[arranged_x, arranged_y, arranged_z] == 0.0:
            C[arranged_x, arranged_y, arranged_z] = -1.0

    return C

def thresholding_C(C:torch.Tensor, brick_size, thresh):
    if thresh is None:
        return C

    if type(C) == torch.Tensor:
        C = C.detach().clone()
        
    C = C.reshape((-1,))
    w_b, h_b = brick_size

    # thresholding
    C[(C < thresh * w_b * h_b) & (C > 0)] = 0.0

    C = C.reshape((64, 64, 64))

    return C

def get_brick_range(brick_info:Tuple):
    brick_size, brick_pos = brick_info
    w, h = brick_size
    x_min = brick_pos[0] - w / 2
    x_max = brick_pos[0] + w / 2
    z_min = brick_pos[2] - h / 2
    z_max = brick_pos[2] + h / 2
    return x_min, x_max, z_min, z_max

def is_brick_range_intersect(brick1_range, brick2_range):
    x1_min, x1_max, z1_min, z1_max = brick1_range
    x2_min, x2_max, z2_min, z2_max = brick2_range

    if (x1_min >= x2_max or x1_max <= x2_min):
        return False

    if (z1_min >= z2_max or z1_max <= z2_min):
        return False

    return True

def is_connected(brick1_info, brick2_info):
    _, brick_pos1 = brick1_info
    _, brick_pos2 = brick2_info
    brick1_range = get_brick_range(brick1_info)
    brick2_range = get_brick_range(brick2_info)
    y1 = brick_pos1[1]
    y2 = brick_pos2[1]
    if abs(y1 - y2) == 1 and is_brick_range_intersect(brick1_range, brick2_range):
        return True
    return False

def get_graph_from_pos(brick_info_list:List[Tuple]):
    edges = {idx: set() for idx in range(len(brick_info_list))}
    for idx1, brick1_info in enumerate(brick_info_list):
        for idx2, brick2_info in enumerate(brick_info_list):
            if idx1 == idx2:
                continue
            if is_connected(brick1_info, brick2_info):
                edges[idx1].add(idx2)
                edges[idx2].add(idx1)
    return edges

# get_pivot_to_remove(self.brick_list, C_mix_batch, brick_position, brick_translation)
def get_pivot_to_remove(brick_list, C, brick_info_list, brick_translation):
    pivot_scores = []
    edges = get_graph_from_pos(brick_info_list)
    for brick_idx, brick_info in enumerate(brick_info_list):
        brick_size, brick_pos = brick_info
        arranged_x, arranged_y, arranged_z = np.multiply(np.array(brick_pos, dtype = 'int'), np.array([1,1,1])) + \
                                                np.array(brick_translation, dtype = 'int')
        score = C[brick_size][arranged_x, arranged_y, arranged_z]
        if len(edges[brick_idx]) != 1:
            score = float("inf")
        pivot_scores.append(score.item() if type(score)==torch.Tensor else score)

    score_min = min(pivot_scores)

    if score_min < 0:
        return np.argmin(pivot_scores)
    
    return None

def serial_3d_conv(conv_size,voxel):
    output_vox = torch.zeros_like(voxel)
    for x in range(voxel.shape[0]):
        x_min = max(0, x - conv_size[0]//2 + 1)
        x_max = min(voxel.shape[0], x + math.ceil(conv_size[0]/2) + 1)
        for y in range(voxel.shape[1]):
            y_min = max(0, y - conv_size[1]//2)
            y_max = min(voxel.shape[1], y + math.ceil(conv_size[1]/2))
            for z in range(voxel.shape[2]):
                z_min = max(0, z - conv_size[2]//2 + 1)
                z_max = min(voxel.shape[2], z + math.ceil(conv_size[2]/2) + 1)
                
                # if y == 0:
                #     print(x, y, z)
                #     print(x_min,x_max, y_min,y_max, z_min,z_max)
                val = torch.sum(voxel[x_min:x_max, y_min:y_max, z_min:z_max]) 
                output_vox[x, y, z] = val
    return output_vox

class BrickGenerator:
    def __init__(self, config):
        self.parallel = config.get("lego_parallel", True)
        self.brick_list = config.get('lego_brick_list', [(2,4), (4,2)])
        self.brick_list = [tuple(brick_size) for brick_size in self.brick_list]
        print(f"[debug] self.brick_list:{self.brick_list}")
        self.config = config
        if self.parallel:
            self.pool = Pool(config['batch_size'])
            print("[debug] paralle mode")

        self.thresh = config.get('lego_thresh')
        self.quantize_thresh = config.get('lego_quantize_thresh')
        self.pred_thresh = config.get('lego_pred_thresh')
        self.no_random = config.get('lego_no_random')
        self.only_argmax = config.get('lego_only_argmax', False)
        self.n_bricks_every_step = config['n_bricks_every_step']
        self.lego_max_n_brick = config.get("lego_max_n_brick")
        self.no_pivot = config.get("lego_no_pivot", False)
        self.no_squeeze_in = config.get("lego_no_squeeze_in", False)
        self.stable_n_pivot_cand = config.get("lego_stable_n_pivot_cand")
        self.stable_n_rule_cand = config.get("lego_stable_n_rule_cand")
        self.stable_use_mask = config.get("lego_stable_use_mask", False)
        self.epsilon = config.get("epsilon", 0.0)
        self.no_valid_check = config.get('lego_no_valid_check', False)
        self.no_parallelization = config.get('lego_no_parallelization', False)

        # budget
        self.budget = config.get("lego_budget", None)
        self.budget_scaling = config.get("lego_budget_scaling", False)

        # rollback
        self.rollback_thres = config.get('lego_rollback_thres')
        self.rollback_steps = config.get('lego_rollback_steps')
        self.rollback_method = config.get('lego_rollback_method')

        self.warn_config_items()
        self.validate_budget(self.brick_list, self.budget)

        self.conv_map = dict()
        self.conv_thick_map = dict()
        self.kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.conv_no_squeeze_in = torch.nn.Conv3d(1, 1, (1, 3, 1), bias=False, device="cuda", padding="same")
        self.init_brick_sized_conv()

    def init_brick_sized_conv(self):
        nn.init.constant_(self.conv_no_squeeze_in.weight, 1.0)
        for param in self.conv_no_squeeze_in.parameters():
            param.requires_grad = False

        for brick_size in self.brick_list:
            # create w_b, h_b dense conv
            conv_torch = torch.nn.Conv3d(1, 1, (brick_size[0], 1, brick_size[1]), bias=False, device="cuda", padding="same")
            conv_torch_thick = torch.nn.Conv3d(1, 1, (brick_size[0], 3, brick_size[1]), bias=False, device="cuda", padding="same")
            nn.init.constant_(conv_torch.weight, 1.0)
            nn.init.constant_(conv_torch_thick.weight, 1.0)

            # freeze conv
            for param in conv_torch.parameters():
                param.requires_grad = False

            # freeze conv
            for param in conv_torch_thick.parameters():
                param.requires_grad = False

            self.conv_map[brick_size] = conv_torch
            self.conv_thick_map[brick_size] = conv_torch_thick

    def warn_config_items(self):
        if self.no_pivot:
            print(f"[warn] no pivot!")
        
        if self.no_valid_check:
            print(f"[warn] validity check is disabled!")
        
        if self.no_parallelization:
            print(f"[warn] parallelization is disabled!")

    def validate_budget(self, brick_list, budget):
        if budget is not None:
            assert len(brick_list) == len(budget)

    def generate_next_state_parallel(self,
            A_sampled_sparse:List[ME.SparseTensor],
            A_gt_sparse:List[ME.SparseTensor],
            initial_brick_info_list_list:List[List[Tuple]],
            infusion_rates:List[float],
            n_bricks:List[int]
        ) -> List[torch.Tensor]:
        
        batch_size:int = len(infusion_rates)
        initial_brick_info_list_list = deepcopy(initial_brick_info_list_list)
        n_bricks = deepcopy(n_bricks) if self.lego_max_n_brick is None else [self.lego_max_n_brick] * len(n_bricks)
        max_n_bricks = max(n_bricks)

        # batch_size assertion
        assert len(A_sampled_sparse) == batch_size
        assert len(A_gt_sparse) == batch_size
        assert len(initial_brick_info_list_list) == batch_size
        assert len(n_bricks) == batch_size

        args = DotDict(
            {
                "rule": "all_24",
                "brick_voxel_size": [64, 64, 64],
                "initial_brick": False,
            }
        )

        A_gt = torch.cat([sparseTensorToDense(x) for x in A_gt_sparse], dim=0)
        A_sampled = torch.cat([sparseTensorToDense(x) for x in A_sampled_sparse], dim=0)

        boards = [MultiBrickTypeLegoBoard(args, board=(None, brick_info_list)) for brick_info_list in initial_brick_info_list_list]

        for brick_idx in range(max_n_bricks):
            if all(n_brick < brick_idx + 1 for n_brick in n_bricks):
                break
            use_gt_list = [random.random() < rate for rate in infusion_rates]
            A = torch.cat([A_gt[batch_idx, None, ...] if use_gt else A_sampled[batch_idx, None, ...] for batch_idx, use_gt in enumerate(use_gt_list)], dim=0)
            assert A.shape[1:] == (1, 64, 64, 64)

            for _ in range(self.n_bricks_every_step):
                
                B = np.concatenate([board.brick_voxel[None, None, ...] for board in boards], axis=0)
                B = torch.tensor(B, dtype=torch.float32, device="cuda")
                C = dict()
                C_rand = dict()

                for brick_size in self.brick_list:
                    C[brick_size] = self.calculate_masked_prob(*brick_size, B, A)
                    C_rand[brick_size] = self.calculate_masked_prob(*brick_size, B, 1-B)
                    C[brick_size] = C[brick_size].cpu().numpy()
                    C_rand[brick_size] = C_rand[brick_size].cpu().numpy()

                subprocess_args = []
                for batch_idx in range(batch_size):
                    n_brick = n_bricks[batch_idx]

                    if n_brick < brick_idx + 1:
                        continue

                    C_batch = dict()
                    C_rand_batch = dict()
                    brick_info_list = boards[batch_idx].brick_positions
                    brick_translation = boards[batch_idx].translation
                    brick_rule = boards[batch_idx].rule
                    budget = self.budget[batch_idx]
                    for brick_size in self.brick_list:
                        C_batch[brick_size] = C[brick_size][batch_idx, 0, ...]
                        C_rand_batch[brick_size] = C_rand[brick_size][batch_idx, 0, ...]
                    subprocess_args.append((batch_idx, self.brick_list, C_batch, C_rand_batch, brick_info_list, brick_translation, brick_rule, self.no_random, self.only_argmax, self.epsilon, budget, self.budget_scaling))

                # parallelly run below
                if self.parallel:
                    results = self.pool.imap_unordered(select_pivot_and_connection, subprocess_args)
                else:
                    results = [select_pivot_and_connection(sub_arg) for sub_arg in subprocess_args]

                for batch_idx, pivot, rule, selected_brick_size in results:
                    if pivot is None:
                        # brick is filled!
                        n_bricks[batch_idx] = brick_idx
                        continue
                    boards[batch_idx].execute_move((pivot, rule))

                    # change budget
                    if self.budget is not None:
                        budget = self.budget[batch_idx]
                        if selected_brick_size is not None:
                            if selected_brick_size[0]!=selected_brick_size[1]:
                                budget[(selected_brick_size[1], selected_brick_size[0])] -= 1

                            budget[selected_brick_size] -= 1
        
        # return coordinate list
        return [self.brickVoxelToCoord(board.brick_voxel) for board in boards]

    def transition_parallel(self,
                            A_sparse: List[ME.SparseTensor],
                            brick_info_list_list: List[List[Tuple]],
                            remained_budget: List[List[int]] = None
                            ) -> Tuple:
        batch_size:int = len(brick_info_list_list)
        brick_info_list_list = deepcopy(brick_info_list_list)
        
        args = DotDict(
            {
                "rule": "all_24",
                "brick_voxel_size": [64, 64, 64],
                "initial_brick": False,
            }
        )

        loss_name = self.config.get('lego_loss')

        A = torch.cat([sparseTensorToDense(x) for x in A_sparse], dim=0)

        if self.lego_max_n_brick is None:
            boards = [MultiBrickTypeLegoBoard(args, board=(None, brick_info_list)) for brick_info_list in brick_info_list_list]
            max_n_bricks = 1
        else:
            boards = [MultiBrickTypeLegoBoard(args, board=(None, brick_info_list[:1])) for brick_info_list in brick_info_list_list]
            max_n_bricks = self.lego_max_n_brick

        if remained_budget is None:
            remained_budget = [None] * batch_size

        assert len(remained_budget) == batch_size

        for brick_idx in range(max_n_bricks):
            for _ in range(self.n_bricks_every_step):
                B = np.concatenate([board.brick_voxel[None, None, ...] for board in boards], axis=0)
                B = torch.tensor(B, dtype=torch.float32, device="cuda")
                C = dict()
                score = dict()
                C_rand = dict()

                for brick_size in self.brick_list:
                    C[brick_size], score[brick_size] = self.calculate_masked_prob(*brick_size, B, A, with_score=True)
                    C_rand[brick_size] = self.calculate_masked_prob(*brick_size, B, 1-B)
                    C[brick_size] = C[brick_size].cpu().numpy()
                    C_rand[brick_size] = C_rand[brick_size].cpu().numpy()


                subprocess_args = []
                for batch_idx in range(batch_size):

                    C_batch = dict()
                    score_batch = dict()
                    C_rand_batch = dict()
                    brick_info_list = boards[batch_idx].brick_positions
                    brick_translation = boards[batch_idx].translation
                    brick_rule = boards[batch_idx].rule
                    budget = remained_budget[batch_idx] 
                    for brick_size in self.brick_list:
                        C_batch[brick_size] = thresholding_C(C[brick_size][batch_idx, 0, ...], brick_size, self.pred_thresh)
                        score_batch[brick_size] = score[brick_size][batch_idx, 0, ...]
                        C_rand_batch[brick_size] = C_rand[brick_size][batch_idx, 0, ...]


                    if loss_name == 'mse_with_removal_loss' or loss_name == 'l1_with_removal_loss':
                        pivot_idx_to_remove = get_pivot_to_remove(self.brick_list, score_batch, brick_info_list, brick_translation)
                        if pivot_idx_to_remove is not None:
                            boards[batch_idx].remove_brick(pivot_idx_to_remove)
                            brick_info_list = boards[batch_idx].brick_positions

                    subprocess_args.append((batch_idx, self.brick_list, C_batch, C_rand_batch, brick_info_list, brick_translation, brick_rule, self.no_random, self.only_argmax, self.epsilon, budget, self.budget_scaling))

                # parallelly run below
                if self.parallel:
                    results = self.pool.imap_unordered(select_pivot_and_connection, subprocess_args)
                else:
                    results = [select_pivot_and_connection(sub_arg) for sub_arg in subprocess_args]

                for batch_idx, pivot, rule, selected_brick_size in results:
                    if pivot is None:
                        # brick is filled!
                        continue
                    boards[batch_idx].execute_move((pivot, rule))

                    # change budget
                    budget = remained_budget[batch_idx]
                    if budget is not None and selected_brick_size is not None:
                        if selected_brick_size[0]!=selected_brick_size[1]:
                            budget[(selected_brick_size[1], selected_brick_size[0])] -= 1

                        budget[selected_brick_size] -= 1

        # return coordinate list
        return [denseArrayToSparse(board.brick_voxel) for board in boards], [board.brick_positions for board in boards]

    def transition_parallel_no_pivot(self, A_sparse:List[ME.SparseTensor], brick_info_list_list:List[List[Tuple]])->Tuple:
        assert self.no_pivot
        batch_size:int = len(brick_info_list_list)
        brick_info_list_list = deepcopy(brick_info_list_list)
        
        args = DotDict(
            {
                "rule": "all_24",
                "brick_voxel_size": [64, 64, 64],
                "initial_brick": False,
            }
        )

        loss_name = self.config.get('lego_loss')

        A = torch.cat([sparseTensorToDense(x) for x in A_sparse], dim=0)

        if self.lego_max_n_brick is None:
            boards = [MultiBrickTypeLegoBoard(args, board=(None, brick_info_list)) for brick_info_list in brick_info_list_list]
            max_n_bricks = 1
        else:
            boards = [MultiBrickTypeLegoBoard(args, board=(None, brick_info_list[:1])) for brick_info_list in brick_info_list_list]
            max_n_bricks = self.lego_max_n_brick

        for brick_idx in range(max_n_bricks):
            for _ in range(self.n_bricks_every_step):
                B = np.concatenate([board.brick_voxel[None, None, ...] for board in boards], axis=0)
                B = torch.tensor(B, dtype=torch.float32, device="cuda")
                C = dict()
                score = dict()
                C_rand = dict()

                for brick_size in self.brick_list:
                    C[brick_size], score[brick_size] = self.calculate_masked_prob(*brick_size, B, A, with_score=True)
                    C_rand[brick_size] = self.calculate_masked_prob(*brick_size, B, 1-B)
                    C[brick_size] = C[brick_size].cpu().numpy()
                    C_rand[brick_size] = C_rand[brick_size].cpu().numpy()


                subprocess_args = []
                for batch_idx in range(batch_size):
                    C_batch = dict()
                    score_batch = dict()
                    C_rand_batch = dict()
                    brick_info_list = boards[batch_idx].brick_positions
                    brick_translation = boards[batch_idx].translation
                    brick_rule = boards[batch_idx].rule
                    budget = deepcopy(self.budget)
                    for brick_size in self.brick_list:
                        C_batch[brick_size] = thresholding_C(C[brick_size][batch_idx, 0, ...], brick_size, self.pred_thresh)
                        score_batch[brick_size] = score[brick_size][batch_idx, 0, ...]
                        C_rand_batch[brick_size] = C_rand[brick_size][batch_idx, 0, ...]


                    # if loss_name == 'mse_with_removal_loss' or loss_name == 'l1_with_removal_loss':
                    #     pivot_idx_to_remove = get_pivot_to_remove(self.brick_list, score_batch, brick_info_list, brick_translation)
                    #     if pivot_idx_to_remove is not None:
                    #         boards[batch_idx].remove_brick(pivot_idx_to_remove)
                    #         brick_info_list = boards[batch_idx].brick_positions

                    subprocess_args.append((batch_idx, self.brick_list, C_batch, C_rand_batch, brick_info_list, brick_translation, brick_rule, self.no_random, self.only_argmax, self.epsilon, budget))

                # parallelly run below
                if self.parallel:
                    results = self.pool.imap_unordered(select_pivot_and_connection, subprocess_args)
                else:
                    results = [select_pivot_and_connection(sub_arg) for sub_arg in subprocess_args]

                for batch_idx, pivot, rule in results:
                    if pivot is None:
                        # brick is filled!
                        continue
                    boards[batch_idx].execute_move((pivot, rule))

        # return coordinate list
        return [denseArrayToSparse(board.brick_voxel) for board in boards], [board.brick_positions for board in boards]

    def calculate_masked_prob(self, w_b:int, h_b:int, B:torch.Tensor, A:torch.Tensor, differentiable=False, with_score=False) -> torch.Tensor:
        """
        Calculates masked probability, i.e., inv(conv(B)) * \sigma(conv(A))

        Args:
            B (torch.Tensor): shape of (B, C, D, D, D)
            A (torch.Tensor): shape of (B, C, D, D, D)
        """
        # shape check
        assert B.shape[2] == B.shape[3] == B.shape[4]
        assert B.shape == A.shape

        # create w_b, h_b dense conv
        conv_torch = self.conv_map[(w_b, h_b)]
        conv_torch_thick = self.conv_thick_map[(w_b, h_b)]

        with torch.no_grad() if not differentiable else contextlib.nullcontext():
            if self.no_valid_check:
                brick_map = torch.ones_like(B, dtype=torch.float32)
            else:
                if self.no_parallelization:
                    brick_map = []
                    for B_batch in B:
                        brick_map.append(serial_3d_conv((w_b, 1, h_b), B_batch[0])[None, ...])
                    brick_map = torch.stack(brick_map)

                else:
                    brick_map = torch.tensor((conv_torch(B) == 0).clone().detach(), dtype=torch.float32)
            if self.no_pivot:
                thick_brick_map = torch.tensor((conv_torch_thick(B) != 0).clone().detach(), dtype=torch.float32)
                assert brick_map.shape == thick_brick_map.shape
                brick_map *= thick_brick_map
            if self.no_squeeze_in:
                brick_map = torch.tensor((self.conv_no_squeeze_in(brick_map) == 2.0).clone().detach(), dtype=torch.float32)
            
            score_map = conv_torch(A)

            # align for even number kernel 
            # x axis
            if w_b % 2 == 0 or True:
                brick_map = torch.roll(brick_map, shifts=(1,), dims=(2,))
                score_map = torch.roll(score_map, shifts=(1,), dims=(2,))
            if h_b % 2 == 0 or True:
                brick_map = torch.roll(brick_map, shifts=(1,), dims=(4,))
                score_map = torch.roll(score_map, shifts=(1,), dims=(4,))

            # thresholding
            if self.thresh is not None:
                score_map[score_map < self.thresh * w_b * h_b] = 0.0

        if with_score:
            return brick_map * score_map, score_map
        
        return brick_map * score_map

    def brickVoxelToCoord(self, brick_voxel:np.ndarray) -> List[torch.Tensor]:
        coords = np.concatenate([x[..., None] for x in np.where(brick_voxel != 0.0)], axis=-1) - np.array([31, 31, 31])
        return torch.tensor(coords, dtype=torch.int32, device="cpu") # (N, 3)

    def generate_next_state_parallel_with_loss(self,
            A_pred_sparse:List[ME.SparseTensor],
            A_gt_sparse:List[ME.SparseTensor],
            brick_info_list_list:List[List[Tuple]],
            infusion_rates:List[float],
        ) -> Tuple[List[torch.Tensor], List[List[np.ndarray]], List[torch.Tensor]]:
        
        batch_size:int = len(infusion_rates)
        brick_info_list_list = deepcopy(brick_info_list_list)
        losses = []

        # batch_size assertion
        assert len(A_pred_sparse) == batch_size
        assert len(A_gt_sparse) == batch_size
        assert len(brick_info_list_list) == batch_size

        args = DotDict(
            {
                "rule": "all_24",
                "brick_voxel_size": [64, 64, 64],
                "initial_brick": False,
            }
        )

        loss_name = self.config.get('lego_loss')
        fill_neg = self.config.get('lego_fill_neg', False)

        A_gt = torch.cat([sparseTensorToDense(x) for x in A_gt_sparse], dim=0)
        A_pred = torch.cat([sparseTensorToDense(x) for x in A_pred_sparse], dim=0)

        A_pred_sq = A_pred ** 2
        A_gt_sq = A_gt ** 2

        boards = [MultiBrickTypeLegoBoard(args, board=(None, brick_info_list)) for brick_info_list in brick_info_list_list]

        for brick_idx in range(self.n_bricks_every_step):
            
            B = np.concatenate([board.brick_voxel[None, None, ...] for board in boards], axis=0)
            B = torch.tensor(B, dtype=torch.float32, device="cuda")
            C_pred = dict()
            score_pred = dict()
            C_pred_sq = dict()
            C_gt = dict()
            C_gt_sq = dict()
            score_gt = dict()

            for brick_size in self.brick_list:
                C_pred[brick_size], score_pred[brick_size] = self.calculate_masked_prob(*brick_size, B, A_pred, differentiable=True, with_score=True)
                C_gt[brick_size], score_gt[brick_size] = self.calculate_masked_prob(*brick_size, B, A_gt, with_score=True)
                C_pred_sq[brick_size] = self.calculate_masked_prob(*brick_size, B, A_pred_sq, differentiable=True) if loss_name == 'mse_with_removal_loss' else None
                C_gt_sq[brick_size] = self.calculate_masked_prob(*brick_size, B, A_gt_sq, differentiable=True) if loss_name == 'mse_with_removal_loss' else None

            for batch_idx in range(batch_size):
                C_pred_batch = dict()
                C_gt_batch = dict()
                C_pred_sq_batch = dict()
                C_gt_quantized_batch = dict()
                C_gt_quantized_sq_batch = dict()
                score_mix_batch = dict()
                score_gt_batch = dict()
                score_pred_batch = dict()
                infusion_rate = infusion_rates[batch_idx]

                # board info
                brick_info_list = boards[batch_idx].brick_positions
                brick_translation = boards[batch_idx].translation
                brick_rules = boards[batch_idx].rule

                pivot_score_pred_list = []
                pivot_score_pred_sq_list = []
                pivot_score_gt_list = []
                pivot_idxs_list = []

                for brick_size in self.brick_list:
                    C_pred_batch[brick_size] = C_pred[brick_size][batch_idx, 0, ...]
                    C_gt_batch[brick_size] = C_gt[brick_size][batch_idx, 0, ...]
                    score_pred_batch[brick_size] = score_pred[brick_size][batch_idx, 0, ...]
                    score_gt_batch[brick_size] = score_gt[brick_size][batch_idx, 0, ...]
                    C_gt_quantized_batch[brick_size] = quantize_C_gt(C_gt_batch[brick_size], score_gt_batch[brick_size], brick_size, brick_info_list, brick_translation, fill_neg=fill_neg, quantize_thresh=self.quantize_thresh) if loss_name == 'mse_with_removal_loss' or loss_name == 'l1_with_removal_loss' else None
                    C_pred_sq_batch[brick_size] = C_pred_sq[brick_size][batch_idx, 0, ...] if loss_name == 'mse_with_removal_loss' else None
                    score_mix_batch[brick_size] = (score_pred_batch[brick_size] * (1-infusion_rate) + score_gt_batch[brick_size] * infusion_rate) if loss_name == 'mse_with_removal_loss' or loss_name == 'l1_with_removal_loss' else None

                if loss_name == 'mse_with_removal_loss' or loss_name == 'l1_with_removal_loss':
                    pivot_idx_to_remove = get_pivot_to_remove(self.brick_list, score_mix_batch, brick_info_list, brick_translation)
                    if pivot_idx_to_remove is not None:
                        boards[batch_idx].remove_brick(pivot_idx_to_remove)
                        brick_info_list = boards[batch_idx].brick_positions

                for brick_size in self.brick_list:
                    pivot_score_pred, pivot_idxs_pred = get_pivot_scores_and_indices_differentiable(*brick_size, C_pred_batch[brick_size], brick_info_list, brick_translation, filter_nonzero=False)
                    pivot_score_pred_sq, _ = get_pivot_scores_and_indices_differentiable(*brick_size, C_pred_sq_batch[brick_size], brick_info_list, brick_translation, filter_nonzero=False) if loss_name == 'mse_with_removal_loss' else (None, None)
                    pivot_score_gt, pivot_idxs_gt = get_pivot_scores_and_indices_differentiable(*brick_size, C_gt_batch[brick_size], brick_info_list, brick_translation, filter_nonzero=False)
                    assert pivot_idxs_pred == pivot_idxs_gt

                    pivot_score_pred_list.extend(pivot_score_pred)
                    pivot_score_pred_sq_list.extend(pivot_score_pred_sq) if loss_name == 'mse_with_removal_loss' else None
                    pivot_score_gt_list.extend(pivot_score_gt)
                    pivot_idxs_list.extend(pivot_idxs_pred)

                if np.sum([x.cpu().numpy() for x in pivot_score_gt_list]) == 0:
                    continue
                
                pivot_score_pred_t = torch.cat([x[None, ...] for x in pivot_score_pred_list])[None, ...]
                pivot_score_gt_t = torch.cat([x[None, ...] for x in pivot_score_gt_list])[None, ...]

                max_gt_pivot = pivot_idxs_list[torch.argmax(pivot_score_gt_t)]
                pivot_score_target = torch.tensor(pivot_idxs_list.index(max_gt_pivot), dtype=torch.long, device="cuda")[None, ...]

                if loss_name == 'ce_loss':
                    losses.append(self.ce_loss(pivot_score_pred_t, pivot_score_target))
                elif loss_name == 'mse_with_removal_loss':
                    n_voxel = A_pred_sparse[batch_idx].C.shape[0]
                    for brick_size in self.brick_list:
                        w_b, h_b = brick_size
                        losses.append((C_pred_sq_batch[brick_size] - 2 * C_gt_quantized_batch[brick_size] * C_pred_batch[brick_size] + w_b * h_b * C_gt_quantized_batch[brick_size].pow(2)).sum() / n_voxel / w_b / h_b)
                elif loss_name == 'l1_with_removal_loss':
                    n_voxel = A_pred_sparse[batch_idx].C.shape[0]
                    for brick_size in self.brick_list:
                        w_b, h_b = brick_size
                        losses.append(torch.abs(C_gt_quantized_batch[brick_size] - C_pred_batch[brick_size]).sum() / n_voxel / w_b / h_b)
                elif loss_name is None:
                    pass
                else:
                    raise ValueError(f"Undefined lego_loss: {loss_name}")

                pivot_score_total_list = (pivot_score_pred_t * (1-infusion_rate) + pivot_score_gt_t * infusion_rate)[0].detach().cpu().numpy()
                # negative filtering
                pivot_score_total_list = [(x if x >= 0 else 0) for x in pivot_score_total_list]

                if sum(pivot_score_total_list) == 0:
                    print("[debug] sum(pivot_score) == 0")
                    continue

                pivot:int = np.random.choice(pivot_idxs_list, p=normalize(pivot_score_total_list))

                #####################done

                rule_score_pred_list = []
                rule_score_gt_list = []
                rule_idxs_list = []

                for brick_size in self.brick_list:
                    rule_scores_pred, rule_indices_pred = get_connection_scores_and_indices_differentiable(*brick_size, C_pred_batch[brick_size], pivot, brick_info_list, brick_rules, brick_translation, filter_nonzero=False)
                    rule_scores_gt, rule_indices_gt = get_connection_scores_and_indices_differentiable(*brick_size, C_gt_batch[brick_size], pivot, brick_info_list, brick_rules, brick_translation, filter_nonzero=False)
                    assert rule_indices_pred == rule_indices_gt

                    rule_score_pred_list.extend(rule_scores_pred)
                    rule_score_gt_list.extend(rule_scores_gt)
                    rule_idxs_list.extend(rule_indices_pred)

                rule_score_pred_t = torch.cat([x[None, ...] for x in rule_score_pred_list])[None, ...]
                rule_score_gt_t = torch.cat([x[None, ...] for x in rule_score_gt_list])[None, ...]

                if torch.sum(rule_score_gt_t) != 0:

                    max_gt_rule = rule_idxs_list[torch.argmax(rule_score_gt_t)]
                    rule_score_target = torch.tensor(rule_idxs_list.index(max_gt_rule), dtype=torch.long, device="cuda")[None, ...]

                    if loss_name == 'ce_loss':
                        losses.append(self.ce_loss(rule_score_pred_t, rule_score_target))
                    elif loss_name == 'mse_with_removal_loss' or loss_name == 'l1_with_removal_loss' or loss_name is None:
                        pass
                    else:
                        raise ValueError(f"Undefined lego_loss: {loss_name}")


                rule_score_total_list = (rule_score_pred_t * (1-infusion_rate) + rule_score_gt_t * infusion_rate)[0].detach().cpu().numpy()
                # negative filtering
                rule_score_total_list = [(x if x >= 0 else 0) for x in rule_score_total_list]


                # sampling rule
                rule:int = rule_idxs_list[np.argmax(rule_score_total_list)]

                # execute move
                boards[batch_idx].execute_move((pivot, rule))
        
        # return B, coordinate list, loss
        return [self.brickVoxelToCoord(board.brick_voxel) for board in boards], [board.brick_positions for board in boards], losses
