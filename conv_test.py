import os
from typing import List, Tuple
import torch
import torch.nn as nn
from lego import binvox_rw
import numpy as np
from scipy.ndimage import zoom, convolve
from scipy.special import softmax
from lego.LegoLogic import LegoMnistBoard
import MinkowskiEngine as ME
from MinkowskiEngine.MinkowskiKernelGenerator import KernelGenerator
import yaml
from models import MODEL
from torch.utils.tensorboard import SummaryWriter
from utils.visualization import (
	sparse_tensors2tensor_imgs, save_tensor_img
)
from torch_scatter import scatter_mul, scatter_add
import sys
import math

num_max_bricks = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
model_path = sys.argv[2] if len(sys.argv) > 2 else "gca-sofa/ckpts/ckpt-step-200000"
print("[info] max_bricks:", num_max_bricks)

class DotDict(dict):
    """dot.notation access to dictionary attributes"""

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError from e

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def sparseTensorToDense(sparse_tensor):
    coord = sparse_tensor.C
    feat = sparse_tensor.F
    merged = ME.SparseTensor(feat, coord + torch.tensor([0,31,31,31], dtype=torch.int32, device="cuda"))
    return merged.dense(shape=torch.Size([1,1,64,64,64], min_coordinate=torch.Size([0,0,0])))[0]

def range_in_voxel(x1, x2, y1, y2, z1, z2, voxel, offset=0):
    return offset <= x1 < voxel.shape[0] - offset and offset < x2 <= voxel.shape[0] - offset \
        and offset <= y1 < voxel.shape[1] - offset and offset < y2 <= voxel.shape[1] - offset \
        and offset <= z1 < voxel.shape[2] - offset and offset < z2 <= voxel.shape[2] - offset

def restrict_idx(idx, voxel, offset=0):
    return max(min(idx, voxel.shape[0] - offset), offset)


def get_pivot_and_conn(board_obj, model):
    model.eval()
    brick_coords = np.concatenate([x[..., None] for x in np.where(board_obj.brick_voxel==True)], axis=-1) - np.array([31, 31, 31])
    # add batch dimentsion
    brick_coords = [brick_coords]
    brick_coords = ME.utils.batched_coordinates(brick_coords)
    feats = torch.ones(brick_coords.shape[0], 1)
    s = ME.SparseTensor(
        features = feats,
        coordinates = brick_coords,
        device="cuda"
    )
    with torch.no_grad():
        data = model.forward(s)
        write_sparse_img(data, fname=f"pred_{board_obj.X.shape[0]}.png")

    # create 2x4 dense conv
    conv2x4_torch = torch.nn.Conv3d(1, 1, (2, 4, 1), bias=False, device="cuda", padding="same")
    nn.init.constant_(conv2x4_torch.weight, 1.0)

    with torch.no_grad():
        # brick_map = torch.tensor(conv2x4_torch(sparseTensorToDense(s)), dtype=torch.float32)
        brick_map = torch.tensor(conv2x4_torch(sparseTensorToDense(s)) == 0, dtype=torch.float32)
        # score_map = conv2x4_torch(sparseTensorToDense(s))
        score_map = conv2x4_torch(sparseTensorToDense(data))
        score_map = torch.sigmoid(score_map)

        brick_map = torch.roll(brick_map, shifts=(1,1), dims=(2,3))
        score_map = torch.roll(score_map, shifts=(1,1), dims=(2,3))


    print(brick_map.dtype)
    print(score_map.dtype)

    dense_merged = brick_map * score_map
    dense_merged = dense_merged[0,0]

    # get max pivot & conn
    pivot_scores = []
    pivot_idxs = []
    arranging_coordinate = board_obj.translation
    for pivot in range(board_obj.X.shape[0]):
        pivot_coordinate = board_obj.X[pivot]

        arranged_x, arranged_y, arranged_z, dir = np.multiply(np.array(pivot_coordinate, dtype = 'int'), np.array([1,1,1,1])) + \
                                                        np.array(arranging_coordinate, dtype = 'int')

        pivot_x_min = restrict_idx(arranged_x - 1, board_obj.brick_voxel, offset=7)
        pivot_x_max = restrict_idx(arranged_x + 1, board_obj.brick_voxel, offset=7)
        pivot_y_min = restrict_idx(arranged_y - 3, board_obj.brick_voxel, offset=7)
        pivot_y_max = restrict_idx(arranged_y + 3, board_obj.brick_voxel, offset=7)
        pivot_z_lower = restrict_idx(arranged_z - 1, board_obj.brick_voxel, offset=7)
        pivot_z_upper = restrict_idx(arranged_z + 1, board_obj.brick_voxel, offset=7)

        pivot_score = torch.sum(dense_merged[pivot_x_min:pivot_x_max+1,
                pivot_y_min:pivot_y_max+1,
                pivot_z_lower])
        pivot_score += torch.sum(dense_merged[pivot_x_min:pivot_x_max+1,
                pivot_y_min:pivot_y_max+1,
                pivot_z_upper])

        if pivot_score > 0:
            pivot_scores.append(pivot_score.item())
            pivot_idxs.append(pivot)

    assert len(pivot_idxs) > 0

    # selected_pivot = pivot_idxs[np.argmax(pivot_scores)]
    selected_pivot = np.random.choice(pivot_idxs, p=softmax(pivot_scores))
    pivot_coordinate = board_obj.X[selected_pivot]
    arranging_coordinate = board_obj.translation

    print("[INFO] pivot_idxs:", pivot_idxs)
    print("[INFO] pivot_scores:", pivot_scores)
    print("[INFO] selected_pivot:", selected_pivot)

    move_scores = []
    move_idxs = []
    for height_diff in [0, 1]:
        for move_idx, move in enumerate(board_obj.rule):
            positional_change =  np.concatenate((np.array(move[1][1]),
                                                np.array([1 - 2 * height_diff])))
            directional_change = np.array([move[1][0]])
            if directional_change == 1:
                continue

            change_coordinate = np.concatenate([positional_change, directional_change]).tolist()

            new_brick_coordinate = np.concatenate([pivot_coordinate[:-1], [0]]) + change_coordinate

            arranged_x, arranged_y, arranged_z, dir = np.multiply(np.array(new_brick_coordinate, dtype = 'int'), np.array([1,1,1,1])) + \
                                                            np.array(arranging_coordinate, dtype = 'int')
            
            if not range_in_voxel(
                arranged_x, arranged_x + 1,
                arranged_y, arranged_y + 3,
                arranged_z, arranged_z, board_obj.brick_voxel, offset=3):
                continue
            
            move_score = dense_merged[arranged_x, arranged_y, arranged_z]


            if move_score != 0:
                move_scores.append(move_score.item())
                move_idxs.append(move_idx + len(board_obj.rule) * height_diff)


    assert len(move_scores) != 0

    # selected_move = move_idxs[np.argmax(move_scores)]
    selected_move = np.random.choice(move_idxs, p=softmax(move_scores))
    print("[INFO] move_idxs:", move_idxs)
    print("[INFO] move_scores:", move_scores)
    print("[INFO] selected_move:", selected_move)

    return selected_pivot, selected_move

def write_sparse_img(sparse_tensor, fname=None):
    batch_size = 1
    out_imgs = sparse_tensors2tensor_imgs(sparse_tensor, config['data_dim'],
                    config['vis']['vis_collated_imgs']['vis_3d'], batch_size
                )

    for batch_idx in range(batch_size):
        save_tensor_img(
            out_imgs[batch_idx],
            os.path.join(
                ".",
                f"{batch_idx}.png" if fname is None else fname
            )
        )

def calculate_masked_prob(w_b:int, h_b:int, B:torch.Tensor, A:torch.Tensor) -> torch.Tensor:
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
    conv_torch = torch.nn.Conv3d(1, 1, (w_b, h_b, 1), bias=False, device="cuda", padding="same")
    nn.init.constant_(conv_torch.weight, 1.0)

    # freeze conv: ? 
    # for param in conv_torch.parameters():
    #     param.requires_grad = False

    with torch.no_grad():
        brick_map = torch.tensor((conv_torch(B) == 0).clone().detach(), dtype=torch.float32)
        score_map = torch.sigmoid(conv_torch(A))

        # align for even number kernel 
        brick_map = torch.roll(brick_map, shifts=(1,1), dims=(2,3))
        score_map = torch.roll(score_map, shifts=(1,1), dims=(2,3))

    return brick_map * score_map

def clip(value, min_value, max_value):
    return max(min(value, max_value), min_value)

def reposition(coordinate):
    prev_x, prev_y, z, prev_dir = coordinate

    # rad = 3/2 * math.pi
    rad = 1/2 * math.pi

    new_x = math.cos(rad) * prev_x - math.sin(rad) * prev_y
    new_y = math.sin(rad) * prev_x + math.cos(rad) * prev_y

    new_dir = (prev_dir + 1) % 2

    return np.array([round(new_x), round(new_y), z, new_dir], dtype = 'int')

def normalize(array):
    sum_all = sum(array)
    return [x / sum_all for x in array]

def get_pivot_scores_and_indices(w_b:int, h_b:int, C:torch.Tensor, X:np.ndarray, arranging_coordinate:np.ndarray) -> Tuple[List[float], List[int]]:
    """
    Calculates pivot scores and returns scores and corresponding indices

    Args:
        C (torch.Tensor): shape of (D, D, D)
        X (np.ndarray): shape of (N, 4)
    """
    assert C.shape[0] == C.shape[1] == C.shape[2]

    pivot_scores = []
    pivot_idxs = []
    for pivot in range(X.shape[0]):
        pivot_coordinate = X[pivot]

        arranged_x, arranged_y, arranged_z, dir = np.multiply(np.array(pivot_coordinate, dtype = 'int'), np.array([1,1,1,1])) + \
                                                        np.array(arranging_coordinate, dtype = 'int')

        # get pivot size
        if dir == 0:
            w_p, h_p = 2, 4
        elif dir == 1:
            w_p, h_p = 4, 2

        # set pivot brick range
        voxel_size = C.shape[0]
        offset = 7
        pivot_x_min = clip(arranged_x - (((w_b + w_p) // 2) - 1), offset, voxel_size - offset)
        pivot_x_max = clip(arranged_x + (((w_b + w_p) // 2) - 1), offset, voxel_size - offset)
        pivot_y_min = clip(arranged_y - (((h_b + h_p) // 2) - 1), offset, voxel_size - offset)
        pivot_y_max = clip(arranged_y + (((h_b + h_p) // 2) - 1), offset, voxel_size - offset)
        pivot_z_lower = clip(arranged_z - 1, offset, voxel_size - offset)
        pivot_z_upper = clip(arranged_z + 1, offset, voxel_size - offset)

        # filter out clipped range
        if pivot_x_min == pivot_x_max or pivot_y_min == pivot_y_max:
            continue

        # aggregate pivot scores
        pivot_score = torch.sum(C[pivot_x_min:pivot_x_max+1,
                pivot_y_min:pivot_y_max+1,
                pivot_z_lower])
        pivot_score += torch.sum(C[pivot_x_min:pivot_x_max+1,
                pivot_y_min:pivot_y_max+1,
                pivot_z_upper])

        if pivot_score > 0:
            pivot_scores.append(pivot_score.item())
            pivot_idxs.append(pivot)

    return pivot_scores, pivot_idxs

def get_connection_scores_and_indices(w_b:int, h_b:int, C:torch.Tensor, pivot_idx:int, X:np.ndarray, rules, arranging_coordinate):
    """
    Calculates connection scores and returns scores and corresponding connection indices

    Args:
        C (torch.Tensor): shape of (D, D, D)
        X (np.ndarray): shape of (N, 4)
    """
    move_scores = []
    move_idxs = []
    pivot_coordinate = X[pivot_idx]
    pivot_position, pivot_direction = pivot_coordinate[:-1], pivot_coordinate[-1]

    scratch_pad = np.zeros((20,20,2), dtype=np.int32)

    for height_diff in [0, 1]:
        for move_idx, move in enumerate(rules):
            positional_change =  np.concatenate((np.array(move[1][1]),
                                                np.array([1 - 2 * height_diff])))
            directional_change = np.array([move[1][0]])
            new_brick_dir = (pivot_direction + move[1][0]) % 2
            
            if not ((new_brick_dir == 0 and w_b == 2 and h_b == 4) or (new_brick_dir == 1 and w_b == 4 and h_b == 2)):
                continue

            change_coordinate = np.concatenate([positional_change, directional_change]).tolist()

            if pivot_direction == 1:
                change_coordinate = reposition(change_coordinate)
            
            new_brick_coordinate = np.concatenate([pivot_position, [0]]) + change_coordinate



            arranged_x, arranged_y, arranged_z, dir = np.multiply(np.array(new_brick_coordinate, dtype = 'int'), np.array([1,1,1,1])) + \
                                                            np.array(arranging_coordinate, dtype = 'int')
            
            offset = 3
            voxel_size = C.shape[0]

            if not all(offset <= value < voxel_size - offset for value in [arranged_x - math.ceil(w_b / 2), arranged_x + (w_b // 2), arranged_y - math.ceil(h_b / 2), arranged_y + (h_b // 2), arranged_z]):
                continue
            
            move_score = C[arranged_x, arranged_y, arranged_z]

            scratch_pad[positional_change[0]+10,positional_change[1]+10,height_diff] = 1

            if move_score != 0:
                move_scores.append(move_score.item())
                move_idxs.append(move_idx + len(rules) * height_diff)
    
    return move_scores, move_idxs

def get_pivot_and_conn_reduced(board:LegoMnistBoard, model:torch.nn.Module)->Tuple[int, int]:
    model.eval()
    brick_coords = np.concatenate([x[..., None] for x in np.where(board.brick_voxel==True)], axis=-1) - np.array([31, 31, 31])
    # add batch dimentsion
    brick_coords = [brick_coords]
    brick_coords = ME.utils.batched_coordinates(brick_coords)
    # make feature
    feats = torch.ones(brick_coords.shape[0], 1)
    B = ME.SparseTensor(
        features = feats,
        coordinates = brick_coords,
        device="cuda"
    )
    with torch.no_grad():
        A = model.forward(B)
    
    B = sparseTensorToDense(B)
    A = sparseTensorToDense(A)
    C = dict()

    brick_list = [(2,4), (4,2)]

    for brick_size in brick_list:
        C[brick_size] = calculate_masked_prob(*brick_size, B, A)[0,0]

    pivot_info = dict()
    for brick_size in brick_list:
        pivot_scores, pivot_indices = get_pivot_scores_and_indices(*brick_size, C[brick_size], board.X, board.translation)
        for pivot_score, pivot_index in zip(pivot_scores, pivot_indices):
            pivot_info[pivot_index] = pivot_info.get(pivot_index, 0.0) + pivot_score

    pivot_indices = list(pivot_info.keys())
    pivot_scores = [pivot_info[pivot] for pivot in pivot_indices]

    # argmax style
    # pivot:tuple = pivot_indices[np.argmax(pivot_scores)]
    # sampling style
    # pivot:int = np.random.choice(pivot_indices, p=softmax(pivot_scores))
    pivot:int = np.random.choice(pivot_indices, p=normalize(pivot_scores))

    rule_info = dict()
    for brick_size in brick_list:
        rule_scores, rule_indices = get_connection_scores_and_indices(*brick_size, C[brick_size], pivot, board.X, board.rule, board.translation)
        for rule_score, rule_index in zip(rule_scores, rule_indices):
            rule_info[rule_index] = rule_info.get(rule_index, 0.0) + rule_score

    # for debug
    # if len(rule_info.keys()) == 0:
    #     for brick_size in brick_list:
    #         pivot_scores, pivot_indices = get_pivot_scores_and_indices(*brick_size, C[brick_size], board.X, board.translation)
    #         for pivot_score, pivot_index in zip(pivot_scores, pivot_indices):
    #             pivot_info[pivot_index] = pivot_info.get(pivot_index, 0.0) + pivot_score
                
    #     for brick_size in brick_list:
    #         rule_scores, rule_indices = get_connection_scores_and_indices(*brick_size, C[brick_size], pivot, board.X, board.rule, board.translation)
    #         for rule_score, rule_index in zip(rule_scores, rule_indices):
    #             rule_info[rule_index] = rule_info.get(rule_index, 0.0) + rule_score
    
    rule_indices = list(rule_info.keys())
    rule_scores = [rule_info[rule] for rule in rule_indices]

    # argmax style
    # rule:tuple = rule_indices[np.argmax(rule_scores)]
    # sampling style
    # rule:int = np.random.choice(rule_indices, p=softmax(rule_scores))
    try:
        rule:int = np.random.choice(rule_indices, p=normalize(rule_scores))
    except:
        print("[debug] pivot", pivot)
        print("[debug] pivot location", board.X[pivot] + board.translation)
        print("[debug] selected pivot score", normalize(pivot_scores)[pivot_indices.index(pivot)])

        print("[debug] rule_indices:", rule_indices)
        print("[debug] rule_scores", rule_scores)

        for brick_size in brick_list:
            pivot_scores, pivot_indices = get_pivot_scores_and_indices(*brick_size, C[brick_size], board.X, board.translation)
            for pivot_score, pivot_index in zip(pivot_scores, pivot_indices):
                pivot_info[pivot_index] = pivot_info.get(pivot_index, 0.0) + pivot_score
        
        for brick_size in brick_list:
            rule_scores, rule_indices = get_connection_scores_and_indices(*brick_size, C[brick_size], pivot, board.X, board.rule, board.translation)
            for rule_score, rule_index in zip(rule_scores, rule_indices):
                rule_info[rule_index] = rule_info.get(rule_index, 0.0) + rule_score

        raise

    return pivot, rule


args = DotDict(
    {
        "num_max_bricks": num_max_bricks,
        "dataset": "",
        "rule": "all_24",
        "disc_feat_dir": None,
        "disc_target_class": None,
        "brick_voxel_size": [64, 64, 64],
        "disc_input_centerize": False,
        "disc_input_normalize": False,
    }
)

board = LegoMnistBoard(None, args)
board._make_brick_voxel()

# load model
config_path = "gca-sofa/config.yaml"
config = yaml.load(open(config_path), Loader=yaml.FullLoader)
writer = SummaryWriter(config['log_dir'])
model = MODEL[config['model']](config, writer)
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.cuda()

for idx in range(num_max_bricks):
    try:
        board.execute_move(get_pivot_and_conn_reduced(board, model))
    except:
        prefix = model_path.split("/")[0]
        board._write_brick_voxel(f"brick_vox_{prefix}_{idx}.binvox")
        raise
