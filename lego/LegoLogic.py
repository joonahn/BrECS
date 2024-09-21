import math
import sys
from typing import List, Tuple, Union

import torch
sys.path.append("..")
import sys
sys.path.append("./lego")
import numpy as np
import brick as brick
import bricks as bricks
from rules_mnist import LIST_RULES_2_4 as mnist_24_rule
from rules_artificial import LIST_RULES_2_4 as artificial_24_rule
from rules import LIST_RULES_2_4 as all_24_rule
try:
    import MinkowskiEngine as ME
except:
    pass

import binvox_rw
from scipy.ndimage import zoom, convolve
from lego.utils import brick_position_to_ldr, brick_position_to_ldr_2x4

class LegoMnistBoard():
    def __init__(self, board=None, args=None):
        self.voxel_dim = args.brick_voxel_size
        self.num_max_bricks = args.num_max_bricks
        self.args = args

        self.target_voxel = self.load_target_voxel(args)
        self.rule = self.load_rule(args)
        self.dist_target_feat = self.load_dist_target_feat(args)
        self.translation = self.calculate_translation(args, self.target_voxel)

        self.invalid_pivots = []

        if board is None:
            brick_ = brick.Brick()

            brick_.set_position([0, 0, 0])
            brick_.set_direction(0)

            self.bricks_ = bricks.Bricks(self.num_max_bricks)
            self.bricks_.add(brick_)
            self.X, self.A = self.bricks_.get_graph()
        else:
            self.X, self.A, self.invalid_pivots = board
            self.bricks_ = bricks.convert_to_bricks(self.X, self.A, max_bricks=args.num_max_bricks, do_validate=False)

        self.brick_voxel = None

    def load_target_voxel(self, args) -> np.ndarray:
        if args.dataset == "":
            target_voxel = None
        elif args.dataset == "mnist":
            target_voxel = np.load('./lego/mnist_easy_dataset/target_voxel_train/{}.npy'.format(100))
        elif args.dataset == "mnist_bridge":
            target_voxel = np.load('./lego/mnist_toy_dataset/bridge.npy')
        elif args.dataset == "mnist_arrow":
            target_voxel = np.load('./lego/mnist_toy_dataset/arrow.npy')
        elif args.dataset == "mnist_triangle":
            target_voxel = np.load('./lego/mnist_toy_dataset/triangle.npy')
        elif args.dataset == "mnist_tree":
            target_voxel = np.load('./lego/mnist_toy_dataset/tree.npy')
        elif args.dataset == "mnist_bridge_large":
            target_voxel = np.load('./lego/mnist_toy_dataset/bridge_large.npy')
        elif args.dataset == "mnist_arrow_large":
            target_voxel = np.load('./lego/mnist_toy_dataset/arrow_large.npy')
        elif args.dataset == "mnist_triangle_large":
            target_voxel = np.load('./lego/mnist_toy_dataset/triangle_large.npy')
        elif args.dataset == "mnist_tree_large":
            target_voxel = np.load('./lego/mnist_toy_dataset/tree_large.npy')
        elif args.dataset == "5bricks_vline":
            target_voxel = np.load('./lego/5brick_sanity_check/vline.npy')
        elif args.dataset == "5bricks_hline":
            target_voxel = np.load('./lego/5brick_sanity_check/hline.npy')
        elif args.dataset == "5bricks_stairs":
            target_voxel = np.load('./lego/5brick_sanity_check/stairs.npy')
        elif args.dataset == "5bricks_thickline":
            target_voxel = np.load('./lego/5brick_sanity_check/thickline.npy')
        elif args.dataset == "10bricks_vline":
            target_voxel = np.load('./lego/10brick_sanity_check/vline.npy')
        elif args.dataset == "10bricks_stairs":
            target_voxel = np.load('./lego/10brick_sanity_check/stairs.npy')
        elif args.dataset == "10bricks_thickline":
            target_voxel = np.load('./lego/10brick_sanity_check/thickline.npy')
        elif args.dataset == "modelnet40_subset":
            with open(args.dataset_dir, "rb") as f:
                target_voxel = binvox_rw.read_as_3d_array(f, fix_coords=False).data 
                assert args.target_voxel_scale <= 1.0

                if args.target_voxel_conv != 0:
                    conv_filter = np.ones((args.target_voxel_conv,)*3)
                    target_voxel = convolve(target_voxel.astype(bool), conv_filter, mode='constant', cval=0.0)

                if args.target_voxel_scale < 1.0:
                    original_target_voxel_size = target_voxel.shape[0]
                    target_voxel = zoom(target_voxel, (args.target_voxel_scale,) * 3, order=0)
                    n_pad_l = (original_target_voxel_size - target_voxel.shape[0]) // 2
                    n_pad_r = math.ceil((original_target_voxel_size - target_voxel.shape[0]) / 2)
                    target_voxel = np.pad(target_voxel, ((n_pad_l, n_pad_r), (n_pad_l, n_pad_r), (n_pad_l, n_pad_r)), mode='constant', constant_values=0)
                    assert original_target_voxel_size == target_voxel.shape[0], f"wrong target voxel shape: {target_voxel.shape[0]}"
                    
                if not args.disc_input_centerize and not args.disc_input_normalize:
                    target_voxel = self.bottom_voxel(target_voxel)
        else:
            raise Exception(f"undefined dataset name : {args.dataset}")
        return target_voxel

    def load_rule(self, args) -> list:
        if args.rule == "mnist_24":
            rule = mnist_24_rule
        elif args.rule == "artificial_24":
            rule = artificial_24_rule
        elif args.rule == "all_24":
            rule = all_24_rule
        else:
            raise Exception(f"undefined rule name : {args.rule}")
        return rule

    def load_dist_target_feat(self, args) -> np.ndarray:
        disc_target_feat = None
        if args.disc_feat_dir is not None and args.disc_target_class is not None:
            disc_target_feat = np.load(args.disc_feat_dir)[args.disc_target_class]

        return disc_target_feat

    def calculate_translation(self, args, target_voxel) -> np.ndarray:
        if target_voxel is not None and \
            (args.dataset != "modelnet40_subset" or (not args.disc_input_centerize and not args.disc_input_normalize)):
            target_voxel = target_voxel
            if not args.disc_input_centerize and not args.disc_input_normalize:
                scale_factor = args.brick_voxel_size[0] / target_voxel.shape[0]
                target_voxel = zoom(target_voxel, (scale_factor,) * 3, order=0)
            btv_index = np.min(np.where(np.sum(target_voxel, axis=(0,1)) > 0))
            btv = target_voxel[:,:,btv_index]
            bottom_trans_pad = np.array(list(map(lambda i : (list(map(lambda j : np.sum(btv[i-1, j-1:j+2] +
                                                                                        btv[i, j-1:j+2] +
                                                                                        btv[i+1, j-1:j+2]), range(1, btv.shape[1]-1)))),
                                                range(1, btv.shape[0]-1))))

            bottom_trans = np.zeros((btv.shape[0], btv.shape[1])).astype(bottom_trans_pad.dtype)
            bottom_trans[1:-1, 1:-1] = bottom_trans_pad

            x_s, y_s = np.where(bottom_trans == np.max(bottom_trans))

            translation = np.array([x_s[0] + 1, y_s[0] + 1, btv_index, 0])
        else:
            translation = np.array([args.brick_voxel_size[0]//2, args.brick_voxel_size[1]//2, args.brick_voxel_size[2]//2, 0])
        return translation

    def get_reward(self, disc=None):
        self._make_brick_voxel()

        brick_voxel = self.brick_voxel

        if self.args.disc_input_centerize or self.args.disc_input_normalize:
            lego_min, lego_max = self.get_voxel_bounding_box(self.brick_voxel)

            if self.args.disc_input_centerize:
                brick_voxel = self.centerize_voxel(brick_voxel, lego_min, lego_max)
            
            if self.args.disc_input_normalize and all(lego_max != lego_min):
                brick_voxel = self.normalize_voxel(brick_voxel, lego_min, lego_max)

            assert brick_voxel.shape == self.brick_voxel.shape

        if disc is None:
            if self.args.dataset == "modelnet40_subset":
                scale_factor = self.target_voxel.shape[0] / self.brick_voxel.shape[0]
                brick_voxel = zoom(brick_voxel, (scale_factor,) * 3, order=0)
            return np.sum(np.logical_and(brick_voxel, self.target_voxel)) / np.sum(np.logical_or(brick_voxel, self.target_voxel))
        else:
            # upscale if needed
            if self.args.disc_voxel_size is not None:
                scale_factor = self.args.disc_voxel_size / brick_voxel.shape[0]
                brick_voxel = zoom(brick_voxel, (scale_factor,) * 3, order=0)

            disc.eval()
            with torch.no_grad():
                if self.args.disc_model == "classifier":
                    brick_voxel = torch.FloatTensor(brick_voxel[None, ...]) # adds batch dimension
                    brick_voxel = brick_voxel.contiguous().cuda()
                    return torch.nn.functional.softmax(disc(brick_voxel), dim=1).cpu().numpy()[0, self.args.disc_target_class] # soft label??
                elif self.args.disc_model == "contrastive_minkowski":
                    assert brick_voxel.shape[0] == 64
                    torch.cuda.empty_cache()
                    coords = [np.concatenate([x[None, ...] for x in np.where(brick_voxel == 1.0)], axis=0).T]
                    feats = [((x - ((brick_voxel.shape[0] - 1) / 2)) / (brick_voxel.shape[0] / 2)).astype(np.float32) for x in coords]
                    coords, feats = ME.utils.sparse_collate(
                        coords,
                        feats,
                        dtype=torch.float32
                    )
                    sin = ME.TensorField(
                        features=feats,
                        coordinates=coords,
                        device="cuda"
                    )
                    output_feat = disc(sin).cpu().numpy()
                    if self.args.disc_distance == "euclidean":
                        return 1 - np.sum((self.disc_target_feat - output_feat[0]) ** 2) * 8
                    elif self.args.disc_distance == "cosine":
                        return np.dot(self.disc_target_feat, output_feat[0]) / (np.linalg.norm(self.disc_target_feat) * (np.linalg.norm(output_feat[0])))
                    else:
                        raise Exception(f"undefined distance: {self.args.disc_distance}")
                else:
                    raise Exception(f"undefined disc_model: {self.args.disc_model}")

    def get_board(self):
        return (self.X, self.A, self.invalid_pivots)

    def get_voxel_bounding_box(self, voxel:np.ndarray):
        x_axis = np.where(np.sum(np.sum(voxel, axis=1), axis=1) != 0)
        y_axis = np.where(np.sum(np.sum(voxel, axis=0), axis=1) != 0)
        z_axis = np.where(np.sum(np.sum(voxel, axis=0), axis=0) != 0)

        x_min, x_max = np.min(x_axis), np.max(x_axis)
        y_min, y_max = np.min(y_axis), np.max(y_axis)
        z_min, z_max = np.min(z_axis), np.max(z_axis)

        return np.array([x_min, y_min, z_min]), np.array([x_max, y_max, z_max])

    def centerize_voxel(self, voxel:np.ndarray, lego_min=None, lego_max=None):
        if lego_min is None or lego_max is None:
            lego_min, lego_max = self.get_voxel_bounding_box(voxel)

        offset = (-lego_max - lego_min + np.array(voxel.shape)) / 2
        offset = offset.astype(np.int32) 
        non = lambda s: s if s<0 else None
        mom = lambda s: max(0,s)
        new_voxel = np.zeros_like(voxel)
        new_voxel[mom(offset[0]):non(offset[0]), mom(offset[1]):non(offset[1]), mom(offset[2]):non(offset[2])] = voxel[mom(-offset[0]):non(-offset[0]), mom(-offset[1]):non(-offset[1]), mom(-offset[2]):non(-offset[2])]
        return new_voxel

    def normalize_voxel(self, voxel:np.ndarray, lego_min=None, lego_max=None):
        if lego_min is None or lego_max is None:
            lego_min, lego_max = self.get_voxel_bounding_box(voxel)

        scale_factor = np.min(voxel.shape[0] / (lego_max - lego_min))
        new_voxel = zoom(voxel, (scale_factor,) * 3, order=0)
        new_offset = (new_voxel.shape[0] - voxel.shape[0]) // 2
        return new_voxel[new_offset:new_offset + voxel.shape[0], new_offset:new_offset + voxel.shape[1],new_offset:new_offset + voxel.shape[2]]

    def bottom_voxel(self, voxel:np.ndarray):
        vertical = np.sum(np.sum(voxel, axis=1), axis=0)
        offset = np.min(np.where(vertical > 0.0))
        new_voxel = np.zeros_like(voxel)
        new_voxel[:, :, :-offset] = voxel[:, :, offset:]
        return new_voxel

    def _make_brick_voxel(self):
        if self.brick_voxel is None:
            self.brick_voxel = np.zeros(shape = self.voxel_dim, dtype=np.int32)
            for coord in self.bricks_.get_positions():
                self._occupy(coord, self.brick_voxel)

    def _write_brick_voxel(self, filename):
        self._make_brick_voxel()
        with open(filename, "wb") as f:
            vox = self.brick_voxel
            if vox.shape == (14, 8, 14):
                vox = np.pad(vox, ((0,0), (3,3), (0,0)), mode='constant', constant_values=0)

            lego_min, lego_max = self.get_voxel_bounding_box(self.brick_voxel)

            if self.args.disc_input_centerize:
                vox = self.centerize_voxel(vox, lego_min, lego_max)
            
            if self.args.disc_input_normalize and all(lego_max != lego_min):
                vox = self.normalize_voxel(vox, lego_min, lego_max)

            vox = binvox_rw.Voxels(vox, [self.voxel_dim[0]] * 3, [-1.0]*3, 2.0, 'xzy')
            binvox_rw.write(vox, f)

    def _write_target_voxel(self, filename):
        if self.target_voxel is None:
            pass
        with open(filename, "wb") as f:
            vox = self.target_voxel
            if vox.shape == (14, 8, 14):
                vox = np.pad(vox, ((0,0), (3,3), (0,0)), mode='constant', constant_values=0)
            vox = binvox_rw.Voxels(vox.astype(np.int32), [vox.shape[0]] * 3, [-1.0]*3, 2.0, 'xzy')
            binvox_rw.write(vox, f)


    def execute_move(self, move):
        """Perform the given move on the board; flips pieces as necessary.
        color gives the color pf the piece to play (1=white,-1=black)
        """
        pivot_fragment_index, relative_action = move
        new_brick_coordinate = self._add_node_and_edge(pivot_fragment_index, relative_action)
        self._update_graph(new_brick_coordinate)

    def get_legal_moves_for_a_pivot(self, pivot_fragment_index) -> List[List]:
        self._make_brick_voxel()

        pivot_coordinate = self.X[pivot_fragment_index]

        pivot_position, pivot_direction = pivot_coordinate[:-1], pivot_coordinate[-1]
        legal_moves = []
        for height_diff in [0, 1]:
            for move_idx, move in enumerate(self.rule):
                positional_change =  np.concatenate((np.array(move[1][1]),
                                                    np.array([1 - 2 * height_diff])))
                directional_change = np.array([move[1][0]])

                change_coordinate = np.concatenate([positional_change, directional_change]).tolist()

                if pivot_direction == 1:
                    change_coordinate = self._reposition(change_coordinate)

                new_brick_coordinate = np.concatenate([pivot_position, [0]]) + change_coordinate
                if self._check_possibility(new_brick_coordinate, self.brick_voxel):
                    legal_moves.append(move_idx + len(self.rule) * height_diff)

        if len(legal_moves) == 0:
            self.invalid_pivots.append(pivot_fragment_index)

        return legal_moves

    def has_legal_moves(self):
        return self.X.shape[0] < self.num_max_bricks

    def _add_node_and_edge(self, pivot_fragment_index, relative_action):
        pivot_coordinate = self.X[pivot_fragment_index]

        pivot_position, pivot_direction = pivot_coordinate[:-1], pivot_coordinate[-1]

        rules_index, height_diff = (int(relative_action - len(self.rule)), 1) if relative_action >= len(self.rule) else (int(relative_action), 0)

        positional_change =  np.concatenate((np.array(self.rule[rules_index][1][1]),
                                             np.array([1 - 2 * height_diff])))
        directional_change = np.array([self.rule[rules_index][1][0]])

        change_coordinate = np.concatenate([positional_change, directional_change]).tolist()

        if pivot_direction == 1:
            change_coordinate = self._reposition(change_coordinate)

        new_brick_coordinate = np.concatenate([pivot_position, [0]]) + change_coordinate

        new_brick_ = brick.Brick()
        new_brick_.set_position(new_brick_coordinate[:-1])
        new_brick_.set_direction(new_brick_coordinate[-1])

        self.bricks_.add(new_brick_)

        return new_brick_coordinate

    def _update_graph(self, new_brick_coordinate):
        self.X, self.A = self.bricks_.get_graph_from_prev_graph(self.X, self.A)

        if self.brick_voxel is not None:
            self._occupy(new_brick_coordinate, self.brick_voxel)
        else:
            self._make_brick_voxel()


    def _reposition(self, new_coordinate):
        prev_x, prev_y, z, prev_dir = new_coordinate

        # rad = 3/2 * math.pi
        rad = 1/2 * math.pi

        new_x = math.cos(rad) * prev_x - math.sin(rad) * prev_y
        new_y = math.sin(rad) * prev_x + math.cos(rad) * prev_y

        new_dir = (prev_dir + 1) % 2

        return np.array([round(new_x), round(new_y), z, new_dir], dtype = 'int')

    def _occupy(self, new_coordinate, voxel):
        arranging_coordinate = self.translation

        arranged_x, arranged_y, arranged_z, dir = np.multiply(np.array(new_coordinate, dtype = 'int'), np.array([1,1,1,1])) + \
                                                  np.array(arranging_coordinate, dtype = 'int')

        voxel[(arranged_x - 1 - (dir == 1)):(arranged_x + 1 + (dir == 1)),
              (arranged_y - 1 - (dir == 0)):(arranged_y + 1 + (dir == 0)),
              arranged_z:(arranged_z + 1)] = 1

    def _check_possibility(self, new_coordinate, voxel):
        arranging_coordinate = self.translation

        arranged_x, arranged_y, arranged_z, dir = np.multiply(np.array(new_coordinate, dtype = 'int'), np.array([1,1,1,1])) + \
                                                  np.array(arranging_coordinate, dtype = 'int')


        return self._range_in_voxel(
            arranged_x - 1 - (dir == 1), arranged_x + 1 + (dir == 1),
            arranged_y - 1 - (dir == 0), arranged_y + 1 + (dir == 0),
            arranged_z, arranged_z + 1, voxel) and \
            np.sum(voxel[(arranged_x - 1 - (dir == 1)):(arranged_x + 1 + (dir == 1)),
              (arranged_y - 1 - (dir == 0)):(arranged_y + 1 + (dir == 0)),
              arranged_z:(arranged_z + 1)]) == 0 and \
                  new_coordinate[2] >=0 # cannot go under the ground

    def _range_in_voxel(self, x1, x2, y1, y2, z1, z2, voxel):
        return 0 <= x1 < voxel.shape[0] and 0 < x2 <= voxel.shape[0] \
            and 0 <= y1 < voxel.shape[1] and 0 < y2 <= voxel.shape[1] \
            and 0 <= z1 < voxel.shape[2] and 0 < z2 <= voxel.shape[2]

class SimpleLegoBoard():
    def __init__(self, args, board=None):
        self.voxel_dim = args.brick_voxel_size
        self.translation = np.array([args.brick_voxel_size[0]//2 - 1, args.brick_voxel_size[1]//2 - 1, args.brick_voxel_size[2]//2 - 1, 0])
        self.rule = self.load_rule(args)
        self.brick_voxel = None
        if board is None:
            self.brick_positions = []
            if args.initial_brick:
                self.add_initial_brick()
            self._make_brick_voxel()
        else:
            self.brick_voxel, self.brick_positions = board
            if self.brick_voxel is None:
                self._make_brick_voxel()
            else:
                self.brick_voxel = self.brick_voxel.astype(np.int32)

    def add_initial_brick(self):
        self.brick_positions.append(np.array([0,0,0,0]))

    def load_rule(self, args) -> list:
        if args.rule == "mnist_24":
            rule = mnist_24_rule
        elif args.rule == "artificial_24":
            rule = artificial_24_rule
        elif args.rule == "all_24":
            rule = all_24_rule
        else:
            raise Exception(f"undefined rule name : {args.rule}")
        return rule

    def execute_move(self, move):
        """Perform the given move on the board; flips pieces as necessary.
        color gives the color pf the piece to play (1=white,-1=black)
        """
        pivot_fragment_index, relative_action = move
        pivot_coordinate = self.brick_positions[pivot_fragment_index]
        pivot_position, pivot_direction = pivot_coordinate[:-1], pivot_coordinate[-1]

        rules_index, height_diff = (int(relative_action - len(self.rule)), 1) if relative_action >= len(self.rule) else (int(relative_action), 0)

        positional_change = np.array([self.rule[rules_index][1][1][0], 1 - 2 * height_diff, self.rule[rules_index][1][1][1]])
        directional_change = np.array([self.rule[rules_index][1][0]])

        change_coordinate = np.concatenate([positional_change, directional_change]).tolist()

        if pivot_direction == 1:
            change_coordinate = self._reposition(change_coordinate)

        new_brick_coordinate = np.concatenate([pivot_position, [0]]) + change_coordinate

        self.brick_positions.append(new_brick_coordinate)

        if self.brick_voxel is not None:
            self._occupy(new_brick_coordinate, self.brick_voxel)
        else:
            self._make_brick_voxel()

    def remove_brick(self, brick_idx):
        del self.brick_positions[brick_idx]
        self.brick_voxel = None
        self._make_brick_voxel()

    def add_brick_manually(self, position):
        assert (position.shape == (4,)) or type(position) == tuple

        self.brick_positions.append(position)
        
        if self.brick_voxel is not None:
            self._occupy(position, self.brick_voxel)
        else:
            self._make_brick_voxel()

    def _make_brick_voxel(self):
        if self.brick_voxel is None:
            self.brick_voxel = np.zeros(shape = self.voxel_dim, dtype=np.int32)
            for coord in self.brick_positions:
                self._occupy(coord, self.brick_voxel)

    def _reposition(self, new_coordinate):
        prev_x, y, prev_z, prev_dir = new_coordinate

        # rad = 3/2 * math.pi
        rad = 1/2 * math.pi

        new_x = math.cos(rad) * prev_x - math.sin(rad) * prev_z
        new_z = math.sin(rad) * prev_x + math.cos(rad) * prev_z

        new_dir = (prev_dir + 1) % 2

        return np.array([round(new_x), y, round(new_z), new_dir], dtype = 'int')

    def _occupy(self, new_coordinate, voxel):
        arranging_coordinate = self.translation

        arranged_x, arranged_y, arranged_z, dir = np.multiply(np.array(new_coordinate, dtype = 'int'), np.array([1,1,1,1])) + \
                                                  np.array(arranging_coordinate, dtype = 'int')

        voxel[(arranged_x - 1 - (dir == 1)):(arranged_x + 1 + (dir == 1)),
                arranged_y:(arranged_y + 1),
                (arranged_z - 1 - (dir == 0)):(arranged_z + 1 + (dir == 0))] = 1

    def _write_brick_voxel(self, filename:str):
        self._make_brick_voxel()
        with open(filename, "wb") as f:
            vox = self.brick_voxel
            vox = binvox_rw.Voxels(vox, [self.voxel_dim[0]] * 3, [-1.0]*3, 2.0, 'xyz')
            binvox_rw.write(vox, f)

    def _write_brick_ldr(self, filename:str):
        brick_position_to_ldr_2x4(filename, [np.swapaxes(x, 1, 2) for x in self.brick_positions]) # xyz -> xzy

class MultiBrickTypeLegoBoard(SimpleLegoBoard):
    def __init__(self, args, board=None):
        super().__init__(args, board)
        self.translation = self.translation[:3]

    def add_initial_brick(self):
        self.brick_positions.append(((2, 4), np.array([0,0,0])))

    def load_rule(self, args):
        return None

    def execute_move(self, move):
        pivot_fragment_index, relative_action = move
        pivot_size, pivot_pos = self.brick_positions[pivot_fragment_index]
        brick_size, positional_change = relative_action

        new_brick_pos = pivot_pos + positional_change

        new_brick_info = (brick_size, new_brick_pos)

        self.brick_positions.append(new_brick_info)

        if self.brick_voxel is not None:
            self._occupy(new_brick_info, self.brick_voxel)
        else:
            self._make_brick_voxel()

    def _occupy(self, brick_info:np.ndarray, voxel):
        arranging_coordinate = self.translation
        brick_size, brick_pos = brick_info
        w_b, h_b = brick_size

        arranged_x, arranged_y, arranged_z = np.array(brick_pos, dtype = 'float') + \
                                                  np.array(arranging_coordinate, dtype = 'float')[:3]

        voxel_x_min = math.floor(arranged_x - w_b / 2)
        voxel_x_max = math.floor(arranged_x + w_b / 2)
        voxel_z_min = math.floor(arranged_z - h_b / 2)
        voxel_z_max = math.floor(arranged_z + h_b / 2)
        arranged_y = math.floor(arranged_y)

        voxel[voxel_x_min:voxel_x_max,
                arranged_y:(arranged_y + 1),
                voxel_z_min:voxel_z_max] = 1

    def _write_brick_ldr(self, filename:str):
        brick_position_to_ldr(filename, self.brick_positions)