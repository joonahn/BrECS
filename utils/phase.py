import torch
import MinkowskiEngine as ME
from lego.brick_generator import BrickGenerator
from lego.utils import is_pos_the_same

class Phase:
    '''
    A scheduler that manages the phase with a Queue
    '''
    def __init__(self, max_phase, equilibrium_max_phase):
        self.max_phase = max_phase
        self.equilibrium_max_phase = equilibrium_max_phase
        self.phase = 0
        self.equilibrium_phase = 0
        self.equilibrium_mode = False

    def __repr__(self):
        return 'phase: {}, equilibrium_phase: {}'.format(
            self.phase, self.equilibrium_phase
        )

    def __str__(self):
        return str(self.phase)

    def __add__(self, other):
        if self.equilibrium_mode:
            self.equilibrium_phase += other
        self.phase += other
        return self

    def set_complete(self):
        if not self.equilibrium_mode:
            self.equilibrium_phase += 1
        self.equilibrium_mode = True

    @property
    def finished(self):
        return (self.phase > self.max_phase) \
               or (self.equilibrium_phase > self.equilibrium_max_phase)

class LegoPhase(Phase):
    def __init__(self, config:dict, brick_generator:BrickGenerator, max_phase, equilibrium_max_phase, initial_voxel, initial_pos, gt_voxel):
        super().__init__(max_phase, equilibrium_max_phase)
        self.config = config
        self.voxel_list = [initial_voxel]
        self.pos_list = [initial_pos]
        self.gt_tensor = self._coord_to_sparse_tensor(gt_voxel)
        self.brick_generator = brick_generator
        self.n_target_skip = config.get('lego_sup_n_skip', 1)
        self.no_pivot = config.get("lego_no_pivot", False)

    def _coord_to_sparse_tensor(self, gt_voxel_coord):
        coord = ME.utils.batched_coordinates([gt_voxel_coord.type(torch.IntTensor)])
        feat = torch.ones(coord.shape[0], 1)
        return ME.SparseTensor(
            feat, coord,
            device=self.config['device']
        )
        
    def _generate_next_state(self):
        pos_t = self.pos_list[-1]
        if self.no_pivot:
            b_next_t, pos_next_t = self.brick_generator.transition_parallel_no_pivot([self.gt_tensor], [pos_t])
        else:
            b_next_t, pos_next_t = self.brick_generator.transition_parallel([self.gt_tensor], [pos_t])
        b_next_t = b_next_t[0]
        b_next_t_coord = b_next_t.C[:, 1:]
        pos_next_t = pos_next_t[0]

        self.voxel_list.append(b_next_t_coord)
        self.pos_list.append(pos_next_t)

    def __add__(self, other):
        super().__add__(other)
        self._generate_next_state()
        return self

    @property
    def target_voxel(self):
        # lazy initialization 
        if len(self.voxel_list) == 1:
            for _ in range(self.n_target_skip):
                self._generate_next_state()

        assert len(self.voxel_list) == self.phase + self.n_target_skip + 1
        assert self.voxel_list[-1].shape[1] == 3
        return self.voxel_list[-1]

    @property
    def current_voxel(self):
        assert self.voxel_list[self.phase].shape[1] == 3
        return self.voxel_list[self.phase]


    @property
    def saturated(self):
        return is_pos_the_same([self.pos_list[self.phase]], [self.pos_list[self.phase + 1]])
