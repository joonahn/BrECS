from multiprocessing.sharedctypes import Value
import pickle
from typing import List
import numpy as np
from datetime import datetime
import os

def get_coordinates(bricks, axes):
    coordinates = []

    for brick in bricks.get_bricks():
        vertices = brick.get_vertices()

        new_vertices = []
        for vertex in vertices:
            new_vertices.append(vertex[axes])
        new_vertices = np.array(new_vertices)
        new_vertices = np.unique(new_vertices, axis=0)

        coordinates.append(new_vertices)

    coordinates = np.array(coordinates)
    return coordinates

def shuffle_routine(squares):
    new_squares = []

    for square in squares:
        cur_square = [square[0], square[1], square[3], square[2]]
#        cur_square = [square[0], square[1], square[3], square[2], square[0]]
        new_squares.append(cur_square)
    
    new_squares = np.array(new_squares)
    return new_squares

def check_overlap_1d(min_max_1, min_max_2):
    assert isinstance(min_max_1, tuple)
    assert isinstance(min_max_2, tuple)

    return min_max_1[1] > min_max_2[0] and min_max_2[1] > min_max_1[0]

def check_overlap_2d(min_max_1, min_max_2):
    assert len(min_max_1) == 2
    assert len(min_max_2) == 2

    return check_overlap_1d(min_max_1[0], min_max_2[0]) and check_overlap_1d(min_max_1[1], min_max_2[1])

def check_overlap_3d(min_max_1, min_max_2):
    assert len(min_max_1) == 3
    assert len(min_max_2) == 3

    return check_overlap_1d(min_max_1[0], min_max_2[0]) and check_overlap_1d(min_max_1[1], min_max_2[1]) and check_overlap_1d(min_max_1[2], min_max_2[2])

def get_min_max_3d(vertices):
    min_max = [
        (np.min(vertices[:, 0]), np.max(vertices[:, 0])),
        (np.min(vertices[:, 1]), np.max(vertices[:, 1])),
        (np.min(vertices[:, 2]), np.max(vertices[:, 2]))
    ]

    return min_max

def normalize_points(batched_points):
    assert len(batched_points.shape) == 3

    new_batched_points = []

    for points in batched_points:
        center = np.mean(points, axis=0)

        points = points - center

        norms = np.linalg.norm(points, axis=1)
        max_norms = np.max(norms)

        points = points / max_norms

        new_batched_points.append(points)

    new_batched_points = np.array(new_batched_points)
    return new_batched_points

def save_bricks(bricks_, str_path, str_file=None):
    str_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    if str_file is None:
        str_save = os.path.join(str_path, 'bricks_{}.npy'.format(str_time))
    else:
        str_save = os.path.join(str_path, str_file + '.npy')

    np.save(str_save, bricks_)

def brick_position_to_ldr_2x4(filename:str, brick_positions, color:int=None):
    LDR_UNITS_PER_STUD = 20
    LDR_UNITS_PER_PLATE = 8
    PLATES_PER_BRICK = 3
    assert filename.endswith(".ldr")
    with open(filename, 'w') as file:
        for brick in brick_positions:
            if brick[-1] == 1:
                transformation_string = "1 0 0 0 1 0 0 0 1"
            else:
                transformation_string = "0 0 -1 0 1 0 1 0 0"
            coords = brick[:3]
            x_coord = coords[0] * LDR_UNITS_PER_STUD
            y_coord = -coords[2] * LDR_UNITS_PER_PLATE * PLATES_PER_BRICK
            z_coord = coords[1] * LDR_UNITS_PER_STUD
            brick_color = np.random.randint(0, 10) if color is None else color
            file.write('1 {} {} {} {} {} 3001.dat\n'.format(brick_color, x_coord, y_coord, z_coord, transformation_string))

def brick_position_to_pickle(filename:str, brick_positions):
    with open(filename, 'wb') as f:
        pickle.dump(brick_positions, f)

def brick_position_to_ldr(filename:str, brick_positions, color:int=None, emph_idx=None):
    LDR_UNITS_PER_STUD = 20
    LDR_UNITS_PER_PLATE = 8
    PLATES_PER_BRICK = 3
    assert filename.endswith(".ldr")
    with open(filename, 'w') as file:
        for brick_idx, brick_info in enumerate(brick_positions):
            brick_size, brick_position = brick_info

            # set rotation matrix
            if brick_size in ((2, 4), (2, 2), (1, 2)):
                transformation_string = "1 0 0 0 1 0 0 0 1"
            elif brick_size in ((4, 2), (2, 1)):
                transformation_string = "0 0 -1 0 1 0 1 0 0"

            # set brick coordinate
            coords = brick_position
            x_coord = coords[2] * LDR_UNITS_PER_STUD
            y_coord = -coords[1] * LDR_UNITS_PER_PLATE * PLATES_PER_BRICK
            z_coord = coords[0] * LDR_UNITS_PER_STUD

            # set brick model and adjust coordinate
            if brick_size in [(2, 4), (4, 2)]:
                brick_model = "3001.dat"
            elif brick_size in [(1, 2)]:
                brick_model = "3004.dat"
                z_coord -= 0.5 * LDR_UNITS_PER_STUD
            elif brick_size in [(2, 1)]:
                brick_model = "3004.dat"
                x_coord -= 0.5 * LDR_UNITS_PER_STUD
            elif brick_size in [(2, 2)]:
                brick_model = "3003.dat"
            else:
                raise ValueError(f"Undefined brick size: {brick_size}")

            brick_color = np.random.randint(0, 10) if color is None else color
            # emph specific brick idx
            if emph_idx is not None:
                brick_color = 0 # dark gray
                if emph_idx == brick_idx:
                    brick_color = 3 # red
            file.write('1 {} {} {} {} {} {}\n'.format(brick_color, x_coord, y_coord, z_coord, transformation_string, brick_model))

def brick_position_to_gif(filename:str, brick_positions:List[np.ndarray]):
    assert filename.endswith(".gif")

    # import pyrender
    if "pyrender" not in dir():
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        os.environ['MESA_GL_VERSION_OVERRIDE'] = '4.1'
        try:
            import pyrender
            import trimesh
            import imageio
        except:
            print("[ERROR] cannot import pyrender")
            return 
    frames = []

    # create renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480, point_size=1.0)

    # set scene
    scene = pyrender.Scene()
    camera = pyrender.OrthographicCamera(xmag=300, ymag=300, zfar=1000)
    camera_pose = np.eye(4) @ \
        trimesh.transformations.rotation_matrix(-np.pi / 4, [0,1,0]) @ \
        trimesh.transformations.rotation_matrix(np.pi, [1,0,0]) @ \
        trimesh.transformations.rotation_matrix(-np.pi / 8, [1,0,0])
    camera_pose[0:3,3] = np.array([400, -200, -500])
    scene.add(camera, pose=camera_pose)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=30.0)
    scene.add(light, pose=camera_pose)

    LDR_UNITS_PER_STUD = 20
    LDR_UNITS_PER_PLATE = 8
    PLATES_PER_BRICK = 3
    
    for coords in brick_positions:
        if coords[-1] == 1:
            transformation_string = "1 0 0 0 1 0 0 0 1"
        else:
            transformation_string = "0 0 -1 0 1 0 1 0 0"

        mat = np.eye(4)
        mat[:3, :3] = np.array([int(x) for x in transformation_string.split()]).reshape(3,3)
        mat[0,3] = coords[0] * LDR_UNITS_PER_STUD * 1.6
        mat[1,3] = -coords[2] * LDR_UNITS_PER_PLATE * PLATES_PER_BRICK * 1.6
        mat[2,3] = coords[1] * LDR_UNITS_PER_STUD * 1.6

        brick = trimesh.load("3001.stl")
        brick = pyrender.Mesh.from_trimesh(brick,
            smooth=False,
            material=pyrender.MetallicRoughnessMaterial(metallicFactor=1.0, roughnessFactor=0.5, baseColorFactor=trimesh.visual.random_color())
        )
        scene.add(brick, pose=mat)

        color, _ = renderer.render(scene)
        frames.append(color)
    
    renderer.delete()
    imageio.mimsave(filename, frames, fps=2)


def is_pos_the_same(pos1, pos2):
    if len(pos1) != len(pos2):
        return False

    for pos_batch1, pos_batch2 in zip(pos1, pos2):
        if len(pos_batch1) != len(pos_batch2):
            return False

        for brick1, brick2 in zip(pos_batch1, pos_batch2):
            size1, pos1 = brick1
            size2, pos2 = brick2
            if size1 != size2:
                return False
            if any(pos1 != pos2):
                return False

    return True