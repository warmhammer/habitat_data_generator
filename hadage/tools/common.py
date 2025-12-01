import quaternion
import numpy as np


def get_state_transform_matrix(sensor_state):
    quat = sensor_state.rotation
    trans = sensor_state.position

    rot_mat = quaternion.as_rotation_matrix(quat)
    
    matrix = np.zeros((4, 4), dtype=rot_mat.dtype)
    matrix[:3, :3] = rot_mat
    matrix[:3, 3] = trans
    matrix[3, 3] = 1

    matrix[:3, 1] *= -1
    matrix[:3, 2] *= -1

    # coords_rotation = np.diag([1, -1, -1, 1])
    # matrix = coords_rotation @ matrix

    return matrix


def get_camera_matrix(agent):
    sensor_spec = agent.agent_config.sensor_specifications[0]

    w = sensor_spec.resolution[1]
    h = sensor_spec.resolution[0]

    hfov = float(sensor_spec.hfov) * np.pi / 180.

    f = (1 / np.tan(hfov / 2.)) * w / 2

    K = np.array([
        [f, 0., (w-1) / 2],
        [0., f, (h-1) / 2],
        [0., 0., 1]
    ])

    return K
