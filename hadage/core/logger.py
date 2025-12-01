import json
import os
import shutil
import yaml

import numpy as np
from PIL import Image

from hadage.core.settings import load_sim_settings, load_light_settings
from hadage.tools.common import get_camera_matrix
from hadage.tools.visual import display_sample


def get_output_paths(dir_path, dataset_name, scene_name, label):
    output_path = os.path.join(
        dir_path, dataset_name, label, scene_name
    )

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        shutil.rmtree(output_path)

    image_output_path = os.path.join(output_path, 'results/')

    if not os.path.exists(image_output_path):
        os.makedirs(image_output_path)

    sim_settings_output_path = os.path.join(output_path, 'configs/sim_settings/')
    if not os.path.exists(sim_settings_output_path):
        os.makedirs(sim_settings_output_path)

    lights_output_path = os.path.join(output_path, 'configs/lighting/')
    if not os.path.exists(lights_output_path):
        os.makedirs(lights_output_path)


    return output_path, image_output_path, sim_settings_output_path, lights_output_path


class ExperimentLogger:
    def __init__(self, sim_settings_filename, package_dir_path='./', data_dir='/data', output_path='generated/', display=False, display_freq=50) -> None:
        self.__sim_settings_filename = sim_settings_filename

        configs_path = os.path.join(package_dir_path, 'configs/')
        sim_settings_path = os.path.join(configs_path, 'sim_settings/', sim_settings_filename)

        # self.data_dir = os.path.join(package_dir_path, '../data/generated/') #######
        self.data_dir = data_dir

        self.sim_settings = load_sim_settings(sim_settings_path, self.data_dir)

        light_settings_path = os.path.join(configs_path, 'lighting/', self.sim_settings['light_settings_filename'])
        self.light_settings = load_light_settings(light_settings_path)

        self.output_dir_path, self.image_output_path, sim_settings_output_path, lights_output_path = get_output_paths(
            output_path, 
            self.sim_settings['dataset_name'], 
            self.sim_settings['scene_name'],
            self.sim_settings['label']
        )

        shutil.copy(sim_settings_path, sim_settings_output_path)
        shutil.copy(light_settings_path, lights_output_path)

        self.depth_scale = self.sim_settings['depth_scale']

        self._step_index = 0

        self._display = display
        self._display_freq = display_freq

    def __str__(self) -> str:
        return self.__sim_settings_filename
    
    def set_display(self, display: bool) -> None:
        self._display = display

    def save_step(self, observations, transformation_matrix, display=None):
        rgb = observations.get("color_sensor", None)
        depth = observations.get("depth_sensor", None)
        semantic = observations.get("semantic_sensor", None)
    
        if rgb is not None:
            rgb_image = Image.fromarray(rgb).convert('RGB')
            rgb_image.save(os.path.join(
                self.image_output_path, 
                f'frame{self.get_str_index()}.jpg'
            ))

        if depth is not None:
            depth_image = Image.fromarray(
                depth.astype(float) * self.depth_scale
            ).convert('I')

            depth_image.save(os.path.join(
                self.image_output_path, 
                f'depth{self.get_str_index()}.png'
            ))

        if semantic is not None:
            if self.sim_settings['remap_classes']:
                semantic = self.sim_settings['classes_remapping'][semantic]

            semantic_image = Image.fromarray(
                semantic
            ).convert('I')

            semantic_image.save(os.path.join(
                self.image_output_path, 
                f'semantic{self.get_str_index()}.png'
            ))

        with open(os.path.join(self.output_dir_path, "traj.txt"), "a") as traj_file:
            np.savetxt(traj_file, transformation_matrix, newline=" ")
            traj_file.write("\n")

        display = self._display if display is None else display

        if display and (self._step_index % self._display_freq == 0):
                display_sample(rgb, depth_obs=depth, semantic_obs=semantic)

        self._step_index += 1

    def save_camera_params(self, sim, agent_index=None, printed=None):
        if agent_index is None:
            agent_index = self.sim_settings['default_agent']
            
        camera_matrix = get_camera_matrix(sim.agents[agent_index])
        
        printed = self._display if printed is None else printed

        depth = self.sim_settings['depth_scale']

        message = f"camera_matrix = \n{camera_matrix}\n\ndepth_scale = {depth}"

        if printed:
            print(message)

        with open(os.path.join(self.output_dir_path, "camera_params.txt"), "w") as file:
            file.write(message)
            
        
        width = self.sim_settings['width']
        height = self.sim_settings['height']
        fx = camera_matrix[0][0]
        fy = camera_matrix[1][1]
        cx = camera_matrix[0][2]
        cy = camera_matrix[1][2]
        
        
        camera_params = {
            'camera_params':
                {
                    'image_height': height, 
                    'image_width': width, 
                    'fx': float(fx),
                    'fy': float(fy),
                    'cx': float(cx),
                    'cy': float(cy),
                    'png_depth_scale': depth,
                    'crop_edge': 0
                }
        }

        with open(os.path.join(self.output_dir_path, "camera_params.yaml"), 'w') as yaml_file:
            yaml.dump(camera_params, yaml_file, default_flow_style=False)

    def save_classes_list(self):
        semseg_classes = self.sim_settings.get('semseg_classes', None)

        if semseg_classes is None:
            raise ValueError(
                    f"{self.sim_settings['dataset_name']} dataset is not supported yet. No semseg_classes"
            )

        with open(os.path.join(self.output_dir_path, "embed_semseg_classes.json"), 'w') as classes_file:
            json.dump({'classes': semseg_classes}, classes_file, indent=2)

    def add_entry(self, message, printed=None):
        printed = self._display if printed is None else printed

        if printed:
            print(message)

        with open(os.path.join(self.output_dir_path, "log.txt"), "a") as log_file:
            log_file.write(f'[{self.get_str_index()}] {message}\n')

    def get_step_index(self):
        return self._step_index
    
    def get_str_index(self):
        return str(self._step_index).zfill(6)

    def get_settings(self):
        return self.sim_settings, self.light_settings
    
    def get_settings_name(self):
        return self.__sim_settings_filename