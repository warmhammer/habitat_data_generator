import argparse
import os
from pathlib import Path

import hadage as hdg


def get_parser():
    parser = argparse.ArgumentParser(description="Generate data based on simulation settings.")
    
    parser.add_argument(
        "sim_settings_path",
        type=Path, 
        help="Path to simulation settings file or directory"
    )
    
    parser.add_argument(
        "--package_dir_path",
        type=Path,
        default=Path('./'),
        help="Path to the package directory"
    )
    parser.add_argument(
        "--data_dir", 
        type=Path, 
        help="Path to the data directory",
        default=Path('/data/')
    )
    parser.add_argument(
        "--output_path", 
        type=Path, 
        help="Path to save the output",
        default=Path('generated/')
    )
    
    return parser
    

def create_experiment(sim_settings_filename):
    logger = hdg.ExperimentLogger(sim_settings_filename)
    sim_settings, _ = logger.get_settings()

    sim = hdg.create_simulator(sim_settings)

    return {'logger': logger, 'sim': sim}


def run_experiment(experiment, start_rotation=None, bar_inds=[0, 1], display=False):
    logger = experiment['logger']
    sim = experiment['sim']

    sim_settings, light_settings = logger.get_settings()
    start_point, navigatable_points = hdg.generate_scenario(sim, sim_settings, display=display)

    sim = hdg.place_agent(sim, sim_settings['default_agent'], start_point, start_rotation=start_rotation)
    hdg.run_scenario(sim, sim_settings, light_settings, navigatable_points, logger, display=display, bar_inds=bar_inds)
    
    logger.save_camera_params(sim, sim_settings['default_agent'])
    logger.save_classes_list()
    

def main(args):
    if str(args.sim_settings_path).endswith('.json'):
        sim_settings_pathes = [str(args.sim_settings_path)]
    else:
        sim_settings_filenames = sorted(os.listdir(
            os.path.join(args.package_dir_path, 'configs/sim_settings/', args.sim_settings_path)
        ))
        
        sim_settings_pathes = [
            os.path.join(args.sim_settings_path, filename)
                for filename in sim_settings_filenames
        ]
        
    for i, sim_settings_filename in enumerate(sim_settings_pathes):
        experiment = create_experiment(sim_settings_filename)
        run_experiment(experiment, bar_inds=[i*2, i*2 + 1])
        

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    main(args)
