
# Habitat Data Sequences Generation Scripts
We present **Hadage** (\[Ha\]bitat \[Da\]ta \[Ge\]nerator). This generator is part of [OSMa-Bench](https://be2rlab.github.io/OSMa-Bench/) pipeline.

Hadage supports generation based on [Replica](https://github.com/facebookresearch/Replica-Dataset), [ReplicaCAD](https://aihabitat.org/datasets/replica_cad/) and [Habitat Matterport 3D](https://aihabitat.org/datasets/hm3d/) datasets.

## Project file structure
Folder tree should be the same.
```bash
habitat-data-generator/
    configs/
        lighting/         # lights configs folder
            ...
        sim_settings/     # sim (scene and agent) configs folder
            ...
```
Make sure to name your files correctly.
```bash
data/
    hm3d/
        hm3d_annotated_basis.scene_dataset_config.json
        minival/
            hm3d_annotated_minival_basis.scene_dataset_config.json
            00800-TEEsavR23oF/
            00801-HaxA7YrQdEC/
            ...
    replica_cad/
        ...
    replica/
        ...
```

## Datasets installation

### HM3D
Follow [installation guide](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#habitat-matterport-3d-research-dataset-hm3d). If you already have access to Matterport datasets:

```bash
python -m habitat_sim.utils.datasets_download --username <api-token-id> --password <api-token-secret> --uids hm3d_minival_v0.2
```

After installation:
* improve `shader_type` field in `scene_dataset_config` to `material` or `phong` (lights reconfiguring don't work with other shader types)

### Augmented Replica CAD
As part of [OSMa-Bench](https://be2rlab.github.io/OSMa-Bench/) we presented ReplicaCAD with augmented semantics annotations. You can easily download the dataset from [this repository](https://github.com/warmhammer/replica_cad_dataset).

```bash
git clone https://github.com/warmhammer/replica_cad_dataset replica_cad
```

### Original Replica CAD
Dataset might be installed in two ways. [Better one](https://aihabitat.org/datasets/replica_cad/):
```bash
# with conda install
python -m habitat_sim.utils.datasets_download --uids replica_cad_dataset --data-path data/

# with source
python /path/to/habitat_sim/src_python/habitat_sim/utils/datasets_download.py --uids replica_cad_dataset --data-path data/
```

And another one:
```bash
git lfs install
huggingface-cli lfs-enable-largefiles .
git clone https://huggingface.co/datasets/ai-habitat/ReplicaCAD_dataset -b main
```

Pay attention that we need ReplicaCAD **without** backed lights.

## Replica
Follow Replica dataset [installation guide](https://github.com/facebookresearch/Replica-Dataset?tab=readme-ov-file#download-on-mac-os-and-linux).

## Habitat Sim and Lab installation
We recommended to use proposed docker container to run Habitat.

```bash
make build-habitat  # building docker image

make up-habitat     # starting contatiner
make prepare        # prepare terminal for visualization
make into-habitat   # attach terminal to container

make stop-habitat   # stop contatiner
```


Another way is to follow [habitat-sim](https://github.com/facebookresearch/habitat-sim?tab=readme-ov-file#installation) and [habitat-lab](https://github.com/facebookresearch/habitat-lab?tab=readme-ov-file#installation) installation guides.

## Data Generation

```bash
python3 generate.py "examples/sim_settings_replica_cad.json"  # single config
python3 generate.py "examples/"                               # full directory
```

Also jupyter notebooks are available in `notebooks/` directory.

## Citing Hadage

Using Hadage in your research? Please cite the following paper: [arxiv](https://arxiv.org/abs/2503.10331)

```bibtex
@inproceedings{popov2025osmabench,
    title={OSMa-Bench: Evaluating Open Semantic Mapping Under Varying Lighting Conditions},
    author={Popov, Maxim and Kurkova, Regina and Iumanov, Mikhail and Mahmoud, Jaafar and Kolyubin, Sergey},
    booktitle={2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
    year={2025}
}
```