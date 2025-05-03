# GraphEQA
This repo provides code for GraphEQA, a novel approach for utilizing 3D scene graphs for embodied question answering (EQA), introduced in the paper [GraphEQA: Using 3D Semantic Scene Graphs for Real-time Embodied Question Answering](https://www.arxiv.org/abs/2412.14480).

<div align="center">
    <img src="doc/grapheqa.gif">
</div>

* Website: https://grapheqa.github.io/

If you find GraphEQA relevant or useful for your research, please use the following citation:

```bibtex
@misc{2024grapheqausing3dsemantic,
      title={GraphEQA: Using 3D Semantic Scene Graphs for Real-time Embodied Question Answering}, 
      author={Anonymous Authors},
      year={2025}
}
```

#### Funding and Disclaimer
This work was in part supported by the National Science Foundation
under Grant No. CMMI-1925130 and in part by the EU Horizon 2020
research and innovation program under grant agreement No. 101017274
(DARKO). Any opinions, findings, and conclusions or recommendations
expressed in this material are those of the author(s) and do not necessarily
reflect the views of the NSF.

## GraphEQA Workspace Configuration
Below are instructions for how to set up a workspace to run GraphEQA. We provide instructions for both Docker and a local setup if you happen to be running Ubuntu 20.04.

Owners and collaborators of this repo are not claiming to have developed anything original to Hydra or any other MIT Spark lab tools.

### Docker setup
#### Prerequisites
1. Install docker.

2. Run `git clone https://github.com/grapheqa/graph_eqa.git`. GraphEQA is also pip installable should you prefer installing it in a Python environment (see below).

#### Set up Docker workspace for running GraphEQA for Habitat-Sim
There are two Docker images for GraphEQA to support simulation-based experiments in Habitat-Sim and embodied experiments on the Hello Robot Stretch platform.

Run the following script to build the docker image locally.

```bash
./docker/docker_build_with_habitat.sh
```

The following script will run a container:

```bash
./docker/docker_run_habitat.sh
```

If the docker container is already running and you want to execute the container in another terminal instance, do:

``` bash
docker exec -it grapheqa-for-habitat bash
```

You will need to download the HM3D dataset and set up `grapheqa_habitat.yaml` to point to the directories local to your system that contain the HM3D dataset, Explore-EQA dataset, etc.

#### Set up Docker workspace for running GraphEQA on Hello Robot's Stretch
Run the following script to build the docker image locally.

```bash
./docker/docker_build.sh
```

The following script will run a container 

```bash
./docker/docker_run.sh
```

If the docker container is already running and you want to execute the container in another terminal instance, do:

``` bash
docker exec -it grapheqa-for-stretch bash
```

### Setting up Hydra on Ubuntu 20.04
This set of instructions is only for local Ubuntu 20.04 installations.

0) Install our fork of Hydra following the instructions at [this link](https://github.com/grapheqa/Hydra). Verify that you are on the `grapheqa` branch.

1) If you do not have conda, install it. Then create a conda environment:

``` bash
conda create -n "grapheqa" python=3.10
conda activate grapheqa
```

2) Follow the instructions for [installing the Hydra Python bindings](https://github.com/MIT-SPARK/Hydra/blob/main/python/README.md) inside of the conda environment created above. Before installing, be sure to source the hydra catkin workspace, otherwise the installation of the python bindings will fail.

3) [Install Habitat Simulator](https://github.com/facebookresearch/habitat-sim#installation) in the 'grapheqa' conda environment.

### Download the HM3D dataset
The HM3D dataset along with semantic annotations can be downloaded from [here](https://github.com/matterport/habitat-matterport-3dresearch). Follow the instructions [here](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#habitat-matterport-3d-research-dataset-hm3d) to download together the train/val scenes, semantic annotations and configs. The data folder should look as follows:

```graphql
hm3d/train
├─ hm3d-train-semantic-annots-v0.2
|    ├─ ...
├─ 00366-fxbzYAGkrtm
|    ├─ fxbzYAGkrtm.basis.glb
|    ├─ fxbzYAGkrtm.basis.navmesh
|    ├─ fxbzYAGkrtm.semantic.glb
|    ├─ fxbzYAGkrtm.semantic.txt
├─ ...
```

### Download Explore-EQA dataset
Navigate to [this repo](https://github.com/grapheqa/explore-eqa_semnav/tree/master/data) and download `questions.csv` and `scene_init_poses.csv` into a directory in your workspace.  

### Install the Stretch AI package from Hello Robot
If running GraphEQA on Stretch RE2 platform, follow the install instructions at our fork of `stretch_ai` [found here](https://github.com/grapheqa/stretch_ai).

### Installing GraphEQA
Clone and install GraphEQA in the 'grapheqa' conda environment:

```bash
git clone git@github.com:grapheqa/graph_eqa.git
cd graph_eqa
pip install -e .
```

The OpenAI API requires an API key. Add the following line to your .bashrc:

`export OPENAI_API_KEY=<YOUR_OPENAI_KEY>`

If using Google's Gemini, add the following line to your .bashrc:

`export GOOGLE_API_KEY=<YOUR_GOOGLE_KEY>`

If using Anthropic's Claude, add the following line to your .bashrc:

`export ANTHROPIC_API_KEY=<YOUR_GOOGLE_KEY>`

## Running GraphEQA with Habitat

### Config updates

Update paths to Explore-EQA data: Change `question_data_path` and `init_pose_data_path` fields in `grapheqa_habitat.yaml` to correspond to the locations where you downloaded the `questions.csv` and `scene_init_poses.csv` files from the [Explore-EQA dataset](#download-explore-eqa-dataset).

Update paths to HM3D data: Change `scene_data_path` and `semantic_annot_data_path` fields in `grapheqa_habitat.yaml` to correspond to the directories where you downloaded the [HM3D data](#download-the-hm3d-dataset).

### Run script
To run GraphEQA with Habitat Sim, run:
```bash
python scripts/run_vlm_planner_eqa_habitat.py -cf grapheqa_habitat
```
Results will be saved in the `graph_eqa/outputs` directory.

## Running GraphEQA with Habitat Hello Robot's Stretch RE2
To run GraphEQA on Hello Robot's Stretch platform, you will need to run the server on the Stretch robot following the instructions at [this fork](https://github.com/grapheqa/stretch_ai). Once you have successfully launched the server, open a terminal on your computer (client side) and run:

```bash
cd graph_eqa
python scripts/run_vlm_planner_eqa_stretch.py -cf grapheqa_stretch
```
