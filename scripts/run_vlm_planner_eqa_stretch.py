from omegaconf import OmegaConf
import os
from pathlib import Path
import torch

import hydra_python

from graph_eqa.utils.hydra_utils import initialize_hydra_pipeline_stretch
from graph_eqa.stretch_ai_agent.robot_hydra_agent import RobotHydraAgent
from graph_eqa.stretch_ai_agent.utils import load_stretch_questions_data
from graph_eqa.scene_graph.scene_graph_sim import SceneGraphSim
from graph_eqa.planners import VLMPlannerEQAGemini, VLMPlannerEQAGPT

from stretch.agent.zmq_client import HomeRobotZmqClient
from stretch.core import get_parameters
from stretch.perception import create_semantic_sensor


def main(stretch_parameter_file, cfg):

    output_path = hydra_python.resolve_output_path(Path(__file__).resolve().parent.parent / cfg.output_path)
    os.makedirs(output_path, exist_ok=True)

    results_filename = output_path / f'{cfg.results_filename}.json'

    questions_data = load_stretch_questions_data(Path(__file__).resolve().parent.parent / cfg.data.question_data_path)
    question_data = questions_data[cfg.question]
    vlm_question, vlm_pred_candidates = question_data['vlm_question'], question_data['vlm_pred_candidates']
    choices, answer, clean_ques_ans, enrich_labels = question_data['choices'], question_data['answer'], question_data['clean_ques_ans'], question_data['enrich_labels']

    # Need to define these arguments
    # Create robot
    parameters = get_parameters(stretch_parameter_file)
    robot = HomeRobotZmqClient(
        robot_ip=parameters.data['robot_ip'],
        use_remote_computer=True,
        output_path=output_path,
        parameters=parameters,
        enable_rerun_server=parameters.data['enable_rerun_server'],
        publish_observations=parameters.data['enable_realtime_updates'],
    )

    print("- Create semantic sensor based on detic")
    device_id = parameters.data['device_id_sem_sensor']
    if parameters.data['use_semantic_sensor']:
        semantic_sensor = create_semantic_sensor(
            parameters=parameters,
            device_id=device_id,
            verbose=True,
        )
        sensor_categories_mapping = semantic_sensor.seg_id_to_name
        # write_config_yaml(sensor_categories_mapping)
        # import ipdb; ipdb.set_trace()
    else:
        semantic_sensor = None
        sensor_categories_mapping = None
        
    obs = robot.get_observation()
    hydra_pipeline = initialize_hydra_pipeline_stretch(
        cfg.hydra, 
        obs, 
        output_path, 
        sensor_categories_mapping=sensor_categories_mapping
    )

    device = f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu"

    sg_sim = SceneGraphSim(
        cfg, 
        output_path, 
        hydra_pipeline, 
        rr_logger=None, 
        device=device, 
        clean_ques_ans=clean_ques_ans,
        enrich_object_labels=enrich_labels)

    agent = RobotHydraAgent(
        robot, 
        parameters, 
        hydra_pipeline, 
        sg_sim,
        semantic_sensor=semantic_sensor, 
        output_path=output_path,
        enable_realtime_updates=parameters.data['enable_realtime_updates']
    )
    agent.start()
    agent.update()
   
    if parameters["agent"]["in_place_rotation_steps"] > 0:
        agent.rotate_in_place(
            steps=parameters["agent"]["in_place_rotation_steps"],
            visualize=False,
        )
    else:
        agent.initialize_at_init_pose(
            steps=parameters["agent"]["in_place_static_steps"],
            visualize=False,
        )
    agent.sg_step()

    # print("============writing pickle file")
    # write_to_pickle(agent.obs_history, 'data_with_semantics')
    # click.secho(f'Location: Bosch Pittsburgh lab',fg="green",)

    if 'gpt' in cfg.vlm.name.lower():
        vlm_planner = VLMPlannerEQAGPT(
            cfg.vlm,
            sg_sim,
            vlm_question, vlm_pred_candidates, choices, answer, 
            output_path)
    elif 'gemini' in cfg.vlm.name.lower():
        vlm_planner = VLMPlannerEQAGemini(
            cfg.vlm,
            sg_sim,
            vlm_question, vlm_pred_candidates, choices, answer, 
            output_path)
    else:
        raise NotImplementedError('VLM planner not implemented.')

    agent.run_eqa_vlm_planner(
        vlm_planner,
        sg_sim,
        manual_wait=False,
        max_planning_steps=cfg.planner.max_planning_steps,
        go_home_at_end=False,
        results_filename=results_filename
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", help="cfg file name", default="", type=str, required=True)
    args = parser.parse_args()

    stretch_config_path = str(Path(__file__).resolve().parent.parent / 'cfg' / 'stretch_interface.yaml')

    cfg_path = Path(__file__).resolve().parent.parent / 'cfg' / f'{args.cfg_file}.yaml'
    cfg = OmegaConf.load(cfg_path)
    OmegaConf.resolve(cfg)

    main(stretch_config_path, cfg)
