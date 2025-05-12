from pathlib import Path
import yaml
import pickle
import datetime
from tqdm import trange
import numpy as np
from graph_eqa.occupancy_mapping.geom import fps
from omegaconf import OmegaConf


def _format_list(name, values, collapse=True, **kwargs):
    indent = kwargs.get("indent", 0)
    prefix = " " * indent + name if indent > 0 else name
    if collapse:
        indent += 2

    args = {
        k: v for k, v in kwargs.items() if k != "indent" and k != "default_flow_style"
    }
    args["indent"] = indent
    value_str = yaml.dump(values, default_flow_style=collapse, **args)
    return f"{prefix}: {value_str}"
    
def write_config_yaml(category_mapping, label_space='detic'):
    package_path = Path(__file__).absolute().parent.parent.parent
    output_path = package_path / f'cfg/label_spaces/{label_space}_label_space.yaml'
    invalid_labels = []
    surface_labels = []
    dynamic_labels = []
    object_labels = []
    output_names = []

    for id, name in enumerate(category_mapping.values()):
        output_names.append({"label": id, "name": name})
        if 'unknown' in name.lower():
            invalid_labels.append(id)
        elif 'floor' in name.lower():
            surface_labels.append(id)
        else:
            object_labels.append(id)

    with output_path.open("w") as fout:
        fout.write("---\n")

        fout.write(yaml.dump({"total_semantic_labels": len(category_mapping.values())}))
        fout.write(_format_list("dynamic_labels", dynamic_labels))
        fout.write(_format_list("invalid_labels", invalid_labels))
        fout.write("object_labels:\n")
        for name in object_labels:
            fout.write("  - " + yaml.dump(repr(name), default_flow_style=True))
        fout.write(_format_list("surface_places_labels", surface_labels))

        fout.write("label_names:\n")
        for name in output_names:
            fout.write("  - " + yaml.dump(name, default_flow_style=True))

def write_to_pickle(obs_history, filename: str):
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    output_pkl_filename = filename + "_" + formatted_datetime + ".pkl"

    """Write out to a pickle file. This is a rough, quick-and-easy output for debugging, not intended to replace the scalable data writer in data_tools for bigger efforts."""
    data: dict[str, Any] = {}
    data["camera_poses"] = []
    data["camera_K"] = []
    # data["base_poses"] = []
    data["xyz"] = []
    # data["world_xyz"] = []
    data["rgb"] = []
    data["depth"] = []
    data["semantic"] = []
    data["instance"] = []
    # data["feats"] = []
    
    print(f"len of obs history = {len(obs_history)}")
    for t in trange(len(obs_history), desc="Processing"):
        frame = obs_history[t]
        # add it to pickle
        data["camera_poses"].append(frame.camera_pose)
        # data["base_poses"].append(frame.base_pose)
        data["camera_K"].append(frame.camera_K)
        data["xyz"].append(frame.xyz)
        # data["world_xyz"].append(frame.full_world_xyz)
        data["rgb"].append(frame.rgb)
        data["depth"].append(frame.depth)
        # data["feats"].append(frame.feats)
        data["semantic"].append(frame.semantic)
        data["instance"].append(frame.instance)

    print("============saving file")
    with open(output_pkl_filename, "wb") as f:
        pickle.dump(data, f)


def cluster_frontiers(frontier_points, min_points_for_clustering, num_clusters, cluster_threshold):
    # # cluster, or return none
    if len(frontier_points) < min_points_for_clustering:
        return frontier_points

    clusters = fps(frontier_points, num_clusters)

    # merge clusters if too close to each other
    clusters_new = np.empty((0, 3))
    for cluster in clusters:
        if len(clusters_new) == 0:
            clusters_new = np.vstack((clusters_new, cluster))
        else:
            clusters_array = np.array(clusters_new)
            dist = np.sqrt(np.sum((clusters_array - cluster) ** 2, axis=1))
            if np.min(dist) > cluster_threshold:
                clusters_new = np.vstack((clusters_new, cluster))
    return clusters_new

def load_stretch_questions_data(filepath):
    questions_data_file = OmegaConf.load(filepath)
    OmegaConf.resolve(questions_data_file)

    questions_data = []
    for k, v in questions_data_file.items():
        vlm_question = clean_ques_ans = v.question
        vlm_pred_candidates = v.choices.keys()
        choices = v.choices.values()
        for token, choice in zip(vlm_pred_candidates, choices):
            vlm_question += "\n" + token + "." + " " + choice
            if ("do not choose" not in choice.lower()) and (choice.lower() not in ['yes', 'no']):
                clean_ques_ans += "  " + token + "." + " " + choice
        
        questions_data.append(
            {
                "vlm_question": vlm_question,
                "clean_ques_ans": clean_ques_ans,
                "vlm_pred_candidates": vlm_pred_candidates,
                "choices": choices,
                "answer": v.answer,
                "enrich_labels": v.enrich_labels

            }
        )
    return questions_data