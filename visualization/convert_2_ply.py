import os
import yaml
import numpy as np
import open3d as o3d
import random
from datasets.scannet200.owis_splits import KNOWN_CLASSES_LABELS,UNKNOWN_CLASSES_LABELS
from datasets.scannet200.scannet200_constants import CLASS_LABELS_200,VALID_CLASS_IDS_200
from copy import copy
# OUTPUT_2 = "output-scans/predictions-ours-2classes/"
label2id = {l:id for id,l in zip(VALID_CLASS_IDS_200, CLASS_LABELS_200)}
def generate_color_palette(x):
    palette = []

    while len(palette) < x:
        r, g, b = random.sample(range(256), 3)
        color = (r/255.0, g/255.0, b/255.0)
        if color[2] > 0:
            continue
        palette.append(color)

    return palette

def read_ply_file(file_path):
    # Read the ply file
    point_cloud = o3d.io.read_triangle_mesh(file_path)
    return point_cloud

model_name = "experiment_name"

for split in ['A','B','C']:
    for task in ['task1','task2','task3']:
        PATH = f"eval_output/{split}/{task}/instance_evaluation_{model_name}_0/decoder_-1/"
        label_database = "data/processed/scannet200/label_database.yaml"
        OUTPUT = f"output-scans/{model_name}/{split}/{task}/"
        OUTPUT_2 = f"output-scans/gt/{split}/{task}/"
        OUTPUT_3 = f"output-scans/input/{split}/{task}/"
        with open(label_database) as f:
            x = yaml.safe_load(f) 
            
        # read label database
        cls_ids = x.keys()
        # from dict keys to list
        cls_ids = [int(i) for i in cls_ids]
        for file in os.listdir(PATH):
            pc = []
            pco = []
            SCENE = file.split(".")[0]
            # if SCENE in good_scenes:
            PATH_PLY = f"data/raw/scannet_test_segments/scans/{SCENE}/{SCENE}_vh_clean_2.labels.ply"
            PATH_PLY_1 = f"data/raw/scannet_test_segments/scans/{SCENE}/{SCENE}_vh_clean_2.ply"
            
            # read label database
            try:
                label = f"data/processed/scannet200/instance_gt/validation/{SCENE}.txt"
                # read label file
                with open(label) as f:
                    lines = f.readlines()
                unique_ids = np.unique(np.asarray(lines))
                unique_ids = [int(i[:-1]) for i in unique_ids]
                classes = [i//1000 for i in unique_ids if i !=0]
                # if task != 'task3':
                if UNKNOWN_CLASSES_LABELS[split][task] != None:
                    UNKNOWNS =  [label2id[i] for i in UNKNOWN_CLASSES_LABELS[split][task]]
                else:
                    UNKNOWNS = []
                    
                instances = []
                if not os.path.exists(OUTPUT):
                    os.makedirs(OUTPUT)
                if not os.path.exists(OUTPUT_2):
                    os.makedirs(OUTPUT_2)
                if not os.path.exists(OUTPUT_3):
                    os.makedirs(OUTPUT_3)  
                    
                with open(f"{PATH}{SCENE}.txt") as f:
                    instances = f.readlines()
                # sort instances by probablity instances.split(" ")[2]
                instances = sorted(instances, key=lambda x: float(x.split(" ")[2]), reverse=False)

                

                pc = read_ply_file(PATH_PLY)
                pco = read_ply_file(PATH_PLY_1)
                points = np.asarray(pc.vertices)
                indices = []
                for instance in instances:
                    path = instance.split(" ")[0].strip()
                    cls = instance.split(" ")[1].strip()
                    # if(int(cls) not in cls_ids):
                    #     continue
                    
                    with open(os.path.join(PATH, path), "r") as f:
                        mask = f.readlines()
                    for idx, point in enumerate(mask):
                        if(point == "1\n"):
                            indices.append(idx)
                            # change color of point based on its instance id
                            if(int(cls) == 0 or int(cls) == 1 or int(cls) == 3):
                                pc.vertex_colors[idx] = [0.5, 0.5, 0.5]
                            elif(int(cls) == 3000):
                                pc.vertex_colors[idx] = [31/255,119/255,180/255]
                            else:
                                pc.vertex_colors[idx] = [42/255,157/255,37/255]
                    indices = np.unique(np.asarray(indices))
                        # indices that are not in the list
                    left_indices = np.setdiff1d(np.arange(len(mask)), indices)
                    for index in left_indices:
                        pc.vertex_colors[index] = [0.5, 0.5, 0.5]
                    # save new point cloud
                    o3d.io.write_triangle_mesh(os.path.join(OUTPUT, SCENE + "-3d_owis.ply"), pc)
                            
                        
                        
                    pc = read_ply_file(PATH_PLY)
                    points = np.asarray(pc.vertices)
                    for i in range(points.shape[0]):
                    # change color of point based on its instance id
                        if(int(lines[i][:-1])//1000 == 0 or int(lines[i][:-1])//1000 == 3 or int(lines[i][:-1])//1000 == 1): # make gray if 0 (nothing) or 3 (floor) or 1 (wall)
                            pc.vertex_colors[i] = [0.5, 0.5, 0.5]
                        elif(int(lines[i][:-1])//1000 in UNKNOWNS):
                            pc.vertex_colors[i] = [31/255,119/255,180/255]#[23/255,124/255,172/255]
                        else:
                            pc.vertex_colors[i] = [42/255,157/255,37/255] # green
                    o3d.io.write_triangle_mesh(os.path.join(OUTPUT_2, SCENE + "-gt.ply"), pc)
                    o3d.io.write_triangle_mesh(os.path.join(OUTPUT_3, SCENE + "-input.ply"), pco)
            except:
                pass