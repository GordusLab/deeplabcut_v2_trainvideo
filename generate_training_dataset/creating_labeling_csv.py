import numpy as np
import deeplabcut
import os, glob, matplotlib.pyplot as plt
import pandas as pd
from motmot.SpiderMovie import SpiderMovie

def creating_labeling_csv(cfg_path, basepath, videoname, joint_filename):
    if not joint_filename.endswith('s_dlc_abs.npy'):
        raise Exception('Unsupported input file. Only *s_dlc_abs.npy files allowed.')

    ## Reading joint data
    joints_data = np.load(joint_filename)
    N_time = joints_data.shape[0]
    N_joints = joints_data.shape[1]
    N_xyp = joints_data.shape[2]
    cfg = deeplabcut.auxiliaryfunctions.read_config(cfg_path)
    # Reading config file and its variables


    ## Reading congfig file
    scorer = cfg['scorer']
    bodyparts = cfg['bodyparts']
    videos = cfg['video_sets'].keys()
    markerSize = cfg['dotsize']
    alpha = cfg['alphavalue']
    colormap = plt.get_cmap(cfg['colormap'])
    colormap = colormap.reversed()
    project_path = cfg['project_path']

    ## Reading video
    vid_name = [os.path.join(basepath, 'raw', videoname)][0]
    mov = SpiderMovie(vid_name)
    relativeimagenames = ['labeled-data/' + videoname + '/' + str(n) for n in range(len(mov))]

    dataframe = None
    count = 0

    print("CREATING-SOME LABELS FOR THE FRAMES")
    frames = os.mkdir(os.path.join(cfg["project_path"], "labeled-data", videoname))

    for bodypart in bodyparts:
        columnindex = pd.MultiIndex.from_product(
            [[scorer], [bodypart], ["x", "y"]], names=["scorer", "bodyparts", "coords"]
        )
        frame = pd.DataFrame(joints_data[:, count, 0:2], columns=columnindex, index=relativeimagenames)
        dataframe = pd.concat([dataframe, frame], axis=1)
        count = count + 1

    dataframe.to_csv(
            os.path.join(
                cfg["project_path"],
                "labeled-data",
                videoname,
                "CollectedData_" + scorer + ".csv",
            )
    )

    dataframe.to_hdf(
        os.path.join(
            cfg["project_path"],
            "labeled-data",
            videoname,
            "CollectedData_" + scorer + ".h5",
        ),
        "df_with_missing",
        format="table",
        mode="w",
    )


