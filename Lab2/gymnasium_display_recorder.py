import gymnasium as gym
from gymnasium.wrappers import RecordVideo

import glob
import io
import base64

from IPython.display import HTML
from IPython import display as ipythondisplay


def gym_make(env_name, run_name, render_fps=30):

    env = gym.make(
        env_name,
        render_mode="rgb_array"
    )

    env.metadata["render_fps"] = render_fps

    env = RecordVideo(
        env,
        video_folder="./videos",
        episode_trigger=lambda episode_id: True,
        name_prefix=f"video_{run_name}"
    )

    return env


def show(run_name, episode_id=0):

    videos = glob.glob(
        f"./videos/video_{run_name}*{episode_id}.mp4"
    )

    if not videos:
        raise FileNotFoundError(
            f"Cannot find video to run '{run_name}', "
            f"episode {episode_id}."
        )

    with open(videos[0], "rb") as video_file:
        video = video_file.read()

    encoded = base64.b64encode(video)

    ipythondisplay.display(
        HTML(
            f"""
            <video width="640" height="480" controls>
                <source
                    src="data:video/mp4;base64,{encoded.decode('ascii')}"
                    type="video/mp4"
                />
            </video>
            """
        )
    )
