import argparse

import cv2
import gymnasium as gym
import os
import moviepy.editor as mpy
import wandb
import wrapper
from random_agent import random_agent
# *****Extend_Auf4)b*****
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from agent import DQNAgent
import gymnasium.wrappers.frame_stack as fs
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

wandb.require("core")


# *****Extend_Auf4)b*****
# Added agent_type
def create_video(env_name, agent, output_dir, cam,  steps_done, num_episodes=1, render_mode='rgb_array',
                 agent_type='random', log_wandb=False):
    # output_dir,
    print("in create_video")
    """ 
    Creates a video of an agent's performance in a Gym environment.

    Args:
        env_name (str): Name of the Gym environment.
        agent (object): Agent object that interacts with the environment.
        output_dir (str): Directory to save the generated videos.
        num_episodes (int, optional): Number of episodes to record. Defaults to 1.
        render_mode (str, optional): Rendering mode:
            - 'human' for live rendering.
            - 'rgb_array' for video recording. Defaults to 'rgb_array'.
        agent_type (str, optional): The type of agent (e.g., 'random', 'dqn'). Defaults to 'random'.    
        log_wandb (bool, optional): to log results to Weights & biases. Default is set to False
    """

    env = gym.make(env_name, render_mode=render_mode)
    e = wrapper.Wrapper(envi=env)
    env = e.env

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        # *****Extend_Auf4)b*****
        action_values = []  # List to store action values for plotting

        frames = []
        cam_frames = []
        states = []
        actions = []
        steps = 0
        env.step(0)

        while not done:
            if steps > 100:
                break
            steps += 1
            if render_mode == 'human':
                r = env.render()  # Render the environment for human viewing
            else:
                r = env.render()
                frames.append(r)  # Capture frames for video

            action = agent.select_action(state)
            state, reward, done, truncated, info = env.step(action)
            states.append(state)
            actions.append(action)

            #print(f"state.shape: {state.shape}")
            #print(isinstance(state, fs.LazyFrames))
            img_array = Image.fromarray(state[3])
            #img_array.show()
            img_array = np.array(img_array)/255
            #print(img_array)
            cam_state = np.array(img_array)
            #print(f"cam_state.dtype bf: {cam_state.dtype}")
            #print(f"cam_state.shape bf: {cam_state.shape}")
            cam_state = torch.from_numpy(cam_state).float().unsqueeze(0)
            cam_state = cam_state.permute(1,2,0)
            cam_state = np.float32(cam_state)

            #print(f"cam_state.dtype: {cam_state.dtype}")
            #print(f"cam_state.shape: {cam_state.shape}")
            cam_frames.append(cam_state)

            # *****Extend_Auf4)b*****
            if agent_type == 'dqn':
                # Store the Q-value for the chosen action in that state
                # action_values.append(q_value)
                q_value = agent.compute_action_value(state, action)
                action_values.append(q_value)

        if render_mode == 'rgb_array':
            # Create video file using MoviePy ('pip install MoviePy' -if needed)
            video_path = os.path.join(output_dir, f"{env_name}_episode_{steps_done}.mp4")
            print(f"Saving video to {video_path}")
            # clip = mpy.ImageSequenceClip(frames, fps=env.metadata.get('render_fps', 30))
            # clip.write_videofile(video_path, codec='libx264')

            # *****Extend_Auf4)b*****
            if agent_type == 'random':
                clip = mpy.ImageSequenceClip(frames, fps=env.metadata.get('render_fps', 30))
            elif agent_type == 'dqn':
                # Create frames with environment and action value plots
                # side_by_side_frames = create_side_by_side_frames(frames, action_values)
                side_by_side_frames = create_side_by_side_frames(frames, cam_frames, action_values, cam, states, actions)
                # action_values,
                clip = mpy.ImageSequenceClip(side_by_side_frames, fps=env.metadata.get('render_fps', 30))

            clip.write_videofile(video_path, codec='libx264')

            if log_wandb:
                if os.path.exists(video_path):
                    print(f"Logging video to wandb: {video_path}")
                    wandb.log({f"video_{steps_done}": wandb.Video(video_path)})
                else:
                    print(f"Failed to save video: {video_path}")

        env.close()


# *****Extend_Auf4)b*****
def create_side_by_side_frames(env_frames, cam_frames, action_values, cam, astate, acrions):
    #  action_values,
    # To store the combined frames
    side_by_side_frames = []

    # Use the height of the environment frame as the standard height
    standard_height = env_frames[0].shape[0]

    for i, env_frame in enumerate(env_frames):
        # Create a new figure and axis for plotting
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(range(i + 1), action_values[:i + 1], label='Q-value of chosen action')
        ax.legend(loc='upper left')
        ax.set_title('Q-values Over Time')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Q-values')

        # Convert the plot to an image
        plt_frame = fig_to_image(fig)
        plt.close(fig)

        # Resize the plot frame to match the height of the environment frame
        plt_frame_resized = resize_image(plt_frame, standard_height)

        # Resize CAM image to match the height of the environment frame
        cam_image = cv2.resize(cam_frames[i], (env_frame.shape[1], standard_height))

        # Process state for GradCAM
        state = astate[i]
        action = acrions[i]
        state = np.array(state)
        state = torch.from_numpy(state).float().unsqueeze(0)#.requires_grad_(True)
        #print(f"state.shape: {state.shape}")
        grayscale_cam = cam(input_tensor=state, targets=[ClassifierOutputTarget(action)])
        grayscale_cam = grayscale_cam[0, :]
        img = np.array(cam_frames[i])
        cam_image = show_cam_on_image(img=cam_frames[i], mask=grayscale_cam, use_rgb=False)
        #print(cam_image.shape)
       #cam_image = cv2.cvtColor(cam_image, cv2.COLOR_GRAY2RGB)

        # Ensure CAM image is resized to match the height of the environment frame
        if cam_image.shape[0] != standard_height:
            cam_image = cv2.resize(cam_image, (env_frame.shape[1], standard_height))
        #img_array = Image.fromarray(cam_image)
        #img_array.show()

        #print(env_frame.shape)
        #print(plt_frame_resized.shape)
        #tetsddded


        # Combine the environment frame and plot frame side-by-side
        combined_frame = np.hstack((env_frame, plt_frame_resized, cam_image))
        # plt_frame_resized
        side_by_side_frames.append(combined_frame)
        #state = state.requires_grad_(False)

    return side_by_side_frames



def fig_to_image(fig):
    """
    Convert a matplotlib figure to a numpy array representation of an RGB image.

    Parameters:
    - fig: The matplotlib figure to convert.

    Returns:
    - np.array: Numpy array representing the RGB image.
    """

    # Draw the figure onto the canvas
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    # Convert the canvas to a numpy array of RGB pixels
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
    return image


def resize_image(image, target_height):
    """
    Resize a numpy array image to a target height while maintaining aspect ratio.

    Parameters:
    - image (np.array): Input numpy array representing an image.
    - target_height (int): Desired height of the resized image.

    Returns:
    - np.array: Resized numpy array representing the image.
    """

    # Convert the numpy array image to a PIL Image
    img = Image.fromarray(image)
    aspect_ratio = img.width / img.height
    target_width = int(target_height * aspect_ratio)
    # Resize
    img_resized = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
    # Convert the resized PIL Image back to a numpy array and return
    return np.array(img_resized)


# *****Extend_Auf4)b*****
# Added agent_type
if __name__ == "__main__":

    # Argument parsing
    parser = argparse.ArgumentParser(description="Create videos of an agent in a Gymnasium environment.")
    parser.add_argument("--env", type=str, required=True, help="The Gymnasium environment to use.")
    parser.add_argument("--episodes", type=int, default=1, help="The number of episodes to record.")
    parser.add_argument("--output_dir", type=str, default="./videos", help="Directory to save the videos.")
    parser.add_argument("--render_mode", type=str, choices=['human', 'rgb_array'], default='rgb_array',
                        help="Render mode: 'human' for live rendering, 'rgb_array' for video recording.")
    # *****Extend_Auf4)b*****
    parser.add_argument("--agent", type=str, choices=["random", "dqn"], default="random",
                        help="The type of agent to use.")

    parser.add_argument("--log_wandb", action="store_true", default=False, help="Log results to Weights & Biases.")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Initialize wandb
    if args.log_wandb and args.render_mode == 'rgb_array':
        wandb.init(project="my-awesome-project", entity="tudortmundg3-tu-dortmund")

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Extract arguments
    env_name = args.env
    num_episodes = args.episodes
    output_dir = args.output_dir
    render_mode = args.render_mode
    log_wandb = args.log_wandb
    # *****Extend_Auf4)b*****
    agent_type = args.agent

    # Create a random agent
    # agent = random_agent(gym.make(env_name).action_space)

    # *****Extend_Auf4)b*****
    # Create the appropriate agent based on user selection
    env = gym.make(env_name)
    if agent_type == 'random':
        agent = random_agent(env.action_space)
    elif agent_type == 'dqn':
        agent = DQNAgent(env, input_dim=env.observation_space.shape[0])
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")

    # Create the video
    create_video(env_name, agent, output_dir, num_episodes, render_mode, agent_type, args.log_wandb)

"""
Example usage:
    python videos.py --env CartPole-v1 --episodes 5 --output_dir ./videos --render_mode rgb_array
    python videos.py --env CartPole-v1 --episodes 5 --output_dir ./videos --render_mode rgb_array  --log_wandb
    or
    python videos.py --env CartPole-v1 --episodes 1 --render_mode human

    #*****Extend_Auf4)b*****
    python videos.py --env CartPole-v1 --episodes 3 --output_dir ./videos --render_mode rgb_array --agent random

    python videos.py --env CartPole-v1 --episodes 3 --output_dir ./videos --render_mode rgb_array --agent dqn


"""