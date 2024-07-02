import argparse
import gymnasium as gym
import os
import moviepy.editor as mpy
from random_agent import random_agent

def create_video(env_name, agent, output_dir, num_episodes=1, render_mode='rgb_array'):
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
    """
    
    env = gym.make(env_name, render_mode=render_mode)

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False

        frames = []
        while not done:
            if render_mode == 'human':
                env.render() # Render the environment for human viewing
            else:
                frames.append(env.render()) # Capture frames for video

            action = agent.select_action(state)
            state, reward, done, truncated, info = env.step(action)

        if render_mode == 'rgb_array':
            # Create video file using MoviePy ('pip install MoviePy' -if needed)
            video_path = os.path.join(output_dir, f"{env_name}_episode_{episode + 1}.mp4")
            print(f"Saving video to {video_path}")
            clip = mpy.ImageSequenceClip(frames, fps=env.metadata.get('render_fps', 30))
            clip.write_videofile(video_path, codec='libx264')

        env.close()


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Create videos of an agent in a Gymnasium environment.")
    parser.add_argument("--env", type=str, required=True, help="The Gymnasium environment to use.")
    parser.add_argument("--episodes", type=int, default=1, help="The number of episodes to record.")
    parser.add_argument("--output_dir", type=str, default="./videos", help="Directory to save the videos.")
    parser.add_argument("--render_mode", type=str, choices=['human', 'rgb_array'], default='rgb_array', help="Render mode: 'human' for live rendering, 'rgb_array' for video recording.")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Extract arguments
    env_name = args.env
    num_episodes = args.episodes
    output_dir = args.output_dir
    render_mode = args.render_mode

    # Create a random agent
    agent = random_agent(gym.make(env_name).action_space)

    # Create the video
    create_video(env_name, agent, output_dir, num_episodes, render_mode)

"""
Example usage:
    python videos.py --env CartPole-v1 --episodes 5 --output_dir ./videos --render_mode rgb_array
    or
    python videos.py --env CartPole-v1 --episodes 1 --render_mode human
"""
