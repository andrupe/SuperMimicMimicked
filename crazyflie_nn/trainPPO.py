import functools
import os
import argparse
from datetime import datetime
from pathlib import Path
import time
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import model
from orbax import checkpoint as ocp
import matplotlib.pyplot as plt
import yaml
import subprocess
from environment.env import DroneBrax 

from rich.console import Console
from rich.table import Table
console = Console()

POLICY_NETWORK = (64,)
eval_reward_storage = []
eval_len_storage = []
pos_reward_storage = []



def write_enviroment_data(env,name,args):
    '''Writes Environment parameters as well as training hyperparameters.'''
    args_dict = vars(args)
    args_dict.update({'Policy NN size': str(POLICY_NETWORK)})

    data = {
        'Hyperparameters':args_dict,
        'Environment Variables': env.reward_parameters
    }
    yaml_path = name.with_suffix('.yaml')
    try:
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    except:
        print("Failed to write yaml!")
        exit()

def plot_training_results(eval_reward_storage, eval_len_storage, pos_reward_storage, savepath):
    '''Plots evaluation metrics in a clean, vertically stacked layout.'''
    
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('ggplot')
    
    # Create a figure with 3 subplots (stacked vertically)
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Main title for the entire figure
    fig.suptitle('Training Evaluation Metrics Over Time', fontsize=16, fontweight='bold', y=0.95)
    
    # Create an x-axis representing the evaluation steps (1, 2, 3...)
    steps = range(1, len(eval_reward_storage) + 1)
    
    # --- Plot 1: Eval Reward ---
    axes[0].plot(steps, eval_reward_storage, color='dodgerblue', linewidth=2, marker='o', markersize=4)
    axes[0].set_title('Evaluation Reward', fontsize=14)
    axes[0].set_ylabel('Reward', fontsize=12)
    
    # --- Plot 2: Episode Length ---
    axes[1].plot(steps, eval_len_storage, color='darkorange', linewidth=2, marker='s', markersize=4)
    axes[1].set_title('Average Episode Length', fontsize=14)
    axes[1].set_ylabel('Steps', fontsize=12)
    
    # --- Plot 3:Position Reward ---
    axes[2].plot(steps, pos_reward_storage, color='mediumorchid', linewidth=2, marker='^', markersize=4)
    axes[2].set_title('Position Reward', fontsize=14)
    axes[2].set_ylabel('Reward', fontsize=12)
    axes[2].set_xlabel('Evaluation Checkpoints', fontsize=12)
    
    # Tidy up the layout so nothing overlaps
    plt.tight_layout()
    plt.subplots_adjust(top=0.90) # Give the suptitle a bit of breathing room
    
    # Display the plot
    plt.savefig(savepath)
    plt.show()



def progress(num_steps, metrics, total_reward=None):

    eval_reward = metrics.get('eval/episode_reward', float('nan'))
    eval_len    = metrics.get('eval/avg_episode_length', float('nan'))
    pos_reward  = metrics.get('eval/episode_reward_pos', float('nan'))

    eval_reward_storage.append(eval_reward)
    eval_len_storage.append(eval_len)
    pos_reward_storage.append(pos_reward)

    fps = metrics.get('training/sps', metrics.get('eval/sps', float('nan')))
    
    v_loss = metrics.get('training/value_loss', float('nan'))
    if total_reward is not None:
        total_reward.append(eval_reward)

    # Create a Rich Table
    table = Table(
        title=f"Training Progress (Step: {num_steps})", 
        title_style="bold magenta",
        header_style="bold cyan",
        style="dim"
    )

    # Define Columns
    table.add_column("Metric", justify="left", style="cyan", no_wrap=True)
    table.add_column("Value", justify="right", style="green")

    # Add Rows
    table.add_row("eval_reward", f"{eval_reward:.2f}")
    table.add_row("eval_ep_len", f"{eval_len:.2f}")
    table.add_row("pos_reward", f"{pos_reward:.2f}")

    console.print(table)
    print() 


def train( args ):
    env = DroneBrax()
    
    #Create path to store the model
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S") 
    base_path    = Path(args.base_log_folder).resolve()
    os.makedirs(base_path, exist_ok=True)

    subfolder_path = Path( base_path / f"{args.custom_name}_{current_time}" )
    file_path      = Path( subfolder_path / f"{args.custom_name}_{current_time}" )
    os.makedirs( subfolder_path, exist_ok = True )

    ckpt_dir = subfolder_path / "checkpoints"
    ckpt_path = Path(ckpt_dir)

    ckpt_path.mkdir(parents=True, exist_ok=True)
    checkpointer = ocp.PyTreeCheckpointer()


    def save_callback(current_step, make_policy, params):
        step_path = ckpt_path / f"step_{current_step}"
        checkpointer.save(step_path, params, force=True)
        print(f"Saved checkpoint: {step_path}")
        brax_path = step_path / 'params'
        model.save_params(brax_path, params)

    train_fn = functools.partial(
        ppo.train,
        num_timesteps          = args.num_timesteps,
        num_evals              = args.num_evals,
        reward_scaling         = args.reward_scaling,
        episode_length         = args.episode_length,
        normalize_observations = args.normalize_observations,
        action_repeat          = args.action_repeat,
        unroll_length          = args.unroll_length,
        num_minibatches        = args.num_minibatches,
        num_updates_per_batch  = args.num_updates_per_batch,
        discounting            = args.discounting,
        learning_rate          = args.learning_rate,
        entropy_cost           = args.entropy_cost,
        num_envs               = args.num_envs,
        batch_size             = args.batch_size,
        network_factory        = functools.partial( ppo_networks.make_ppo_networks, policy_hidden_layer_sizes = POLICY_NETWORK ),
        policy_params_fn       = save_callback,
        progress_fn            = progress
    )
    
    print(f"Starting training for {args.num_timesteps} steps...")
    
    if args.checkpoint != None:
        restore_path = Path(args.checkpoint).resolve()
        print( f"Restoring from {restore_path}" )
    else:
        print( "Starting from clean network..." )
        restore_path = None

    start = time.time()
    make_inference_fn, params, metrics = train_fn(
        environment             = env,
        eval_env                = env,
        restore_checkpoint_path = restore_path)
    end = time.time()
    print(f"Training Completed in {(end - start) / 60:.2f}m")

    checkpointer.save(subfolder_path / "final", params)

    model.save_params(file_path, params)

    yaml_path = Path( subfolder_path / f"{args.custom_name}" )
    reward_path = Path( subfolder_path / f"{args.custom_name}_reward")
    plot_training_results(eval_reward_storage, eval_len_storage, pos_reward_storage, reward_path)
    write_enviroment_data(env, yaml_path,args)
    abs_subfolder = subfolder_path.resolve()
    csv_path = abs_subfolder / "csv"
    output_file = abs_subfolder / "matrices.h"
    # Convert network to csv
    command  = f"python tools/extract_nn.py --path {str(file_path.resolve())} --out {str(subfolder_path.resolve())}"
    subprocess.run(command, shell = True, capture_output=True, text=True)
    tomatrix = f"python tools/csvtohead.py {csv_path} -o {output_file}"    
    subprocess.run(tomatrix, shell=True, capture_output=True, text=True)
    return make_inference_fn, params


def parse_args():
    parser = argparse.ArgumentParser(description="PPO training drone environment")
    parser.add_argument("--num_timesteps",          type = int,   default = 30_000_000)
    parser.add_argument("--num_evals",              type = int,   default = 15)
    parser.add_argument("--reward_scaling",         type = float, default = 1.0)
    parser.add_argument("--episode_length",         type = int,   default =  500)
    parser.add_argument("--normalize_observations", type = bool,  default = True)
    parser.add_argument("--action_repeat",          type = int,   default = 1)
    parser.add_argument("--unroll_length",          type = int,   default = 20)
    parser.add_argument("--num_minibatches",        type = int,   default = 32)
    parser.add_argument("--num_updates_per_batch",  type = int,   default = 4)
    parser.add_argument("--discounting",            type = float, default = 0.99)
    parser.add_argument("--learning_rate",          type = float, default = 3e-4)
    parser.add_argument("--entropy_cost",           type = float, default = 1e-3)
    parser.add_argument("--num_envs",               type = int,   default =  4096)
    parser.add_argument("--batch_size",             type = int,   default = 256)
    parser.add_argument("--seed",                   type = int,   default = 0)
    parser.add_argument("--base_log_folder",        type = str,   default = 'logs')
    parser.add_argument("--custom_name",            type = str,   default = 'params')
    parser.add_argument("--checkpoint",             type = str,   default = None)

    return parser.parse_args()


def main():
    args = parse_args()
    train(args)

if __name__ == "__main__":
    main()