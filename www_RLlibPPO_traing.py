import os
import ray
import pandas as pd
import matplotlib.pyplot as plt
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from www_test_Rllib import MultipleRL_Env  # Import the custom multi-agent environment
import warnings

# Suppress RLlib Deprecation Warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Environment Configuration
DEFAULT_ENV_CONFIG = {
    "num_nodes": 50,  # Adjusted for efficiency
    "num_vnf_request1": 3,
    "num_vnf_request2": 5,
    "num_vnf_request3": 7,
    "num_vnf_request4": 6,
    "num_agents": 3,
}

def env_creator(config):
    """Create a new instance of the Multi-Agent RL Environment."""
    config = {**DEFAULT_ENV_CONFIG, **config}  # Merge default & provided config
    return MultipleRL_Env(config)

register_env("MultipleRL_Env", env_creator)

# Initialize Ray
ray.init(ignore_reinit_error=True, log_to_driver=False)

# Policy Configuration
def policy_mapping_fn(agent_id, *args, **kwargs) -> str:
    return f"policy_{int(agent_id) % 3}"  # Three policies

# Define Multi-Agent Policies
env_instance = env_creator(DEFAULT_ENV_CONFIG)
sample_agent_id = list(env_instance.observation_spaces.keys())[0]

policies = {
    f"policy_{i}": PolicySpec(
        observation_space=env_instance.observation_spaces[sample_agent_id],
        action_space=env_instance.action_spaces[sample_agent_id],
        config={}
    ) for i in range(3)
}

# RLlib PPO Training Configuration
config = (
    PPOConfig()
    .environment(env="MultipleRL_Env", env_config=DEFAULT_ENV_CONFIG, disable_env_checking=True)
    .framework("torch")
    .env_runners(num_env_runners=0)  # Force CPU training
    .resources(num_gpus=0, num_cpus_per_worker=0.5)  # Limit CPU usage
    .training(train_batch_size=500, num_epochs=5, lambda_=0.95, vf_clip_param=10.0)
    .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn, policies_to_train=list(policies.keys()))
    .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
)

# Initialize PPO Trainer
trainer = config.build_algo()

# Train PPO Algorithm
num_train_steps = 10
all_results = []

for i in range(num_train_steps):
    results = trainer.train()
    timestep, mean_reward = results["timesteps_total"], results["episode_reward_mean"]
    print(f"Iteration {i}: Timestep = {timestep}, Mean Reward = {mean_reward:.3f}")
    all_results.append((timestep, mean_reward))

# Save Model & Results
model_filename = "ppo_multiplesfc_env_model.pt"
trainer.save(model_filename)
df = pd.DataFrame(all_results, columns=["Timestep", "Mean Reward"])
df.to_csv("www_PPO_vnf_migration_results.csv", index=False)

# Plot Training Performance
plt.figure(figsize=(12, 6))
plt.plot(df["Timestep"], df["Mean Reward"], marker='o', label="Mean Reward")
plt.title("RLlib PPO Training: Timesteps vs Cumulative Reward", fontsize=16)
plt.xlabel("Timesteps", fontsize=14)
plt.ylabel("Cumulative Reward", fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

# Cleanup
ray.shutdown()
os.system("pkill -9 ray")
os.system("rm -rf /tmp/ray/")
print("Training complete. Ray processes cleaned up.")
