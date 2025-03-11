# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ray.rllib.env import EnvContext
from www_test_RLlib3env import MultipleRL_Env

# Configuration for the environment
config = {
    "num_nodes": 50,
    "num_vnf_request1": 5,
    "num_vnf_request2": 6,
    "num_vnf_request3": 4,
    "num_vnf_request4": 3,
    "num_agents": 3
}

# Create the environment
env = MultipleRL_Env(config)

# Reset the environment to get initial observations
observations, infos = env.reset()

# Initialize variables to track rewards and steps
num_steps = 50  # Number of random steps to test
total_rewards = {agent: 0.0 for agent in env.agent_ids}
total_steps = {agent: 0 for agent in env.agent_ids}

# Data collection for CSV and plotting
data = []

# Run the random action test
for step in range(num_steps):
    print(f"\nStep {step + 1}")
    
    # Generate random actions for all agents
    actions = {agent: env.action_spaces[agent].sample() for agent in env.agent_ids}
    
    # Step the environment with random actions
    observations, rewards, dones, truncated, infos = env.step(actions)
    
    # Accumulate rewards and count steps for each agent
    for agent, reward in rewards.items():
        total_rewards[agent] += reward
        total_steps[agent] += 1

    # Calculate average rewards for this step
    avg_rewards = {agent: total_rewards[agent] / max(total_steps[agent], 1) for agent in env.agent_ids}
    
    # Save the data for each step
    data.append({
        "Step": step + 1,
        **{f"Agent_{agent}_Reward": rewards[agent] for agent in env.agent_ids},
        **{f"Agent_{agent}_Average_Reward": avg_rewards[agent] for agent in env.agent_ids},
        **{f"Agent_{agent}_Steps": total_steps[agent] for agent in env.agent_ids}
    })
    
    # Check if the episode is done
    if dones["__all__"]:
        print("All agents are done. Resetting environment.")
        observations, infos = env.reset()

# Convert collected data to a DataFrame
df = pd.DataFrame(data)

# Save data to a CSV file
csv_filename = "www_testt_random_action_test_results.csv"
df.to_csv(csv_filename, index=False)
print(f"Data saved to {csv_filename}")

# Plotting the average rewards over steps
plt.figure(figsize=(10, 6))
for agent in env.agent_ids:
    plt.plot(df["Step"], df[f"Agent_{agent}_Average_Reward"], label=f"Agent {agent} Avg Reward")

plt.xlabel("Step")
plt.ylabel("Average Reward")
plt.title("Average Reward per Agent over Time Steps")
plt.legend()
plt.grid()
plt.savefig("average_rewards_plot.png")
plt.show()

# Plotting the steps taken by each agent
plt.figure(figsize=(10, 6))
for agent in env.agent_ids:
    plt.plot(df["Step"], df[f"Agent_{agent}_Steps"], label=f"Agent {agent} Steps")

plt.xlabel("Step")
plt.ylabel("Steps Taken")
plt.title("Steps Taken by Each Agent over Time Steps")
plt.legend()
plt.grid()
plt.savefig("steps_taken_plot.png")
plt.show()

print("Plots saved as 'average_rewards_plot.png' and 'steps_taken_plot.png'")
