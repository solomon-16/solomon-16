import gymnasium as gym
from gym.envs.registration import register
from gym.wrappers import FlattenObservation
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from gymnasium.spaces import  Box, Dict, Discrete,MultiDiscrete
import random

# Function to create the network for customized topology
def network_graph():
    
    g = nx.MultiGraph()
    edges = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('A', 'D')]
    g.add_edges_from(edges)
    node_attributes = {
        '1': {'cpu': '150 GHz', 'ram': '100 GB', 'storage': '2TB'},
        '2': {'cpu': '200 GHz', 'ram': '150 GB', 'storage': '512 GB'},
        '3': {'cpu': '300 GHz', 'ram': '200 GB', 'storage': '512 GB'},
        '3': {'cpu': '400 GHz', 'ram': '300 GB', 'storage': '1 TB'}
    }
    nx.set_node_attributes(g, node_attributes)
    edge_attributes = {
        ('1', '2',0): {'bandwidth': '90 Gbps', 'latency': '10 '},
        ('2', '3',0): {'bandwidth': '90 Gbps', 'latency': '8 '},
        ('3', '4',0): {'bandwidth': '70 Gbps', 'latency': '7 '},
        ('1', '4',0): {'bandwidth': '100 Gbps', 'latency': '5'}
    }
    nx.set_edge_attributes(g, edge_attributes)
    return g
# function to cereate  service function chain graph
def sfc_graph():
   
    sfc_graph = nx.MultiGraph()
    vnf_attributes = {
        '1': {'cpu': '2 GHz', 'ram': '4 GB', 'storage': '10 GB', 'processing_delay': '1'},
        '2': {'cpu': '3 GHz', 'ram': '8 GB', 'storage': '20 GB', 'processing_delay': '2 '},
        '3': {'cpu': '2.5 GHz', 'ram': '6 GB', 'storage': '30 GB', 'processing_delay': '2.5 '}
    }
    sfc_graph.add_nodes_from((vnf, attrs) for vnf, attrs in vnf_attributes.items())
    sfc_edges = [('1', '2'), ('2', '3')]
    sfc_graph.add_edges_from(sfc_edges)
    return sfc_graph


class VNFMigrationEnv(gym.Env):
    def __init__(self, network_graph, sfc_graph):
        super(VNFMigrationEnv, self).__init__()
        self.network_graph = network_graph
        self.sfc_graph = sfc_graph
        self.nodes = list(network.nodes())
        self.vnf_nodes = list(sfc.nodes())
        self.num_nodes = len(self.nodes)
        self.num_vnf_nodes = len(self.vnf_nodes)
        self.node_link = len(network.edges()) if network.edges() else 0
        self.edge_link = len(sfc.edges())
        self.action_space = MultiDiscrete([self.num_nodes]*self.num_vnf_nodes)
        self.observation_space = MultiDiscrete([self.num_nodes]*self.num_vnf_nodes)
        self.intial_observation = [0] * self.num_vnf_nodes
        self.threshold_latency=10
    # Initialize the state of the environment
    def reset(self):
        self.current_observation = [0] * self.num_vnf_nodes  # Reset state to all zeros
        return self.current_observation 
    
    # define step function
    def step(self,action):
          self.reward = 0
         # Select 3 nodes from self.nodes
          action = random.choices(self.nodes, self.num_vnf_nodes)
          # Update state based on action
          self.intial_observation[0:2]=action[0],action[1],action[2] 
          self.next_observation=self.intial_observation
          latency = self.calculate_total_latency(action)
          # Allocate VNFs on the network graph
          for i, node in enumerate(action):
                  if latency < self.latency_threshold:
                    self.reward+=1 # reward
                    vnf_node = self.vnf_nodes[i]
                    network.nodes[node]['vnf'] = vnf_node
                  else:
                    self.reward-=1 # penality
     
          terminated = False  
          truncated = False  
          info = {}  # Placeholder, additional information
          return self.next_observation, self.reward, terminated, truncated, info
    # Calculate total latency including link latency and VNF processing delay
    def calculate_total_latency(self, action):
        total_latency = 0
        for i in range(len(action) - 1):
           source = action[i]
           target = action[i + 1]
           # Extract link latency
           link_latency = int(network.get_edge_data(source, target)[0]['latency'])
           # Calculate VNF processing delay
           for node in action:
           # Extract VNF processing delay
            vnf_processing_delay = int(sfc.nodes[node]['processing_delay'])
            #calculate total latency
        total_latency = link_latency + vnf_processing_delay
        return total_latency
        
          
    
# Create the SFC graph

network = network_graph()
sfc = sfc_graph()
env = VNFMigrationEnv(network, sfc)
