import gym
import networkx as nx
import numpy as np
from gym import spaces
from gymnasium.spaces import Graph, Box, Discrete

# Function to create the network for customized topology
def network_graph():
    # Create graph
    g = nx.MultiGraph()
    # Create edges
    edges = [('A', 'B'),('A', 'm'), ('A', 'n'),('B', 'C'),('B', 'r'), ('B', 's'),('C', 'D'), ('C', 'l'),('C', 'o'),('D', 'u'),('D', 'x'),('A', 'E'), ('F', 'B'), ('C', 'G'), ('D', 'H'), ('E', 'F'),('F', 'G'), ('G', 'H'), ('E', 'W'),('F', 'W'), ('G', 'W'), ('H', 'W')]
    g.add_edges_from(edges)
    node_attributes = {
        'A': {'cpu': '5 GHz', 'ram': '64 GB', 'storage': '2TB'},
        'B': {'cpu': '10 GHz', 'ram': '32 GB', 'storage': '512 GB'},
        'C': {'cpu': '15 GHz', 'ram': '24 GB', 'storage': '512 GB'},
        'D': {'cpu': '20 GHz', 'ram': '16 GB', 'storage': '1 TB'},
        'E': {'cpu': '5 GHz', 'ram': '64 GB', 'storage': '2TB'},
        'F': {'cpu': '10 GHz', 'ram': '32 GB', 'storage': '512 GB'},
        'G': {'cpu': '15 GHz', 'ram': '24 GB', 'storage': '512 GB'},
        'H': {'cpu': '20 GHz', 'ram': '16 GB', 'storage': '1 TB'},
        'W': {'cpu': '30 GHz', 'ram': '128 GB', 'storage': '20 TB'}
    }
    nx.set_node_attributes(g,node_attributes)
    edge_attributes = {
        ('E', 'F', 0): {'bandwidth': '50 Gbps', 'latency': '10 ms'},
        ('F', 'G', 0): {'bandwidth': '30 Gbps', 'latency': '8 ms'},
        ('G', 'H', 0): {'bandwidth': '70 Gbps', 'latency': '7 ms'},
        ('H', 'W', 0): {'bandwidth': '100 Gbps', 'latency': '5 ms'},
        ('F', 'W', 0): {'bandwidth': '1020 Gbps', 'latency': '5 ms'},
        ('E', 'W', 0): {'bandwidth': '200 Gbps', 'latency': '3 ms'},
    }
    nx.set_edge_attributes(g, edge_attributes)
    return g


def sfc_graph():
    sfc_graph = nx.MultiGraph()
    vnf_nodes = ['VNF1', 'VNF2', 'VNF3', 'VNF4', 'VNF5']
    # Define resource attributes for each VNF
    vnf_attributes = {
        'VNF1': {'cpu': '2 GHz', 'ram': '4 GB', 'storage': '100 GB'},
        'VNF2': {'cpu': '3 GHz', 'ram': '8 GB', 'storage': '200 GB'},
        'VNF3': {'cpu': '2.5 GHz', 'ram': '6 GB', 'storage': '150 GB'},
        'VNF4': {'cpu': '2.2 GHz', 'ram': '12 GB', 'storage': '250 GB'},
        'VNF5': {'cpu': '4 GHz', 'ram': '16 GB', 'storage': '300 GB'}
    }
    # Add nodes with resource attributes
    sfc_graph.add_nodes_from((vnf, attrs) for vnf, attrs in vnf_attributes.items())
    # Add edges with attributes, including bandwidth and latency
    sfc_edges = [
        ('VNF1', 'VNF2', {'bandwidth': '50 Gbps', 'latency': '10 ms'}),
        ('VNF2', 'VNF3', {'bandwidth': '30 Gbps', 'latency': '8 ms'}),
        ('VNF3', 'VNF1', {'bandwidth': '40 Gbps', 'latency': '15 ms'}),
        ('VNF2', 'VNF5', {'bandwidth': '60 Gbps', 'latency': '12 ms'}),
        ('VNF4', 'VNF2', {'bandwidth': '20 Gbps', 'latency': '6 ms'})
    ]
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
        node_link = len(network.edges()) if network .edges() else 0
        edge_link = len(sfc.edges())
        self.total_node = self.num_nodes + self.num_vnf_nodes
        self.total_links = edge_link + node_link
        self.action_space = spaces.Discrete(3)  
            
        self.observation_space = Graph(
            node_space=Box(low=0, high=10, shape=(self.total_node,)),
            edge_space=Box(low=0, high=10, shape=(self.total_links,))
        )
        

    def reset(self):
        observation = np.zeros((self.total_node, self.observation_space.shape[1]))  # Reset observation space to zeros
        return observation
        
    
    def step(self, action):
        # Generate a random action
       
        observation = np.zeros((self.total_node, self.observation_space.shape[1]))  # Initialize observation space
        reward = 0
        done = False
        info = {}
        return observation, reward, done, info

# Testing the environment
network = network_graph()
sfc = sfc_graph()
env = VNFMigrationEnv(network, sfc)

    
        
        
