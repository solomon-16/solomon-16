import gymnasium as gym
from gymnasium.spaces import MultiDiscrete,Box
import networkx as nx
import numpy as np
import random
from community import community_louvain
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform, pdist
from matplotlib.patches import Ellipse
from ray.rllib.env import EnvContext
import numpy as np
import random
import networkx as nx
from gymnasium.spaces import Box, Discrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv
class MultipleRL_Env(MultiAgentEnv):
    """
    Custom RLlib Multi-Agent Environment.
    """
    def __init__(self, config:EnvContext):
        super().__init__()
        self.config = config or {}
        # Extract environment parameters dynamically from config
        self.num_nodes = self.config["num_nodes"]
        self.num_vnf_request1 = self.config["num_vnf_request1"]
        self.num_vnf_request2 = self.config["num_vnf_request2"]
        self.num_vnf_request3 = self.config["num_vnf_request3"]
        self.num_vnf_request4 = self.config["num_vnf_request4"]
        self.num_agent = self.config["num_agents"]
        self.network_graph = self.create_network_graph(self.num_nodes)
        self.service_request1 = self.Service_request_generation__1(self.num_vnf_request1)
        self.service_request2 = self.Service_request_generation_2( self.num_vnf_request2)
        self.service_request3 = self.Service_request_generation_3(self.num_vnf_request3)
        self.service_request4 = self.Service_request_generation_4(self.num_vnf_request4)
        # create  a pool of service request
        self.service_type =(self.service_request1, self.service_request2,self.service_request4, self.service_request3)
        # Create list of nodes with service request 
        self.service_requist1_nodes=list(self.service_request1.nodes())
        self.service_requist2_nodes=list(self.service_request2.nodes())
        self.service_request3_nodes=list(self.service_request3.nodes())
        self.service_request4_nodes=list(self.service_request4.nodes())
        # Find maximum number of VNFs across all requests
        self.maximun_VNF_nodes = self.get_max_nodes_in_service_type()
        #self.selected_request = self.select_sfc_graph()s
        self.request1,self.request2,self.request3 = self.select_sfc_graph()
        # Network infrastructure nodes
        self.nodes = list(self.network_graph.nodes())
        self.vnf_nodes1 = list( self.request1.nodes())
        self.vnf_nodes2 = list(self.request2.nodes())
        self.vnf_nodes3 = list(self.request3.nodes())
        # Define agents ID for  each agents
        self.agents = [f"{i}" for i in range(self.num_agent)]
        self.agent_ids = self.agents
        self.possible_agents=self.agents
        self.action_spaces = {agent: MultiDiscrete([len(self.nodes) - 1]* ((self.maximun_VNF_nodes))) for agent in self.agent_ids}
        # Create  an observation space to observe all network 
        self.observation_spaces = {
                                  agent: MultiDiscrete([len(self.nodes)] * len(self.nodes), dtype=np.int32)
                                        for agent in self.agent_ids
                                       }
    def create_network_graph(self,num_nodes):
        """
        Create a scalable and connected graph representing the USANET topology.
        """
        g = nx.Graph()
        # Add nodes with hardware attributes
        for i in range(num_nodes):
                g.add_node(i, 
                        cpu=f'{random.uniform(2.0, 4.0):.2f} GHz',
                        ram=f'{random.randint(50,120)} GB',
                        storage=f'{random.randint(100, 1000)} GB',
                        energy=f'{random.randint(20,50)}'
                )

            # Define a backbone structure with 30 interconnected nodes
        backbone_nodes = min(30, num_nodes)
        base_edges = [
                (0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6),
                (3, 7), (4, 8), (5, 9), (6, 10), (7, 11), (8, 11),
                (9, 11), (10, 11), (7, 8), (9, 10), (4, 5), (8, 9),
                (0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (5, 10), (6, 11),
                (3, 6), (2, 4), (1, 9), (8, 10), (7, 9), (10, 12), (11, 13),
                (12, 14), (13, 15), (14, 16), (15, 17), (16, 18), (17, 19),
                (18, 20), (19, 21), (20, 22), (21, 23), (22, 24), (23, 25),
                (24, 26), (25, 27), (26, 28), (27, 29), (28, 0), (29, 1)
            ]

            # Add backbone edges
        g.add_edges_from(base_edges)
        # Dynamically expand the network if num_nodes > 30
        if num_nodes > backbone_nodes:
                for i in range(backbone_nodes, num_nodes):
                    # Connect new nodes to backbone nodes for strong connectivity
                    backbone_node = random.choice(range(backbone_nodes))
                    g.add_edge(i, backbone_node)

                    # Additional random connections for better redundancy
                    for _ in range(random.randint(1, 3)):
                        another_node = random.choice(range(num_nodes))
                        if another_node != i:
                            g.add_edge(i, another_node)

            # Assign bandwidth and latency to edges
        for u, v in g.edges():
                g[u][v]['bandwidth'] = f'{random.randint(50, 200)} Gbps'
                g[u][v]['latency'] = 0  # Adjusted range for better visualization

        return g
    
    #service request generation 1
    def Service_request_generation__1(self,num_vnfs1):
        sfc_graph = nx.Graph()
        for i in range(num_vnfs1):
            sfc_graph.add_node(i, cpu=f'{1.0 + 0.5 * i} GHz',
                               ram=f'{2 + 4 * i} GB',
                               storage=f'{5 + 3 * i} GB',
                               processing_delay=0.5)
        edges = [(i, i + 1) for i in range(num_vnfs1 - 1)]
        sfc_graph.add_edges_from(edges)
        return sfc_graph
    # service request generation 2
    def Service_request_generation_2(self, num_vnfs2):
        sfc_graph = nx.Graph()
        for i in range(num_vnfs2):
            sfc_graph.add_node(i, cpu=f'{0.4 + 0.2 * i} GHz',
                               ram=f'{2 + 2 * i} GB',
                               storage=f'{5 + 5 * i} GB',
                               processing_delay=0.5)
        edges = [(i, i + 1) for i in range(num_vnfs2 - 1)]
        sfc_graph.add_edges_from(edges)
        return sfc_graph
    #service request generation 3
    def Service_request_generation_3(self, num_vnfs3):
        sfc_graph = nx.Graph()
        for i in range(num_vnfs3):
            sfc_graph.add_node(i, cpu=f'{1.0 + 0.5 * i} GHz',
                               ram=f'{2 + 2 * i} GB',
                               storage=f'{5 + 3 * i} GB',
                               processing_delay=0.5)
        edges = [(i, i + 1) for i in range(num_vnfs3 - 1)]
        sfc_graph.add_edges_from(edges)
        return sfc_graph
    #service request generation 4
    def Service_request_generation_4(self, num_vnf_request4):
        sfc_graph = nx.Graph()
        for i in range(num_vnf_request4):
            sfc_graph.add_node(i, cpu=f'{1.0 + 0.5 * i} GHz',
                               ram=f'{2 + 2 * i} GB',
                               storage=f'{5 + 3 * i} GB',
                               processing_delay=0.5)
        edges = [(i, i + 1) for i in range(num_vnf_request4 - 1)]
        sfc_graph.add_edges_from(edges)
        return sfc_graph
    #Select service requests from teh pool of requets
    def select_sfc_graph(self):
        """Selects three SFC graphs using random sampling."""
        valid_graphs = [graph for graph in self.service_type if graph.number_of_nodes() > 0]
        if len(valid_graphs) < 3:
            raise ValueError("Not enough valid SFC graphs available to select from.")
        
        selected_graphs = random.sample(valid_graphs, k=3)
        return selected_graphs
    def get_max_nodes_in_service_type(self):
        """
        Finds the maximum number of nodes across all service request graphs in self.service_type.
        """
        max_nodes = 0
        for graph in self.service_type:
            num_nodes = graph.number_of_nodes()
            if num_nodes > max_nodes:
                max_nodes = num_nodes
        return max_nodes
    # individul agent observation 
    def observe(self, agent):
       """Returns the next observation for an agent based on its last action."""
       if agent not in self.observation_spaces:
          raise KeyError(f"Agent {agent} does not have a defined observation space.")
        # Sample an observation from the space
       next_observation = self.observation_spaces[agent].sample()
       return np.array(next_observation, dtype=np.int32)

    def _get_info(self,agent):
        for agent in self.agent_ids:
         return({agent: {} })
    
    def reset(self, seed=300, options=None):
        """
        Resets the environment for a new episode.
        The observations are structured as a dictionary where the keys are agent IDs
        and the values are dictionaries of node and edge attributes.
        All node and edge attributes are initialized to zero.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed) 

        self.time_step = 0
        self.rewards = {agent: 0.0 for agent in self.agent_ids}  # Reset individual rewards  
    
        # Initialize dictionaries
        self.observations = {}  
        self.rewards = {}
        self.dones = {}
        self.terminations = {}
        self.truncations = {}
        self.infos = {}

        # Create a base observation structure with all zeroed attributes
        for agent in self.agent_ids:
         for agent in self.agent_ids:
          observation_size = self.observation_spaces[agent].shape[0]
        self.observations[agent] = np.full(observation_size, 10, dtype=np.int32)
        # Initialize agent-specific metadata
        for agent in self.agent_ids:
            self.rewards[agent] = 0.0
            self.dones[agent] = False
            self.terminations[agent] = False
            self.truncations[agent] = False
            self.infos[agent] = {f"agent_{agent}": []}

        print(f"Reset observations with zeroed attributes: {self.observations}")
        return self.observations, self.infos


    def step(self, actions):
        #print(f"Actions data type: {type(actions)}")
        #print(f"Received actions: {actions}")
        
        rewards = {}
        if isinstance(actions, dict):
            actions = {agent: np.array(val) if isinstance(val, (list, np.ndarray)) else val for agent, val in actions.items()}
        elif isinstance(actions, (list, np.ndarray)):
            
            actions = {agent: val for agent, val in zip(self.agent_ids, actions)}
        else:
            raise TypeError(f"Invalid action format: {type(actions)}. Expected dict, list, or numpy array.")

        self.deployment_action = self._process_actions(actions)
        #print(f"The deployment action: {self.deployment_action}")

        done = {agent: False for agent in self.agent_ids}
        done["__all__"] = False
        
        truncated = {agent: False for agent in self.agent_ids}
        truncated["__all__"] = False  

        
        for agent_ID in self.agent_ids:
            reward_latency = self.get_reward_latency(agent_ID)
            reward_processing = self.get_reward_processing_latency(agent_ID)
            #print(f"the rewarrd  due to latency {reward_latency }")
            #print(f"the reward  due to proccessing {reward_processing}")
            total_reward = reward_latency + reward_processing
            rewards[agent_ID] = -total_reward
            #self._cumulative_rewards[agent_ID] += -total_reward

        # Generate observations
        for u, v in self.network_graph.edges():
          self.network_graph.edges[u, v]['latency'] = random.randint(2,12)
        # Ensure info only contains relevant keys
        observations = {agent: self.observe(agent) for agent in self.agent_ids}
        info = {agent: {} for agent in observations.keys()}
        #print(f"Infos: {list(info.keys())}")
        #print(f"observations{observations}, rewards{rewards}, done{done}, truncated{truncated}, info{info}")
        return observations, rewards, done, truncated, info

    # Pre proccessing the action to assign individual  SFC for each Agent
    def _process_actions(self, actionn):
        processed_actions = {}
        
        if isinstance(actionn, dict):
            action_dict = {str(agent_ID): np.array(val, dtype=np.int32) for agent_ID, val in actionn.items()}
            #print(f"Action is Dictionary given below: {action_dict}")
        elif isinstance(actionn, (list, np.ndarray)):
            action_dict = {str(agent_ID): np.array(val, dtype=np.int32) 
                        for agent_ID, val in zip(self.agent_ids, actionn)}
           # print(f"Converted NumPy array to dictionary: {action_dict}")
        else:
            raise TypeError(f"Invalid action format: {type(actionn)}. Expected dictionary or NumPy array.")

        #print(f"Action Dictionarized: {action_dict}")  # Debugging print statement

        # Iterate over all agents and pad or truncate actions properly
        for agent_ID, val in action_dict.items():
            if len(val) < self.maximun_VNF_nodes:
                # Pad with zeros if the action array is shorter
                padded_val = np.pad(val, (0, self.maximun_VNF_nodes - len(val)), 'constant', constant_values=0)
            elif len(val) > self.maximun_VNF_nodes:
                # Truncate if the action array is too long
                padded_val = val[:self.maximun_VNF_nodes]
            else:
                padded_val = val

            processed_actions[agent_ID] = padded_val
            #print(f"Processed action for agent {agent_ID}: {processed_actions[agent_ID]}")

        return processed_actions
    # calculating teh reward value
    def get_reward_latency(self, agent_ID):
        """Calculates the network latency based on the deployment of VNFs."""
        all_latencies = []
        #val =self.deployment_action(agent_ID)
        node_label = list(self.deployment_action.get(agent_ID, []))
        #print(f"the action value {node_label}")
        # Iterate over the selected nodes to compute latency
        for i in range(len(node_label) - 1):
            source = self.nodes[node_label[i]]
            target = self.nodes[node_label[i + 1]]
    
            if nx.has_path(self.network_graph, source, target):
                shortest_path = nx.shortest_path(self.network_graph, source, target)
                path_latency = sum(
                    float(self.network_graph[u][v]['latency']) for u, v in zip(shortest_path[:-1], shortest_path[1:])
                )
                all_latencies.append(path_latency)

        return min(all_latencies) if all_latencies else 0

    def get_reward_processing_latency(self,agent_ID):
        # Ensure actions is a dictionary
        total_processing_latency = 0
        if agent_ID not in self.deployment_action:
            return 0.0
            
        actions = self.deployment_action[agent_ID]
        vnf_nodes = list(actions)
        if agent_ID == "0":
           vnf_indices = list(self.vnf_nodes1)
           for vnf_id in vnf_indices:
               if vnf_id in self.request1.nodes:
                  vnf_attributes = self.request1.nodes[vnf_id]
                  processing_delay = vnf_attributes.get('processing_delay', 0.0)
                  total_processing_latency += processing_delay
               else:
                print(f"VNF ID {vnf_id} does not exist in the graph.")        
        elif agent_ID == "1":
             vnf_indices = list(self.vnf_nodes2)
             for vnf_id in vnf_indices:
                if vnf_id in self.request2.nodes:
                  vnf_attributes = self.request2.nodes[vnf_id]
                  processing_delay = vnf_attributes.get('processing_delay', 0.0)
                  total_processing_latency += processing_delay
                else:
                 print(f"VNF ID {vnf_id} does not exist in the graph.")
        elif agent_ID == "2":
           vnf_indices = list(self.vnf_nodes3)
           for vnf_id in vnf_indices:
                if vnf_id in self.request3.nodes:
                  vnf_attributes = self.request3.nodes[vnf_id]
                  processing_delay = vnf_attributes.get('processing_delay', 0.0)
                  total_processing_latency += processing_delay
                else:
                 print(f"VNF ID {vnf_id} does not exist in the graph.")        
        return total_processing_latency

