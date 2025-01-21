import networkx as nx
import numpy as np
import gymnasium as gym


class PTSPEnv(gym.Env):
    def __init__(
        self,
        graph,
        *,
        start_node=0,
        self_selection_penalty=100,
        goal_state_reward=100,
        distance_multiplier=1,
        terminate_on_self_selection=True,
    ):
        self.graph = graph
        self.start_node = start_node

        self.self_selection_penalty = self_selection_penalty
        self.goal_state_reward = goal_state_reward
        self.distance_multiplier = distance_multiplier
        self.terminate_on_self_selection = terminate_on_self_selection

        if self.graph.get_prob(start_node) != 0:
            raise ValueError("Start node must have zero probability")

        self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,))
        self.observation_space = gym.spaces.Dict(
            spaces={
                "current_node": gym.spaces.Discrete(self.graph.num_nodes),
                "visited": gym.spaces.MultiBinary(self.graph.num_nodes),
                "prob": gym.spaces.Box(low=0, high=1, shape=(self.graph.num_nodes,)),
                "dist": gym.spaces.Box(
                    low=1,
                    high=100,
                    shape=(len(self.graph.edges),),
                ),
                # add randomized graphs after base solved
            }
        )

        self.reset()

    @classmethod
    def fromlist(cls, nodes, edges, **kwargs):
        graph = PTSPGraph(nodes, edges)
        return cls(graph, **kwargs)

    def reset(self, **kwargs):
        self.current_node = self.start_node

        self.visited = np.array([0] * self.graph.num_nodes)
        self.visited[self.current_node] = 1

        self.cost = 0

        self.end_node = np.random.choice(
            range(self.graph.num_nodes), p=self.graph.nodes
        )

        self.found, self.terminated = False, False

        return self._state(), {}

    def step(self, action, ignore_termination=False):
        action = self._get_next_node(action)

        if (self.found or self.terminated) and not ignore_termination:
            raise ValueError("Cannot step in a finished episode. Call reset.")

        reward = self._reward(self._state(), action)

        if self.terminate_on_self_selection and self.visited[action]:
            self.terminated = True
        elif action == self.end_node:
            self.found = True

        self.current_node = action
        self.visited[self.current_node] = 1

        return self._state(), reward, self.found, self.terminated, {}

    def _state(self):
        state = {
            "current_node": self.current_node,
            "visited": self.visited,
            "prob": self.graph.nodes,
            "dist": self.graph.edges,
        }
        return state

    def _reward(self, state, action):
        if state["visited"][action]:
            return -self.self_selection_penalty
        return (
            -self.graph.get_dist(self.current_node, action) * self.distance_multiplier
        ) + (self.goal_state_reward if action == self.end_node else 0)

    def _get_next_node(self, action):
        if action.ndim > 0:
            action = action[0]
        free = []
        for i in range(self.graph.num_nodes):
            if not self.visited[i]:
                free.append(i)
        node = int(action * (self.graph.num_nodes - sum(self.visited)))
        node = min(node, len(free) - 1)

        return free[node]


class PTSPValEnv(PTSPEnv):
    def step(self, action):
        action = self._get_next_node(action)

        if sum(self.visited) == self.graph.num_nodes:
            raise ValueError("Cannot step in a finished episode. Call reset.")

        reward = self._reward(self._state(), action)

        self.current_node = action
        self.visited[self.current_node] = 1

        return self._state(), reward, self.found, self.terminated, {}


class PTSPGraph:
    def __init__(self, nodes, edges):
        if not np.isclose(sum(nodes), 1):
            raise ValueError("Nodes probabilities must sum to 1")

        self.graph = nx.Graph()

        self.nodes = np.array(nodes)
        self.edges = np.array(edges)

        self.num_nodes = len(nodes)

        for i in range(self.num_nodes):
            self.graph.add_node(i, weight=nodes[i])

        c = 0
        for i in range(self.num_nodes - 1):
            for j in range(i + 1, self.num_nodes):
                self.graph.add_edge(i, j, weight=edges[c])
                c += 1

    def get_dist(self, a, b):
        return self.graph[a][b]["weight"]

    def get_prob(self, a):
        return self.nodes[a]
