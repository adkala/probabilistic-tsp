from stable_baselines3 import SAC
from sklearn.cluster import KMeans


import gymnasium as gym
import networkx as nx
import json
import numpy as np

# requirements: pip install stable-baselines3 gymnasium networkx numpy

CHECKPOINT = "models/SAC_truerand_2024-04-10-17-42-20/logs/rl_model_1050000_steps.zip"
# NODES = [
#     0,
#     0.1,
#     0.2,
#     0.7,
# ]  # note: first node should be 0 (starting pos), set non-existant nodes to some low value (i.e 1e-3), number of nodes has to be 6 including the start node
# EDGES = list(
#     range(1, 7)
# )  # note: keep edge weights between 1 and 100 (use scaling, necessary since what it was trained on), if you have less than 6 total nodes, set edge weights to 100,

DESC_FP = "description.json"

# ---


def parse_json(fp):
    with open(fp) as fp:
        r = []
        for i in json.load(fp)["scenario_objective"]["entities_of_interest"]:
            for j in i["entity_priors"]["location_belief_map"]:
                r.append((j["polygon_vertices"][0][:-1], j["probability"]))
    return r


def sort_inner_tuples(d):
    i = [d[i][2] for i in range(len(d))]
    mi = 0
    for j in range(1, len(i)):
        if i[j] > i[mi]:
            mi = j

    i[mi], i[0] = i[0], 0
    d[mi], d[0] = d[0], d[mi]
    j = [
        np.sqrt((d[k][0][0] - d[j][0][0]) ** 2 + (d[k][0][1] - d[j][0][1]) ** 2)
        for k in range(len(d))
        for j in range(k + 1, len(d))
    ]

    if len(j) > 1 and sum(j) > 0:
        j = get_list(
            list(np.array(i) / sum(i)),
            list((np.array(j) - min(j)) / (max(j) - min(j)) * 99 + 1),
        )
        d = [d[i] for i in j]
    return d


def get_sorted_belief_maps(d, sf=sort_inner_tuples):
    def km(a, kmeans=KMeans(n_clusters=6, init="random")):
        if len(a) < 6:
            return a
        groups = [[] for _ in range(6)]
        for a, b in zip(a, kmeans.fit_predict([np.mean(i[0], axis=0) for i in a])):
            groups[b].append(a)
        for b in range(6):
            if len(groups[b]) == 0:
                groups = groups[:b]
                break
            groups[b] = km(groups[b])
        return groups

    def recurse(a):
        r = []
        for a in a:
            if isinstance(a, tuple):
                r.append((list(np.mean(a[0], axis=0)), [a[0]], a[1]))
            else:
                a = sf(recurse(a))
                r.append(
                    [
                        list(np.mean([a[0] for a in a], axis=0)),
                        [a for a in a for a in a[1]],
                        sum(a[2] for a in a),
                    ]
                )
        return r

    a = [a for a in sf(recurse(km(d))) for a in a[1]]

    i = 0
    n = []
    while i < len(a):
        if i == 0 or a[i] != a[i - 1]:
            n.append(a[i])
        i += 1
    return n


# ---


def fix_list(nodes, edges, len_nodes=6):
    og_len = len(nodes)

    nodes += [0] * (len_nodes - og_len)
    _edges = []

    for i in range(len_nodes):
        for j in range(i + 1, len_nodes):
            if i < og_len and j < og_len:
                _edges.append(edges.pop(0))
            else:
                _edges.append(100)

    return nodes, _edges


def get_list(nodes, edges):
    # assert len(nodes) == 6, "Number of nodes should be 6"
    assert nodes[0] == 0, "First node should be 0"
    assert all(0 <= node <= 1 for node in nodes), "Nodes should be between 0 and 1"

    # assert len(edges) == 15, "Number of edges should be 15 -> (n * (n - 1) // 2)"
    assert len(edges) == len(nodes) * (len(nodes) - 1) // 2
    assert all(1 <= edge <= 100 for edge in edges), "Edges should be between 1 and 100"

    if len(nodes) < 6:
        nodes, edges = fix_list(nodes, edges)

    if len(nodes) <= 6:
        env = PTSPValEnv.fromlist(nodes, edges)
        env.reset()
        model = SAC.load(CHECKPOINT)

        _path = get_policy_path(model, env)
        path = []
        for n in _path:
            if nodes[n] != 0:
                path.append(n)
        return path
    else:
        return (
            np.random.permutation(len(nodes) - 1) + 1
        ).tolist()  # model not compatible with more than 6 nodes


# ------------------------------------------------------------


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


def get_policy_path(policy, env, num_nodes=6):
    state, _ = env.reset()

    path = [state["current_node"]]
    while len(path) < num_nodes:
        action, _ = policy.predict(state, deterministic=True)
        state, _, _, _, _ = env.step(action)
        path.append(state["current_node"])
    env.reset()
    return path


if __name__ == "__main__":
    with open("dump.json", "w") as f:
        json.dump({"bm_list": get_sorted_belief_maps(parse_json(DESC_FP))}, f)
