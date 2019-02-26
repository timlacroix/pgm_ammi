import numpy as np

class Edge:

    def __init__(self, pair_potential):
        self.pair_potential = pair_potential
        self.message = None


class Node:

    def __init__(self, name, id, potentials, verbose = False):
        self.name = name
        self.id = id
        self.potentials = potentials
        self.verbose = verbose

        self.out_edges = []
        self.in_edges = []
        self.children = []
        self.parent = None
        self.sampled_value = None
        self.marginal = None
        self.z = None

    def send_messages(self):
        to_send = set(range(len(self.in_edges)))
        for i, e in enumerate(self.in_edges):
            if e.message is not None:
                to_send.remove(i)

        sent = 0
        if len(to_send) == 1:
            id_edge = list(to_send)[0]
            sent += self.send_log(id_edge)
        elif len(to_send) == 0:
            for i, _e in enumerate(self.out_edges):
                sent += self.send_log(i)
        return sent

    def send_log(self, edge_id):
        if self.out_edges[edge_id].message is not None:
            return 0  # already sent
        msg_in = np.log(self.potentials)
        if len(self.in_edges) > 1:
            # Compute product of received messages
            msg_in = np.sum(np.vstack([
                x.message for i, x in enumerate(self.in_edges)
                if i != edge_id  # For each neighbor that is not the one we're sending to
            ]), axis=0) + np.log(self.potentials)

        edge = self.out_edges[edge_id]
        message = np.log(edge.pair_potential) + msg_in[None, :]
        maxi = np.max(message, axis=1)

        edge.message = maxi + np.log(np.sum(
            np.exp(message-maxi[:, None]), axis=1
        ))
        return 1

    def get_marginal(self):
        self.marginal = np.exp(np.log(self.potentials) + np.sum(
            np.vstack([x.message for x in self.in_edges]), axis=0
        ))

        self.z = np.sum(self.marginal)
        self.marginal /= self.z
        return self.marginal, self.z


class ProbabilisticGraph:

    def __init__(self):
        self.nodes = {}

        self.name_to_node = {}
        self.n_nodes = 0
        self.sample_order = None

    def add_node(self, name, potentials):
        node = Node(name, self.n_nodes, potentials)
        self.nodes[self.n_nodes] = node
        self.name_to_node[name] = self.n_nodes

        self.n_nodes += 1
        return node

    def add_edge(self, a, b, pair_potential):
        # Verify that arguments are given in the right order
        assert pair_potential.shape[1] == self.nodes[a].potentials.shape[0]
        assert pair_potential.shape[0] == self.nodes[b].potentials.shape[0]

        self.nodes[b].parent = a

        a_to_b = Edge(pair_potential)
        self.nodes[a].out_edges.append(a_to_b)
        self.nodes[a].children.append((self.nodes[b], a_to_b))

        self.nodes[b].in_edges.append(a_to_b)

        b_to_a = Edge(pair_potential.T)
        self.nodes[b].out_edges.append(b_to_a)
        self.nodes[a].in_edges.append(b_to_a)

    def get_sampling_order(self):
        if self.sample_order is not None:
            return self.sample_order

        roots = []
        for n in self.nodes.values():
            if n.parent is None:
                roots.append(n)

        final_order = []

        for root in roots:
            stack = [root]
            while stack:
                cur = stack.pop()
                final_order.append(cur)
                for c, _e in cur.children:
                    stack.append(c)

        self.sample_order = final_order
        return self.sample_order

    def reset_messages(self):
        for i, n in self.nodes.items():
            for e in n.out_edges:
                e.message = None
            for e in n.in_edges:
                e.message = None

    def send_messages(self) -> int:
        messages_sent= 0
        for _i, n in self.nodes.items():
            n_messages = n.send_messages()
            if n.verbose and n_messages > 0:
                print("Node {node_name} sent {n_messages} messages.".format(
                    node_name=n.name, n_messages=n_messages
                ))
            messages_sent += n_messages
        return messages_sent

    def sample(self, observed = None):
        order = self.get_sampling_order()
        res = []
        to_observe = None
        if observed:
            res = np.empty(len(observed))
            to_observe = {x: i for i, x in enumerate(observed)}
        for node in order:
            if node.parent is None:
                node.sampled_value = np.random.choice(
                    node.potentials.shape[0], p=node.potentials
                )
            for child, edge in node.children:
                child.sampled_value = np.random.choice(
                    node.potentials.shape[0],
                    p=edge.pair_potential[node.sampled_value]
                )
            if to_observe is None:
                res.append(node.sampled_value)
            elif node.name in to_observe:
                res[to_observe[node.name]] = node.sampled_value

        return np.array(res)

    def get_node(self, name):
        return self.nodes[self.name_to_node[name]]


from scipy.linalg import expm

def build_graph_from_tree(
        pg, tree_node, pg_node, mutate_proba
):
    for x in tree_node.children:
        child = pg.add_node(x.name, np.ones(4))

        mutate = expm(x.distance_to_parent * mutate_proba)
        assert (np.abs((np.sum(mutate, axis=1) - 1)) < 1e-10).all()
        # mutate /= np.sum(mutate, axis=1)[:, None]
        pg.add_edge(pg_node.id, child.id, mutate)
        build_graph_from_tree(pg, x, child, mutate_proba)

    return pg


class Tree:

    def __init__(self, name, distance, parent=None):
        self.parent = parent
        self.distance_to_parent = distance
        self.children = []
        self.name = name

    def print_tree(self, indentation=''):
        res = indentation + self.name + '\n'
        for x in self.children:
            res += x.print_tree(indentation+'\t')
        return res

    def gen_all_trees(self, root, child_id, res, pi, mutate_proba):
        if self.parent:
            old_parent = self.parent
            # Try inserting above me
            edge_name = self.parent.name + "->" + self.name

            # insert
            new_node = Tree(
                "new_ancestor", self.distance_to_parent / 2, self.parent
            )
            self.parent.children[child_id] = new_node

            observed = Tree(
                "observed",
                0.01,  # self.parent.closest_leaf(),
                new_node
            )

            new_node.children = [self, observed]

            self.distance_to_parent /= 2
            self.parent = new_node
            # -> gen Proba Tree
            pg = ProbabilisticGraph()
            pg_root = pg.add_node("root", pi)
            build_graph_from_tree(pg, root, pg_root, mutate_proba)

            res[edge_name] = [pg, root.print_tree()]
            # go back to normal
            self.parent = old_parent
            self.distance_to_parent *= 2
            self.parent.children[child_id] = self

        for k, child in enumerate(self.children):
            child.gen_all_trees(root, k, res, pi, mutate_proba)
        return res


def gen_tree(t, tree):
    t.name = tree[0]
    for x in tree[2]:
        c = Tree(x[0], x[1], t)
        t.children.append(c)
        gen_tree(c, x)


def gen_candidate_trees():
    tree = Tree(".", 0, None)
    gen_tree(
        tree,
        ("root", 0, [
            ("child1", 0.25, [
                ("child3", 0.8, [
                    ("human", 0.2, []), ("baboon", 0.27, []),
                 ]),
                ("child4", 2.9, [
                    ("mouse", 0.7, []), ("rat", 0.76, []),
                 ]),
            ]),
            ("child2", 0.25, [
               ("child5", 0.3, [
                    ("cow", 1.1, []), ("pig", 1, []),
                ]),
               ("child6", 0.5, [
                    ("cat", 0.76, []), ("dog", 1.1, []),
                ]),
            ]),
        ])
    )

    pi = np.array([0.39, 0.12, 0.11, 0.38])

    def q_from_pi(pi, kappa):
        temp = np.array([
            [0, pi[1], kappa * pi[2], pi[3]],
            [pi[0], 0, pi[2], kappa * pi[3]],
            [kappa * pi[0], pi[1], 0, pi[3]],
            [pi[0], pi[1], kappa * pi[2], 0],
        ])
        for i, row in enumerate(temp):
            row[i] = -np.sum(row)
        return temp

    return tree.gen_all_trees(tree, 0, {}, pi, q_from_pi(pi, 2.7))
