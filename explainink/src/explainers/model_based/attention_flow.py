from typing import Dict, List, Optional

import torch

try:
    import networkx as nx
except ImportError:
    print("NetworkX is not installed, install it if required.")

import numpy as np

from ...common import InvalidModelForExplainer, get_torch_aggregation
from ...models import ClassificationModel
from ...types import ModelExplainerOutput
from .base import ModelExplainer

# Code taken from the repository of the paper https://arxiv.org/pdf/2005.00928
# https://github.com/samiraabnar/attention_flow/blob/master/attention_graph_util.py


def get_adjmat(mat, input_tokens):
    n_layers, length, _ = mat.shape
    adj_mat = np.zeros(((n_layers + 1) * length, (n_layers + 1) * length))
    labels_to_index = {}
    for k in np.arange(length):
        labels_to_index[str(k) + "_" + input_tokens[k]] = k

    for i in np.arange(1, n_layers + 1):
        for k_f in np.arange(length):
            index_from = (i) * length + k_f
            label = "L" + str(i) + "_" + str(k_f)
            labels_to_index[label] = index_from
            for k_t in np.arange(length):
                index_to = (i - 1) * length + k_t
                adj_mat[index_from][index_to] = mat[i - 1][k_f][k_t]

    return adj_mat, labels_to_index


def convert_adjmat_tomats(adjmat, n_layers, length):
    mats = np.zeros((n_layers, length, length))

    for i in np.arange(n_layers):
        mats[i] = adjmat[
            (i + 1) * length : (i + 2) * length, i * length : (i + 1) * length
        ]

    return mats


def draw_attention_graph(adjmat, labels_to_index, n_layers, length):
    A = adjmat
    G = nx.from_numpy_array(A, create_using=nx.DiGraph())
    for i in np.arange(A.shape[0]):
        for j in np.arange(A.shape[1]):
            nx.set_edge_attributes(G, {(i, j): A[i, j]}, "capacity")

    pos = {}
    label_pos = {}
    for i in np.arange(n_layers + 1):
        for k_f in np.arange(length):
            pos[i * length + k_f] = ((i + 0.4) * 2, length - k_f)
            label_pos[i * length + k_f] = (i * 2, length - k_f)

    index_to_labels = {}
    for key in labels_to_index:
        index_to_labels[labels_to_index[key]] = key.split("_")[-1]
        if labels_to_index[key] >= length:
            index_to_labels[labels_to_index[key]] = ""

    all_weights = []
    # 4 a. Iterate through the graph nodes to gather all the weights
    for _, _, data in G.edges(data=True):
        all_weights.append(
            data["weight"]
        )  # we'll use this when determining edge thickness

    # 4 b. Get unique weights
    unique_weights = list(set(all_weights))

    # 4 c. Plot the edges - one by one!
    for weight in unique_weights:
        # 4 d. Form a filtered list with just the weight you want to draw
        weighted_edges = [
            (node1, node2)
            for (node1, node2, edge_attr) in G.edges(data=True)
            if edge_attr["weight"] == weight
        ]
        # 4 e. I think multiplying by [num_nodes/sum(all_weights)] makes the graphs edges look cleaner
        w = weight
        width = w
        nx.draw_networkx_edges(
            G, pos, edgelist=weighted_edges, width=width, edge_color="darkblue"
        )

    return G


def compute_joint_attention(att_mat, add_residual=True):
    if add_residual:
        residual_att = np.eye(att_mat.shape[1])[None, ...]
        aug_att_mat = att_mat + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(axis=-1)[..., None]
    else:
        aug_att_mat = att_mat

    joint_attentions = np.zeros(aug_att_mat.shape)

    layers = joint_attentions.shape[0]
    joint_attentions[0] = aug_att_mat[0]
    for i in np.arange(1, layers):
        joint_attentions[i] = aug_att_mat[i].dot(joint_attentions[i - 1])

    return joint_attentions


def compute_flows(G, labels_to_index, input_nodes, length):
    number_of_nodes = len(labels_to_index)
    flow_values = np.zeros((number_of_nodes, number_of_nodes))
    for key in labels_to_index:
        if key not in input_nodes:
            current_layer = int(labels_to_index[key] / length)
            pre_layer = current_layer - 1
            u = labels_to_index[key]
            for inp_node_key in input_nodes:
                v = labels_to_index[inp_node_key]
                flow_value = nx.maximum_flow_value(
                    G, u, v, flow_func=nx.algorithms.flow.edmonds_karp
                )
                flow_values[u][pre_layer * length + v] = flow_value
            flow_values[u] /= flow_values[u].sum()

    return flow_values


class AttentionFlowExplainer(ModelExplainer):
    """
    Attention Flow from https://arxiv.org/pdf/2005.00928

    Can only be used with models whose `forward` method returns `attentions`.

    Attributes:
        model (ClassificationModel): a classification model.
        kwargs (Dict): dict with additional parameters.

    """

    def __init__(self, model: ClassificationModel, **kwargs):
        super().__init__(model, **kwargs)
        self.aggregation = get_torch_aggregation(
            self.kwargs.get("aggregation", "mean")
        )

    def _explain(
        self, features: Dict, targets: Optional[List[int]] = None, **kwargs
    ) -> ModelExplainerOutput:
        with torch.inference_mode():
            output = self.model.forward(**features, output_attentions=True)

        attentions = output.attentions

        if attentions is None:
            raise InvalidModelForExplainer(
                self.model.__class__.__name__, self.__class__.__name__
            )

        pred_probs = output.logits.softmax(-1).max(-1).values.tolist()
        pred_labels = output.logits.argmax(-1).tolist()

        # From num_layers tensors of (batch_size, num_heads, sequence_length, sequence_length)
        # to (batch_size, num_layers, num_headers, sequence_length, sequence_length)
        attentions = torch.stack(attentions, dim=0)
        attentions = torch.einsum("lbhij->blhij", attentions)

        # Fuse attention heads in each layer (see Section 3 of the paper)
        attentions = self.aggregation(attentions, dim=2)

        # Follow the pipeline of the notebooks in:
        # https://github.com/samiraabnar/attention_flow
        attentions = attentions.detach().cpu().numpy()
        tokens = [str(i) for i in range(attentions.shape[-1])]

        scores = []
        for id_example in range(attentions.shape[0]):
            example_attentions = attentions[id_example, ...]
            example_attentions = compute_joint_attention(example_attentions)
            adj_mat, labels_to_index = get_adjmat(example_attentions, tokens)

            input_nodes = []
            for key in labels_to_index:
                if not key.startswith("L"):
                    input_nodes.append(key)

            G = draw_attention_graph(
                adj_mat,
                labels_to_index,
                example_attentions.shape[0],
                example_attentions.shape[-1],
            )
            flow_values = compute_flows(
                G, labels_to_index, input_nodes, example_attentions.shape[-1]
            )
            flow_att_map = convert_adjmat_tomats(
                flow_values,
                example_attentions.shape[0],
                example_attentions.shape[-1],
            )
            scores.append(flow_att_map.mean(0).mean(0))

        scores = np.vstack(scores)

        return ModelExplainerOutput(
            scores=scores,
            pred_labels=pred_labels,
            pred_probs=pred_probs,
        )
