"""
Pytest configuration and fixtures for fa2_modified tests
"""

import numpy as np
import pytest


@pytest.fixture
def simple_adjacency_matrix():
    """Simple 4-node graph as adjacency matrix"""
    return np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])


@pytest.fixture
def weighted_adjacency_matrix():
    """Simple weighted graph as adjacency matrix"""
    return np.array(
        [
            [0.0, 2.5, 0.0, 1.0],
            [2.5, 0.0, 3.0, 0.0],
            [0.0, 3.0, 0.0, 1.5],
            [1.0, 0.0, 1.5, 0.0],
        ]
    )


@pytest.fixture
def triangle_graph():
    """Triangle graph (3 nodes, all connected)"""
    return np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])


@pytest.fixture
def star_graph():
    """Star graph (1 hub connected to 4 leaves)"""
    n = 5
    G = np.zeros((n, n))
    for i in range(1, n):
        G[0, i] = 1
        G[i, 0] = 1
    return G


@pytest.fixture
def disconnected_graph():
    """Graph with two disconnected components"""
    return np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])


@pytest.fixture
def ring_graph():
    """Ring graph (nodes in a circle)"""
    n = 6
    G = np.zeros((n, n))
    for i in range(n):
        G[i, (i + 1) % n] = 1
        G[(i + 1) % n, i] = 1
    return G


@pytest.fixture
def random_seed():
    """Set random seed for reproducible tests"""
    import random

    np.random.seed(42)
    random.seed(42)
    yield
    # Reset after test
    np.random.seed(None)
    random.seed(None)


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line(
        "markers", "requires_networkx: requires networkx to be installed"
    )
    config.addinivalue_line(
        "markers", "requires_igraph: requires igraph to be installed"
    )
