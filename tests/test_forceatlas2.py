"""
Tests for ForceAtlas2 main class and layout algorithms
"""

import numpy as np
import pytest
import scipy.sparse

from fa2_modified import ForceAtlas2


class TestForceAtlas2Initialization:
    """Test ForceAtlas2 class initialization"""

    def test_default_initialization(self):
        """Test ForceAtlas2 with default parameters"""
        fa2 = ForceAtlas2()

        assert fa2.outboundAttractionDistribution is False
        assert fa2.linLogMode is False
        assert fa2.adjustSizes is False
        assert fa2.edgeWeightInfluence == 1.0
        assert fa2.jitterTolerance == 1.0
        assert fa2.barnesHutOptimize is True
        assert fa2.barnesHutTheta == 1.2
        assert fa2.scalingRatio == 2.0
        assert fa2.strongGravityMode is False
        assert fa2.gravity == 1.0
        assert fa2.verbose is True

    def test_custom_parameters(self):
        """Test ForceAtlas2 with custom parameters"""
        fa2 = ForceAtlas2(
            outboundAttractionDistribution=True,
            edgeWeightInfluence=0.5,
            jitterTolerance=2.0,
            barnesHutOptimize=False,
            barnesHutTheta=1.5,
            scalingRatio=3.0,
            strongGravityMode=True,
            gravity=2.0,
            verbose=False,
        )

        assert fa2.outboundAttractionDistribution is True
        assert fa2.edgeWeightInfluence == 0.5
        assert fa2.jitterTolerance == 2.0
        assert fa2.barnesHutOptimize is False
        assert fa2.barnesHutTheta == 1.5
        assert fa2.scalingRatio == 3.0
        assert fa2.strongGravityMode is True
        assert fa2.gravity == 2.0
        assert fa2.verbose is False

    def test_unimplemented_features_raise_error(self):
        """Test that unimplemented features raise assertion error"""
        with pytest.raises(AssertionError):
            ForceAtlas2(linLogMode=True)

        with pytest.raises(AssertionError):
            ForceAtlas2(multiThreaded=True)

    def test_adjustSizes_feature_works(self):
        """Test that adjustSizes is now an implemented feature"""
        fa2 = ForceAtlas2(adjustSizes=True, nodeSize=2.0)
        assert fa2.adjustSizes is True
        assert fa2.nodeSize == 2.0


class TestForceAtlas2Init:
    """Test the init method that prepares graph data"""

    def test_init_numpy_array(self):
        """Test init with numpy array"""
        G = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])

        fa2 = ForceAtlas2(verbose=False)
        nodes, edges = fa2.init(G)

        assert len(nodes) == 4
        assert len(edges) == 4  # 4 unique edges in undirected graph

        # Check node masses (1 + number of connections)
        assert nodes[0].mass == 3.0  # 1 + 2 edges
        assert nodes[1].mass == 3.0
        assert nodes[2].mass == 3.0
        assert nodes[3].mass == 3.0

    def test_init_sparse_matrix(self):
        """Test init with scipy sparse matrix"""
        G = scipy.sparse.lil_matrix(
            [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]]
        )

        fa2 = ForceAtlas2(verbose=False)
        nodes, edges = fa2.init(G)

        assert len(nodes) == 4
        assert len(edges) == 4

    def test_init_with_positions(self):
        """Test init with initial positions"""
        G = np.array([[0, 1], [1, 0]])
        pos = np.array([[0.0, 0.0], [1.0, 1.0]])

        fa2 = ForceAtlas2(verbose=False)
        nodes, edges = fa2.init(G, pos=pos)

        assert nodes[0].x == 0.0
        assert nodes[0].y == 0.0
        assert nodes[1].x == 1.0
        assert nodes[1].y == 1.0

    def test_init_without_positions(self):
        """Test init without initial positions (random)"""
        G = np.array([[0, 1], [1, 0]])

        fa2 = ForceAtlas2(verbose=False)
        nodes, edges = fa2.init(G, pos=None)

        # Should have random positions between 0 and 1
        assert 0 <= nodes[0].x <= 1
        assert 0 <= nodes[0].y <= 1
        assert 0 <= nodes[1].x <= 1
        assert 0 <= nodes[1].y <= 1

    def test_init_weighted_graph(self):
        """Test init with weighted edges"""
        G = np.array([[0.0, 2.5, 0.0], [2.5, 0.0, 1.5], [0.0, 1.5, 0.0]])

        fa2 = ForceAtlas2(verbose=False)
        nodes, edges = fa2.init(G)

        assert len(edges) == 2
        # Check edge weights
        weights = sorted([e.weight for e in edges])
        assert weights == [1.5, 2.5]

    def test_init_invalid_input(self):
        """Test init with invalid input"""
        fa2 = ForceAtlas2(verbose=False)

        # Non-square matrix
        with pytest.raises(AssertionError):
            G = np.array([[0, 1], [1, 0], [0, 1]])
            fa2.init(G)

        # Non-symmetric matrix
        with pytest.raises(AssertionError):
            G = np.array([[0, 1], [0, 0]])
            fa2.init(G)

        # Invalid type
        with pytest.raises(AssertionError):
            fa2.init([[0, 1], [1, 0]])


class TestForceAtlas2Layout:
    """Test the main forceatlas2 layout method"""

    def test_simple_graph_layout(self):
        """Test layout on simple graph"""
        G = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]])

        fa2 = ForceAtlas2(verbose=False)
        pos = fa2.forceatlas2(G, iterations=10)

        assert len(pos) == 4
        assert all(len(p) == 2 for p in pos)  # Each position is (x, y)
        assert all(isinstance(p[0], float) and isinstance(p[1], float) for p in pos)

    def test_layout_with_initial_positions(self):
        """Test layout with provided initial positions"""
        G = np.array([[0, 1], [1, 0]])
        initial_pos = np.array([[0.0, 0.0], [10.0, 0.0]])

        fa2 = ForceAtlas2(verbose=False)
        pos = fa2.forceatlas2(G, pos=initial_pos, iterations=10)

        assert len(pos) == 2
        # Positions should have changed from initial
        # (they should be pulled together by attraction)

    def test_layout_barnes_hut_enabled(self):
        """Test layout with Barnes-Hut optimization"""
        G = np.array([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0]])

        fa2 = ForceAtlas2(barnesHutOptimize=True, verbose=False)
        pos = fa2.forceatlas2(G, iterations=10)

        assert len(pos) == 4

    def test_layout_barnes_hut_disabled(self):
        """Test layout without Barnes-Hut optimization"""
        G = np.array([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0]])

        fa2 = ForceAtlas2(barnesHutOptimize=False, verbose=False)
        pos = fa2.forceatlas2(G, iterations=10)

        assert len(pos) == 4

    def test_layout_strong_gravity(self):
        """Test layout with strong gravity mode"""
        G = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])

        fa2 = ForceAtlas2(strongGravityMode=True, gravity=2.0, verbose=False)
        pos = fa2.forceatlas2(G, iterations=50)

        assert len(pos) == 3
        # With strong gravity, nodes should be pulled toward origin

    def test_layout_outbound_attraction_distribution(self):
        """Test layout with outbound attraction distribution (hub dissuasion)"""
        # Create a star graph (one hub connected to all others)
        n = 5
        G = np.zeros((n, n))
        for i in range(1, n):
            G[0, i] = 1
            G[i, 0] = 1

        fa2 = ForceAtlas2(outboundAttractionDistribution=True, verbose=False)
        pos = fa2.forceatlas2(G, iterations=50)

        assert len(pos) == n

    def test_layout_edge_weight_influence(self):
        """Test layout with different edge weight influence"""
        G = np.array([[0.0, 1.0, 5.0], [1.0, 0.0, 1.0], [5.0, 1.0, 0.0]])

        # With weight influence = 1, weights should matter
        fa2_weighted = ForceAtlas2(edgeWeightInfluence=1.0, verbose=False)
        pos_weighted = fa2_weighted.forceatlas2(G, iterations=50)

        # With weight influence = 0, weights should be ignored
        fa2_unweighted = ForceAtlas2(edgeWeightInfluence=0.0, verbose=False)
        pos_unweighted = fa2_unweighted.forceatlas2(G, iterations=50)

        assert len(pos_weighted) == 3
        assert len(pos_unweighted) == 3

    def test_layout_different_iterations(self):
        """Test that more iterations produce different results"""
        G = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])

        fa2 = ForceAtlas2(verbose=False)

        # Set seed for reproducibility
        np.random.seed(42)
        pos_10 = fa2.forceatlas2(G, iterations=10)

        np.random.seed(42)
        pos_100 = fa2.forceatlas2(G, iterations=100)

        # Results should be different with different iteration counts
        assert pos_10 != pos_100

    def test_layout_sparse_matrix(self):
        """Test layout with scipy sparse matrix"""
        G = scipy.sparse.lil_matrix(
            [[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 1], [0, 0, 1, 0]]
        )

        fa2 = ForceAtlas2(verbose=False)
        pos = fa2.forceatlas2(G, iterations=10)

        assert len(pos) == 4

    def test_layout_single_node(self):
        """Test layout with single node"""
        G = np.array([[0]])

        fa2 = ForceAtlas2(verbose=False)
        pos = fa2.forceatlas2(G, iterations=10)

        assert len(pos) == 1

    def test_layout_disconnected_nodes(self):
        """Test layout with disconnected nodes"""
        G = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

        fa2 = ForceAtlas2(verbose=False)
        pos = fa2.forceatlas2(G, iterations=50)

        assert len(pos) == 4
        # Two disconnected components


class TestForceAtlas2Parameters:
    """Test different parameter combinations"""

    def test_varying_scaling_ratio(self):
        """Test with different scaling ratios"""
        G = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])

        for ratio in [0.5, 1.0, 2.0, 5.0]:
            fa2 = ForceAtlas2(scalingRatio=ratio, verbose=False)
            pos = fa2.forceatlas2(G, iterations=10)
            assert len(pos) == 3

    def test_varying_gravity(self):
        """Test with different gravity values"""
        G = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

        for grav in [0.1, 1.0, 5.0, 10.0]:
            fa2 = ForceAtlas2(gravity=grav, verbose=False)
            pos = fa2.forceatlas2(G, iterations=10)
            assert len(pos) == 3

    def test_varying_jitter_tolerance(self):
        """Test with different jitter tolerance values"""
        G = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])

        for jitter in [0.5, 1.0, 2.0]:
            fa2 = ForceAtlas2(jitterTolerance=jitter, verbose=False)
            pos = fa2.forceatlas2(G, iterations=10)
            assert len(pos) == 3

    def test_varying_barnes_hut_theta(self):
        """Test with different Barnes-Hut theta values"""
        G = np.array([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0]])

        for theta in [0.5, 1.0, 1.2, 2.0]:
            fa2 = ForceAtlas2(barnesHutTheta=theta, verbose=False)
            pos = fa2.forceatlas2(G, iterations=10)
            assert len(pos) == 4


class TestForceAtlas2Reproducibility:
    """Test reproducibility with same random seed"""

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results"""
        G = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])

        fa2 = ForceAtlas2(verbose=False)

        # First run
        np.random.seed(42)
        import random

        random.seed(42)
        pos1 = fa2.forceatlas2(G, iterations=10)

        # Second run with same seed
        np.random.seed(42)
        random.seed(42)
        pos2 = fa2.forceatlas2(G, iterations=10)

        # Results should be identical
        for p1, p2 in zip(pos1, pos2):
            assert abs(p1[0] - p2[0]) < 1e-10
            assert abs(p1[1] - p2[1]) < 1e-10


class TestForceAtlas2EdgeCases:
    """Test edge cases and boundary conditions"""

    def test_zero_iterations(self):
        """Test with zero iterations"""
        G = np.array([[0, 1], [1, 0]])

        fa2 = ForceAtlas2(verbose=False)
        pos = fa2.forceatlas2(G, iterations=0)

        assert len(pos) == 2

    def test_large_graph(self):
        """Test with larger graph"""
        n = 50
        G = np.zeros((n, n))
        # Create a ring graph
        for i in range(n):
            G[i, (i + 1) % n] = 1
            G[(i + 1) % n, i] = 1

        fa2 = ForceAtlas2(verbose=False)
        pos = fa2.forceatlas2(G, iterations=10)

        assert len(pos) == n

    def test_complete_graph(self):
        """Test with complete graph"""
        n = 10
        G = np.ones((n, n)) - np.eye(n)

        fa2 = ForceAtlas2(verbose=False)
        pos = fa2.forceatlas2(G, iterations=10)

        assert len(pos) == n

    def test_no_edges(self):
        """Test with graph with no edges"""
        G = np.zeros((5, 5))

        fa2 = ForceAtlas2(verbose=False)
        pos = fa2.forceatlas2(G, iterations=10)

        assert len(pos) == 5
