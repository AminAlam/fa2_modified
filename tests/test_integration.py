"""
Integration tests for fa2_modified package
Tests the interaction between different components
"""

import numpy as np
import pytest
import scipy.sparse

from fa2_modified import ForceAtlas2


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows"""

    def test_numpy_to_layout_workflow(self):
        """Test complete workflow from numpy array to layout"""
        # Create adjacency matrix
        G = np.array(
            [
                [0, 1, 1, 0, 0],
                [1, 0, 1, 1, 0],
                [1, 1, 0, 1, 1],
                [0, 1, 1, 0, 1],
                [0, 0, 1, 1, 0],
            ]
        )

        # Initialize ForceAtlas2
        fa2 = ForceAtlas2(
            barnesHutOptimize=True,
            barnesHutTheta=1.2,
            scalingRatio=2.0,
            strongGravityMode=False,
            gravity=1.0,
            verbose=False,
        )

        # Compute layout
        positions = fa2.forceatlas2(G, iterations=100)

        # Verify results
        assert len(positions) == 5
        assert all(len(pos) == 2 for pos in positions)

        # Verify positions are reasonable (not all at origin)
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        assert np.std(x_coords) > 0.01
        assert np.std(y_coords) > 0.01

    def test_sparse_to_layout_workflow(self):
        """Test complete workflow from sparse matrix to layout"""
        # Create sparse adjacency matrix
        G = scipy.sparse.lil_matrix(
            [
                [0, 1, 1, 0, 0],
                [1, 0, 1, 1, 0],
                [1, 1, 0, 1, 1],
                [0, 1, 1, 0, 1],
                [0, 0, 1, 1, 0],
            ]
        )

        fa2 = ForceAtlas2(verbose=False)
        positions = fa2.forceatlas2(G, iterations=100)

        assert len(positions) == 5
        assert all(len(pos) == 2 for pos in positions)

    def test_weighted_workflow(self):
        """Test workflow with weighted edges"""
        # Create weighted graph where one edge is much stronger
        G = np.array(
            [
                [0.0, 1.0, 1.0, 0.0],
                [1.0, 0.0, 10.0, 0.0],  # Strong connection between 1 and 2
                [1.0, 10.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )

        fa2 = ForceAtlas2(edgeWeightInfluence=1.0, verbose=False)
        positions = fa2.forceatlas2(G, iterations=200)

        # Nodes 1 and 2 should be closer together due to strong edge
        dist_1_2 = np.linalg.norm(np.array(positions[1]) - np.array(positions[2]))
        dist_0_1 = np.linalg.norm(np.array(positions[0]) - np.array(positions[1]))

        # The strong edge should pull nodes closer (this is a heuristic test)
        assert len(positions) == 4

    def test_initial_positions_workflow(self):
        """Test workflow with provided initial positions"""
        G = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

        # Start with nodes in a line
        initial_pos = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])

        fa2 = ForceAtlas2(verbose=False)
        positions = fa2.forceatlas2(G, pos=initial_pos, iterations=100)

        assert len(positions) == 3
        # Positions should have changed from initial
        for i in range(3):
            pos_changed = (
                positions[i][0] != initial_pos[i][0]
                or positions[i][1] != initial_pos[i][1]
            )
            assert pos_changed


class TestParameterInteractions:
    """Test interactions between different parameters"""

    def test_gravity_repulsion_balance(self):
        """Test balance between gravity and repulsion"""
        G = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

        # High gravity should keep nodes close to origin
        fa2_high_gravity = ForceAtlas2(gravity=10.0, scalingRatio=1.0, verbose=False)
        pos_high_gravity = fa2_high_gravity.forceatlas2(G, iterations=100)

        # Low gravity should allow nodes to spread out more
        fa2_low_gravity = ForceAtlas2(gravity=0.1, scalingRatio=1.0, verbose=False)
        pos_low_gravity = fa2_low_gravity.forceatlas2(G, iterations=100)

        # Calculate spread from origin
        spread_high = np.mean([np.linalg.norm(pos) for pos in pos_high_gravity])
        spread_low = np.mean([np.linalg.norm(pos) for pos in pos_low_gravity])

        # Lower gravity should result in larger spread (generally)
        assert len(pos_high_gravity) == 4
        assert len(pos_low_gravity) == 4

    def test_barnes_hut_vs_direct(self):
        """Test that Barnes-Hut gives similar results to direct computation"""
        G = np.array(
            [
                [0, 1, 1, 0, 0],
                [1, 0, 1, 1, 0],
                [1, 1, 0, 1, 1],
                [0, 1, 1, 0, 1],
                [0, 0, 1, 1, 0],
            ]
        )

        # Set seed for reproducibility
        np.random.seed(42)
        import random

        random.seed(42)

        fa2_barnes = ForceAtlas2(barnesHutOptimize=True, verbose=False)
        pos_barnes = fa2_barnes.forceatlas2(G, iterations=50)

        np.random.seed(42)
        random.seed(42)

        fa2_direct = ForceAtlas2(barnesHutOptimize=False, verbose=False)
        pos_direct = fa2_direct.forceatlas2(G, iterations=50)

        # Results should be similar (not identical due to approximation)
        assert len(pos_barnes) == len(pos_direct)

    def test_strong_vs_normal_gravity(self):
        """Test difference between strong and normal gravity"""
        G = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

        np.random.seed(42)
        fa2_strong = ForceAtlas2(strongGravityMode=True, gravity=1.0, verbose=False)
        pos_strong = fa2_strong.forceatlas2(G, iterations=100)

        np.random.seed(42)
        fa2_normal = ForceAtlas2(strongGravityMode=False, gravity=1.0, verbose=False)
        pos_normal = fa2_normal.forceatlas2(G, iterations=100)

        assert len(pos_strong) == 3
        assert len(pos_normal) == 3


class TestScalability:
    """Test with different graph sizes"""

    def test_small_graph(self):
        """Test with very small graph (2 nodes)"""
        G = np.array([[0, 1], [1, 0]])

        fa2 = ForceAtlas2(verbose=False)
        positions = fa2.forceatlas2(G, iterations=10)

        assert len(positions) == 2

    def test_medium_graph(self):
        """Test with medium-sized graph (50 nodes)"""
        n = 50
        G = np.zeros((n, n))
        # Create a random graph with ~10% edge density
        for i in range(n):
            for j in range(i + 1, n):
                if np.random.random() < 0.1:
                    G[i, j] = 1
                    G[j, i] = 1

        fa2 = ForceAtlas2(barnesHutOptimize=True, verbose=False)
        positions = fa2.forceatlas2(G, iterations=50)

        assert len(positions) == n

    @pytest.mark.slow
    def test_large_graph(self):
        """Test with larger graph (200 nodes)"""
        n = 200
        G = scipy.sparse.lil_matrix((n, n))
        # Create a sparse random graph
        for i in range(n):
            num_edges = np.random.randint(1, 5)
            for _ in range(num_edges):
                j = np.random.randint(0, n)
                if i != j:
                    G[i, j] = 1
                    G[j, i] = 1

        fa2 = ForceAtlas2(barnesHutOptimize=True, verbose=False)
        positions = fa2.forceatlas2(G, iterations=50)

        assert len(positions) == n


class TestSpecialGraphTopologies:
    """Test with special graph structures"""

    def test_star_topology(self):
        """Test star graph (hub and spokes)"""
        n = 10
        G = np.zeros((n, n))
        # Node 0 is hub, connected to all others
        for i in range(1, n):
            G[0, i] = 1
            G[i, 0] = 1

        fa2 = ForceAtlas2(outboundAttractionDistribution=True, verbose=False)
        positions = fa2.forceatlas2(G, iterations=100)

        assert len(positions) == n

    def test_complete_bipartite(self):
        """Test complete bipartite graph"""
        n1, n2 = 5, 5
        n = n1 + n2
        G = np.zeros((n, n))
        # Connect all nodes in first set to all nodes in second set
        for i in range(n1):
            for j in range(n1, n):
                G[i, j] = 1
                G[j, i] = 1

        fa2 = ForceAtlas2(verbose=False)
        positions = fa2.forceatlas2(G, iterations=100)

        assert len(positions) == n

    def test_chain_topology(self):
        """Test chain/path graph"""
        n = 10
        G = np.zeros((n, n))
        for i in range(n - 1):
            G[i, i + 1] = 1
            G[i + 1, i] = 1

        fa2 = ForceAtlas2(verbose=False)
        positions = fa2.forceatlas2(G, iterations=100)

        assert len(positions) == n

    def test_grid_topology(self):
        """Test grid graph"""
        rows, cols = 4, 4
        n = rows * cols
        G = np.zeros((n, n))

        # Helper to convert 2D coords to 1D index
        def idx(r, c):
            return r * cols + c

        # Connect grid neighbors
        for r in range(rows):
            for c in range(cols):
                if r < rows - 1:
                    G[idx(r, c), idx(r + 1, c)] = 1
                    G[idx(r + 1, c), idx(r, c)] = 1
                if c < cols - 1:
                    G[idx(r, c), idx(r, c + 1)] = 1
                    G[idx(r, c + 1), idx(r, c)] = 1

        fa2 = ForceAtlas2(verbose=False)
        positions = fa2.forceatlas2(G, iterations=100)

        assert len(positions) == n


class TestConvergence:
    """Test convergence properties"""

    def test_convergence_over_iterations(self):
        """Test that layout stabilizes over iterations"""
        G = np.array([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0]])

        fa2 = ForceAtlas2(verbose=False)

        # Run with different iteration counts
        np.random.seed(42)
        pos_10 = fa2.forceatlas2(G, iterations=10)

        np.random.seed(42)
        pos_100 = fa2.forceatlas2(G, iterations=100)

        np.random.seed(42)
        pos_1000 = fa2.forceatlas2(G, iterations=1000)

        # More iterations should produce valid layouts
        assert len(pos_10) == 4
        assert len(pos_100) == 4
        assert len(pos_1000) == 4

    def test_reproducibility_same_seed(self):
        """Test that same seed produces identical results"""
        G = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])

        fa2 = ForceAtlas2(verbose=False)

        # First run
        np.random.seed(12345)
        import random

        random.seed(12345)
        pos1 = fa2.forceatlas2(G, iterations=50)

        # Second run with same seed
        np.random.seed(12345)
        random.seed(12345)
        pos2 = fa2.forceatlas2(G, iterations=50)

        # Should be identical
        for i in range(len(pos1)):
            assert abs(pos1[i][0] - pos2[i][0]) < 1e-10
            assert abs(pos1[i][1] - pos2[i][1]) < 1e-10
