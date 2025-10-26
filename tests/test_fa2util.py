"""
Tests for fa2util module - testing force calculations and utility classes
"""
from math import sqrt

import numpy as np
import pytest

from fa2_modified import fa2util


class TestNode:
    """Test Node class"""
    
    def test_node_initialization(self):
        """Test Node object initialization"""
        node = fa2util.Node()
        assert node.mass == 0.0
        assert node.old_dx == 0.0
        assert node.old_dy == 0.0
        assert node.dx == 0.0
        assert node.dy == 0.0
        assert node.x == 0.0
        assert node.y == 0.0
    
    def test_node_attributes(self):
        """Test Node attribute assignment"""
        node = fa2util.Node()
        node.mass = 5.0
        node.x = 10.0
        node.y = 20.0
        node.dx = 1.0
        node.dy = 2.0
        
        assert node.mass == 5.0
        assert node.x == 10.0
        assert node.y == 20.0
        assert node.dx == 1.0
        assert node.dy == 2.0


class TestEdge:
    """Test Edge class"""
    
    def test_edge_initialization(self):
        """Test Edge object initialization"""
        edge = fa2util.Edge()
        assert edge.node1 == -1
        assert edge.node2 == -1
        assert edge.weight == 0.0
    
    def test_edge_attributes(self):
        """Test Edge attribute assignment"""
        edge = fa2util.Edge()
        edge.node1 = 0
        edge.node2 = 1
        edge.weight = 2.5
        
        assert edge.node1 == 0
        assert edge.node2 == 1
        assert edge.weight == 2.5


class TestLinRepulsion:
    """Test linear repulsion force function"""
    
    def test_repulsion_basic(self):
        """Test basic repulsion between two nodes"""
        n1 = fa2util.Node()
        n1.x = 0.0
        n1.y = 0.0
        n1.mass = 1.0
        
        n2 = fa2util.Node()
        n2.x = 1.0
        n2.y = 0.0
        n2.mass = 1.0
        
        fa2util.linRepulsion(n1, n2, coefficient=1.0)
        
        # Nodes should be pushed apart
        assert n1.dx < 0  # n1 pushed left
        assert n2.dx > 0  # n2 pushed right
        assert abs(n1.dx) == abs(n2.dx)  # Equal and opposite forces
        assert n1.dy == 0.0  # No vertical movement
        assert n2.dy == 0.0
    
    def test_repulsion_zero_distance(self):
        """Test repulsion when nodes are at the same position"""
        n1 = fa2util.Node()
        n1.x = 0.0
        n1.y = 0.0
        n1.mass = 1.0
        
        n2 = fa2util.Node()
        n2.x = 0.0
        n2.y = 0.0
        n2.mass = 1.0
        
        fa2util.linRepulsion(n1, n2, coefficient=1.0)
        
        # No force should be applied when distance is zero
        assert n1.dx == 0.0
        assert n1.dy == 0.0
        assert n2.dx == 0.0
        assert n2.dy == 0.0
    
    def test_repulsion_different_masses(self):
        """Test repulsion with different node masses"""
        n1 = fa2util.Node()
        n1.x = 0.0
        n1.y = 0.0
        n1.mass = 2.0
        
        n2 = fa2util.Node()
        n2.x = 1.0
        n2.y = 0.0
        n2.mass = 1.0
        
        fa2util.linRepulsion(n1, n2, coefficient=1.0)
        
        # Forces should be equal and opposite despite different masses
        assert abs(n1.dx) == abs(n2.dx)


class TestLinGravity:
    """Test linear gravity force function"""
    
    def test_gravity_basic(self):
        """Test basic gravity pulling node toward origin"""
        n = fa2util.Node()
        n.x = 10.0
        n.y = 0.0
        n.mass = 1.0
        
        fa2util.linGravity(n, g=1.0)
        
        # Node should be pulled toward origin
        assert n.dx < 0
        assert n.dy == 0.0
    
    def test_gravity_at_origin(self):
        """Test gravity when node is at origin"""
        n = fa2util.Node()
        n.x = 0.0
        n.y = 0.0
        n.mass = 1.0
        
        fa2util.linGravity(n, g=1.0)
        
        # No force when at origin
        assert n.dx == 0.0
        assert n.dy == 0.0
    
    def test_gravity_diagonal(self):
        """Test gravity pulling diagonally"""
        n = fa2util.Node()
        n.x = 3.0
        n.y = 4.0
        n.mass = 1.0
        
        fa2util.linGravity(n, g=1.0)
        
        # Both components should pull toward origin
        assert n.dx < 0
        assert n.dy < 0
        
        # Check that the direction is correct
        distance = sqrt(n.x**2 + n.y**2)
        expected_dx = -(n.x / distance) * n.mass
        expected_dy = -(n.y / distance) * n.mass
        
        assert abs(n.dx - expected_dx) < 1e-10
        assert abs(n.dy - expected_dy) < 1e-10


class TestStrongGravity:
    """Test strong gravity force function"""
    
    def test_strong_gravity_basic(self):
        """Test strong gravity force"""
        n = fa2util.Node()
        n.x = 10.0
        n.y = 5.0
        n.mass = 1.0
        
        fa2util.strongGravity(n, g=1.0, coefficient=1.0)
        
        # Should pull toward origin
        assert n.dx < 0
        assert n.dy < 0
    
    def test_strong_gravity_at_origin(self):
        """Test strong gravity when at origin"""
        n = fa2util.Node()
        n.x = 0.0
        n.y = 0.0
        n.mass = 1.0
        
        fa2util.strongGravity(n, g=1.0, coefficient=1.0)
        
        # No force at origin
        assert n.dx == 0.0
        assert n.dy == 0.0


class TestLinAttraction:
    """Test linear attraction force function"""
    
    def test_attraction_basic(self):
        """Test basic attraction between connected nodes"""
        n1 = fa2util.Node()
        n1.x = 0.0
        n1.y = 0.0
        n1.mass = 1.0
        
        n2 = fa2util.Node()
        n2.x = 10.0
        n2.y = 0.0
        n2.mass = 1.0
        
        fa2util.linAttraction(n1, n2, e=1.0, distributedAttraction=False, coefficient=1.0)
        
        # Nodes should be pulled together
        assert n1.dx > 0  # n1 pulled right
        assert n2.dx < 0  # n2 pulled left
        assert abs(n1.dx) == abs(n2.dx)  # Equal and opposite
    
    def test_attraction_distributed(self):
        """Test distributed attraction (hub dissuasion)"""
        n1 = fa2util.Node()
        n1.x = 0.0
        n1.y = 0.0
        n1.mass = 10.0  # Hub node with high mass
        
        n2 = fa2util.Node()
        n2.x = 10.0
        n2.y = 0.0
        n2.mass = 1.0
        
        fa2util.linAttraction(n1, n2, e=1.0, distributedAttraction=True, coefficient=1.0)
        
        # With distributed attraction, force on n1 is reduced by its mass
        assert n1.dx > 0
        assert n2.dx < 0
    
    def test_attraction_weighted_edge(self):
        """Test attraction with edge weight"""
        n1 = fa2util.Node()
        n1.x = 0.0
        n1.y = 0.0
        n1.mass = 1.0
        
        n2 = fa2util.Node()
        n2.x = 10.0
        n2.y = 0.0
        n2.mass = 1.0
        
        fa2util.linAttraction(n1, n2, e=2.0, distributedAttraction=False, coefficient=1.0)
        
        # Stronger edge weight should result in stronger force
        assert n1.dx > 0
        assert n2.dx < 0


class TestApplyForces:
    """Test force application functions"""
    
    def test_apply_repulsion(self):
        """Test applying repulsion to multiple nodes"""
        nodes = []
        for i in range(3):
            n = fa2util.Node()
            n.x = float(i)
            n.y = 0.0
            n.mass = 1.0
            nodes.append(n)
        
        fa2util.apply_repulsion(nodes, coefficient=1.0)
        
        # Middle node should have zero net force (symmetry)
        # End nodes should be pushed outward
        assert nodes[0].dx < 0  # Left node pushed left
        assert nodes[2].dx > 0  # Right node pushed right
    
    def test_apply_gravity(self):
        """Test applying gravity to multiple nodes"""
        nodes = []
        for i in range(3):
            n = fa2util.Node()
            n.x = float(i + 1)
            n.y = float(i + 1)
            n.mass = 1.0
            nodes.append(n)
        
        fa2util.apply_gravity(nodes, gravity=1.0, scalingRatio=1.0, useStrongGravity=False)
        
        # All nodes should be pulled toward origin
        for node in nodes:
            assert node.dx < 0
            assert node.dy < 0
    
    def test_apply_strong_gravity(self):
        """Test applying strong gravity to multiple nodes"""
        nodes = []
        for i in range(3):
            n = fa2util.Node()
            n.x = float(i + 1)
            n.y = float(i + 1)
            n.mass = 1.0
            nodes.append(n)
        
        fa2util.apply_gravity(nodes, gravity=1.0, scalingRatio=1.0, useStrongGravity=True)
        
        # All nodes should be pulled toward origin
        for node in nodes:
            assert node.dx < 0
            assert node.dy < 0
    
    def test_apply_attraction(self):
        """Test applying attraction forces along edges"""
        nodes = []
        for i in range(3):
            n = fa2util.Node()
            n.x = float(i * 10)
            n.y = 0.0
            n.mass = 1.0
            nodes.append(n)
        
        edges = []
        # Connect node 0 to node 1
        e = fa2util.Edge()
        e.node1 = 0
        e.node2 = 1
        e.weight = 1.0
        edges.append(e)
        
        fa2util.apply_attraction(nodes, edges, distributedAttraction=False, 
                                coefficient=1.0, edgeWeightInfluence=1.0)
        
        # Node 0 and 1 should be pulled together
        assert nodes[0].dx > 0
        assert nodes[1].dx < 0
        # Node 2 should have no force (not connected)
        assert nodes[2].dx == 0.0


class TestRegion:
    """Test Barnes-Hut Region class"""
    
    def test_region_initialization(self):
        """Test Region initialization with nodes"""
        nodes = []
        for i in range(4):
            n = fa2util.Node()
            n.x = float(i)
            n.y = float(i)
            n.mass = 1.0
            nodes.append(n)
        
        region = fa2util.Region(nodes)
        
        assert region.mass > 0
        assert region.size > 0
        assert len(region.nodes) == 4
        assert len(region.subregions) == 0
    
    def test_region_mass_center(self):
        """Test region mass center calculation"""
        nodes = []
        # Create nodes in a square pattern
        positions = [(0, 0), (0, 10), (10, 0), (10, 10)]
        for x, y in positions:
            n = fa2util.Node()
            n.x = float(x)
            n.y = float(y)
            n.mass = 1.0
            nodes.append(n)
        
        region = fa2util.Region(nodes)
        
        # Center should be at (5, 5)
        assert abs(region.massCenterX - 5.0) < 1e-10
        assert abs(region.massCenterY - 5.0) < 1e-10
    
    def test_region_build_subregions(self):
        """Test building subregions for Barnes-Hut"""
        nodes = []
        # Create nodes in different quadrants
        positions = [(0, 0), (0, 10), (10, 0), (10, 10)]
        for x, y in positions:
            n = fa2util.Node()
            n.x = float(x)
            n.y = float(y)
            n.mass = 1.0
            nodes.append(n)
        
        region = fa2util.Region(nodes)
        region.buildSubRegions()
        
        # Should have created subregions
        assert len(region.subregions) > 0
    
    def test_region_single_node(self):
        """Test region with single node"""
        n = fa2util.Node()
        n.x = 5.0
        n.y = 5.0
        n.mass = 1.0
        
        region = fa2util.Region([n])
        region.buildSubRegions()
        
        # Single node should not create subregions
        assert len(region.subregions) == 0


class TestAdjustSpeedAndApplyForces:
    """Test speed adjustment and force application"""
    
    def test_adjust_speed_basic(self):
        """Test basic speed adjustment"""
        nodes = []
        for i in range(5):
            n = fa2util.Node()
            n.x = float(i)
            n.y = float(i)
            n.mass = 1.0
            n.dx = 0.1
            n.dy = 0.1
            n.old_dx = 0.0
            n.old_dy = 0.0
            nodes.append(n)
        
        result = fa2util.adjustSpeedAndApplyForces(nodes, speed=1.0, 
                                                   speedEfficiency=1.0, 
                                                   jitterTolerance=1.0)
        
        assert 'speed' in result
        assert 'speedEfficiency' in result
        assert result['speed'] > 0
        assert result['speedEfficiency'] > 0
    
    def test_adjust_speed_applies_forces(self):
        """Test that forces are applied to node positions"""
        nodes = []
        for i in range(3):
            n = fa2util.Node()
            n.x = float(i)
            n.y = float(i)
            n.mass = 1.0
            n.dx = 1.0
            n.dy = -1.0
            n.old_dx = 0.5
            n.old_dy = -0.5
            nodes.append(n)
        
        old_positions = [(n.x, n.y) for n in nodes]
        
        result = fa2util.adjustSpeedAndApplyForces(nodes, speed=1.0,
                                                   speedEfficiency=1.0,
                                                   jitterTolerance=1.0)
        
        # Positions should have changed
        for i, n in enumerate(nodes):
            assert n.x != old_positions[i][0] or n.y != old_positions[i][1]


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_nodes_list(self):
        """Test with empty nodes list"""
        nodes = []
        fa2util.apply_repulsion(nodes, coefficient=1.0)
        fa2util.apply_gravity(nodes, gravity=1.0, scalingRatio=1.0)
        # Should not raise errors
    
    def test_single_node_forces(self):
        """Test forces on single node"""
        nodes = []
        n = fa2util.Node()
        n.x = 5.0
        n.y = 5.0
        n.mass = 1.0
        nodes.append(n)
        
        fa2util.apply_repulsion(nodes, coefficient=1.0)
        fa2util.apply_gravity(nodes, gravity=1.0, scalingRatio=1.0)
        
        # Gravity should pull toward origin
        assert n.dx < 0
        assert n.dy < 0
    
    def test_edge_weight_zero(self):
        """Test attraction with zero edge weight"""
        nodes = []
        for i in range(2):
            n = fa2util.Node()
            n.x = float(i * 10)
            n.y = 0.0
            n.mass = 1.0
            nodes.append(n)
        
        edges = []
        e = fa2util.Edge()
        e.node1 = 0
        e.node2 = 1
        e.weight = 0.0
        edges.append(e)
        
        fa2util.apply_attraction(nodes, edges, distributedAttraction=False,
                                coefficient=1.0, edgeWeightInfluence=1.0)
        
        # Zero weight edge should produce no force
        assert nodes[0].dx == 0.0
        assert nodes[1].dx == 0.0
    
    def test_edge_weight_influence_variations(self):
        """Test different edge weight influence values"""
        nodes = []
        for i in range(2):
            n = fa2util.Node()
            n.x = float(i * 10)
            n.y = 0.0
            n.mass = 1.0
            nodes.append(n)
        
        edges = []
        e = fa2util.Edge()
        e.node1 = 0
        e.node2 = 1
        e.weight = 2.0
        edges.append(e)
        
        # Test with edgeWeightInfluence = 0 (ignore weights)
        fa2util.apply_attraction(nodes, edges, distributedAttraction=False,
                                coefficient=1.0, edgeWeightInfluence=0.0)
        force_0 = nodes[0].dx
        
        # Reset
        for n in nodes:
            n.dx = 0.0
            n.dy = 0.0
        
        # Test with edgeWeightInfluence = 1 (use weights)
        fa2util.apply_attraction(nodes, edges, distributedAttraction=False,
                                coefficient=1.0, edgeWeightInfluence=1.0)
        force_1 = nodes[0].dx
        
        # With weight=2 and influence=1, force should be stronger
        assert abs(force_1) > abs(force_0)

