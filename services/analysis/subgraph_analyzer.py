#!/usr/bin/env python3

from typing import Dict, Set, List
import networkx as nx
import numpy as np
import pandas as pd


class SubgraphAnalyzer:
    """Handles subgraph analysis and insights"""

    def __init__(self):
        pass

    def get_subgraph_info(self, nodes: Set[int], feature_embeddings: Dict[str, np.ndarray], 
                         df: pd.DataFrame, graph: nx.Graph) -> Dict:
        """Get detailed information about a subgraph"""
        if not nodes:
            return {"size": 0, "density": 0.0, "members": []}

        # Get members info
        members = []
        for node in nodes:
            person_data = df.iloc[node]
            members.append({
                "name": person_data["Person Name"],
                "company": person_data["Person Company"],
                "role": person_data.get("Professional Identity - Role Specification", ""),
                "experience": person_data.get("Professional Identity - Experience Level", ""),
                "industry": person_data.get("Company Identity - Industry Classification", ""),
                "linkedin": person_data.get("Person Linkedin URL", "")
            })

        # Calculate subgraph density  
        subgraph = graph.subgraph(nodes)
        density = self.calculate_subgraph_density(nodes, graph)
        
        # Calculate average edge weight
        edges_with_weights = subgraph.edges(data=True)
        if edges_with_weights:
            total_weight = sum(data.get("weight", 0.0) for _, _, data in edges_with_weights)
            avg_weight = total_weight / len(edges_with_weights)
        else:
            avg_weight = 0.0

        result = {
            "nodes": nodes,
            "size": len(nodes),
            "density": density,
            "avg_edge_weight": avg_weight,
            "members": members,
            "edges": len(edges_with_weights)
        }

        # Add centroid analysis if we have embeddings
        if feature_embeddings:
            result["centroid_insights"] = self.analyze_subgraph_centroids(nodes, feature_embeddings)

        # Add subgroup analysis
        result.update(self.analyze_subgroups(nodes, graph, df))

        return result

    def calculate_subgraph_density(self, nodes: Set[int], graph: nx.Graph) -> float:
        """Calculate density of a subgraph given a set of nodes"""
        if len(nodes) < 2:
            return 0.0

        subgraph = graph.subgraph(nodes)
        num_edges = subgraph.number_of_edges()
        num_nodes = len(nodes)
        max_possible_edges = num_nodes * (num_nodes - 1) / 2

        return num_edges / max_possible_edges if max_possible_edges > 0 else 0.0

    def analyze_subgraph_centroids(self, nodes: Set[int], feature_embeddings: Dict[str, np.ndarray]) -> Dict:
        """Analyze centroids of the dense subgraph to find representative tags"""
        centroids = {}
        
        for feature_name, embeddings in feature_embeddings.items():
            # Get embeddings for subgraph nodes
            subgraph_embeddings = embeddings[list(nodes)]
            
            # Calculate centroid
            centroid = np.mean(subgraph_embeddings, axis=0)
            
            # Find closest tags to centroid
            # For now, just store the centroid (would need tag embeddings to populate closest_tags)
            centroids[feature_name] = {
                "centroid": centroid,
                "closest_tags": []  # Would need tag embeddings to populate
            }
        
        return centroids

    def analyze_subgroups(self, nodes: Set[int], graph: nx.Graph, df: pd.DataFrame) -> Dict:
        """Analyze cohesive subgroups within the dense subgraph"""
        if len(nodes) < 4:
            return {"subgroups": [], "subgroup_summary": {"total_subgroups": 0, "strongest_subgroup_strength": 0.0, "avg_subgroup_density": 0.0}}
        
        subgraph = graph.subgraph(nodes)
        
        # Use community detection to find subgroups
        try:
            import networkx.algorithms.community as nx_comm
            communities = list(nx_comm.greedy_modularity_communities(subgraph, weight='weight'))
        except:
            # Fallback: use connected components
            communities = list(nx.connected_components(subgraph))
        
        subgroups = []
        total_density = 0.0
        max_strength = 0.0
        
        for i, community in enumerate(communities):
            if len(community) >= 3:  # Only consider subgroups with 3+ people
                subgroup_density = self.calculate_subgraph_density(community, graph)
                total_density += subgroup_density
                
                # Calculate connection strength (average edge weight)
                community_subgraph = subgraph.subgraph(community)
                edges_with_weights = community_subgraph.edges(data=True)
                if edges_with_weights:
                    avg_weight = sum(data.get("weight", 0.0) for _, _, data in edges_with_weights) / len(edges_with_weights)
                    max_strength = max(max_strength, avg_weight)
                else:
                    avg_weight = 0.0
                
                # Get member info
                members = []
                for node in community:
                    person_data = df.iloc[node]
                    members.append({
                        "name": person_data["Person Name"],
                        "company": person_data["Person Company"],
                        "role": person_data.get("Professional Identity - Role Specification", "")
                    })
                
                subgroups.append({
                    "nodes": community,
                    "size": len(community),
                    "density": subgroup_density,
                    "connection_strength": avg_weight,
                    "members": members
                })
        
        # Sort by connection strength
        subgroups.sort(key=lambda x: x["connection_strength"], reverse=True)
        
        avg_density = total_density / len(subgroups) if subgroups else 0.0
        
        return {
            "subgroups": subgroups,
            "subgroup_summary": {
                "total_subgroups": len(subgroups),
                "strongest_subgroup_strength": max_strength,
                "avg_subgroup_density": avg_density
            }
        }