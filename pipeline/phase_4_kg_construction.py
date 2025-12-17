"""
Phase 4: Knowledge Graph Construction (CORE MODULE)
====================================================
This module integrates all outputs from previous phases to build the final
Onto-MedKG (Ontology-grounded Medical Knowledge Graph).

GRAPH STRUCTURE:
- Nodes: Entities, Events, and Grounded Concepts
- Edges: Relationships (E-E, E-Ev, Ev-Ev) from triple extraction
- Attributes: Ontology IDs, semantic types, induced concepts
"""

import networkx as nx
from typing import List, Dict
import re
import unicodedata

canonical_event_map = {}  

def normalize_text_for_matching(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r'^\[?(Event|Entity):\s*', '', s, flags=re.I)
    s = re.sub(r'\]$', '', s)

    s = unicodedata.normalize('NFKD', s)
    s = ''.join([c for c in s if not unicodedata.combining(c)])
    s = s.lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def get_canonical_event_name(raw_event_name: str) -> str:
    norm = normalize_text_for_matching(raw_event_name)
    if norm in canonical_event_map:
        return canonical_event_map[norm]

    clean = re.sub(r'^\[?(Event|Entity):\s*', '', raw_event_name.strip(), flags=re.I)
    clean = re.sub(r'\]$', '', clean)

    canonical = f"[Event: {clean}]"
    canonical_event_map[norm] = canonical
    return canonical




def build_knowledge_graph(all_triples: List[Dict], grounded_nodes: Dict[str, Dict]) -> nx.MultiDiGraph:
    """
    Build the final knowledge graph with event canonicalization (FIX4 + FIX5).
    """

    kg = nx.MultiDiGraph()
    print("  Building graph structure...")

    print(f"  Adding {len(grounded_nodes)} nodes with attributes...")

    for original_name, attributes in grounded_nodes.items():

        node_name = original_name

        if node_name.startswith("[Event:"):
            node_name = get_canonical_event_name(node_name)


        node_type = _determine_node_type(node_name, all_triples)

        kg.add_node(
            node_name,
            ontology_id=attributes.get("ontology_id", "UNKNOWN"),
            induced_concept=attributes.get("induced_concept", ""),
            ontology_name=attributes.get("ontology_name", node_name),
            semantic_type=attributes.get("semantic_type", "Medical Concept"),
            node_type=node_type
        )

    print(f"  Adding {len(all_triples)} relationships (edges)...")

    for triple in all_triples:

        head = triple["head"]
        tail = triple["tail"]
        relation = triple["relation"]

        if head.startswith("[Event:"):
            head = get_canonical_event_name(head)

        if tail.startswith("[Event:"):
            tail = get_canonical_event_name(tail)

        kg.add_edge(
            head,
            tail,
            relation=relation,
            triple_type=triple.get("type", ""),
            head_type=triple.get("head_type", "entity"),
            tail_type=triple.get("tail_type", "entity"),
            segment_id=triple.get("segment_id", 0),
            confidence=triple.get("confidence", 1.0)
        )

    print("  Graph construction complete!")
    return kg



def _determine_node_type(node_name: str, all_triples: List[Dict]) -> str:
    """
    Determine node type using deterministic cues first:
      1) If name has explicit [Event:] prefix -> 'event'
      2) If name has explicit [Entity:] prefix -> 'entity'
      3) Else fallback to counting context in triples (majority)
    """
    if isinstance(node_name, str):
        low = node_name.strip()
        if low.startswith("[Event:") or low.startswith("Event:"):
            return 'event'
        if low.startswith("[Entity:") or low.startswith("Entity:"):
            return 'entity'

    entity_count = 0
    event_count = 0
    for triple in all_triples:
        if triple.get('head') == node_name:
            if triple.get('head_type') == 'entity':
                entity_count += 1
            elif triple.get('head_type') == 'event':
                event_count += 1
        if triple.get('tail') == node_name:
            if triple.get('tail_type') == 'entity':
                entity_count += 1
            elif triple.get('tail_type') == 'event':
                event_count += 1

    return 'entity' if entity_count >= event_count else 'event'



def get_graph_statistics(kg: nx.MultiDiGraph) -> Dict:
    """
    Compute statistics about the knowledge graph.
    
    Args:
        kg (nx.MultiDiGraph): The knowledge graph
        
    Returns:
        Dict: Statistics about the graph structure
    """
    stats = {
        'total_nodes': kg.number_of_nodes(),
        'total_edges': kg.number_of_edges(),
        'density': nx.density(kg),
        'is_connected': nx.is_weakly_connected(kg),
    }
    
    entity_nodes = [n for n, d in kg.nodes(data=True) if d.get('node_type') == 'entity']
    event_nodes = [n for n, d in kg.nodes(data=True) if d.get('node_type') == 'event']
    
    stats['entity_nodes'] = len(entity_nodes)
    stats['event_nodes'] = len(event_nodes)
    
    edge_types = {}
    for u, v, data in kg.edges(data=True):
        triple_type = data.get('triple_type', 'unknown')
        edge_types[triple_type] = edge_types.get(triple_type, 0) + 1
    
    stats['edge_types'] = edge_types
    
    if kg.number_of_nodes() > 0:
        degrees = dict(kg.degree())
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
        stats['top_connected_nodes'] = top_nodes
    
    return stats


def export_graph_to_formats(kg: nx.MultiDiGraph, output_dir: str) -> Dict[str, str]:
    """
    Export the knowledge graph to various formats for analysis.
    
    Args:
        kg (nx.MultiDiGraph): The knowledge graph to export
        output_dir (str): Directory to save the exported files
        
    Returns:
        Dict[str, str]: Mapping of format names to file paths
    """
    import json
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    exported_files = {}
    
    graphml_path = os.path.join(output_dir, "knowledge_graph.graphml")
    nx.write_graphml(kg, graphml_path)
    exported_files['graphml'] = graphml_path
    
    json_path = os.path.join(output_dir, "knowledge_graph.json")
    graph_data = nx.node_link_data(kg)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=2)
    exported_files['json'] = json_path
    
    edgelist_path = os.path.join(output_dir, "knowledge_graph_edges.txt")
    nx.write_edgelist(kg, edgelist_path, data=True)
    exported_files['edgelist'] = edgelist_path
    
    return exported_files


def export_graph_to_neo4j_csv(kg: nx.MultiDiGraph, output_dir: str) -> Dict[str, str]:
    """
    Export nodes and relationships to Neo4j bulk import CSVs.

    Nodes CSV columns:
      :ID,name,labels,ontology_id,ontology_name,semantic_type,induced_concept,original_node,uri

    Relationships CSV columns:
      :START_ID,:END_ID,:TYPE,relation,confidence,segment_id,doc_id

    Args:
        kg (nx.MultiDiGraph): Knowledge graph
        output_dir (str): Directory to write CSV files

    Returns:
        Dict[str, str]: Paths to the written CSV files
    """
    import os
    import csv

    os.makedirs(output_dir, exist_ok=True)

    nodes_path = os.path.join(output_dir, "neo4j_nodes.csv")
    rels_path = os.path.join(output_dir, "neo4j_relationships.csv")

    with open(nodes_path, 'w', newline='', encoding='utf-8-sig') as nf:
        writer = csv.writer(nf)
        writer.writerow([':ID', 'name', 'labels', 'ontology_id', 'ontology_name', 'semantic_type', 'induced_concept', 'original_node', 'uri'])
        for node_id, data in kg.nodes(data=True):
            name = str(node_id)
            labels = 'Entity' if data.get('node_type') == 'entity' else 'Event'
            ontology_id = data.get('ontology_id', '')
            ontology_name = data.get('ontology_name', '')
            semantic_type = data.get('semantic_type', '')
            induced_concept = data.get('induced_concept', '')
            original_node = name
            uri = data.get('uri', '')
            writer.writerow([name, name, labels, ontology_id, ontology_name, semantic_type, induced_concept, original_node, uri])

    with open(rels_path, 'w', newline='', encoding='utf-8-sig') as rf:
        writer = csv.writer(rf)
        writer.writerow([':START_ID', ':END_ID', ':TYPE', 'relation', 'confidence', 'segment_id', 'doc_id'])
        for u, v, key, data in kg.edges(keys=True, data=True):
            start_id = str(u)
            end_id = str(v)
            rel_type = (data.get('relation') or data.get('triple_type') or 'RELATED').upper()
            rel_type = rel_type.replace(' ', '_')
            relation = data.get('relation', '')
            confidence = data.get('confidence', '')
            segment_id = data.get('segment_id', '')
            doc_id = data.get('doc_id', '')
            writer.writerow([start_id, end_id, rel_type, relation, confidence, segment_id, doc_id])

    return {'neo4j_nodes': nodes_path, 'neo4j_rels': rels_path}
