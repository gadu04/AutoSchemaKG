"""
Neo4j CSV Import Script
Imports nodes and relationships from CSV files into Neo4j database.
"""

import os
from pathlib import Path
from neo4j import GraphDatabase
import csv
from typing import List, Dict


class Neo4jImporter:
    def __init__(self, uri: str, user: str, password: str):
        """Initialize Neo4j connection."""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        """Close the Neo4j connection."""
        self.driver.close()
    
    def clear_database(self):
        """Clear all nodes and relationships from the database."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("Database cleared successfully.")
    
    def create_constraints(self):
        """Create constraints and indexes for better performance."""
        with self.driver.session() as session:
            # Create constraint on Entity ID
            try:
                session.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
                print("Created constraint on Entity.id")
            except Exception as e:
                print(f"Constraint may already exist: {e}")
    
    def import_nodes(self, csv_file_path: str):
        """Import nodes from CSV file."""
        if not os.path.exists(csv_file_path):
            print(f"Error: File not found - {csv_file_path}")
            return
        
        print(f"Importing nodes from {csv_file_path}...")
        
        with open(csv_file_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
            reader = csv.DictReader(f)
            # Clean column names by stripping whitespace and BOM
            reader.fieldnames = [name.strip() for name in reader.fieldnames]
            nodes = list(reader)
        
        print(f"Read {len(nodes)} rows from CSV")
        if nodes:
            print(f"First row keys: {list(nodes[0].keys())}")
            print(f"First row sample: ID='{nodes[0].get(':ID', '')}', name='{nodes[0].get('name', '')}'")
        
        with self.driver.session() as session:
            count = 0
            skipped = 0
            for node in nodes:
                # Clean the data
                node_id = node.get(':ID', '').strip()
                name = node.get('name', '').strip()
                labels = node.get('labels', 'Entity').strip()
                ontology_id = node.get('ontology_id', '').strip()
                ontology_name = node.get('ontology_name', '').strip()
                semantic_type = node.get('semantic_type', '').strip()
                induced_concept = node.get('induced_concept', '').strip()
                original_node = node.get('original_node', '').strip()
                uri = node.get('uri', '').strip()
                
                if not node_id or not name:
                    skipped += 1
                    if skipped <= 3:
                        print(f"Skipping node - ID: '{node_id}', Name: '{name}'")
                    continue
                
                # Create node with properties
                query = f"""
                MERGE (n:{labels} {{id: $id}})
                SET n.name = $name,
                    n.ontology_id = $ontology_id,
                    n.ontology_name = $ontology_name,
                    n.semantic_type = $semantic_type,
                    n.induced_concept = $induced_concept,
                    n.original_node = $original_node,
                    n.uri = $uri
                """
                
                session.run(query, {
                    'id': node_id,
                    'name': name,
                    'ontology_id': ontology_id,
                    'ontology_name': ontology_name,
                    'semantic_type': semantic_type,
                    'induced_concept': induced_concept,
                    'original_node': original_node,
                    'uri': uri
                })
                
                count += 1
                if count % 100 == 0:
                    print(f"Imported {count} nodes...")
        
        print(f"Successfully imported {count} nodes. Skipped {skipped} nodes.")
    
    def import_relationships(self, csv_file_path: str):
        """Import relationships from CSV file."""
        if not os.path.exists(csv_file_path):
            print(f"Error: File not found - {csv_file_path}")
            return
        
        print(f"Importing relationships from {csv_file_path}...")
        
        with open(csv_file_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
            reader = csv.DictReader(f)
            # Clean column names by stripping whitespace and BOM
            reader.fieldnames = [name.strip() for name in reader.fieldnames]
            relationships = list(reader)
        
        print(f"Read {len(relationships)} rows from CSV")
        if relationships:
            print(f"First row keys: {list(relationships[0].keys())}")
            print(f"First row sample: Start='{relationships[0].get(':START_ID', '')}', End='{relationships[0].get(':END_ID', '')}'")
        
        with self.driver.session() as session:
            count = 0
            skipped = 0
            for rel in relationships:
                start_id = rel.get(':START_ID', '').strip()
                end_id = rel.get(':END_ID', '').strip()
                rel_type = rel.get(':TYPE', '').strip()
                relation = rel.get('relation', '').strip()
                confidence = rel.get('confidence', '1').strip()
                segment_id = rel.get('segment_id', '').strip()
                doc_id = rel.get('doc_id', '').strip()
                
                if not start_id or not end_id or not rel_type:
                    skipped += 1
                    if skipped <= 3:
                        print(f"Skipping rel - Start: '{start_id}', End: '{end_id}', Type: '{rel_type}'")
                    continue
                
                # Replace spaces and special characters in relationship type
                # Remove/replace characters that are invalid in Neo4j relationship types
                rel_type_clean = rel_type.upper()
                # Replace invalid characters with underscore
                invalid_chars = [' ', '-', ',', '.', '/', '\\', '(', ')', '[', ']', '{', '}', ':', ';', '"', "'"]
                for char in invalid_chars:
                    rel_type_clean = rel_type_clean.replace(char, '_')
                # Remove multiple consecutive underscores
                while '__' in rel_type_clean:
                    rel_type_clean = rel_type_clean.replace('__', '_')
                # Remove leading/trailing underscores
                rel_type_clean = rel_type_clean.strip('_')
                
                # If relationship type is too long or empty, use a default
                if not rel_type_clean or len(rel_type_clean) > 200:
                    rel_type_clean = 'RELATED_TO'
                
                # Create relationship
                query = f"""
                MATCH (start:Entity {{id: $start_id}})
                MATCH (end:Entity {{id: $end_id}})
                MERGE (start)-[r:{rel_type_clean}]->(end)
                SET r.relation = $relation,
                    r.confidence = toFloat($confidence),
                    r.segment_id = $segment_id,
                    r.doc_id = $doc_id
                """
                
                try:
                    session.run(query, {
                        'start_id': start_id,
                        'end_id': end_id,
                        'relation': relation,
                        'confidence': confidence,
                        'segment_id': segment_id,
                        'doc_id': doc_id
                    })
                    count += 1
                    if count % 100 == 0:
                        print(f"Imported {count} relationships...")
                except Exception as e:
                    print(f"Error importing relationship {start_id} -> {end_id}: {e}")
                    skipped += 1
        
        print(f"Successfully imported {count} relationships. Skipped {skipped} relationships.")
    
    def get_statistics(self):
        """Get database statistics."""
        with self.driver.session() as session:
            # Count nodes
            node_count = session.run("MATCH (n) RETURN count(n) as count").single()['count']
            
            # Count relationships
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()['count']
            
            # Get relationship types
            rel_types = session.run("CALL db.relationshipTypes()").data()
            
            print(f"\n=== Database Statistics ===")
            print(f"Total Nodes: {node_count}")
            print(f"Total Relationships: {rel_count}")
            print(f"Relationship Types: {len(rel_types)}")
            print(f"Types: {', '.join([rt['relationshipType'] for rt in rel_types[:10]])}")


def main():
    # Get base directory
    base_dir = Path(__file__).parent.parent.parent
    
    # CSV file paths - using Eval/import/data directory
    data_dir = base_dir / "Eval" / "import" / "data"
    nodes_csv = data_dir / "neo4j_nodes.csv"
    relationships_csv = data_dir / "neo4j_relationships.csv"
    
    # Neo4j connection details
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")
    
    print(f"Base directory: {base_dir}")
    print(f"Connecting to Neo4j at {NEO4J_URI}...")
    
    # Create importer instance
    importer = Neo4jImporter(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        # Optional: Clear existing data
        clear_db = input("Clear existing database? (y/N): ").lower()
        if clear_db == 'y':
            importer.clear_database()
        
        # Create constraints
        importer.create_constraints()
        
        # Import nodes
        if os.path.exists(nodes_csv):
            importer.import_nodes(str(nodes_csv))
        else:
            print(f"Warning: Nodes file not found at {nodes_csv}")
        
        # Import relationships
        if os.path.exists(relationships_csv):
            importer.import_relationships(str(relationships_csv))
        else:
            print(f"Warning: Relationships file not found at {relationships_csv}")
        
        # Show statistics
        importer.get_statistics()
        
        print("\n✓ Import completed successfully!")
        
    except Exception as e:
        print(f"Error during import: {e}")
    finally:
        importer.close()


if __name__ == "__main__":
    main()
