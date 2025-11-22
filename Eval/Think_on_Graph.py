"""
Think on Graph (ToG) Retrieval for AutoSchemaKG
Adapted from the original ToG implementation to work with Neo4j knowledge graphs.
"""

import numpy as np
from typing import Optional, List, Dict, Tuple
import networkx as nx
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import json
import os
from pathlib import Path


class InferenceConfig:
    """Configuration for ToG inference."""
    def __init__(self, Dmax: int = 3):
        self.Dmax = Dmax  # Maximum depth for path exploration


class LLMGenerator:
    """Wrapper for LLM API calls."""
    def __init__(self, use_real_llm: bool = False):
        self.use_real_llm = use_real_llm
        if use_real_llm:
            try:
                from openai import OpenAI
                self.client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
                print("✓ Connected to LM Studio")
            except Exception as e:
                print(f"⚠ Warning: Could not connect to LM Studio: {e}")
                print("  Falling back to stub mode")
                self.use_real_llm = False
        
    def generate_response(self, messages: List[Dict]) -> str:
        """Generate response from LLM."""
        if not self.use_real_llm:
            # Stub response for testing
            last_message = messages[-1]["content"].lower()
            if "named entities" in last_message or "extract" in last_message:
                return '{"entities": []}'
            elif "sufficient" in last_message or "yes or no" in last_message:
                return "Yes"
            return "Response"
        
        # Real LLM implementation with LM Studio
        try:
            response = self.client.chat.completions.create(
                model="local-model",
                messages=messages,
                temperature=0.7,
                max_tokens=1000,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"⚠ LLM generation error: {e}")
            return "Error: Could not generate response from LLM"


class EmbeddingModel:
    """Wrapper for sentence embeddings."""
    def __init__(self, model_name: str = "BAAI/bge-m3", use_docker: bool = False):
        self.use_docker = use_docker
        if use_docker:
            self.api_url = "http://localhost:8080/embed"
        else:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        
    def encode(self, texts: List[str], query_type: str = None) -> np.ndarray:
        """Encode texts to embeddings."""
        if self.use_docker:
            import requests
            response = requests.post(
                self.api_url,
                json={"inputs": texts}
            )
            response.raise_for_status()  # Check for HTTP errors
            embeddings = np.array(response.json(), dtype=np.float32)
            
            # Ensure proper shape
            if len(embeddings.shape) == 1:
                embeddings = embeddings.reshape(1, -1)
            
            return embeddings
        else:
            return self.model.encode(texts, convert_to_numpy=True)


class TogV3Retriever:
    """Think on Graph v3 Retriever for Neo4j knowledge graphs."""
    
    def __init__(self, KG: nx.DiGraph, llm_generator: LLMGenerator, 
                 sentence_encoder: EmbeddingModel, 
                 inference_config: Optional[InferenceConfig] = None,
                 use_qdrant: bool = True,
                 qdrant_url: str = "http://localhost:6333"):
        """
        Initialize ToG retriever.
        
        Args:
            KG: NetworkX DiGraph representation of the knowledge graph
            llm_generator: LLM generator for NER and reasoning
            sentence_encoder: Embedding model for similarity computation
            inference_config: Configuration for inference
            use_qdrant: Whether to use Qdrant for embedding storage
            qdrant_url: Qdrant server URL
        """
        self.KG: nx.DiGraph = KG
        self.node_list = list(self.KG.nodes())
        self.edge_list = list(self.KG.edges)
        
        self.llm_generator = llm_generator
        self.sentence_encoder = sentence_encoder
        self.inference_config = inference_config if inference_config else InferenceConfig()
        
        self.use_qdrant = use_qdrant
        self.collection_name = "kg_nodes"
        
        # Setup Qdrant or precompute embeddings
        if use_qdrant:
            print("Setting up Qdrant for embedding storage...")
            self._setup_qdrant(qdrant_url)
        else:
            print("Computing node embeddings...")
            self.node_embeddings = self._compute_node_embeddings()
            print(f"Node embeddings shape: {self.node_embeddings.shape}")

    def _setup_qdrant(self, url: str):
        """Setup Qdrant collection for embeddings."""
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct
        
        self.qdrant_client = QdrantClient(url=url)
        
        # Check if collection exists
        collections = self.qdrant_client.get_collections().collections
        collection_exists = any(c.name == self.collection_name for c in collections)
        
        if collection_exists:
            print(f"✓ Using existing Qdrant collection '{self.collection_name}'")
            return
        
        # Create collection and index embeddings
        print(f"Creating new Qdrant collection '{self.collection_name}'...")
        
        # Get embedding dimension from a sample
        sample_text = self.KG.nodes[self.node_list[0]].get('name', str(self.node_list[0]))
        sample_embedding = self.sentence_encoder.encode([sample_text])[0]
        embedding_dim = len(sample_embedding)
        
        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
        )
        
        # Index all nodes in batches
        print(f"Indexing {len(self.node_list)} nodes...")
        batch_size = 100
        points = []
        
        for idx, node in enumerate(self.node_list):
            node_data = self.KG.nodes[node]
            text = node_data.get('name', node_data.get('id', str(node)))
            
            # Compute embedding
            if len(points) == 0 or len(points) % batch_size != 0:
                embedding = self.sentence_encoder.encode([text])[0]
            
            point = PointStruct(
                id=idx,
                vector=embedding.tolist(),
                payload={"node_id": str(node), "text": text}
            )
            points.append(point)
            
            # Upload batch
            if len(points) >= batch_size:
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                print(f"  Indexed {idx + 1}/{len(self.node_list)} nodes...")
                points = []
        
        # Upload remaining
        if points:
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
        
        print(f"✓ Indexed all {len(self.node_list)} nodes in Qdrant")
    
    def _compute_node_embeddings(self) -> np.ndarray:
        """Compute embeddings for all nodes (fallback when not using Qdrant)."""
        node_texts = []
        for node in self.node_list:
            node_data = self.KG.nodes[node]
            text = node_data.get('name', node_data.get('id', str(node)))
            node_texts.append(text)
        
        embeddings = self.sentence_encoder.encode(node_texts)
        return embeddings

    def ner(self, text: str) -> Dict:
        """Extract topic entities from the query using LLM."""
        messages = [
            {
                "role": "system",
                "content": "Extract the named entities from the provided question and output them as a JSON object in the format: {\"entities\": [\"entity1\", \"entity2\", ...]}"
            },
            {
                "role": "user",
                "content": f"Extract all the named entities from: {text}"
            }
        ]
        response = self.llm_generator.generate_response(messages)
        try:
            import json
            entities_json = json.loads(response)
        except Exception as e:
            print(f"NER parsing error: {e}")
            return {"entities": []}
        
        if "entities" not in entities_json or not isinstance(entities_json["entities"], list):
            return {"entities": []}
        return entities_json

    def retrieve_topk_nodes(self, query: str, topN: int = 5) -> List:
        """Retrieve top-k most relevant nodes for the query."""
        entities = self.ner(query)
        entities = entities.get("entities", [])

        if len(entities) == 0:
            entities = [query]

        topk_nodes = []
        
        # First, add exact matches
        for entity in entities:
            if entity in self.node_list:
                topk_nodes.append(entity)
        
        # Then add similar nodes
        if self.use_qdrant:
            # Use Qdrant for fast similarity search
            topk_for_each_entity = max(1, topN // len(entities))
            
            for entity in entities:
                topk_for_this_entity = topk_for_each_entity + 1
                entity_embedding = self.sentence_encoder.encode([entity])[0]
                
                results = self.qdrant_client.query_points(
                    collection_name=self.collection_name,
                    query=entity_embedding.tolist(),
                    limit=topk_for_this_entity
                ).points
                
                for result in results:
                    node_id = result.payload["node_id"]
                    topk_nodes.append(node_id)
        else:
            # Fallback to numpy computation
            topk_for_each_entity = max(1, topN // len(entities))
            
            for entity in entities:
                topk_for_this_entity = topk_for_each_entity + 1
                entity_embedding = self.sentence_encoder.encode([entity])
                scores = self.node_embeddings @ entity_embedding[0].T
                top_indices = np.argsort(scores)[-topk_for_this_entity:][::-1]
                topk_nodes.extend([self.node_list[i] for i in top_indices])

        topk_nodes = list(dict.fromkeys(topk_nodes))  # Remove duplicates

        if len(topk_nodes) > 2 * topN:
            topk_nodes = topk_nodes[:2 * topN]
        
        print(f"Retrieved {len(topk_nodes)} initial nodes: {topk_nodes[:5]}...")
        return topk_nodes

    def retrieve(self, query: str, topN: int = 5) -> Tuple[str, List[str]]:
        """
        Retrieve the top N paths that connect the entities in the query.
        
        Args:
            query: The query string
            topN: Number of top results to return
            
        Returns:
            Tuple of (answer, sources_list)
        """
        Dmax = self.inference_config.Dmax
        
        # Step 1: Retrieve top-k initial nodes
        initial_nodes = self.retrieve_topk_nodes(query, topN=topN)
        P = [[e] for e in initial_nodes]
        D = 0

        print(f"Starting ToG search with Dmax={Dmax}")
        
        while D <= Dmax:
            print(f"Depth {D}: {len(P)} paths")
            P = self.search(query, P)
            print(f"After search: {len(P)} paths")
            P = self.prune(query, P, topN)
            print(f"After prune: {len(P)} paths")
            
            if self.reasoning(query, P):
                print(f"Reasoning satisfied at depth {D}")
                generated_text = self.generate(query, P, use_llm=True)
                break
            
            D += 1

        if D > Dmax:
            print(f"Max depth reached")
            generated_text = self.generate(query, P, use_llm=True)

        return generated_text

    def search(self, query: str, P: List[List]) -> List[List]:
        """Expand paths by one hop."""
        new_paths = []
        for path in P:
            tail_entity = path[-1]
            
            try:
                successors = list(self.KG.successors(tail_entity))
            except:
                successors = []
            
            # Remove entities already in path
            successors = [n for n in successors if n not in path]

            if len(successors) == 0:
                new_paths.append(path)
                continue

            for neighbour in successors:
                edge_data = self.KG.edges.get((tail_entity, neighbour), {})
                relation = edge_data.get("relation", "RELATED_TO")
                new_path = path + [relation, neighbour]
                new_paths.append(new_path)

        return new_paths

    def prune(self, query: str, P: List[List], topN: int = 3) -> List[List]:
        """Prune paths to top-N most relevant."""
        if len(P) <= topN:
            return P
        
        path_strings = []
        for path in P:
            formatted_nodes = []
            for i, node_or_relation in enumerate(path):
                if i % 2 == 0:  # Node
                    node_data = self.KG.nodes.get(node_or_relation, {})
                    node_text = node_data.get("name", node_data.get("id", str(node_or_relation)))
                    formatted_nodes.append(node_text)
                else:  # Relation
                    formatted_nodes.append(node_or_relation)
            
            path_string = " ".join(formatted_nodes)
            path_strings.append(path_string)

        # Encode and compute similarity
        query_embedding = self.sentence_encoder.encode([query])[0]
        path_embeddings = self.sentence_encoder.encode(path_strings)

        # Normalize
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        path_embeddings = path_embeddings / np.linalg.norm(path_embeddings, axis=1, keepdims=True)

        # Compute scores
        scores = path_embeddings @ query_embedding

        # Sort and return top-N
        sorted_indices = np.argsort(scores)[::-1]
        sorted_paths = [P[i] for i in sorted_indices[:topN]]

        return sorted_paths

    def reasoning(self, query: str, P: List[List]) -> bool:
        """Check if retrieved knowledge is sufficient to answer the query."""
        triples = []
        for path in P:
            for i in range(0, len(path) - 2, 2):
                node1_data = self.KG.nodes.get(path[i], {})
                node2_data = self.KG.nodes.get(path[i + 2], {})
                
                node1_text = node1_data.get("name", node1_data.get("id", str(path[i])))
                node2_text = node2_data.get("name", node2_data.get("id", str(path[i + 2])))
                
                triples.append((node1_text, path[i + 1], node2_text))

        triples_string = [f"({t[0]}, {t[1]}, {t[2]})" for t in triples]
        triples_string = ". ".join(triples_string)

        prompt = f"Given a question and the associated retrieved knowledge graph triples (entity, relation, entity), you are asked to answer whether it's sufficient for you to answer the question with these triples and your knowledge (Yes or No). Query: {query} \n Knowledge triples: {triples_string}"

        messages = [
            {"role": "system", "content": "Answer the question following the prompt."},
            {"role": "user", "content": prompt}
        ]

        response = self.llm_generator.generate_response(messages)
        return "yes" in response.lower()

    def generate(self, query: str, P: List[List], use_llm: bool = True) -> Tuple[str, List[str]]:
        """Generate final answer from retrieved paths."""
        triples = []
        for path in P:
            for i in range(0, len(path) - 2, 2):
                node1_data = self.KG.nodes.get(path[i], {})
                node2_data = self.KG.nodes.get(path[i + 2], {})
                
                node1_text = node1_data.get("name", node1_data.get("id", str(path[i])))
                node2_text = node2_data.get("name", node2_data.get("id", str(path[i + 2])))
                
                triples.append(f"({node1_text}, {path[i + 1]}, {node2_text})")

        if not use_llm or len(triples) == 0:
            # Return triples as-is without LLM generation
            sources = ["N/A" for _ in range(len(triples))]
            return "\n".join(triples), sources
        
        # Use LLM to generate natural language answer from triples
        triples_context = "\n".join([f"{i+1}. {triple}" for i, triple in enumerate(triples)])
        
        prompt = f"""Based on the provided Knowledge Graph Triples, compose a comprehensive and detailed narrative answer to the question.

Guidelines:
1. **Format:** Write as a continuous, cohesive article (prose only). Do not use bullet points or lists.
2. **Tone:** Use an objective, encyclopedic, and educational tone.
3. **Structure:** Smoothly integrate definitions, symptoms, classifications, and complications. Ensure logical transitions between sentences.
4. **Detail:** Elaborate on the relationships found in the triples to provide a full explanation.

Knowledge Triples:
{triples_context}

Question: {query}

Detailed Answer:"""

        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on knowledge graph information."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            answer = self.llm_generator.generate_response(messages)
        except Exception as e:
            print(f"LLM generation error: {e}")
            answer = "\n".join(triples)
        
        sources = triples
        return answer, sources


def load_kg_from_neo4j(uri: str, user: str, password: str) -> nx.DiGraph:
    """Load knowledge graph from Neo4j into NetworkX."""
    print("Loading KG from Neo4j...")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    G = nx.DiGraph()
    
    with driver.session() as session:
        # Load nodes
        result = session.run("MATCH (n:Entity) RETURN n")
        for record in result:
            node = record["n"]
            node_id = node.get("id", node.element_id)
            G.add_node(node_id, **dict(node))
        
        print(f"Loaded {len(G.nodes())} nodes")
        
        # Load relationships
        result = session.run("MATCH (a:Entity)-[r]->(b:Entity) RETURN a, r, b")
        for record in result:
            source = record["a"].get("id", record["a"].element_id)
            target = record["b"].get("id", record["b"].element_id)
            rel = record["r"]
            
            # Get relation name
            relation = rel.get("relation", rel.type)
            
            # Get all edge attributes except 'relation' to avoid conflict
            edge_attrs = dict(rel)
            edge_attrs['relation'] = relation  # Set the relation attribute
            
            G.add_edge(source, target, **edge_attrs)
        
        print(f"Loaded {len(G.edges())} edges")
    
    driver.close()
    return G


def process_questions_from_csv(retriever: TogV3Retriever, input_csv_path: str, output_csv_path: str):
    """Process questions from CSV and save answers."""
    import csv
    from tqdm import tqdm
    
    print(f"\nProcessing questions from: {input_csv_path}")
    
    # Read questions
    questions = []
    with open(input_csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = row.get('question', '').strip()
            if question:  # Skip empty questions
                questions.append(question)
    
    print(f"Found {len(questions)} questions to process")
    
    # Process each question
    results = []
    for i, question in enumerate(tqdm(questions, desc="Answering questions")):
        
        print(f"\n{'='*80}")
        print(f"[Question {i+1}/{len(questions)}]")
        print(f"Q: {question}")
        print(f"{'='*80}")
        
        try:
            # Get answer from ToG
            answer, sources = retriever.retrieve(question, topN=5)
            print(f"ToG Answer: {answer}")
            print(f"Sources: {len(sources)} triples")
            
            results.append({
                'question': question,
                'tog_answer': answer,
                'num_sources': len(sources),
                'sources': ' | '.join(sources[:5])  # Store first 5 sources
            })
            
        except Exception as e:
            print(f"\nError processing question {i+1}: {e}")
            results.append({
                'question': question,
                'tog_answer': f"ERROR: {str(e)}",
                'num_sources': 0,
                'sources': ''
            })
    
    # Save results
    with open(output_csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        fieldnames = ['question', 'answer']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({
                'question': result['question'],
                'answer': result['tog_answer']
            })
    
    print(f"\n✓ Results saved to: {output_csv_path}")
    print(f"  Total questions: {len(questions)}")
    print(f"  Successfully answered: {sum(1 for r in results if not r['tog_answer'].startswith('ERROR'))}")


def main():
    """Example usage of ToG retrieval."""
    # Configuration
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")
    
    # Load KG from Neo4j
    kg = load_kg_from_neo4j(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    # Initialize components
    print("Initializing LLM and embedding models...")
    use_real_llm = os.getenv("USE_REAL_LLM", "true").lower() == "true"
    print(f"Using real LLM (LM Studio): {use_real_llm}")
    llm_generator = LLMGenerator(use_real_llm=use_real_llm)
    
    # Use local BGE-M3 model (more reliable for large batches)
    print("Loading BGE-M3 model locally...")
    embedding_model = EmbeddingModel(use_docker=False)
    
    # Create retriever
    print("Creating ToG retriever...")
    use_qdrant = os.getenv("USE_QDRANT", "true").lower() == "true"
    print(f"Using Qdrant for embeddings: {use_qdrant}")
    
    retriever = TogV3Retriever(
        KG=kg,
        llm_generator=llm_generator,
        sentence_encoder=embedding_model,
        inference_config=InferenceConfig(Dmax=2),
        use_qdrant=use_qdrant,
        qdrant_url="http://localhost:6333"
    )
    
    # Get base directory and setup paths
    base_dir = Path(__file__).parent.parent
    input_csv = base_dir / "Eval" / "data" / "1000.csv"
    output_csv = base_dir / "Eval" / "data" / "ToG_answer.csv"
    
    # Process questions from CSV
    if input_csv.exists():
        process_questions_from_csv(retriever, str(input_csv), str(output_csv))
    else:
        print(f"\n⚠ Input file not found: {input_csv}")
        print("Running example queries instead...\n")
        
        # Example queries
        queries = [
            "What is (are) keratoderma with woolly hair ?",
            "How many people are affected by keratoderma with woolly hair ?",
            "What are the genetic changes related to keratoderma with woolly hair ?"
        ]
        
        for query in queries:
            print(f"\n{'='*80}")
            print(f"Query: {query}")
            print(f"{'='*80}")
            
            answer, sources = retriever.retrieve(query, topN=5)
            
            print(f"\nAnswer:\n{answer}")
            print(f"\nSources ({len(sources)} triples):")
            for i, source in enumerate(sources[:10]):  # Show first 10
                print(f"{i+1}. {source}")


if __name__ == "__main__":
    main()
