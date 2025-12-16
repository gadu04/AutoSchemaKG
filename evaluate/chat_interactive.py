"""
Think on Graph (ToG) Chat Interface
Simple command-line interface for asking questions to the knowledge graph.
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
        self.Dmax = Dmax


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
            last_message = messages[-1]["content"].lower()
            if "named entities" in last_message or "extract" in last_message:
                return '{"entities": []}'
            elif "sufficient" in last_message or "yes or no" in last_message:
                return "Yes"
            return "Response"
        
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
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        
    def encode(self, texts: List[str], query_type: str = None) -> np.ndarray:
        """Encode texts to embeddings."""
        return self.model.encode(texts, convert_to_numpy=True)


class TogV3Retriever:
    """Think on Graph v3 Retriever."""
    
    def __init__(self, KG: nx.DiGraph, llm_generator: LLMGenerator, 
                 sentence_encoder: EmbeddingModel, 
                 inference_config: Optional[InferenceConfig] = None,
                 use_qdrant: bool = True,
                 qdrant_url: str = "http://localhost:6333"):
        self.KG = KG
        self.node_list = list(self.KG.nodes())
        self.edge_list = list(self.KG.edges)
        self.llm_generator = llm_generator
        self.sentence_encoder = sentence_encoder
        self.inference_config = inference_config if inference_config else InferenceConfig()
        self.use_qdrant = use_qdrant
        self.collection_name = "kg_nodes"
        
        if use_qdrant:
            print("Setting up Qdrant...")
            self._setup_qdrant(qdrant_url)
        else:
            print("Computing node embeddings...")
            self.node_embeddings = self._compute_node_embeddings()

    def _setup_qdrant(self, url: str):
        """Setup Qdrant collection."""
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct
        
        self.qdrant_client = QdrantClient(url=url)
        collections = self.qdrant_client.get_collections().collections
        collection_exists = any(c.name == self.collection_name for c in collections)
        
        if collection_exists:
            print(f"✓ Using existing Qdrant collection '{self.collection_name}'")
            return
        
        print(f"Creating Qdrant collection '{self.collection_name}'...")
        sample_text = self.KG.nodes[self.node_list[0]].get('name', str(self.node_list[0]))
        sample_embedding = self.sentence_encoder.encode([sample_text])[0]
        embedding_dim = len(sample_embedding)
        
        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
        )
        
        # Index nodes in batches
        batch_size = 100
        points = []
        for idx, node in enumerate(self.node_list):
            node_data = self.KG.nodes[node]
            text = node_data.get('name', node_data.get('id', str(node)))
            embedding = self.sentence_encoder.encode([text])[0]
            
            point = PointStruct(
                id=idx,
                vector=embedding.tolist(),
                payload={"node_id": str(node), "text": text}
            )
            points.append(point)
            
            if len(points) >= batch_size:
                self.qdrant_client.upsert(collection_name=self.collection_name, points=points)
                print(f"  Indexed {idx + 1}/{len(self.node_list)} nodes...")
                points = []
        
        if points:
            self.qdrant_client.upsert(collection_name=self.collection_name, points=points)
        
        print(f"✓ Indexed all nodes")
    
    def _compute_node_embeddings(self) -> np.ndarray:
        """Compute embeddings for all nodes."""
        node_texts = []
        for node in self.node_list:
            node_data = self.KG.nodes[node]
            text = node_data.get('name', node_data.get('id', str(node)))
            node_texts.append(text)
        return self.sentence_encoder.encode(node_texts)

    def ner(self, text: str) -> Dict:
        """Extract entities from query."""
        messages = [
            {"role": "system", "content": "Extract the named entities from the provided question and output them as a JSON object in the format: {\"entities\": [\"entity1\", \"entity2\", ...]}"},
            {"role": "user", "content": f"Extract all the named entities from: {text}"}
        ]
        response = self.llm_generator.generate_response(messages)
        try:
            entities_json = json.loads(response)
        except:
            return {"entities": []}
        if "entities" not in entities_json:
            return {"entities": []}
        return entities_json

    def retrieve_topk_nodes(self, query: str, topN: int = 5) -> List:
        """Retrieve top-k relevant nodes."""
        entities = self.ner(query).get("entities", [])
        if len(entities) == 0:
            entities = [query]

        topk_nodes = []
        for entity in entities:
            if entity in self.node_list:
                topk_nodes.append(entity)
        
        if self.use_qdrant:
            topk_for_each_entity = max(1, topN // len(entities))
            for entity in entities:
                entity_embedding = self.sentence_encoder.encode([entity])[0]
                results = self.qdrant_client.query_points(
                    collection_name=self.collection_name,
                    query=entity_embedding.tolist(),
                    limit=topk_for_each_entity + 1
                ).points
                for result in results:
                    topk_nodes.append(result.payload["node_id"])
        else:
            topk_for_each_entity = max(1, topN // len(entities))
            for entity in entities:
                entity_embedding = self.sentence_encoder.encode([entity])
                scores = self.node_embeddings @ entity_embedding[0].T
                top_indices = np.argsort(scores)[-topk_for_each_entity-1:][::-1]
                topk_nodes.extend([self.node_list[i] for i in top_indices])

        topk_nodes = list(dict.fromkeys(topk_nodes))
        if len(topk_nodes) > 2 * topN:
            topk_nodes = topk_nodes[:2 * topN]
        
        return topk_nodes

    def retrieve(self, query: str, topN: int = 5) -> Tuple[str, List[str]]:
        """Retrieve answer for query."""
        Dmax = self.inference_config.Dmax
        initial_nodes = self.retrieve_topk_nodes(query, topN=topN)
        P = [[e] for e in initial_nodes]
        D = 0

        while D <= Dmax:
            P = self.search(query, P)
            P = self.prune(query, P, topN)
            if self.reasoning(query, P):
                return self.generate(query, P, use_llm=True)
            D += 1

        return self.generate(query, P, use_llm=True)

    def search(self, query: str, P: List[List]) -> List[List]:
        """Expand paths by one hop."""
        new_paths = []
        for path in P:
            tail_entity = path[-1]
            try:
                successors = list(self.KG.successors(tail_entity))
            except:
                successors = []
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
        """Prune paths to top-N."""
        if len(P) <= topN:
            return P
        path_strings = []
        for path in P:
            formatted_nodes = []
            for i, node_or_relation in enumerate(path):
                if i % 2 == 0:
                    node_data = self.KG.nodes.get(node_or_relation, {})
                    node_text = node_data.get("name", node_data.get("id", str(node_or_relation)))
                    formatted_nodes.append(node_text)
                else:
                    formatted_nodes.append(node_or_relation)
            path_strings.append(" ".join(formatted_nodes))

        query_embedding = self.sentence_encoder.encode([query])[0]
        path_embeddings = self.sentence_encoder.encode(path_strings)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        path_embeddings = path_embeddings / np.linalg.norm(path_embeddings, axis=1, keepdims=True)
        scores = path_embeddings @ query_embedding
        sorted_indices = np.argsort(scores)[::-1]
        return [P[i] for i in sorted_indices[:topN]]

    def reasoning(self, query: str, P: List[List]) -> bool:
        """Check if knowledge is sufficient."""
        triples = []
        for path in P:
            for i in range(0, len(path) - 2, 2):
                node1_data = self.KG.nodes.get(path[i], {})
                node2_data = self.KG.nodes.get(path[i + 2], {})
                node1_text = node1_data.get("name", node1_data.get("id", str(path[i])))
                node2_text = node2_data.get("name", node2_data.get("id", str(path[i + 2])))
                triples.append((node1_text, path[i + 1], node2_text))
        triples_string = ". ".join([f"({t[0]}, {t[1]}, {t[2]})" for t in triples])
        prompt = f"Given a question and the associated retrieved knowledge graph triples (entity, relation, entity), you are asked to answer whether it's sufficient for you to answer the question with these triples and your knowledge (Yes or No). Query: {query} \n Knowledge triples: {triples_string}"
        messages = [
            {"role": "system", "content": "Answer the question following the prompt."},
            {"role": "user", "content": prompt}
        ]
        response = self.llm_generator.generate_response(messages)
        return "yes" in response.lower()

    def generate(self, query: str, P: List[List], use_llm: bool = True) -> Tuple[str, List[str]]:
        """Generate answer."""
        triples = []
        for path in P:
            for i in range(0, len(path) - 2, 2):
                node1_data = self.KG.nodes.get(path[i], {})
                node2_data = self.KG.nodes.get(path[i + 2], {})
                node1_text = node1_data.get("name", node1_data.get("id", str(path[i])))
                node2_text = node2_data.get("name", node2_data.get("id", str(path[i + 2])))
                triples.append(f"({node1_text}, {path[i + 1]}, {node2_text})")

        if not use_llm or len(triples) == 0:
            return "\n".join(triples), ["N/A"] * len(triples)
        
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
            # If LLM is in stub mode or returns generic response, show triples too
            if answer in ["Response", "Error: Could not generate response from LLM"] or len(answer) < 50:
                triples_str = "\n".join([f"  {i+1}. {t}" for i, t in enumerate(triples)])
                answer = f"{answer}\n\nKnowledge Graph Triples:\n{triples_str}"
        except Exception as e:
            answer = "\n".join([f"  {i+1}. {t}" for i, t in enumerate(triples)])
        
        return answer, triples


def load_kg_from_neo4j(uri: str, user: str, password: str) -> nx.DiGraph:
    """Load KG from Neo4j."""
    print("Loading KG from Neo4j...")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    G = nx.DiGraph()
    
    with driver.session() as session:
        result = session.run("MATCH (n:Entity) RETURN n")
        for record in result:
            node = record["n"]
            node_id = node.get("id", node.element_id)
            G.add_node(node_id, **dict(node))
        print(f"Loaded {len(G.nodes())} nodes")
        
        result = session.run("MATCH (a:Entity)-[r]->(b:Entity) RETURN a, r, b")
        for record in result:
            source = record["a"].get("id", record["a"].element_id)
            target = record["b"].get("id", record["b"].element_id)
            rel = record["r"]
            relation = rel.get("relation", rel.type)
            edge_attrs = dict(rel)
            edge_attrs['relation'] = relation
            G.add_edge(source, target, **edge_attrs)
        print(f"Loaded {len(G.edges())} edges")
    
    driver.close()
    return G


def main():
    """Interactive chat interface."""
    # Configuration
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")
    
    # Load KG
    kg = load_kg_from_neo4j(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    # Initialize components
    print("\nInitializing models...")
    use_real_llm = os.getenv("USE_REAL_LLM", "true").lower() == "true"
    llm_generator = LLMGenerator(use_real_llm=use_real_llm)
    
    print("Loading BGE-M3 model...")
    embedding_model = EmbeddingModel()
    
    print("Creating ToG retriever...")
    use_qdrant = os.getenv("USE_QDRANT", "true").lower() == "true"
    retriever = TogV3Retriever(
        KG=kg,
        llm_generator=llm_generator,
        sentence_encoder=embedding_model,
        inference_config=InferenceConfig(Dmax=2),
        use_qdrant=use_qdrant,
        qdrant_url="http://localhost:6333"
    )
    
    print("\n" + "="*80)
    print("Think on Graph Chat Interface")
    print("="*80)
    print("Ask questions about medical conditions.")
    print("Commands: 'quit', 'exit', 'q' to stop")
    print("="*80 + "\n")
    
    # Chat loop
    while True:
        try:
            query = input("Your question: ").strip()
            
            if not query:
                continue
                
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            answer, sources = retriever.retrieve(query, topN=5)
            
            # Show triples first
            print(f"\nRetrieved Triples ({len(sources)} total):")
            for i, source in enumerate(sources, 1):
                print(f"  {i}. {source}")
            
            # Then show LLM answer
            print(f"\nLLM Answer:\n{answer}\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
