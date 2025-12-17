# Medical-SchemaKG Framework - System Diagrams

This document contains ASCII diagrams to help visualize the framework architecture.

---

## Overall System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                     MEDICAL-SCHEMAKG FRAMEWORK                          │
│                    Four-Phase Pipeline Architecture                     │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
          ┌─────────────────────────────────────────────────┐
          │         INPUT: Medical Text Documents           │
          │              (.txt format)                      │
          └─────────────────────────────────────────────────┘
                                    │
                                    ▼
╔═══════════════════════════════════════════════════════════════════════╗
║                        PHASE 1: INGESTION                             ║
║  ┌─────────────────────────────────────────────────────────────────┐  ║
║  │  • Load text file                                               │  ║
║  │  • Segment into paragraphs                                      │  ║
║  │  • Clean and normalize                                          │  ║
║  └─────────────────────────────────────────────────────────────────┘  ║
╚═══════════════════════════════════════════════════════════════════════╝
                                    │
                                    ▼
                    List[str] - text_segments (e.g., 8 paragraphs)
                                    │
                                    ▼
╔═══════════════════════════════════════════════════════════════════════╗
║                    PHASE 2: TRIPLE EXTRACTION                         ║
║  ┌─────────────────────────────────────────────────────────────────┐  ║
║  │  For each text segment:                                         │  ║
║  │    1. Call LLM with extraction prompt                           │  ║
║  │    2. Parse JSON response                                       │  ║
║  │    3. Collect E-E, E-Ev, Ev-Ev triples                         │  ║
║  │    4. Track unique nodes                                        │  ║
║  │                                                                  │  ║
║  │  LLM API Route: LM Studio                                       │  ║
║  └─────────────────────────────────────────────────────────────────┘  ║
╚═══════════════════════════════════════════════════════════════════════╝
                                    │
                                    ▼
            (List[Dict] - triples, Set[str] - unique_nodes)
                    (e.g., 24 triples, 35 nodes)
                                    │
                                    ▼
╔═══════════════════════════════════════════════════════════════════════╗
║              PHASE 3: SCHEMA INDUCTION & GROUNDING                    ║
║                                                                        ║
║  ┌──────────────────────────────────────────────────────────────┐    ║
  │  PART 3a: CONCEPT INDUCTION                                  │    ║
║  │  ┌────────────────────────────────────────────────────────┐  │    ║
║  │  │  For each unique node:                                  │  │    ║
║  │  │    • Call LLM with concept prompt                       │  │    ║
║  │  │    • Generate abstract concept                          │  │    ║
║  │  │  Example: "Metformin" → "a diabetes medication"         │  │    ║
║  │  └────────────────────────────────────────────────────────┘  │    ║
║  └──────────────────────────────────────────────────────────────┘    ║
║                              ▼                                         ║
║                  Dict[str, str] - induced_concepts                    ║
║                              ▼                                         ║
║  ┌──────────────────────────────────────────────────────────────┐    ║
  │  PART 3b: ONTOLOGY GROUNDING                                 │    ║
  │  ┌────────────────────────────────────────────────────────┐  │    ║
  │  │  For each induced concept:                              │  │    ║
  │  │    • Map to UMLS/SNOMED CT                              │  │    ║
║  │  │    • Add ontology ID and semantic type                  │  │    ║
║  │  │  Example: → UMLS:C0025598, "Pharmacologic Substance"    │  │    ║
║  │  └────────────────────────────────────────────────────────┘  │    ║
║  └──────────────────────────────────────────────────────────────┘    ║
╚═══════════════════════════════════════════════════════════════════════╝
                                    │
                                    ▼
                Dict[str, Dict] - grounded_nodes
                    (35 nodes with full attributes)
                                    │
                                    ▼
╔═══════════════════════════════════════════════════════════════════════╗
║              PHASE 4: KNOWLEDGE GRAPH CONSTRUCTION                    ║
║  ┌─────────────────────────────────────────────────────────────────┐  ║
║  │  1. Create NetworkX MultiDiGraph                               │  ║
║  │  2. Add all nodes with grounded attributes                      │  ║
║  │  3. Add all edges from triples                                  │  ║
║  │  4. Compute graph statistics                                    │  ║
║  │  5. Export to multiple formats                                  │  ║
║  └─────────────────────────────────────────────────────────────────┘  ║
╚═══════════════════════════════════════════════════════════════════════╝
                                    │
                                    ▼
            ┌───────────────────────────────────────┐
            │   OUTPUTS: Onto-MedKG Knowledge Graph │
            │   • Nodes: 35                         │
            │   • Edges: 24                         │
            │   • Formats: PNG, JSON, GraphML       │
            └───────────────────────────────────────┘
```

---

## Data Flow Through Pipeline

```
TEXT SEGMENT:
┌────────────────────────────────────────────────────────────┐
│ "Metformin is used to treat Type 2 Diabetes by            │
│  improving insulin sensitivity in peripheral tissues."     │
└────────────────────────────────────────────────────────────┘
                        │
                        ▼ PHASE 2: TRIPLE EXTRACTION
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
   LLM PROMPT                      JSON RESPONSE
┌──────────────────┐         ┌──────────────────────┐
│ "Extract E-E,    │   →→→   │ {                    │
│  E-Ev, Ev-Ev     │         │   "entity_entity": [ │
│  triples..."     │         │     {head: "...",    │
│                  │         │      relation: "...",│
│                  │         │      tail: "..."}]   │
└──────────────────┘         │ }                    │
                             └──────────────────────┘
                                       │
                                       ▼ PARSE & VALIDATE
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
            EXTRACTED TRIPLES                    UNIQUE NODES
┌─────────────────────────────────┐    ┌──────────────────────┐
│ 1. (Metformin, treats,          │    │ • Metformin          │
│     Type 2 Diabetes)            │    │ • Type 2 Diabetes    │
│ 2. (Metformin, improves,        │    │ • insulin sensitivity│
│     insulin sensitivity)        │    │ • peripheral tissues │
└─────────────────────────────────┘    └──────────────────────┘
                    │                                     │
                    │                                     ▼
                    │                      PHASE 3a: CONCEPT INDUCTION
                    │                                     │
                    │              ┌─────────────────────┴────────────┐
                    │              ▼                                  ▼
                    │      LLM CONCEPT PROMPT                 INDUCED CONCEPTS
                    │    ┌─────────────────────┐      ┌────────────────────┐
                    │    │ "Generate abstract  │ →→→  │ Metformin →        │
                    │    │  concepts for       │      │   "a diabetes med" │
                    │    │  these terms..."    │      │ Type 2 Diabetes →  │
                    │    └─────────────────────┘      │   "a disease"      │
                    │                                 └────────────────────┘
                    │                                            │
                    │                                            ▼
                    │                          PHASE 3b: ONTOLOGY GROUNDING
                    │                                            │
                    │                              ┌─────────────┴──────────┐
                    │                              ▼                        ▼
                    │                     GROUNDED NODES          UMLS IDs
                    │                ┌──────────────────────┐  ┌───────────────┐
                    │                │ Metformin:           │  │ UMLS:C0025598 │
                    │                │   concept: "..."     │  │ UMLS:C0011860 │
                    │                │   ontology_id: "..." │  │ ...           │
                    │                │   semantic_type: "..." │ └───────────────┘
                    │                └──────────────────────┘
                    │                            │
                    └────────────────────────────┴──────────────────┐
                                                                     ▼
                                          PHASE 4: GRAPH CONSTRUCTION
                                                      │
                            ┌─────────────────────────┴─────────────────────┐
                            ▼                                               ▼
                    ADD NODES WITH                                  ADD EDGES FROM
                      ATTRIBUTES                                       TRIPLES
                ┌────────────────────┐                        ┌─────────────────┐
                │ Node: "Metformin"  │                        │ Metformin       │
                │   • ontology_id    │                        │    ↓ treats     │
                │   • concept        │                        │ Type 2 Diabetes │
                │   • semantic_type  │                        └─────────────────┘
                └────────────────────┘
                            │
                            └─────────────────────────┐
                                                      ▼
                                        FINAL KNOWLEDGE GRAPH
                                    ┌──────────────────────────┐
                                    │  NetworkX MultiDiGraph   │
                                    │  • Nodes with attributes │
                                    │  • Edges with metadata   │
                                    │  • Multiple formats      │
                                    └──────────────────────────┘
```

---

## Triple Types Visualization

```
ENTITY-ENTITY (E-E) TRIPLES:
┌─────────────┐                    ┌──────────────────┐
│  Metformin  │ ──── treats ────→  │ Type 2 Diabetes  │
│  (Entity)   │                    │    (Entity)      │
└─────────────┘                    └──────────────────┘
     Static                              Static
    "Thing"                             "Thing"


ENTITY-EVENT (E-Ev) TRIPLES:
┌─────────────┐                    ┌──────────────────────┐
│   Patient   │ ─ participated_in → │  Clinical Trial for  │
│  (Entity)   │                    │  Diabetes Management │
└─────────────┘                    │     (Event)          │
     Static                        └──────────────────────┘
    "Thing"                          "Happening/Action"


EVENT-EVENT (Ev-Ev) TRIPLES:
┌──────────────────┐               ┌─────────────────────┐
│ Initial Diabetes │ ─── led_to ──→│  Metformin          │
│    Diagnosis     │               │  Treatment          │
│    (Event)       │               │  Initiation         │
└──────────────────┘               │  (Event)            │
   "Happening"                     └─────────────────────┘
                                      "Happening"
```

---

## LLM API Routing

```
                        ┌──────────────────────┐
                        │  call_llm_for_*()   │
                        │    (interface.py)    │
                        └──────────┬───────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │  real_call_llm_*()  │
                        │    (real_api.py)    │
                        └──────────┬───────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │  Call LM Studio     │
                        │  API with prompt    │
                        │                     │
                        │  • Real extraction  │
                        │  • 3-5 sec/call     │
                        └─────────────────────┘
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │  Return structured   │
                        │  JSON response       │
                        └──────────────────────┘
```

---

## Module Dependencies

```
main.py
   │
   ├─► pipeline/
   │      ├─► phase_1_ingestion.py
   │      ├─► phase_2_triple_extraction.py ──┐
   │      │                                   │
   │      ├─► phase_3_schema_induction.py ───┤
   │      │                                   │
   │      └─► phase_4_kg_construction.py     │
   │                                          │
   ├─► llm_api/                              │
   │      ├─► interface.py ◄──────────────────┘
   │      └─► real_api.py ──► LM Studio API
   │
   └─► utils/
          └─► visualization.py ──► matplotlib
                                ──► NetworkX
```

---

## Knowledge Graph Structure

```
                    ONTO-MEDKG KNOWLEDGE GRAPH
                  (NetworkX MultiDiGraph Object)

NODES WITH ATTRIBUTES:
┌─────────────────────────────────────────────────────────────┐
│ Node: "Metformin"                                           │
│   ├─ node_name: "Metformin"                                │
│   ├─ ontology_id: "UMLS:C0025598"                          │
│   ├─ induced_concept: "a type of diabetes medication"      │
│   ├─ ontology_name: "Metformin"                            │
│   ├─ semantic_type: "Pharmacologic Substance"              │
│   └─ node_type: "entity"                                   │
└─────────────────────────────────────────────────────────────┘

EDGES WITH METADATA:
┌─────────────────────────────────────────────────────────────┐
│ Edge: "Metformin" → "Type 2 Diabetes"                      │
│   ├─ relation: "treats"                                     │
│   ├─ triple_type: "E-E"                                     │
│   ├─ head_type: "entity"                                    │
│   ├─ tail_type: "entity"                                    │
│   ├─ segment_id: 1                                          │
│   └─ confidence: 0.95                                       │
└─────────────────────────────────────────────────────────────┘

GRAPH PROPERTIES:
┌─────────────────────────────────────────────────────────────┐
│ • Type: MultiDiGraph (multiple directed edges allowed)     │
│ • Directed: Yes (head → tail)                              │
│ • Multi-edges: Supported (same nodes, different relations) │
│ • Self-loops: Allowed                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## Execution Flow

```
USER RUNS: python main.py
          │
          ▼
    ┌──────────────┐
    │  main.py     │
    │  main()      │
    └──────┬───────┘
           │
           ├─► Phase 1 (0.1s)
           │   └─► Load sample_medical_text.txt → 8 segments
           │
           ├─► Phase 2 (30s)
           │   ├─► For each of 8 segments:
           │   │   ├─► real_call_llm_for_triples()
           │   │   │   ├─► Build extraction prompt
           │   │   │   ├─► Call LM Studio API (~3s)
           │   │   │   └─► Parse JSON response
           │   └─► Collect: ~40 triples, ~50 nodes (varies)
           │
           ├─► Phase 3a (15s)
           │   ├─► real_call_llm_for_concepts()
           │   │   ├─► Batch process (20 nodes/call)
           │   │   ├─► Build concept prompt
           │   │   ├─► Call LM Studio API (~3s/batch)
           │   │   └─► Parse JSON response
           │   └─► Generate: 50 concepts
           │
           ├─► Phase 3b (0.1s)
           │   └─► ground_concepts_to_ontology() → UMLS IDs
           │
           ├─► Phase 4 (0.3s)
           │   └─► build_knowledge_graph() → NetworkX graph
           │
           └─► Output (0.5s)
               ├─► Console summary
               ├─► knowledge_graph.png
               ├─► knowledge_graph.json
               └─► knowledge_graph.graphml

TOTAL TIME: ~50 seconds
```

---

## File Organization

```
Framework/
│
├─── Core Pipeline ───────────────────────────────────┐
│    ├── main.py (orchestrator)                       │
│    └── pipeline/                                    │
│        ├── phase_1_ingestion.py                     │
│        ├── phase_2_triple_extraction.py             │
│        ├── phase_3_schema_induction.py              │
│        └── phase_4_kg_construction.py               │
│                                                      │
├─── LLM Integration ─────────────────────────────────┤
│    └── llm_api/                                     │
│        ├── interface.py (router)                    │
│        └── real_api.py (LM Studio)                  │
│                                                      │
├─── Utilities ───────────────────────────────────────┤
│    └── utils/                                        │
│        └── visualization.py                          │
│                                                      │
├─── Testing ─────────────────────────────────────────┤
│    └── test_framework.py                            │
│                                                      │
├─── Scripts ─────────────────────────────────────────┤
│    ├── setup.ps1 (setup script)                     │
│    └── run.bat (quick run)                          │
│                                                      │
├─── Documentation ───────────────────────────────────┤
│    ├── README.md (main guide)                       │
│    ├── QUICKSTART.md (quick setup)                  │
│    ├── TECHNICAL_DOCS.md (developer)                │
│    ├── EXAMPLES.md (usage examples)                 │
│    ├── FAQ.md (Q&A)                                 │
│    ├── PROJECT_SUMMARY.md (overview)                │
│    ├── CHANGELOG.md (history)                       │
│    ├── DOCUMENTATION_INDEX.md (navigation)          │
│    └── DIAGRAMS.md (this file)                      │
│                                                      │
├─── Configuration ───────────────────────────────────┤
│    ├── requirements.txt                              │
│    ├── .env.example                                  │
│    ├── .gitignore                                    │
│    └── LICENSE                                       │
│                                                      │
├─── Data ────────────────────────────────────────────┤
│    └── data/                                         │
│        └── sample_medical_text.txt (auto-generated) │
│                                                      │
└─── Output ──────────────────────────────────────────┘
     └── output/ (created at runtime)
         ├── knowledge_graph.png
         ├── knowledge_graph.json
         ├── knowledge_graph.graphml
         └── knowledge_graph_edges.txt
```

---

*These diagrams are designed to help you visualize and understand the Medical-SchemaKG Framework architecture and data flow.*
