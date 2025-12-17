"""
LLM API Interface
=================
Provides interface for LLM API calls using LM Studio.
"""

from typing import Dict, List
from llm_api.real_api import (
    real_call_llm_for_triples,
    real_call_llm_for_concepts
)


def call_llm_for_triples(text_segment: str, use_real_llm: bool = True) -> Dict:
    """
    Extract triples from a text segment using LLM.
    
    Args:
        text_segment (str): Text to extract triples from
        use_real_llm (bool): Deprecated parameter, kept for compatibility
        
    Returns:
        Dict: Extracted triples in format:
            {
                'entity_entity': [
                    {'head': str, 'relation': str, 'tail': str, 'confidence': float}
                ],
                'entity_event': [...],
                'event_event': [...]
            }
    """
    return real_call_llm_for_triples(text_segment)


def call_llm_for_concepts(node_list: List[str], use_real_llm: bool = True, triples_list: List[Dict] = None) -> Dict[str, str]:
    """
    Generate induced concepts for a list of nodes using LLM.
    
    Implements AutoSchemaKG approach with separate handling for entities and events.
    
    Args:
        node_list (List[str]): List of node names to generate concepts for
        use_real_llm (bool): Deprecated parameter, kept for compatibility
        triples_list (List[Dict], optional): List of triples for context extraction
        
    Returns:
        Dict[str, str]: Mapping of node name to induced concept phrases
            Example: {"Metformin": "medication, drug, pharmaceutical, treatment"}
    """
    return real_call_llm_for_concepts(node_list, triples_list)
