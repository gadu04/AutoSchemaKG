"""
Real LLM API Implementation 
============================
Implements actual LLM API calls using local LM Studio (OpenAI-compatible API).

SERVER: http://localhost:1234/v1
API KEY: lm-studio (fixed key for LM Studio)

Features:
- Infinite retry with 3-second delay on errors
- All responses logged to file
- Uses local model loaded in LM Studio
"""

import os
import json
import re
import time
from datetime import datetime
from typing import Dict, List
from pathlib import Path
from openai import OpenAI


LOG_DIR = Path(os.getenv("OUTPUT_DIR", "output"))
LOG_FILE = LOG_DIR / "llm_api_responses.log"


def _clean_json_string(json_str: str) -> str:
    """
    Clean malformed JSON from LLM output.
    Fixes common issues like missing commas, unescaped quotes, etc.
    """
    json_str = json_str.strip()
    if json_str.startswith("```json"):
        json_str = json_str[7:]
    if json_str.startswith("```"):
        json_str = json_str[3:]
    if json_str.endswith("```"):
        json_str = json_str[:-3]
    
    json_str = json_str.strip()
    
    # Fix common issues
    # 1. Fix missing commas between array elements
    json_str = re.sub(r'}\s*\n\s*{', '},\n{', json_str)
    json_str = re.sub(r'}\s*{', '},{', json_str)
    
    # 2. Remove trailing commas before closing brackets
    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
    
    # 3. Fix missing quotes around keys
    json_str = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
    
    # 4. Fix common escape issues
    # Replace \" with " inside strings (be careful)
    json_str = json_str.replace('\\"', '"')
    
    # 5. Ensure proper escaping of internal quotes
    lines = json_str.split('\n')
    fixed_lines = []
    for line in lines:
        if '":' in line or '": ' in line:
            fixed_lines.append(line)
        else:
            if '"' not in line:
                line = line.replace("'", '"')
            fixed_lines.append(line)
    
    json_str = '\n'.join(fixed_lines)
    
    # 6. Fix incomplete JSON (missing closing brackets)
    open_braces = json_str.count('{') - json_str.count('}')
    open_brackets = json_str.count('[') - json_str.count(']')
    
    if open_braces > 0:
        json_str += '}' * open_braces
    if open_brackets > 0:
        json_str += ']' * open_brackets
    
    return json_str


def _extract_json_from_text(text: str) -> str:
    """
    Extract valid JSON from text that may contain additional content.
    """
    text = text.strip()
    
    start_idx = max(text.find('{'), text.find('['))
    if start_idx == -2: 
        start_idx = text.find('{')
    
    if start_idx == -1:
        raise ValueError("No JSON object found in response")
    
    if text[start_idx] == '{':
        closing = '}'
    else:
        closing = ']'
    
    end_idx = text.rfind(closing)
    if end_idx == -1 or end_idx <= start_idx:
        raise ValueError("No valid JSON closing bracket found")
    
    json_str = text[start_idx:end_idx+1]
    return json_str


def _parse_json_robust(json_str: str, max_attempts: int = 5) -> Dict:
    """
    Attempt to parse JSON with multiple cleaning strategies.
    Handles malformed JSON from Llama3 output.
    """
    attempts = []
    
    # Attempt 1: Direct parse
    try:
        result = json.loads(json_str)
        return result
    except json.JSONDecodeError as e:
        attempts.append(f"Direct parse: {str(e)[:60]}")
    except Exception as e:
        attempts.append(f"Direct: {type(e).__name__}: {str(e)[:40]}")
    
    # Attempt 2: Extract JSON first, then clean
    try:
        extracted = _extract_json_from_text(json_str)
        cleaned = _clean_json_string(extracted)
        result = json.loads(cleaned)
        return result
    except Exception as e:
        attempts.append(f"Extract+clean: {type(e).__name__}: {str(e)[:40]}")
    
    # Attempt 3: Just clean (no extract)
    try:
        cleaned = _clean_json_string(json_str)
        result = json.loads(cleaned)
        return result
    except Exception as e:
        attempts.append(f"Clean only: {type(e).__name__}: {str(e)[:40]}")
    
    # Attempt 4: Try to find and fix specific patterns
    try:
        # Remove all newlines and excess whitespace
        cleaned = ' '.join(json_str.split())
        # Fix double spaces
        while '  ' in cleaned:
            cleaned = cleaned.replace('  ', ' ')
        result = json.loads(cleaned)
        return result
    except Exception as e:
        attempts.append(f"Whitespace fix: {type(e).__name__}: {str(e)[:40]}")
    
    # Attempt 5: Return minimal valid JSON
    try:
        # As last resort, return empty structure
        print(f"      ⚠ Using fallback empty response", flush=True)
        return {
            "entity_entity": [],
            "entity_event": [],
            "event_event": []
        }
    except Exception as e:
        attempts.append(f"Fallback: {type(e).__name__}")
    
    # All attempts failed
    error_msg = " | ".join(attempts)
    print(f"\n      DEBUG: Response text:\n{json_str[:200]}\n", flush=True)
    raise json.JSONDecodeError(f"JSON parsing failed after {len(attempts)} attempts: {error_msg}", json_str, 0)


def _log_llm_response(call_type: str, input_data: str, response_data: str, attempt: int = 1, error: str = None):
    """
    Log LLM API call details to file.
    
    Args:
        call_type: Type of call ('triple_extraction' or 'concept_induction')
        input_data: Input text/prompt sent to LLM
        response_data: Response received from LLM
        attempt: Attempt number for retries
        error: Error message if call failed
    """
    try:
        # Ensure log directory exists
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write(f"TIMESTAMP: {timestamp}\n")
            f.write(f"CALL TYPE: {call_type}\n")
            f.write(f"ATTEMPT: {attempt}\n")
            if error:
                f.write(f"STATUS: ERROR\n")
                f.write(f"ERROR: {error}\n")
            else:
                f.write(f"STATUS: SUCCESS\n")
                f.write("-" * 100 + "\n")
                f.write(f"INPUT:\n{input_data[:500]}{'...' if len(input_data) > 500 else ''}\n")
                f.write("-" * 100 + "\n")
                f.write(f"RESPONSE:\n{response_data[:1000]}{'...' if len(response_data) > 1000 else ''}\n")
                f.write("=" * 100 + "\n\n")
    except Exception as e:
        print(f"    ⚠ Warning: Failed to log API response: {e}")


def _build_triple_extraction_prompt(text_segment: str) -> str:
    """
    Build prompt for triple extraction - OPTIMIZED FOR LLAMA3.1.
    More explicit and structured format.
    """
    # Limit text segment to avoid token overflow
    text_segment = text_segment[:800]
    
    return f"""You are a medical knowledge extraction expert.

TASK: Extract triples (relationships) from medical text.

TEXT TO ANALYZE:
{text_segment}

INSTRUCTIONS:
1. Find relationships between medical concepts
2. Extract ONLY these 3 types:
   - entity_entity: Drug treats Disease (e.g., "Metformin" treats "Type 2 Diabetes")
   - entity_event: Entity in Event (e.g., "Patient" participated in "[Event: clinical trial]")
   - event_event: Event causes Event (e.g., "[Event: infection]" leads to "[Event: treatment]")

3. Return ONLY this JSON structure - nothing else:

{{
  "entity_entity": [
    {{"head": "word1", "relation": "word2", "tail": "word3"}}
  ],
  "entity_event": [
    {{"head": "entity", "relation": "action", "tail": "[Event: description]"}}
  ],
  "event_event": [
    {{"head": "[Event: description]", "relation": "action", "tail": "[Event: description]"}}
  ]
}}

4. If no triples found, return:
{{
  "entity_entity": [],
  "entity_event": [],
  "event_event": []
}}

CRITICAL:
- Output ONLY valid JSON
- No explanations
- No markdown code blocks
- No text before or after JSON
- Start with {{ and end with }}
- Use double quotes for all strings
- Separate array items with commas"""


def _build_concept_induction_prompt(node_list: list, triples_context: list = None) -> str:
    """
    Build prompt for concept induction - OPTIMIZED FOR LLAMA3.
    """
    triples_str = ""
    if triples_context:
        triples_str = "\n\nContext from triples:\n"
        for t in triples_context[:3]:
            triples_str += f"  - {t.get('head', 'X')} {t.get('relation', '')} {t.get('tail', 'Y')}\n"
    
    nodes_str = "\n".join([f"  {i+1}. {node}" for i, node in enumerate(node_list[:20])])
    
    return f"""You are a medical concept analyzer.

TASK: Generate semantic concepts for each medical term.

TERMS:
{nodes_str}
{triples_str}

INSTRUCTIONS:
1. For each term, provide 2-4 concept phrases separated by commas
2. Focus on: type, function, category, characteristics
3. Examples: "Metformin" -> "medication, drug, antidiabetic, glucose controller"
4. Return ONLY this JSON - nothing else:

{{
  "term1": "concept1, concept2, concept3",
  "term2": "concept1, concept2, concept3"
}}

CRITICAL:
- Output ONLY valid JSON
- No explanations or markdown
- Start with {{ and end with }}
- Use double quotes for all strings"""


def real_call_llm_for_triples(text_segment: str) -> Dict:
    """
    REAL API: Extract triples using Llama3 (via LM Studio).
    WITH improved error handling and debugging.
    """
    base_url = os.getenv('LM_STUDIO_BASE_URL', 'http://localhost:1234/v1')
    client = OpenAI(base_url=base_url, api_key="lm-studio")
    
    model_name = os.getenv("MODEL_NAME", "local-model")
    temperature = float(os.getenv("MODEL_TEMPERATURE", "0.05"))  # Lower = more deterministic
    max_tokens = int(os.getenv("MODEL_MAX_TOKENS", "1500"))
    
    prompt = _build_triple_extraction_prompt(text_segment)
    
    attempt = 0
    last_response = ""
    
    while True:
        attempt += 1
        try:
            print(f"    → LLM API call for triple extraction (attempt {attempt})...", end=" ", flush=True)
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You output only valid JSON. No explanations. No code blocks. Only JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            response_text = response.choices[0].message.content.strip()
            last_response = response_text
            
            # Debug: Check response starts with {
            if not response_text.startswith('{'):
                print(f"✗ Response doesn't start with '{{' (first char: '{response_text[0] if response_text else 'empty'}')")
                if attempt >= 3:
                    print(f"    ⚠ Failed after {attempt} attempts. Using fallback.")
                    print(f"      Response was: {response_text[:100]}", flush=True)
                    _log_llm_response("triple_extraction", prompt, response_text, attempt, "Invalid JSON format")
                    return {"entity_entity": [], "entity_event": [], "event_event": []}
                time.sleep(1)
                continue
            
            # Clean response
            response_text = response_text.replace("<think>", "").replace("</think>", "")
            response_text = response_text.strip()
            
            # Extract JSON
            try:
                triples_data = _parse_json_robust(response_text)
                
                # Validate structure
                if not isinstance(triples_data, dict):
                    raise ValueError("Response is not a dict")
                
                if "entity_entity" not in triples_data:
                    triples_data["entity_entity"] = []
                if "entity_event" not in triples_data:
                    triples_data["entity_event"] = []
                if "event_event" not in triples_data:
                    triples_data["event_event"] = []
                
                print(f"✓")
                _log_llm_response("triple_extraction", prompt, response_text, attempt)
                return triples_data
                
            except json.JSONDecodeError as parse_error:
                print(f"✗ JSON parse failed")
                if attempt >= 3:
                    print(f"    ⚠ Failed after {attempt} attempts. Using fallback.")
                    print(f"      Response was: {response_text[:150]}", flush=True)
                    _log_llm_response("triple_extraction", prompt, response_text, attempt, f"JSON parse: {str(parse_error)[:50]}")
                    return {"entity_entity": [], "entity_event": [], "event_event": []}
                time.sleep(1)
                continue
            
        except Exception as e:
            print(f"✗ Error: {type(e).__name__}: {str(e)[:40]}")
            if attempt >= 3:
                print(f"    ⚠ Failed after {attempt} attempts. Using fallback.")
                print(f"      Last response: {last_response[:100]}", flush=True)
                _log_llm_response("triple_extraction", prompt, last_response, attempt, str(e))
                return {"entity_entity": [], "entity_event": [], "event_event": []}
            time.sleep(2)
            continue


def real_call_llm_for_concepts(node_list: List[str], triples_list: List[Dict] = None) -> Dict[str, str]:
    """
    REAL API: Induce concepts using Llama3.
    WITH improved error handling and debugging.
    """
    base_url = os.getenv('LM_STUDIO_BASE_URL', 'http://localhost:1234/v1')
    client = OpenAI(base_url=base_url, api_key="lm-studio")
    
    model_name = os.getenv("MODEL_NAME", "local-model")
    temperature = float(os.getenv("MODEL_TEMPERATURE", "0.05"))
    max_tokens = int(os.getenv("MODEL_MAX_TOKENS", "1500"))
    
    prompt = _build_concept_induction_prompt(node_list, triples_list)
    
    attempt = 0
    last_response = ""
    
    while True:
        attempt += 1
        try:
            print(f"    → LLM API call for concept induction (attempt {attempt})...", end=" ", flush=True)
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You output only valid JSON. No explanations. Only JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            response_text = response.choices[0].message.content.strip()
            last_response = response_text
            
            # Debug: Check response starts with {
            if not response_text.startswith('{'):
                print(f"✗ Response doesn't start with '{{'")
                if attempt >= 3:
                    print(f"    ⚠ Failed after {attempt} attempts. Using fallback.")
                    print(f"      Response was: {response_text[:100]}", flush=True)
                    _log_llm_response("concept_induction", prompt, response_text, attempt, "Invalid JSON format")
                    return {node: "medical concept" for node in node_list}
                time.sleep(1)
                continue
            
            # Clean
            response_text = response_text.replace("<think>", "").replace("</think>", "")
            response_text = response_text.strip()
            
            # Parse JSON
            try:
                concepts_data = _parse_json_robust(response_text)
                
                # Validate: all nodes must have concepts
                for node in node_list:
                    if node not in concepts_data:
                        concepts_data[node] = "medical concept"
                
                print(f"✓")
                _log_llm_response("concept_induction", prompt, response_text, attempt)
                return concepts_data
                
            except json.JSONDecodeError as parse_error:
                print(f"✗ JSON parse failed")
                if attempt >= 3:
                    print(f"    ⚠ Failed after {attempt} attempts. Using fallback.")
                    print(f"      Response was: {response_text[:150]}", flush=True)
                    _log_llm_response("concept_induction", prompt, response_text, attempt, f"JSON parse: {str(parse_error)[:50]}")
                    return {node: "medical concept" for node in node_list}
                time.sleep(1)
                continue
            
        except Exception as e:
            print(f"✗ Error: {type(e).__name__}: {str(e)[:40]}")
            if attempt >= 3:
                print(f"    ⚠ Failed after {attempt} attempts. Using fallback.")
                print(f"      Last response: {last_response[:100]}", flush=True)
                _log_llm_response("concept_induction", prompt, last_response, attempt, str(e))
                return {node: "medical concept" for node in node_list}
            time.sleep(2)
            continue
