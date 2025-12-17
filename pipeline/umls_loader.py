"""
UMLS API Loader Module - Verified Working Version
=================================================
Cấu hình dựa trên file debug_umls.py đã chạy thành công.
"""

import os
from typing import Dict, List, Optional

try:
    import requests
except ImportError:
    requests = None

class UMLSLoader:
    
    SEARCH_URL = "https://uts-ws.nlm.nih.gov/rest/search/current"
    CONTENT_URL = "https://uts-ws.nlm.nih.gov/rest/content/current"
    
    # SỬ DỤNG ĐÚNG DANH SÁCH ĐÃ TEST THÀNH CÔNG TRONG DEBUG
    # Không thêm khoảng trắng sau dấu phẩy
    TARGET_SABS = "SNOMEDCT_US,ICD10CM,RXNORM,LNC,MSH,FMA,GO,HPO,HL7V3.0,NCI,OMIM,HGNC,ATC,ICD10PCS,CVX,HCPCS,MED-RT,CHV"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("UMLS_API_KEY", "").strip()
        self.authenticated = False
        
        if requests is None:
            print("  ⚠ Error: 'requests' library missing.")
            return
        
        if not self.api_key:
            print("  ⚠ Error: UMLS_API_KEY missing.")
            return
        
        self._validate_api_key()
    
    def _validate_api_key(self) -> bool:
        try:
            params = {"apiKey": self.api_key, "string": "fever", "pageSize": 1}
            resp = requests.get(self.SEARCH_URL, params=params, timeout=10)
            
            if resp.status_code == 200:
                self.authenticated = True
                return True
            else:
                print(f"  ⚠ API Validation Failed: {resp.status_code}")
                return False
        except Exception as e:
            print(f"  ⚠ API Connection Error: {e}")
            return False

    def search_concept(self, concept: str) -> List[Dict]:
        """
        Tìm kiếm concept với cấu hình 'chặt' để loại bỏ rác.
        """
        if not self.authenticated:
            return []
        
        try:
            params = {
                "apiKey": self.api_key,
                "string": concept,
                "sabs": self.TARGET_SABS, 
                "searchType": "words",
                "returnIdType": "concept",
                "pageSize": 5
            }
            
            resp = requests.get(self.SEARCH_URL, params=params, timeout=10)
            
            if resp.status_code != 200:
                print(f"  [DEBUG] API Error {resp.status_code} for term '{concept}'")
                return []
            
            data = resp.json()
            results = []
            
            if "result" in data and "results" in data["result"]:
                for item in data["result"]["results"]:
                    name_val = item.get('name', 'NONE')
                    if name_val == 'NONE': continue
                    
                    results.append({
                        'umls_id': item.get('ui'),
                        'name': name_val,
                        'source': item.get('rootSource'),
                        'score': self._calculate_match_score(concept, name_val)
                    })
            
            results.sort(key=lambda x: x['score'], reverse=True)
            return results
        
        except Exception as e:
            print(f"  [DEBUG] Exception: {e}")
            return []

    def get_best_match(self, concept: str, threshold: float = 0.4) -> Optional[Dict]:
        results = self.search_concept(concept)
        if results:
            return results[0]
        return None
    
    def _calculate_match_score(self, search_term: str, result_name: str) -> float:
        s, r = search_term.lower(), result_name.lower()
        if s == r: return 1.0
        if s in r or r in s: return 0.8
        return 0.5 

    def is_available(self) -> bool:
        return self.authenticated

    def get_concept_details(self, umls_id: str) -> Optional[Dict]:
        return None
    
    def get_cui(self, term: str) -> Optional[Dict]:
        res = self.get_best_match(term)
        if res:
            return {
                "ui": res['umls_id'],
                "umls_id": res['umls_id'],
                "name": res['name'],
                "rootSource": res.get('source'),
                "uri": f"https://uts.nlm.nih.gov/uts/umls/concept/{res['umls_id']}"
            }
        return None