"""
Simplified PDF Parser for Medical Textbooks
Strategy: Fixed percentage-based content extraction
- Skip first 1.4% of pages (Intro)
- Skip last 10% of pages (Index/Glossary)
- Parse everything in between
"""

import os
import time
import json
import requests
from pathlib import Path
from typing import Optional, Dict
from requests.adapters import HTTPAdapter, Retry

# Configuration
API_URL = "https://www.datalab.to/api/v1/marker"
# List of API keys for rotation
API_KEYS = [
    # "xkj26_flKZCQFYGcv0XNtUiBHnpDd2P4D3AgykWMQEc", # REMOVED: Over limit
    "833_EemjQe8lQ3zhM2zlUC2XBh0xAsIM8T-oCLICO7M", 
    "NkxPxQ9qW1ZXYN4SLknYxWD_TdAPJOWW8gY9GaKrRMY",
]
CURRENT_KEY_INDEX = 0

def get_current_api_key():
    return API_KEYS[CURRENT_KEY_INDEX]

def rotate_api_key():
    global CURRENT_KEY_INDEX
    CURRENT_KEY_INDEX = (CURRENT_KEY_INDEX + 1) % len(API_KEYS)
    print(f"\n🔄 Rotating API Key to #{CURRENT_KEY_INDEX + 1} ({get_current_api_key()[:8]}...)")

# Configure session with retry handling
session = requests.Session()
retries = Retry(
    total=20,
    backoff_factor=4,
    status_forcelist=[429],
    allowed_methods=["GET", "POST"],
    raise_on_status=False,
)
adapter = HTTPAdapter(max_retries=retries)
session.mount("http://", adapter)
session.mount("https://", adapter)


def submit_and_poll_conversion(
    pdf_path: Path,
    output_format: str = 'markdown',
    use_llm: bool = False,
    page_range: Optional[str] = None,
    paginate: bool = False,
    keep_headers_footers: bool = False
) -> Optional[Dict]:
    """
    Submit PDF for conversion and poll until complete.
    Handles API key rotation on credit exhaustion.
    """
    # Additional config
    additional_config = {}
    if keep_headers_footers:
        additional_config['keep_pageheader_in_output'] = True
        additional_config['keep_pagefooter_in_output'] = True
    
    # Submit initial request with retry/rotation loop
    max_key_retries = len(API_KEYS)
    data = None
    
    with open(pdf_path, 'rb') as f:
        for attempt in range(max_key_retries):
            current_key = get_current_api_key()
            headers = {"X-Api-Key": current_key}
            
            # Reset file pointer for retry
            f.seek(0)
            
            form_data = {
                'file': (pdf_path.name, f, 'application/pdf'),
                'output_format': (None, output_format),
                'use_llm': (None, use_llm),
                'force_ocr': (None, False),
                'paginate': (None, paginate),
                'strip_existing_ocr': (None, False),
                'disable_image_extraction': (None, False)
            }
            
            if page_range:
                form_data['page_range'] = (None, page_range)
            
            if additional_config:
                form_data['additional_config'] = (None, json.dumps(additional_config))
            
            print(f"📤 Submitting {pdf_path.name}" + (f" (pages {page_range})" if page_range else "") + f" [Key: ...{current_key[-4:]}]")
            
            try:
                response = session.post(API_URL, files=form_data, headers=headers)
                
                if response.status_code == 200:
                    data = response.json()
                    break # Success!
                
                elif response.status_code in [402, 403, 401]: # Payment Required / Forbidden / Unauthorized
                    print(f"⚠️ API Key Error ({response.status_code}): {response.text}")
                    rotate_api_key()
                    continue # Try next key
                
                else:
                    print(f"❌ API Error: {response.text}")
                    return None
                    
            except Exception as e:
                print(f"❌ Connection Error: {e}")
                return None
        
        if not data:
            print("❌ All API keys exhausted or failed!")
            return None
    
    # Poll for completion
    max_polls = 1000
    check_url = data["request_check_url"]
    headers = {"X-Api-Key": get_current_api_key()} # Use the successful key
    
    for i in range(max_polls):
        time.sleep(3)
        try:
            response = session.get(check_url, headers=headers)
            result = response.json()
            
            if result['status'] == 'complete':
                print(f"✅ Completed {pdf_path.name}" + (f" (pages {page_range})" if page_range else ""))
                return result
            
            elif result['status'] == 'failed':
                print(f"❌ Failed {pdf_path.name}: {result.get('error', 'Unknown error')}")
                return None
            
            if i % 10 == 0:
                print(f"⏳ Still processing... ({i * 3}s elapsed)")
        except Exception as e:
            print(f"⚠️ Polling error: {e}")
            time.sleep(5)
    
    print(f"⚠️ Timeout processing {pdf_path.name}")
    return None


def get_total_pages(pdf_path: Path) -> int:
    """Get total pages using PyPDF2 (fast/free) or API (fallback)"""
    try:
        import PyPDF2
        with open(pdf_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            return len(pdf_reader.pages)
    except Exception as e:
        print(f"⚠️ PyPDF2 failed ({e}), using API to check page count...")
        # Use API to check page count (cheap/free for metadata)
        result = submit_and_poll_conversion(pdf_path, page_range="0", use_llm=False)
        if result:
            meta = result.get('metadata', {})
            return meta.get('pages') or meta.get('num_pages') or 0
    return 0


def parse_medical_textbook(
    pdf_path: Path,
    output_path: Path,
    use_llm: bool = False,
    dry_run: bool = False
) -> bool:
    """
    Parse medical textbook using fixed percentage logic.
    - Skip first 1.4% (Intro)
    - Skip last 10% (Index)
    """
    print("\n" + "="*70)
    print(f"📚 PARSING MEDICAL TEXTBOOK (SIMPLE MODE)")
    print("="*70)
    print(f"PDF: {pdf_path.name}")
    
    # Step 1: Get total pages
    total_pages = get_total_pages(pdf_path)
    if total_pages == 0:
        print("❌ Could not determine total pages")
        return False
        
    print(f"📄 Total Pages: {total_pages}")
    
    # Step 2: Calculate range
    # Logic: Skip first 1.4%, Skip last 8.5%
    start_page = int(total_pages * 0.014)
    end_page = int(total_pages * (1 - 0.085))
    
    # Ensure valid range
    start_page = max(0, start_page)
    end_page = min(total_pages - 1, end_page)
    
    if start_page >= end_page:
        print("❌ Invalid page range calculated")
        return False
        
    content_pages = end_page - start_page + 1
    
    print(f"\n📊 Calculation:")
    print(f"   Skip Intro (1.4%): Pages 0-{start_page-1}")
    print(f"   Skip Index (10%):  Pages {end_page+1}-{total_pages-1}")
    print(f"   ------------------------------------------")
    print(f"   TARGET CONTENT:    Pages {start_page}-{end_page} (API 0-indexed)")
    print(f"   Total to parse:    {content_pages} pages")
    
    # Cost estimate
    cost_per_page = 0.006 if use_llm else 0.004
    total_cost = content_pages * cost_per_page
    print(f"\n💰 Estimated cost: ${total_cost:.2f}")
    
    if dry_run:
        print(f"\n🛑 Dry run complete. Set dry_run=False to execute.")
        return True
    
    # Step 3: Parse
    print(f"\n🚀 Starting conversion...")
    
    # Split the workload if it's too large for one key (assuming ~$5 limit per key)
    # $5 / $0.004 per page = 1250 pages max per key
    MAX_PAGES_PER_KEY = 1000 # Safe limit
    
    if content_pages > MAX_PAGES_PER_KEY and len(API_KEYS) > 1:
        print(f"⚠️ Large job ({content_pages} pages). Splitting across {len(API_KEYS)} keys...")
        
        # Calculate split point
        mid_point = start_page + (content_pages // 2)
        
        ranges = [
            (start_page, mid_point),
            (mid_point + 1, end_page)
        ]
        
        full_markdown = ""
        all_images = {}
        
        for i, (r_start, r_end) in enumerate(ranges):
            # Force rotate key for next chunk
            if i > 0:
                rotate_api_key()
                
            print(f"\n🔄 Processing Chunk {i+1}/{len(ranges)}: Pages {r_start}-{r_end} ({r_end-r_start+1} pages)")
            
            chunk_result = submit_and_poll_conversion(
                pdf_path=pdf_path,
                page_range=f"{r_start}-{r_end}",
                use_llm=use_llm,
                paginate=False,
                keep_headers_footers=False
            )
            
            if not chunk_result or 'markdown' not in chunk_result:
                print(f"❌ Chunk {i+1} failed")
                return False
                
            full_markdown += chunk_result['markdown'] + "\n\n"
            if 'images' in chunk_result:
                all_images.update(chunk_result['images'])
                
        result = {'markdown': full_markdown, 'images': all_images}
        
    else:
        # Normal single batch processing
        result = submit_and_poll_conversion(
            pdf_path=pdf_path,
            page_range=f"{start_page}-{end_page}",
            use_llm=use_llm,
            paginate=False,
            keep_headers_footers=False
        )
    
    if not result or 'markdown' not in result:
        print("❌ Conversion failed")
        return False
        
    # Step 4: Save
    print(f"\n💾 Saving output...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Post-process markdown
    markdown_content = result['markdown']
    
    # Basic cleanup: remove excessive newlines
    import re
    markdown_content = re.sub(r'\n{3,}', '\n\n', markdown_content)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"# {pdf_path.stem}\n\n")
        f.write(f"**Source:** {pdf_path.name}\n")
        f.write(f"**Pages Parsed:** {start_page}-{end_page} (0-indexed)\n")
        f.write(f"**Total Pages:** {content_pages}\n")
        f.write(f"\n---\n\n")
        f.write(markdown_content)
        
    print(f"✅ Saved to: {output_path}")
    
    # Save images
    if 'images' in result and result['images']:
        import base64
        img_dir = output_path.parent / f"{output_path.stem}_images"
        img_dir.mkdir(exist_ok=True)
        for img_name, img_b64 in result['images'].items():
            (img_dir / img_name).write_bytes(base64.b64decode(img_b64))
        print(f"   Images saved to: {img_dir}")
        
    return True

if __name__ == "__main__":
    # UPDATE THIS PATH
    pdf_file = Path(r"Data/raw/original/AMA_Family_Guide.pdf")
    output_file = Path(r"Data/parsed/AMA_Family_Guide_content.md")
    
    if pdf_file.exists():
        parse_medical_textbook(
            pdf_path=pdf_file,
            output_path=output_file,
            use_llm=False,
            dry_run=False  # Set to False to run for real
        )
    else:
        print(f"❌ File not found: {pdf_file}")
