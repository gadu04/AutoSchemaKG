"""
Streamlit UI for Medical QA System (Think on Graph only)
"""

import streamlit as st
import requests
import os

st.set_page_config(
    page_title="Medical QA - Think on Graph",
    page_icon="🕸️",
    layout="wide"
)

# Title
st.title(" Medical Knowledge Graph QA System")
st.markdown("### Powered by Think on Graph")

# API configuration - read from environment or use default
API_HOST = os.getenv("API_HOST", "localhost")
API_PORT = os.getenv("API_PORT", "8000")
API_URL = f"http://{API_HOST}:{API_PORT}"

# Sidebar
with st.sidebar:
    st.header("About")
    st.info("""
    This system uses **Think on Graph (ToG)** retrieval to answer medical questions by:
    
    1.  Identifying relevant entities
    2.  Exploring knowledge graph paths
    3.  Reasoning about graph structure
    4.  Generating comprehensive answers
    
    **Knowledge Source:** Neo4j Medical KG
    """)
    
    st.markdown("---")
    st.caption(f"**API Endpoint:** {API_URL}")

# Main interface
question = st.text_input(
    " Ask a medical question:",
    placeholder="e.g., What are the symptoms of diabetes?"
)

# Ask button
if st.button(" Get Answer", type="primary"):
    if not question:
        st.warning("Please enter a question.")
    else:
        with st.spinner(" Retrieving from knowledge graph..."):
            try:
                response = requests.post(
                    f"{API_URL}/api/ask",
                    json={"question": question},
                    timeout=120 
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.markdown("---")
                    st.subheader(" Answer")
                    
                    st.success(result["tog_answer"])
                    
                    st.markdown("---")
                    with st.expander(f" Knowledge Triples ({len(result['triples'])} total)", expanded=True):
                        for i, triple in enumerate(result["triples"], 1):
                            st.code(f"{i}. {triple}", language="text")
                else:
                    st.error(f"API Error: {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                st.error(f" Cannot connect to API at {API_URL}. Make sure the server is running.")
            except requests.exceptions.Timeout:
                st.error(f" Request timeout. The query took too long to process.")
            except Exception as e:
                st.error(f" Error: {str(e)}")

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Medical Knowledge Graph QA System | Think on Graph Retrieval</p>
    </div>
    """,
    unsafe_allow_html=True
)