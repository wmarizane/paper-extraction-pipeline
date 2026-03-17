import requests
import json
import xml.etree.ElementTree as ET
from pathlib import Path

# Default local GROBID server URL
GROBID_URL = "http://localhost:8070"

def check_grobid_server(url: str = GROBID_URL) -> bool:
    """Check if the GROBID server is up and running."""
    try:
        response = requests.get(f"{url}/api/isalive", timeout=5)
        return response.status_code == 200 and response.text.lower() == "true"
    except requests.RequestException:
        return False

def parse_pdf_with_grobid(pdf_path: str, url: str = GROBID_URL) -> str:
    """
    Sends a PDF to GROBID's processFulltextDocument endpoint
    and returns the extracted TEI XML.
    """
    endpoint = f"{url}/api/processFulltextDocument"
    
    with open(pdf_path, 'rb') as pdf_file:
        files = {
            'input': (Path(pdf_path).name, pdf_file, 'application/pdf')
        }
        # TEI XML format is GROBID's standard structured output
        # includeRawAffiliations and teiCoordinates are optional
        data = {
            'generateIDs': 'true',
            'consolidateHeader': '0',
            'consolidateCitations': '0',
            'includeRawCitations': '1'
        }
        
        response = requests.post(endpoint, files=files, data=data)
        
        if response.status_code != 200:
            raise Exception(f"GROBID processing failed with status {response.status_code}: {response.text}")
            
        return response.text

def process_tei_xml(xml_content: str) -> str:
    """
    Basic parser for TEI XML to extract linear text.
    For this initial phase, we just extract paragraphs into a single text block
    to emulate the current .txt input format of the prototype.
    """
    # Parse the XML string
    root = ET.fromstring(xml_content)
    
    # TEI XML namespace
    ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
    
    # Extract Title
    title_elem = root.find('.//tei:titleStmt/tei:title', ns)
    title = title_elem.text if title_elem is not None else "Unknown Title"
    
    # Extract Abstract
    abstract_paras = root.findall('.//tei:profileDesc/tei:abstract/tei:p', ns)
    abstract_text = "\n".join([p.text for p in abstract_paras if p.text])
    
    # Extract Body paragraphs
    body_paras = root.findall('.//tei:text/tei:body//tei:p', ns)
    body_text = "\n\n".join([p.text for p in body_paras if p.text])
    
    # Combine into a simple text representation
    full_text = f"Title: {title}\n\n"
    if abstract_text:
        full_text += f"Abstract:\n{abstract_text}\n\n"
    full_text += f"Body:\n{body_text}"
    
    return full_text

if __name__ == "__main__":
    # Example usage for testing
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pdf_parser.py <path_to_pdf>")
        sys.exit(1)
        
    target_pdf = sys.argv[1]
    
    print("Checking GROBID server...")
    if not check_grobid_server():
        print("Error: Local GROBID server is not running on localhost:8070.")
        print("Please ensure you have started it via Docker.")
        sys.exit(1)
    else:
        print("GROBID server OK.")
        
    print(f"Sending {target_pdf} to GROBID...")
    try:
        xml_result = parse_pdf_with_grobid(target_pdf)
        clean_text = process_tei_xml(xml_result)
        
        print("\n--- Extraction Preview ---")
        print(clean_text[:1000] + "...\n")
        print("--- End of Preview ---")
        
    except Exception as e:
        print(f"Failed: {e}")
