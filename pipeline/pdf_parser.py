import requests
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from config import settings

def check_grobid_server(url: str = None) -> bool:
    """Check if the GROBID server is up and running."""
    target_url = url or settings.grobid_url
    try:
        response = requests.get(f"{target_url}/api/isalive", timeout=5)
        return response.status_code == 200 and response.text.lower() == "true"
    except requests.RequestException:
        return False

def parse_pdf_with_grobid(pdf_path: str, url: str = None) -> str:
    """
    Sends a PDF to GROBID's processFulltextDocument endpoint
    and returns the extracted TEI XML.
    """
    target_url = url or settings.grobid_url
    endpoint = f"{target_url}/api/processFulltextDocument"
    
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


def save_tei_xml(xml_content: str, pdf_path: str, output_dir: str = "tei_xml") -> Path:
    """
    Save raw TEI XML content for future reuse and inspection.

    Args:
        xml_content: Raw TEI XML returned by GROBID
        pdf_path: Source PDF path
        output_dir: Directory where TEI XML files are stored

    Returns:
        Path to the saved TEI XML file
    """
    tei_dir = Path(output_dir)
    tei_dir.mkdir(parents=True, exist_ok=True)

    pdf_stem = Path(pdf_path).stem
    tei_path = tei_dir / f"{pdf_stem}_tei.xml"

    tei_path.write_text(xml_content, encoding="utf-8")
    return tei_path

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
        print(f"Error: GROBID server is not running at {settings.grobid_url}.")
        print("Please ensure you have started it via Docker.")
        sys.exit(1)
    else:
        print("GROBID server OK.")
        
    print(f"Sending {target_pdf} to GROBID...")
    try:
        xml_result = parse_pdf_with_grobid(target_pdf)
        tei_path = save_tei_xml(xml_result, target_pdf)
        clean_text = process_tei_xml(xml_result)
        
        print(f"Saved TEI XML: {tei_path}")
        print("\n--- Extraction Preview ---")
        print(clean_text[:1000] + "...\n")
        print("--- End of Preview ---")
        
    except Exception as e:
        print(f"Failed: {e}")
