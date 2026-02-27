import fitz  # PyMuPDF
import os
import json
import uuid
from PIL import Image
import io

class PDFProcessor:
    def __init__(self, storage_dir="data"):
        self.storage_dir = storage_dir
        self.raw_pdfs_dir = os.path.join(storage_dir, "raw_pdfs")
        self.extracted_images_dir = os.path.join(storage_dir, "extracted_images")
        
        os.makedirs(self.raw_pdfs_dir, exist_ok=True)
        os.makedirs(self.extracted_images_dir, exist_ok=True)

    def process_pdf(self, pdf_path, paper_id=None):
        if not paper_id:
            paper_id = os.path.splitext(os.path.basename(pdf_path))[0]
            
        doc = fitz.open(pdf_path)
        
        structured_data = {
            "paper_id": paper_id,
            "sections": [],
            "chunks": [],
            "figures": []
        }
        
        current_section = "Introduction"
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("dict")
            
            # Simple heuristic for section detection (larger font size, bold)
            # This is a baseline and can be improved with layout analysis libraries
            page_text = page.get_text()
            lines = page_text.split('\n')
            
            # Extract Text Chunks
            # For now, we chunk by page, but future implementation can be more granular
            chunk_data = {
                "chunk_id": str(uuid.uuid4()),
                "page_num": page_num + 1,
                "section": current_section,
                "content": page_text
            }
            structured_data["chunks"].append(chunk_data)
            
            # Extract Images
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                image_filename = f"{paper_id}_p{page_num+1}_img{img_index}.{image_ext}"
                image_path = os.path.join(self.extracted_images_dir, image_filename)
                
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                
                # Figure caption heuristic: text below the image
                # In PyMuPDF, we can get the image bounding box
                img_rects = page.get_image_rects(xref)
                caption = "Caption not detected"
                if img_rects:
                    # Look for text near the image rectangle
                    rect = img_rects[0]
                    # Search area slightly below the image
                    search_area = fitz.Rect(rect.x0, rect.y1, rect.x1, rect.y1 + 100)
                    caption_text = page.get_text("text", clip=search_area).strip()
                    if caption_text:
                        caption = caption_text
                
                figure_data = {
                    "figure_id": f"Fig_{xref}",
                    "caption": caption,
                    "image_path": image_path,
                    "page_num": page_num + 1,
                    "referenced_in": [] # To be populated by cross-referencing text
                }
                structured_data["figures"].append(figure_data)
                
        # Link figures to references in text
        self._link_figures_to_text(structured_data)
        
        return structured_data

    def _link_figures_to_text(self, data):
        # Basic regex or string matching to find "Fig. X" or "Figure X" in chunks
        import re
        for figure in data["figures"]:
            # This is a placeholder for more robust pattern matching
            fig_num_match = re.search(r'(Fig\.|Figure)\s*(\d+)', figure["caption"], re.IGNORECASE)
            if fig_num_match:
                fig_label = fig_num_match.group(0)
                for chunk in data["chunks"]:
                    if fig_label in chunk["content"]:
                        figure["referenced_in"].append(chunk["chunk_id"])

if __name__ == "__main__":
    # Test stub
    processor = PDFProcessor()
    # Usage: processor.process_pdf("path/to/paper.pdf")
