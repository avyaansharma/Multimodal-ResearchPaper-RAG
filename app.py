import streamlit as st
import os
import tempfile
from backend.parser import PDFProcessor
from backend.embedder import MultimodalEmbedder
from backend.vectorstore import VectorStore
from backend.retriever import MultimodalRetriever
from backend.vision_module import VisionModule
from dotenv import load_dotenv

# Page config
st.set_page_config(page_title="Multimodal Research Engine", layout="wide")
load_dotenv()

# Initialize classes (st.cache_resource for performance)
@st.cache_resource
def get_backend():
    processor = PDFProcessor()
    embedder = MultimodalEmbedder()
    vector_store = VectorStore()
    vision = VisionModule()
    return processor, embedder, vector_store, vision

processor, embedder, vector_store, vision = get_backend()

st.title("🧠 Multimodal Research Intelligence Engine")
st.markdown("Upload research papers, extract figures, and ask questions with visual awareness.")

# Sidebar for file upload
with st.sidebar:
    st.header("Upload Research Papers")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    
    if st.button("Process Papers"):
        if uploaded_files:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                
                # Save to temp and process
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                    
                data = processor.process_pdf(tmp_path, paper_id=uploaded_file.name.replace(".pdf", ""))
                
                # Embed and index
                status_text.text(f"Embedding {uploaded_file.name}...")
                
                # Text
                text_contents = [c["content"] for c in data["chunks"]]
                text_metadata = [{"content": c["content"], "paper_id": data["paper_id"], "page_num": c["page_num"]} for c in data["chunks"]]
                text_embeddings = embedder.embed_text(text_contents)
                vector_store.add_text_embeddings(text_embeddings, text_metadata)
                
                # Images
                for fig in data["figures"]:
                    img_embedding = embedder.embed_image(fig["image_path"])
                    vector_store.add_image_embeddings(img_embedding, [fig])
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            vector_store.save_indices()
            status_text.text("Done! Indices saved.")
            st.success("Papers processed and indexed successfully.")
        else:
            st.warning("Please upload at least one PDF.")

# Main chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "images" in message:
            for img_path in message["images"]:
                st.image(img_path)

if query := st.chat_input("Ask about the papers (e.g., 'Explain the attention mechanism diagram')"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        # 1. Retrieval
        retriever = MultimodalRetriever()
        results = retriever.retrieve(query)
        
        # 2. Vision analysis for retrieved images
        vision_analyses = []
        image_paths = []
        for res in results["images"]:
            img_path = res["metadata"]["image_path"]
            image_paths.append(img_path)
            analysis = vision.analyze_image(img_path, query)
            vision_analyses.append(f"Analysis for Figure on Page {res['metadata']['page_num']}: {analysis}")
        
        # 3. Final synthesis
        text_contexts = [res["metadata"]["content"] for res in results["text_chunks"]]
        final_answer = vision.synthesize_response(query, text_contexts, vision_analyses)
        
        st.markdown(final_answer)
        for img_path in image_paths:
            st.image(img_path, caption="Retrieved relevant figure")
            
        st.session_state.messages.append({"role": "assistant", "content": final_answer, "images": image_paths})
