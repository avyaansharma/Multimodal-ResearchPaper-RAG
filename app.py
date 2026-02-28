import streamlit as st
import os
import tempfile
from backend.parser import PDFProcessor
from backend.embedder import MultimodalEmbedder
from backend.vectorstore import VectorStore
from backend.retriever import MultimodalRetriever
from backend.vision_module import VisionModule
from backend.agent_system import ResearchAgentSystem
from backend.paper_analyzer import PaperAnalyzer
from dotenv import load_dotenv

# Page config
st.set_page_config(page_title="Multimodal Research Engine", layout="wide", page_icon="🧠")
load_dotenv()

# Initialize classes (st.cache_resource for performance)
@st.cache_resource
def get_backend():
    processor = PDFProcessor()
    embedder = MultimodalEmbedder()
    vector_store = VectorStore()
    vision = VisionModule()
    agent_system = ResearchAgentSystem()
    paper_analyzer = PaperAnalyzer()
    return processor, embedder, vector_store, vision, agent_system, paper_analyzer

processor, embedder, vector_store, vision, agent_system, paper_analyzer = get_backend()

st.title("🧠 Multimodal Research Intelligence Engine")

# --- UI Layout: Tabs for different modes ---
tab_chat, tab_compare, tab_overview = st.tabs([
    "💬 Interactive Research Chat", 
    "🔍 Research Gap & Comparison (Agentic)", 
    "📄 Paper Overview (Short Notes)"
])

# --- TAB 1: INTERACTIVE RESEARCH CHAT ---
with tab_chat:
    st.markdown("### Ask questions about your research library with visual awareness.")
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📚 Your Paper Library")
        uploaded_files = st.file_uploader("Upload PDFs to index", type="pdf", accept_multiple_files=True, key="chat_uploader")
        
        if st.button("Index Papers", key="index_btn"):
            if uploaded_files:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {uploaded_file.name}...")
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                        
                    data = processor.process_pdf(tmp_path, paper_id=uploaded_file.name.replace(".pdf", ""))
                    
                    # Embed and index
                    status_text.text(f"Embedding {uploaded_file.name}...")
                    text_contents = [c["content"] for c in data["chunks"]]
                    text_metadata = [{"content": c["content"], "paper_id": data["paper_id"], "page_num": c["page_num"]} for c in data["chunks"]]
                    text_embeddings = embedder.embed_text(text_contents)
                    vector_store.add_text_embeddings(text_embeddings, text_metadata)
                    
                    for fig in data["figures"]:
                        img_embedding = embedder.embed_image(fig["image_path"])
                        vector_store.add_image_embeddings(img_embedding, [fig])
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                vector_store.save_indices()
                status_text.text("Done! Indices saved.")
                st.success("Library updated and indexed.")
            else:
                st.warning("Please upload papers to index.")

    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "images" in message:
                for img_path in message["images"]:
                    st.image(img_path)

    if query := st.chat_input("Ask about the papers..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving and analyzing..."):
                retriever = MultimodalRetriever()
                results = retriever.retrieve(query)
                
                vision_analyses = []
                image_paths = []
                for res in results["images"]:
                    img_path = res["metadata"]["image_path"]
                    image_paths.append(img_path)
                    analysis = vision.analyze_image(img_path, query)
                    vision_analyses.append(f"Analysis for Figure on Page {res['metadata']['page_num']}: {analysis}")
                
                text_contexts = [res["metadata"]["content"] for res in results["text_chunks"]]
                final_answer = vision.synthesize_response(query, text_contexts, vision_analyses)
                
                st.markdown(final_answer)
                for img_path in image_paths:
                    st.image(img_path, caption="Retrieved relevant figure")
                    
                st.session_state.messages.append({"role": "assistant", "content": final_answer, "images": image_paths})

# --- TAB 2: RESEARCH COMPARISON MODE ---
with tab_compare:
    st.markdown("### Agentic Research Gap Analysis")
    st.info("Upload two specific papers to find similarities, evolution, and potential research gaps.")
    
    col1, col2 = st.columns(2)
    with col1:
        file_a = st.file_uploader("Upload Base Paper (Reference)", type="pdf", key="file_a")
    with col2:
        file_b = st.file_uploader("Upload New Paper (Comparison)", type="pdf", key="file_b")
    
    if st.button("Run Multi-Agent Analysis") and file_a and file_b:
        with st.spinner("Agents are analyzing the papers... this involves multiple stages of reasoning."):
            # Extract text from both
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_a:
                tmp_a.write(file_a.getvalue())
                text_a = processor.process_pdf(tmp_a.name)["chunks"][0]["content"] # Simple harvest for now
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_b:
                tmp_b.write(file_b.getvalue())
                text_b = processor.process_pdf(tmp_b.name)["chunks"][0]["content"]
            
            # Run LangGraph Agent System
            result = agent_system.run_analysis(text_a, text_b)
            
            st.markdown("---")
            st.markdown("## 📋 Agentic Research Report")
            st.markdown(result["final_report"])
            
            # Additional metadata (collapsible)
            with st.expander("View Stage-by-Stage Reasoning"):
                st.subheader("1. Paper A Extraction")
                st.write(result["paper_a_summary"])
                st.subheader("2. Paper B Extraction")
                st.write(result["paper_b_summary"])
                st.subheader("3. Comparative Analysis")
                st.write(result["comparison"])
                st.subheader("4. Research Gap Scout")
                st.write(result["gaps"])
    elif not (file_a and file_b):
        st.warning("Please upload both papers to start the analysis.")

# --- TAB 3: PAPER OVERVIEW (SHORT NOTES) ---
with tab_overview:
    st.markdown("### 📄 Deep Paper Overview & Analysis")
    st.info("Upload a research paper to generate a structured overview including Topic, Claim, Evidence, Method, Dataset, Limitations, and Citation.")
    
    file_single = st.file_uploader("Upload Paper for Overview", type="pdf", key="file_single")
    
    if st.button("Generate Paper Overview") and file_single:
        with st.spinner("Analyzing paper structure... This involves a non-linear multi-agent feedback loop."):
            # Extract text
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file_single.getvalue())
                # Use processor to get all text
                processed_data = processor.process_pdf(tmp_file.name)
                # Combine all text chunks
                full_text = " ".join([c["content"] for c in processed_data["chunks"]])
            
            # Run LangGraph Paper Analyzer
            try:
                result = paper_analyzer.run_analysis(full_text)
                
                st.markdown("---")
                st.markdown(result["final_report"])
                
                # Success feedback
                st.success("Analysis complete!")
                
                # Option to download as markdown
                st.download_button(
                    label="Download Report as Markdown",
                    data=result["final_report"],
                    file_name=f"{file_single.name.replace('.pdf', '')}_overview.md",
                    mime="text/markdown"
                )
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
                st.info("This might be due to rate limits or PDF format. The system has built-in retries.")
    elif not file_single:
        st.warning("Please upload a paper to start.")
