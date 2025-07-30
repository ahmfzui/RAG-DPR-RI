import streamlit as st
import os
import time
from pathlib import Path
from datetime import datetime
from chat import RAGProcessor, ChatBotDPRRI

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="ChatBot DPR RI - Asisten Virtual",
    page_icon="🏛️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# CSS Kustom dengan tema DPR (Orange)
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #ea580c 0%, #f97316 50%, #fb923c 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(234, 88, 12, 0.3);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 3rem;
        font-weight: 800;
    }
    
    .main-header p {
        margin: 0rem 0 0 0;
        opacity: 0.9;
        font-size: 1.2rem;
        font-weight: 400;
    }
    
    .stats-card {
        background: #fff7ed;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ea580c;
        margin: 0.5rem 0;
    }
    
    .method-indicator {
        background: #fed7aa;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.8rem;
        color:white;
        font-weight: 500;
    }
    .metric-card {
        background: linear-gradient(135deg, #ea580c 0%, #f97316 50%, #fb923c 100%);
        border-left: 5px solid white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #ea580c 0%, #f97316 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(234, 88, 12, 0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #c2410c 0%, #ea580c 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(234, 88, 12, 0.4);
        color: white !important;
    }
    .stButton > button:focus, 
    .stButton > button:active {
        outline: none;
        color: white !important;
        background: linear-gradient(135deg, #14532d 0%, #22c55e 100%);
        transform: translateY(0);
        box-shadow: 0 4px 12px rgba(34, 197, 94, 0.4);
    }
    
    .sidebar-section {
        background: #fffbeb;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #fed7aa;
    }
    
    .metric-container {
        background: #fff7ed;
        padding: 0.8rem;
        border-radius: 6px;
        text-align: center;
        border: 1px solid #fed7aa;
    }
    
    .chat-input {
        border-radius: 8px;
        border: 2px solid #fed7aa;
    }
    
    .welcome-message {
        background-color: #262730;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- System Initialization ---
@st.cache_resource(show_spinner=False)
def initialize_system():
    """Initialize RAG system components"""
    try:
        processor = RAGProcessor()
        chatbot = ChatBotDPRRI(processor)
        return processor, chatbot, True
    except Exception as e:
        return None, None, str(e)

def get_system():
    """Get system components with initialization"""
    if "system_ready" not in st.session_state:
        with st.spinner("Memuat sistem ChatBot DPR RI..."):
            processor, chatbot, result = initialize_system()
            
            if processor and chatbot:
                st.session_state.system_ready = True
                st.session_state.processor = processor
                st.session_state.chatbot = chatbot
                st.success("Sistem berhasil dimuat", icon="✅")
                time.sleep(0.3)
            else:
                st.error(f"Gagal memuat sistem: {result}")
                st.info("""
                **Langkah troubleshooting:**
                - Pastikan server Qdrant aktif
                - Pastikan Ollama server berjalan
                - Periksa konfigurasi environment
                """)
                st.stop()
    
    return st.session_state.processor, st.session_state.chatbot

# Initialize system
processor, chatbot = get_system()

# --- Cached Stats Function ---
@st.cache_data(ttl=30, show_spinner=False)
def get_cached_stats():
    """Get database statistics with caching"""
    return processor.get_stats()

# --- Header Section ---
st.markdown("""
<div class="main-header">
    <h1>ChatBot DPR RI</h1>
    <p>Asisten Virtual untuk Informasi Legislatif Indonesia</p>
    <small>""" + datetime.now().strftime("%d %B %Y, %H:%M WIB") + """</small>
</div>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/e/e6/DPR_RI_insignia.png", width=120)
    st.title("Panel Kontrol")
    
    # Status sistem
    st.markdown("---")
    if st.button("Perbarui Data", key="refresh_stats", help="Klik untuk memperbarui statistik", use_container_width=True):
        st.cache_data.clear()
    
    stats = get_cached_stats()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="font-size: 0.9rem; color:white;">Dokumen</div>
                <div style="font-size: 1.8rem; color:white; font-weight: bold;">{stats.get("total_documents", 0)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="font-size: 0.9rem; color:white;">Chunk</div>
                <div style="font-size: 1.8rem; font-weight: bold; color:white;">{stats.get("total_points", 0)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("---")

    # Upload Documents Section
    with st.expander("Upload Dokumen", expanded=False):
        st.markdown("""
        **Format didukung:**
        - Dokumen: PDF, DOCX, TXT, JSON
        - Gambar: JPG, PNG, BMP, TIFF

        """)
        
        uploaded_files = st.file_uploader(
            "Pilih file untuk diproses",
            type=["pdf", "docx", "txt", "json", "jpg", "jpeg", "png", "bmp", "tiff"],
            accept_multiple_files=True,
            help="Sistem akan otomatis memproses berbagai format dokumen"
        )
        
        if st.button("Proses Dokumen", use_container_width=True):
            if uploaded_files:
                total_files = len(uploaded_files)
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                temp_dir = Path("./temp_uploads")
                temp_dir.mkdir(exist_ok=True)
                
                processed_count = 0
                skipped_count = 0
                error_count = 0
                
                try:
                    for i, uploaded_file in enumerate(uploaded_files):
                        progress = (i + 1) / total_files
                        progress_bar.progress(progress)
                        status_text.text(f"Memproses: {uploaded_file.name}")
                        
                        file_path = temp_dir / uploaded_file.name
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        try:
                            success = processor.process_document(str(file_path))
                            if success:
                                processed_count += 1
                            else:
                                skipped_count += 1
                        except Exception as e:
                            error_count += 1
                            st.error(f"Error: {uploaded_file.name}")
                
                finally:
                    try:
                        for file_path in temp_dir.glob('*'):
                            os.remove(file_path)
                        os.rmdir(temp_dir)
                    except:
                        pass
                    
                    progress_bar.empty()
                    status_text.empty()
                
                # Show results
                if processed_count > 0:
                    st.success(f"Berhasil memproses {processed_count} dokumen")
                
                if skipped_count > 0:
                    st.info(f"{skipped_count} dokumen sudah ada")
                
                if error_count > 0:
                    st.warning(f"{error_count} dokumen gagal diproses")
                
                if processed_count > 0:
                    st.cache_data.clear()
                    
            else:
                st.warning("Mohon pilih file terlebih dahulu")

    # Database Management
    with st.expander("Kelola Database", expanded=False):
        # Document list management
        if "show_documents" not in st.session_state:
            st.session_state.show_documents = False
        
        if "documents_list" not in st.session_state:
            st.session_state.documents_list = []
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Lihat Dokumen", use_container_width=True):
                with st.spinner("Mengambil daftar..."):
                    st.session_state.documents_list = processor.list_documents()
                    st.session_state.show_documents = True
        
        with col2:
            if st.button("Tutup Daftar", use_container_width=True):
                st.session_state.show_documents = False
                st.session_state.documents_list = []
        
        # Show documents if enabled
        if st.session_state.show_documents and st.session_state.documents_list:
            doc_options = ["Pilih dokumen..."] + [doc['source'] for doc in st.session_state.documents_list]
            selected_doc = st.selectbox(
                "Pilih dokumen untuk dihapus:",
                options=doc_options,
                key="doc_selector"
            )
            
            # Display document info
            for doc in st.session_state.documents_list:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**{doc['source']}**")
                
                with col2:
                    st.caption(f"{doc['chunk_count']} Chunk")
            
            # Delete selected document
            if selected_doc and selected_doc != "Pilih dokumen...":
                if st.button(f"Hapus {selected_doc}", use_container_width=True, type="primary"):
                    with st.spinner(f"Menghapus {selected_doc}..."):
                        success = processor.delete_document(selected_doc)
                    
                    if success:
                        st.success(f"Berhasil menghapus: {selected_doc}")
                        st.session_state.documents_list = processor.list_documents()
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.error(f"Gagal menghapus: {selected_doc}")
        
        elif st.session_state.show_documents and not st.session_state.documents_list:
            st.info("Tidak ada dokumen di database")

    # Settings
    with st.expander("Pengaturan", expanded=False):
        st.markdown("**Delete Chat History:**")
        if st.button("Bersihkan Chat", use_container_width=True):
            st.session_state.messages = []
            st.success("Riwayat chat dihapus")
        st.markdown("---")
        st.markdown("**Informasi Sistem:**")
        st.caption("Model LLM: ")
        st.caption(f"`{chatbot.model_name}`")
        st.caption("Model Embedding: ")
        st.caption(f"`{getattr(processor, 'embedding_model', 'Tidak diketahui')}`")
        st.caption("Model Reranker: ")
        st.caption(
            f"`{chatbot.reranker_model}`" if chatbot.reranker 
            else "Nonaktif"
        )
        st.caption("Server Qdrant: ")
        st.caption(f"`{processor.qdrant_url}`")
        st.caption("OCR Processor: ")
        st.caption(f"`{'Aktif' if processor.ocr_processor else 'Nonaktif'}`")

# --- Chat Section ---

# Welcome message
def get_welcome_message():
    """Generate welcome message"""
    stats = get_cached_stats()
    
    welcome_text = """
    <strong>Selamat datang di ChatBot DPR RI</strong><br><br>
    Saya adalah asisten virtual yang siap membantu Anda mendapatkan informasi seputar <strong>Dewan Perwakilan Rakyat Republik Indonesia</strong>.<br><br>
    Silakan ajukan pertanyaan Anda tentang DPR RI!
    """.strip()
    
    return welcome_text

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": get_welcome_message(),
            "is_welcome": True
        }
    ]

# Display chat history
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message.get("is_welcome", False):
            st.markdown(f'<div class="welcome-message">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(message["content"])
        
        # Show sources if available
        if "sources" in message and message["sources"]:
            unique_sources = {}
            for source in message["sources"]:
                src_name = source['source']
                if src_name not in unique_sources:
                    unique_sources[src_name] = source

            with st.expander(f"Lihat Sumber ({len(unique_sources)} dokumen)"):
                for src_name, source in unique_sources.items():
                    st.info(f"**{src_name}**\n\n> {source['preview']}")

# Chat input
if prompt := st.chat_input("Tanyakan tentang DPR RI..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        response_placeholder = st.empty()
        sources_placeholder = st.empty()
        metadata_placeholder = st.empty()
        
        full_response = ""
        sources_data = []

        try:
            generator = chatbot.ask_stream(prompt)
            
            for part in generator:
                if part["type"] == "status":
                    status_placeholder.info(part['message'])
                
                elif part["type"] == "sources":
                    sources_data = part.get("sources", [])
                    scan_count = len([s for s in sources_data if "OCR" in s.get("method", "")])
                    sources_info = f"Ditemukan {part['count']} hasil relevan"
                    if scan_count > 0:
                        sources_info += f" (termasuk {scan_count} dokumen scan)"
                    status_placeholder.info(sources_info)
                
                elif part["type"] == "answer_start":
                    status_placeholder.info("Menyusun jawaban...")
                
                elif part["type"] == "answer_chunk":
                    full_response += part["chunk"]
                    response_placeholder.markdown(full_response + "▌")
                
                elif part["type"] == "answer_complete":
                    full_response = part["answer"]
                    response_placeholder.markdown(full_response)
                    status_placeholder.empty()
                    
                elif part["type"] == "error":
                    status_placeholder.error(f"Terjadi kesalahan: {part['message']}")
                    break

        except Exception as e:
            st.error(f"Terjadi kesalahan: {str(e)}")
            full_response = "Maaf, terjadi kesalahan saat memproses pertanyaan Anda."

        # Show sources
        if sources_data:
            # Buat daftar sumber unik lebih dulu
            unique_sources = {}
            for source in sources_data:
                src_name = source['source']
                if src_name not in unique_sources:
                    unique_sources[src_name] = source

            with sources_placeholder.expander(f"Lihat Sumber ({len(unique_sources)} dokumen)"):
                for src_name, source in unique_sources.items():
                    st.info(f"**{src_name}**\n\n> {source['preview']}")
        
        # Add to history
        final_message = {
            "role": "assistant", 
            "content": full_response,
        }
        if sources_data:
            final_message["sources"] = sources_data
        
        st.session_state.messages.append(final_message)

# --- Footer ---
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #9ca3af; font-size: 0.85rem; padding: 1rem;">
    <p>Dikembangkan untuk Dewan Perwakilan Rakyat Republik Indonesia</p>
    <p><strong>Dibuat oleh Ahmad Fauzi</strong></p>
</div>
""", unsafe_allow_html=True)