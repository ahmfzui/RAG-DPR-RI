import os
import hashlib
import uuid
import torch
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Iterator
from dataclasses import dataclass
import time
import logging
import sys
import io

from dotenv import load_dotenv

# Document processing
from unstructured.partition.docx import partition_docx
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.text import partition_text
from langchain_community.document_loaders import JSONLoader

# OCR components
import pytesseract
from PIL import Image, ImageEnhance
import pandas as pd
import re
from pdf2image import convert_from_path

# LangChain components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers.cross_encoder import CrossEncoder
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams, Filter, FieldCondition, MatchValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

load_dotenv()
logging.basicConfig(level=logging.WARNING)

# Konfigurasi Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

@dataclass
class SearchResult:
    content: str
    metadata: Dict[str, Any]
    score: float

class EnhancedCleanOCR:
    """Enhanced OCR processor dengan implementasi OCR baru yang lebih optimal - Fixed untuk Windows"""
    
    def __init__(self):
        pass
    
    def post_process_for_rag(self, text: str) -> list[str]:
        """
        Membersihkan dan menstrukturkan teks OCR mentah menjadi daftar unit semantik.
        """
        lines = text.split('\n')
        
        # Kata kunci yang menandakan awal dari butir/bagian baru
        KEYWORDS = ('Dewan', 'Majelis', 'Pemerintah', 'Presiden', 'Badan', 'Anggota', 'Mengingat', 'Menimbang', 'MEMUTUSKAN', 'Menetapkan')
        HEADING_KEYWORDS = ('BAB', 'Pasal')
        
        processed_lines = []
        list_counter = 1
        in_list_section = False
        
        for line in lines:
            line = re.sub(r'^\d+\.\s*', '', line).strip()
            if not line:
                continue

            if line.endswith('dengan:'):
                in_list_section = True
            
            is_new_item = line.startswith(KEYWORDS) or line.startswith(HEADING_KEYWORDS)
            
            if is_new_item:
                if in_list_section and line.startswith(('Dewan', 'Majelis', 'Pemerintah', 'Presiden', 'Badan', 'Anggota')):
                    processed_lines.append(f"{list_counter}. {line}")
                    list_counter += 1
                else:
                    processed_lines.append(line)
            else:
                if processed_lines:
                    if processed_lines[-1].startswith(HEADING_KEYWORDS):
                         processed_lines.append(line)
                    else:
                         processed_lines[-1] = processed_lines[-1] + " " + line
                else:
                    processed_lines.append(line)

        return processed_lines

    def ocr_to_raw_text_from_image(self, image: Image.Image, language: str = 'ind') -> str:
        """
        OCR yang menghasilkan satu blok teks mentah yang terformat dengan baik dari PIL Image.
        """
        try:
            # Convert to grayscale and enhance
            if image.mode != 'L':
                image = image.convert('L')
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2)
            
            data = pytesseract.image_to_data(image, lang=language, output_type=pytesseract.Output.DICT)
            df = pd.DataFrame(data)
            
            df.dropna(subset=['text'], inplace=True)
            df['text'] = df['text'].str.strip()
            df = df[df['text'] != '']
            df = df[df['conf'] != -1]

            for col in ['left', 'top', 'width', 'height']:
                df[col] = pd.to_numeric(df[col])

            grouped = df.groupby(['block_num', 'par_num', 'line_num'])
            lines_df = pd.DataFrame([
                {'y_pos': g['top'].mean(), 'text': ' '.join(g.sort_values(by='left')['text'])} 
                for _, g in grouped
            ])

            sorted_lines = lines_df.sort_values(by='y_pos')
            structured_text = '\n'.join(sorted_lines['text'])
            
            # Ambil daftar unit semantik yang sudah bersih
            processed_list = self.post_process_for_rag(structured_text)
            
            # Gabungkan daftar menjadi satu string dengan pemisah paragraf ganda
            final_text = '\n\n'.join(processed_list)
            
            return final_text
        except Exception as e:
            print(f"Error processing image: {e}")
            return ""

    def ocr_to_raw_text(self, image_path: str, language: str = 'ind') -> str:
        """
        OCR yang menghasilkan satu blok teks mentah yang terformat dengan baik dari file path.
        """
        try:
            image = Image.open(image_path)
            return self.ocr_to_raw_text_from_image(image, language)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return ""
    
    def extract_text_from_image(self, image: Image.Image, language: str = 'ind+eng') -> str:
        """Extract clean text from PIL Image object menggunakan OCR baru - Tanpa temporary file"""
        try:
            # Langsung proses dari PIL Image tanpa menyimpan file temporary
            return self.ocr_to_raw_text_from_image(image, language.split('+')[0])  # Ambil bahasa utama
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return ""
    
    def extract_text_from_image_path(self, image_path: str, language: str = 'ind+eng') -> str:
        """Extract clean text from image file path menggunakan OCR baru"""
        try:
            return self.ocr_to_raw_text(image_path, language.split('+')[0])  # Ambil bahasa utama
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return ""
    
    def extract_text_from_pdf_memory(self, pdf_path: str, dpi: int = 300, language: str = 'ind+eng') -> str:
        """Extract text from PDF by converting to images in memory - Fixed untuk Windows"""
        try:
            print(f"🔄 Converting PDF to images in memory: {os.path.basename(pdf_path)}")
            
            # Convert PDF to images in memory
            pages = convert_from_path(pdf_path, dpi=dpi)
            
            all_text = []
            total_characters = 0
            
            for i, page_image in enumerate(pages, 1):
                print(f"📄 Processing page {i}/{len(pages)} via OCR...")
                
                try:
                    # Extract text langsung dari PIL Image tanpa menyimpan file temporary
                    page_text = self.ocr_to_raw_text_from_image(page_image, language.split('+')[0])
                    
                    if page_text.strip():
                        all_text.append(page_text.strip())
                        char_count = len(page_text)
                        total_characters += char_count
                        print(f"✅ Page {i}: {char_count} characters extracted")
                    else:
                        print(f"⚠️ Page {i}: No text extracted")
                        
                except Exception as page_error:
                    print(f"❌ Page {i}: Error during OCR - {page_error}")
                    continue
            
            # Combine all pages with double newlines for better separation
            full_text = "\n\n".join(all_text)
            print(f"✅ PDF OCR completed: {len(pages)} pages, {total_characters} total characters")
            
            return full_text
            
        except Exception as e:
            print(f"❌ Error processing PDF {pdf_path}: {e}")
            return ""


class RAGProcessor:
    """Sistem RAG untuk ChatBot DPR RI dengan OCR integration"""
    
    def __init__(self):
        # Load configuration
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.collection_name = os.getenv("QDRANT_COLLECTION_NAME", "dpr_knowledge_base")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large-instruct")
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "1024"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200")) 
        
        self.processed_documents = set()
        
        # Initialize OCR processor
        self.ocr_processor = EnhancedCleanOCR()
        
        self._validate_config()
        self._initialize_system()
        self._load_existing_documents()
    
    def _validate_config(self):
        """Validate required configuration"""
        required_vars = ["QDRANT_URL", "QDRANT_API_KEY"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    def _initialize_system(self):
        """Initialize all components"""
        self._setup_embeddings()
        self._setup_qdrant()
        self._setup_vector_store()
        self._setup_text_splitter()
        print("✅ Sistem RAG dengan OCR berhasil diinisialisasi")
    
    def _setup_embeddings(self):
        """Setup embeddings with proper device handling"""
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True}
            )
            
            test_embedding = self.embeddings.embed_query("test")
            self.vector_size = len(test_embedding)
            
            try:
                self.sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
                self.has_sparse = True
            except Exception:
                self.sparse_embeddings = None
                self.has_sparse = False
                
        except Exception as e:
            raise RuntimeError(f"Failed to initialize embeddings: {e}")
    
    def _setup_qdrant(self):
        """Setup Qdrant client and collection"""
        try:
            client_kwargs = {"url": self.qdrant_url, "timeout": 60}
            if self.qdrant_api_key:
                client_kwargs["api_key"] = self.qdrant_api_key
            if "localhost" not in self.qdrant_url and "127.0.0.1" not in self.qdrant_url:
                client_kwargs["prefer_grpc"] = True
                
            self.client = QdrantClient(**client_kwargs)
            collections = self.client.get_collections()
            
            collection_names = [col.name for col in collections.collections]
            if self.collection_name not in collection_names:
                self._create_collection()
            
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Qdrant: {e}")
    
    def _create_collection(self):
        """Create optimized collection"""
        try:
            if self.has_sparse:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={"dense": VectorParams(size=self.vector_size, distance=Distance.COSINE)},
                    sparse_vectors_config={"sparse": SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False)
                    )}
                )
            else:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
                )
            
            index_fields = [
                ("file_hash", models.PayloadSchemaType.KEYWORD),
                ("source", models.PayloadSchemaType.KEYWORD),
                ("extraction_method", models.PayloadSchemaType.KEYWORD)
            ]
            
            for field_name, field_type in index_fields:
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field_name,
                        field_schema=field_type
                    )
                except Exception:
                    pass
            
        except Exception as e:
            raise RuntimeError(f"Failed to create collection: {e}")
    
    def _setup_vector_store(self):
        """Setup LangChain vector store"""
        try:
            if self.has_sparse:
                self.vector_store = QdrantVectorStore(
                    client=self.client,
                    collection_name=self.collection_name,
                    embedding=self.embeddings,
                    sparse_embedding=self.sparse_embeddings,
                    retrieval_mode=RetrievalMode.HYBRID,
                    vector_name="dense",
                    sparse_vector_name="sparse"
                )
            else:
                self.vector_store = QdrantVectorStore(
                    client=self.client,
                    collection_name=self.collection_name,
                    embedding=self.embeddings,
                    retrieval_mode=RetrievalMode.DENSE
                )
        except Exception as e:
            raise RuntimeError(f"Failed to setup vector store: {e}")
    
    def _setup_text_splitter(self):
        """Setup text splitter"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            add_start_index=True,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def _load_existing_documents(self):
        """Load existing document hashes from database"""
        print("🔄 Loading existing documents from database...")
        try:
            collection_info = self.client.get_collection(self.collection_name)
            total_points = collection_info.points_count
            print(f"📊 Total points in database: {total_points}")

            if total_points == 0:
                print("📋 Database kosong - starting fresh")
                self.processed_documents = set()
                return

            scroll_result, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=total_points,
                with_payload=True,
                with_vectors=False
            )
            
            if not scroll_result:
                self.processed_documents = set()
                return

            all_hashes = set()
            for point in scroll_result:
                metadata = point.payload.get('metadata')
                if metadata and isinstance(metadata, dict):
                    file_hash_value = metadata.get('file_hash')
                    if file_hash_value:
                        all_hashes.add(file_hash_value)
            
            self.processed_documents = all_hashes
            print(f"✅ Loaded {len(self.processed_documents)} unique document hashes")

        except Exception as e:
            print(f"❌ Error loading documents: {e}")
            self.processed_documents = set()
    
    def get_file_hash(self, file_path: str) -> str:
        """Generate SHA256 hash for duplicate detection"""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_sha256.update(chunk)
        except Exception:
            hash_sha256.update(os.path.basename(file_path).encode())
            hash_sha256.update(str(os.path.getsize(file_path)).encode())
        return hash_sha256.hexdigest()
    
    def is_document_exists(self, file_hash: str) -> bool:
        """Check if document exists using in-memory set"""
        return file_hash in self.processed_documents
    
    def _can_extract_text_normally(self, file_path: str) -> bool:
        """Check if PDF can be extracted normally (has selectable text)"""
        try:
            # Try to extract text using unstructured
            elements = partition_pdf(filename=file_path)
            text = "\n\n".join([str(el) for el in elements])
            
            # If we get substantial text, it's not a scanned PDF
            words = len(text.split())
            if words > 50:  # Threshold for meaningful text
                return True
            return False
        except:
            return False
    
    def _extract_text_from_json(self, file_path: str) -> str:
        """Extract text from JSON files"""
        strategies = [
            {"jq_schema": ".content // .text // .message // .description // .body"},
            {"jq_schema": ".messages[]?.content // .messages[]?.text"},
            {"jq_schema": "."}
        ]
        
        for strategy in strategies:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    if not first_line:
                        continue
                    
                    f.seek(0)
                    second_line = f.readline().strip()
                    
                    is_jsonl = False
                    try:
                        json.loads(first_line)
                        if second_line:
                            f.readline()
                            second_line = f.readline().strip()
                            if second_line and json.loads(second_line):
                                is_jsonl = True
                    except:
                        pass
                    f.seek(0)
                
                loader = JSONLoader(
                    file_path=file_path,
                    jq_schema=strategy["jq_schema"],
                    text_content=False,
                    json_lines=is_jsonl
                )
                
                docs = loader.load()
                if docs:
                    text = "\n\n".join([doc.page_content for doc in docs if doc.page_content.strip()])
                    if text.strip():
                        return text
            except Exception:
                continue
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                try:
                    data = json.loads(content)
                    return json.dumps(data, indent=2, ensure_ascii=False)
                except:
                    return content
        except Exception as e:
            raise RuntimeError(f"Failed to extract JSON: {e}")
    
    def extract_text(self, file_path: str) -> tuple[str, str]:
        """Extract text from various file formats, returns (text, extraction_method)"""
        file_ext = Path(file_path).suffix.lower()
        filename = os.path.basename(file_path)
        
        try:
            if file_ext == '.docx':
                elements = partition_docx(filename=file_path)
                text = "\n\n".join([str(el) for el in elements])
                return text, "unstructured"
                
            elif file_ext == '.pdf':
                # First try normal extraction
                if self._can_extract_text_normally(file_path):
                    print(f"📄 {filename}: Using normal text extraction")
                    elements = partition_pdf(filename=file_path)
                    text = "\n\n".join([str(el) for el in elements])
                    return text, "unstructured"
                else:
                    # Use OCR for scanned PDFs
                    print(f"🔍 {filename}: Detected scanned PDF, using enhanced OCR...")
                    text = self.ocr_processor.extract_text_from_pdf_memory(file_path)
                    
                    # Add document info comment at the start
                    if text.strip():
                        doc_info = f"Dokumen: {filename}\nDiekstrak menggunakan Enhanced OCR v2\n\n"
                        text = doc_info + text
                    
                    return text, "ocr"
                    
            elif file_ext == '.txt':
                elements = partition_text(filename=file_path)
                text = "\n\n".join([str(el) for el in elements])
                return text, "unstructured"
                
            elif file_ext == '.json':
                text = self._extract_text_from_json(file_path)
                return text, "json"
                
            elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                # Image files - use enhanced OCR
                print(f"🖼️ {filename}: Using enhanced OCR for image...")
                text = self.ocr_processor.extract_text_from_image_path(file_path)
                
                # Add image info comment at the start
                if text.strip():
                    img_info = f"Gambar: {filename}\nDiekstrak menggunakan Enhanced OCR v2\n\n"
                    text = img_info + text
                
                return text, "ocr"
                
            else:
                raise ValueError(f"Unsupported format: {file_ext}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to extract text from {file_path}: {e}")
    
    def process_document(self, file_path: str, category: str = "general") -> bool:
        """Process single document with enhanced OCR support"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            filename = os.path.basename(file_path)
            file_hash = self.get_file_hash(file_path)
            
            if self.is_document_exists(file_hash):
                print(f"⏭️ Skipping duplicate: {filename}")
                return False
            
            # Extract text with method information
            text_data, extraction_method = self.extract_text(file_path)
            
            if not text_data.strip():
                print(f"⚠️ No text extracted from: {filename}")
                return False
            
            metadata = {
                "source": filename,
                "file_hash": file_hash,
                "extraction_method": extraction_method,
                "category": category,
                "processed_at": datetime.utcnow().isoformat()
            }
            
            chunks = self.text_splitter.create_documents([text_data])
            if not chunks:
                print(f"⚠️ No chunks created from: {filename}")
                return False
            
            documents = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chunk_index": i,
                    "chunk_id": f"{file_hash}_{i}"
                })
                
                documents.append(Document(
                    page_content=chunk.page_content.strip(),
                    metadata=chunk_metadata
                ))
            
            ids = [str(uuid.uuid4()) for _ in range(len(documents))]
            self.vector_store.add_documents(documents=documents, ids=ids)
            
            self.processed_documents.add(file_hash)
            
            method_emoji = "🔍" if extraction_method == "ocr" else "📄"
            print(f"✅ {method_emoji} Processed: {filename} ({len(chunks)} chunks, method: {extraction_method})")
            return True
            
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            return False
    
    def process_folder(self, folder_path: str, category: str = "general") -> Dict[str, int]:
        """Process folder with enhanced OCR support"""
        supported_formats = {'.docx', '.pdf', '.txt', '.json', '.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        stats = {"processed": 0, "skipped": 0, "errors": 0, "ocr_processed": 0}
        
        folder = Path(folder_path)
        if not folder.exists():
            print(f"📁 Folder '{folder_path}' tidak ditemukan")
            return stats
        
        files = [f for f in folder.rglob("*") if f.suffix.lower() in supported_formats and f.is_file()]
        
        if not files:
            print(f"📁 Tidak ada file yang didukung di '{folder_path}'")
            return stats
        
        print(f"📂 Processing {len(files)} files (including images and scanned PDFs with enhanced OCR)...")
        
        for file_path in files:
            try:
                file_ext = file_path.suffix.lower()
                
                # Track OCR usage
                is_image = file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
                is_pdf = file_ext == '.pdf'
                
                success = self.process_document(str(file_path), category)
                
                if success:
                    stats["processed"] += 1
                    # Check if OCR was used (for images or scanned PDFs)
                    if is_image or (is_pdf and not self._can_extract_text_normally(str(file_path))):
                        stats["ocr_processed"] += 1
                else:
                    stats["skipped"] += 1
                    
            except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")
                stats["errors"] += 1
        
        return stats
    
    def get_retriever(self, k: int = 10, score_threshold: float = 0.1):
        """Get retriever for RAG"""
        return self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": k, "score_threshold": score_threshold}
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics with enhanced OCR info"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            total_points = collection_info.points_count
            
            if total_points == 0:
                return {
                    "total_points": 0, "total_documents": 0,
                    "documents": {}, "collection_status": "empty",
                    "has_sparse": self.has_sparse,
                    "processed_hashes": len(self.processed_documents),
                    "ocr_enabled": True,
                    "ocr_version": "Enhanced OCR v2 - Windows Fixed"
                }
            
            # Get extraction method stats
            scroll_result, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=total_points,
                with_payload=True,
                with_vectors=False
            )
            
            extraction_methods = {}
            if scroll_result:
                for point in scroll_result:
                    metadata = point.payload.get('metadata', {})
                    method = metadata.get('extraction_method', 'unknown')
                    extraction_methods[method] = extraction_methods.get(method, 0) + 1
            
            total_documents = len(self.processed_documents)

            return {
                "total_points": total_points,
                "total_documents": total_documents,
                "extraction_methods": extraction_methods,
                "collection_status": collection_info.status.value if hasattr(collection_info.status, 'value') else str(collection_info.status),
                "has_sparse": self.has_sparse,
                "processed_hashes": len(self.processed_documents),
                "ocr_enabled": True,
                "ocr_version": "Enhanced OCR v2 - Windows Fixed"
            }
            
        except Exception as e:
            print(f"❌ Error getting stats: {e}")
            return {
                "total_points": 0, "total_documents": 0,
                "documents": {}, "collection_status": "error",
                "has_sparse": self.has_sparse,
                "processed_hashes": len(self.processed_documents),
                "ocr_enabled": True,
                "ocr_version": "Enhanced OCR v2 - Windows Fixed"
            }
    
    def delete_document(self, filename: str) -> bool:
        """Delete document by filename - Fixed to match actual metadata structure"""
        try:
            print(f"🔍 Searching for document: {filename}")
            
            # First, let's find the document using the same structure as _load_existing_documents
            scroll_result, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,  # Get all points to search through
                with_payload=True,
                with_vectors=False
            )
            
            if not scroll_result:
                print(f"❌ No documents found in database")
                return False
            
            # Find points that match the filename
            matching_points = []
            file_hash_to_remove = None
            
            for point in scroll_result:
                # Check the metadata structure like in _load_existing_documents
                metadata = point.payload.get('metadata')
                if metadata and isinstance(metadata, dict):
                    source_file = metadata.get('source')
                    if source_file == filename:
                        matching_points.append(point)
                        # Get the file hash for processed_documents cleanup
                        if not file_hash_to_remove:
                            file_hash_to_remove = metadata.get('file_hash')
            
            if not matching_points:
                print(f"❌ Document '{filename}' not found in database")
                return False
            
            print(f"🔍 Found {len(matching_points)} chunks for '{filename}'")
            
            # Extract point IDs for deletion
            point_ids = [point.id for point in matching_points]
            
            # Delete the points
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=point_ids)
            )
            
            # Remove from processed documents set if hash is found
            if file_hash_to_remove and file_hash_to_remove in self.processed_documents:
                self.processed_documents.remove(file_hash_to_remove)
                print(f"🗑️ Removed hash from processed_documents: {file_hash_to_remove[:16]}...")
            
            print(f"✅ Successfully deleted: {filename} ({len(point_ids)} chunks)")
            return True
            
        except Exception as e:
            print(f"❌ Error deleting {filename}: {e}")
            return False

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the database with their metadata"""
        try:
            print("📋 Listing all documents...")
            
            scroll_result, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,
                with_payload=True,
                with_vectors=False
            )
            
            if not scroll_result:
                return []
            
            # Group by source file
            documents = {}
            
            for point in scroll_result:
                metadata = point.payload.get('metadata')
                if metadata and isinstance(metadata, dict):
                    source = metadata.get('source')
                    if source:
                        if source not in documents:
                            documents[source] = {
                                'source': source,
                                'file_hash': metadata.get('file_hash'),
                                'extraction_method': metadata.get('extraction_method'),
                                'category': metadata.get('category'),
                                'processed_at': metadata.get('processed_at'),
                                'chunk_count': 0
                            }
                        documents[source]['chunk_count'] += 1
            
            return list(documents.values())
            
        except Exception as e:
            print(f"❌ Error listing documents: {e}")
            return []


class ChatBotDPRRI:
    """ChatBot DPR RI - Asisten Virtual untuk Informasi Legislatif dengan Enhanced OCR"""
    
    def __init__(self, processor: RAGProcessor):
        self.processor = processor
        self.model_name = os.getenv("OLLAMA_MODEL", "gemma3:4b")
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
        self.reranker_model = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-large")
        
        self._setup_llm()
        self._setup_reranker()
        self._setup_chain()
    
    def _setup_llm(self):
        """Setup Ollama LLM"""
        try:
            self.llm = OllamaLLM(
                model=self.model_name,
                base_url=self.base_url,
                temperature=self.temperature
            )
            self.llm.invoke("Hi")
        except Exception as e:
            raise RuntimeError(f"Failed to setup LLM: {e}")
    
    def _setup_reranker(self):
        """Setup reranker model"""
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.reranker = CrossEncoder(
                self.reranker_model,
                device=device,
                trust_remote_code=True
            )
            print(f"✅ Reranker model loaded successfully: {self.reranker_model}")
        except Exception as e:
            print(f"⚠️ Warning: Failed to setup reranker ({self.reranker_model}): {e}")
            self.reranker = None
    
    def _rerank_documents(self, question: str, docs: List[Document], top_k: int = 5, batch_size: int = 32) -> List[Document]:
        """Rerank retrieved documents using configured reranker model"""
        if not docs or not self.reranker:
            return docs[:top_k]

        try:
            pairs = [[question, doc.page_content] for doc in docs]

            all_scores = []
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]
                batch_scores = self.reranker.predict(batch_pairs)
                all_scores.extend(batch_scores)

            scored_docs = [(doc, score) for doc, score in zip(docs, all_scores)]
            scored_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
            reranked_docs = [doc for doc, _ in scored_docs[:top_k]]

            return reranked_docs

        except Exception as e:
            logging.warning(f"Reranking failed: {e}. Returning original documents.")
            return docs[:top_k]
    
    def _setup_chain(self):
        """Setup LangChain chain"""
        prompt_template = """Kamu adalah Asisten Virtual DPR RI yang bertugas membantu masyarakat Indonesia memahami informasi tentang lembaga legislatif, yaitu DPR RI.

KARAKTER & PERAN ASISTEN:
- Merupakan chatbot resmi yang ramah, informatif, dan mudah dipahami, berbasis RAG (Retrieval-Augmented Generation) dengan dukungan Enhanced OCR v2 untuk dokumen scan.
- Fokus memberikan informasi akurat, detail, lengkap, dan relevan tentang DPR RI kepada masyarakat, sesuai dokumen resmi yang tersedia.
- Menggunakan Bahasa Indonesia yang formal, hangat, dan mudah dimengerti oleh publik.

PANDUAN DALAM MENJAWAB:
1. WAJIB menggunakan hanya informasi dari konteks yang tersedia. Tidak boleh mengarang atau menambahkan informasi lain.
2. Pilih dan cocokkan informasi dari konteks sesuai dengan pertanyaan yang diajukan.
3. Jika informasi tidak tersedia atau tidak ditemukan, sampaikan dengan sopan bahwa informasi tersebut belum tersedia.
4. JANGAN menyebutkan istilah teknis seperti "dokumen", "file", "JSON", "OCR", dsb.
5. JANGAN menjelaskan proses pencarian atau cara kerja sistem.
6. Fokus hanya pada substansi informasi yang ditanyakan.
7. Gunakan format rapi dan jelas, seperti bullet points atau penomoran jika diperlukan.

GAYA PENYAMPAIAN:
- Gunakan sapaan yang sopan dan ramah.
- Jangan menggunakan kata sambutan seperti "Assalamualaikum" atau "Selamat pagi/siang/sore".
- Hindari kalimat teknis seperti "Berdasarkan konteks yang diberikan".
- Jelaskan dengan bahasa yang mudah dipahami oleh masyarakat awam.

INFORMASI KONTEKS RAG:
{context}

PERTANYAAN DARI MASYARAKAT:
{question}

JAWABAN (dalam Bahasa Indonesia yang lengkap, jelas, dan informatif):"""
        
        self.prompt = ChatPromptTemplate.from_template(prompt_template)
        self.retriever = self.processor.get_retriever(k=10, score_threshold=0.1)
        
        self.chain = (
            {
                "context": self.retriever | self._format_docs,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _format_docs(self, docs):
        """Format retrieved documents"""
        if not docs:
            return "Tidak ada informasi relevan ditemukan."
        
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            method = doc.metadata.get('extraction_method', 'unknown')
            method_indicator = "🔍" if method == "ocr" else "📄"
            context_parts.append(f"DOKUMEN {i} ({method_indicator} {source}):\n{doc.page_content.strip()}")
        
        return "\n\n".join(context_parts)
    
    def ask_stream(self, question: str) -> Iterator[Dict[str, Any]]:
        """Ask question with streaming response and reranking"""
        try:
            start_time = time.time()
            
            yield {
                "type": "status",
                "message": "🔍 Mencari informasi relevan...",
                "stage": "retrieval"
            }
            
            context_docs = self.retriever.invoke(question)
            
            if self.reranker:
                yield {
                    "type": "status",
                    "message": f"⚡ Mengurutkan ulang dokumen...",
                    "stage": "reranking"
                }
                context_docs = self._rerank_documents(question, context_docs, top_k=5)
            else:
                context_docs = context_docs[:5]
            
            sources = []
            for doc in context_docs:
                method = doc.metadata.get('extraction_method', 'unknown')
                method_indicator = "🔍 Enhanced OCR v2" if method == "ocr" else "📄 Text"
                sources.append({
                    "source": doc.metadata.get('source', 'Unknown'),
                    "method": method_indicator,
                    "preview": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                })
            
            yield {
                "type": "sources",
                "sources": sources,
                "count": len(sources)
            }
            
            yield {
                "type": "status",
                "message": "🧠 Menghasilkan jawaban...",
                "stage": "generation"
            }
            
            context = self._format_docs(context_docs)
            formatted_prompt = self.prompt.format(context=context, question=question)
            
            llm_stream = OllamaLLM(
                model=self.model_name,
                base_url=self.base_url,
                temperature=self.temperature
            )
            
            yield {
                "type": "answer_start"
            }
            
            full_response = ""
            try:
                for chunk in llm_stream.stream(formatted_prompt):
                    if chunk:
                        full_response += chunk
                        yield {
                            "type": "answer_chunk",
                            "chunk": chunk
                        }
            except Exception:
                full_response = llm_stream.invoke(formatted_prompt)
                yield {
                    "type": "answer_chunk",
                    "chunk": full_response
                }
            
            yield {
                "type": "answer_complete",
                "answer": full_response.strip(),
                "metadata": {
                    "total_time": time.time() - start_time,
                    "sources_found": len(sources),
                    "question": question,
                    "timestamp": datetime.utcnow().isoformat(),
                    "reranker_used": self.reranker_model if self.reranker else None,
                    "ocr_sources": len([s for s in sources if "Enhanced OCR v2" in s["method"]]),
                    "ocr_version": "Enhanced OCR v2 - Windows Fixed"
                }
            }
            
        except Exception as e:
            yield {
                "type": "error",
                "error": str(e),
                "message": f"Maaf, terjadi kesalahan: {str(e)}"
            }
    
    def ask(self, question: str) -> Dict[str, Any]:
        """Non-streaming version for backward compatibility with reranking"""
        try:
            start_time = time.time()
            
            context_docs = self.retriever.invoke(question)
            
            if self.reranker:
                context_docs = self._rerank_documents(question, context_docs, top_k=5)
            else:
                context_docs = context_docs[:5]
                
            answer = self.chain.invoke({"context": self._format_docs(context_docs), "question": question})
            
            sources = []
            for doc in context_docs:
                method = doc.metadata.get('extraction_method', 'unknown')
                method_indicator = "🔍 Enhanced OCR v2" if method == "ocr" else "📄 Text"
                sources.append({
                    "source": doc.metadata.get('source', 'Unknown'),
                    "method": method_indicator,
                    "preview": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                })
            
            return {
                "answer": answer.strip(),
                "sources": sources,
                "metadata": {
                    "total_time": time.time() - start_time,
                    "sources_found": len(sources),
                    "question": question,
                    "timestamp": datetime.utcnow().isoformat(),
                    "reranker_used": self.reranker_model if self.reranker else None,
                    "ocr_sources": len([s for s in sources if "Enhanced OCR v2" in s["method"]]),
                    "ocr_version": "Enhanced OCR v2 - Windows Fixed"
                }
            }
        except Exception as e:
            return {
                "answer": f"Maaf, terjadi kesalahan: {str(e)}",
                "sources": [],
                "metadata": {"error": str(e)}
            }
    
    def chat(self):
        """Interactive chat interface with Enhanced OCR support"""
        history = []
        
        print("="*60)
        print("🤖 ChatBot | Dewan Perwakilan Rakyat Republik Indonesia")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💬 Model: {self.model_name}")
        print(f"🔄 Reranker: {self.reranker_model if self.reranker else 'Disabled'}")
        print(f"🔍 OCR: Enhanced OCR v2 - Windows Fixed")
        print("="*60)
        print("Perintah: help | stats | history | clear | delete | quit")
        print("✨ Mode streaming - jawaban real-time dengan reranking dan Enhanced OCR")
        print("="*60)
    
        while True:
            try:
                user_input = input(f"\n😄 Anda: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print(f"\n👋 Terima kasih telah menggunakan ChatBot DPR RI!")
                    print(f"📊 Total pertanyaan: {len(history)}")
                    break
                
                elif user_input.lower() == 'help':
                    stats = self.processor.get_stats()
                    print(f"\n📖 INFORMASI SISTEM:")
                    print(f"  📚 Dokumen: {stats.get('total_documents', 0)}")
                    print(f"  📝 Total chunk: {stats.get('total_points', 0)}")
                    print(f"  🔍 Mode pencarian: {'Hybrid' if stats.get('has_sparse') else 'Dense'}")
                    print(f"  ✨ Reranking: {self.reranker_model if self.reranker else 'Disabled'}")
                    print(f"  🔍 OCR: {stats.get('ocr_version', 'Enhanced OCR v2 - Windows Fixed')}")
                    print(f"  ✨ Streaming: Aktif")
                    print(f"  🔄 Hash tracking: {stats.get('processed_hashes', 0)} dokumen")
                    
                    # Show extraction methods
                    extraction_methods = stats.get('extraction_methods', {})
                    if extraction_methods:
                        print(f"  📊 Metode ekstraksi:")
                        for method, count in extraction_methods.items():
                            icon = "🔍" if method == "ocr" else "📄"
                            print(f"    {icon} {method}: {count} chunk")
                    
                    print(f"\n💡 Tips:")
                    print(f"  • Tanyakan tentang fungsi, tugas, atau prosedur DPR RI")
                    print(f"  • Sistem mendukung dokumen scan dan gambar dengan OCR terbaru")
                    print(f"  • OCR telah diperbaiki untuk Windows")
                    continue
                
                elif user_input.lower() == 'stats':
                    stats = self.processor.get_stats()
                    print(f"\n📊 STATISTIK DATABASE:")
                    print(f"  📚 Total dokumen: {stats.get('total_documents', 0)}")
                    print(f"  📝 Total chunk: {stats.get('total_points', 0)}")
                    print(f"  🔄 Hash tracking: {stats.get('processed_hashes', 0)} dokumen")
                    print(f"  🔄 Reranker: {self.reranker_model if self.reranker else 'Disabled'}")
                    print(f"  🔍 OCR: {stats.get('ocr_version', 'Enhanced OCR v2 - Windows Fixed')}")
                    
                    # Show extraction methods breakdown
                    extraction_methods = stats.get('extraction_methods', {})
                    if extraction_methods:
                        print(f"  📊 Breakdown metode ekstraksi:")
                        for method, count in extraction_methods.items():
                            icon = "🔍" if method == "ocr" else "📄"
                            print(f"    {icon} {method}: {count} chunk")
                    continue
                
                elif user_input.lower() == 'history':
                    if history:
                        print(f"\n📚 RIWAYAT PERCAKAPAN (5 terakhir):")
                        for i, entry in enumerate(history[-5:], 1):
                            time_str = entry['metadata']['timestamp'].split('T')[1][:8]
                            ocr_count = entry['metadata'].get('ocr_sources', 0)
                            ocr_indicator = f" 🔍{ocr_count}" if ocr_count > 0 else ""
                            print(f"  {i}. [{time_str}]{ocr_indicator} {entry['question'][:50]}...")
                    else:
                        print("\n📝 Belum ada riwayat percakapan")
                    continue
                
                elif user_input.lower() == 'clear':
                    history.clear()
                    print("🗑️ Riwayat percakapan dihapus")
                    continue
                
                elif user_input.lower() == 'delete':
                    filename = input("📁 Nama file yang akan dihapus: ").strip()
                    if filename:
                        success = self.processor.delete_document(filename)
                        print(f"{'✅ Berhasil' if success else '❌ Gagal'} menghapus '{filename}'")
                    continue
                
                elif not user_input:
                    continue
                
                print(f"\n🤖 ChatBot DPR RI:")
                
                sources = []
                answer = ""
                metadata = {}
                current_status = ""
                
                for chunk in self.ask_stream(user_input):
                    if chunk["type"] == "status":
                        current_status = chunk["message"]
                        print(f"\r{current_status}", end='', flush=True)
                    
                    elif chunk["type"] == "sources":
                        sources = chunk["sources"]
                        ocr_count = len([s for s in sources if "Enhanced OCR v2" in s["method"]])
                        ocr_info = f" (termasuk {ocr_count} hasil Enhanced OCR v2)" if ocr_count > 0 else ""
                        print(f"\r🔍 Ditemukan {chunk['count']} potongan dari sumber relevan{ocr_info}")
                        print()
                    
                    elif chunk["type"] == "answer_start":
                        if current_status:
                            print(f"\r{' ' * len(current_status)}\r", end='')
                    
                    elif chunk["type"] == "answer_chunk":
                        print(chunk["chunk"], end='', flush=True)
                        answer += chunk["chunk"]
                    
                    elif chunk["type"] == "answer_complete":
                        answer = chunk["answer"]
                        metadata = chunk["metadata"]
                        print()
                    
                    elif chunk["type"] == "error":
                        print(f"\r❌ {chunk['message']}")
                        break
                
                if sources:
                    unique_sources = []
                    seen = set()
                    for item in sources:    
                        src_name = item['source'] 
                        if src_name not in seen: 
                            seen.add(src_name)
                            unique_sources.append((src_name, item.get('method', '📄 Text')))
                        if len(unique_sources) == 5:
                            break
                    
                    print(f"\n📚 Daftar Sumber ({len(unique_sources)} dokumen):")
                    for i, (src_name, method) in enumerate(unique_sources, 1):
                        print(f"  {i}. {method} {src_name}")
                
                if metadata:
                    total_time = metadata.get('total_time', 0)
                    reranker_used = metadata.get('reranker_used')
                    ocr_sources = metadata.get('ocr_sources', 0)
                    ocr_version = metadata.get('ocr_version')
                    print(f"\n⌛ Waktu: {total_time:.1f} detik")
                    if reranker_used:
                        print(f"🔄 Reranker: {reranker_used}")
                    if ocr_sources > 0:
                        print(f"🔍 OCR sources: {ocr_sources} ({ocr_version})")
                    print("=" * 60)
                    
                    history.append({
                        "question": user_input,
                        "answer": answer,
                        "metadata": metadata
                    })
                    
                    if len(history) > 20:
                        history = history[-20:]
                
            except KeyboardInterrupt:
                print("\n\n⛔ Sesi dihentikan oleh pengguna")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")


def main():
    print("\n🚀 CHATBOT DPR RI - SISTEM RAG DENGAN ENHANCED OCR V2 (WINDOWS FIXED) DAN RERANKING")
    print("="*60)
    
    try:
        print("🔧 Menginisialisasi sistem...")
        processor = RAGProcessor()
        
        folder_path = "External Documents"
        if os.path.exists(folder_path):
            print(f"📂 Memproses dokumen dari: {folder_path}")
            stats = processor.process_folder(folder_path)
            
            if stats["processed"] > 0:
                print(f"✅ {stats['processed']} dokumen baru diproses")
            if stats["ocr_processed"] > 0:
                print(f"🔍 {stats['ocr_processed']} dokumen menggunakan Enhanced OCR v2 (Windows Fixed)")
            if stats["skipped"] > 0:
                print(f"⏭️ {stats['skipped']} dokumen sudah ada (duplikat dihindari)")
            if stats["errors"] > 0:
                print(f"⚠️ {stats['errors']} dokumen gagal diproses")
        else:
            print(f"📁 Folder '{folder_path}' tidak ditemukan")
            os.makedirs(folder_path, exist_ok=True)
            print(f"📁 Folder '{folder_path}' dibuat - silakan tambahkan dokumen")
        
        db_stats = processor.get_stats()
        print(f"\n📊 DATABASE SIAP:")
        print(f"  📚 Dokumen: {db_stats.get('total_documents', 0)}")
        print(f"  📝 Chunk: {db_stats.get('total_points', 0)}")
        print(f"  🔍 Mode: {'Hybrid' if db_stats.get('has_sparse') else 'Dense'}")
        print(f"  🔄 Hash tracking: {db_stats.get('processed_hashes', 0)} dokumen")
        print(f"  🔍 OCR: {db_stats.get('ocr_version', 'Enhanced OCR v2 - Windows Fixed')}")
        
        # Show extraction methods
        extraction_methods = db_stats.get('extraction_methods', {})
        if extraction_methods:
            print(f"  📊 Metode ekstraksi:")
            for method, count in extraction_methods.items():
                icon = "🔍" if method == "ocr" else "📄"
                print(f"    {icon} {method}: {count} chunk")
        
        print("\n🚀 Memulai ChatBot DPR RI...")
        chatbot = ChatBotDPRRI(processor)
        chatbot.chat()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🛠️ Troubleshooting:")
        print("  1. Pastikan Qdrant server berjalan")
        print("  2. Periksa konfigurasi .env")
        print("  3. Jalankan: ollama serve")
        print("  4. Periksa model: ollama list")
        print("  5. Install Tesseract OCR jika belum ada")

if __name__ == "__main__":
    main()