import hashlib
import io
import os
import re
from pathlib import Path

import faiss
import gdown
import numpy as np
import streamlit as st
from docx import Document
from google import genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# -----------------------------
# App settings
# -----------------------------
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📚",
    layout="wide",
)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 800
DEFAULT_OVERLAP = 120
DEFAULT_TOP_K = 5

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "i", "in", "is", "it", "of", "on", "or", "that", "the",
    "this", "to", "was", "what", "when", "where", "which", "who",
    "why", "with", "you", "your", "about", "does", "do", "can"
}


# -----------------------------
# Document extraction
# -----------------------------
def extract_pdf(file_bytes, filename):
    records = []
    reader = PdfReader(io.BytesIO(file_bytes))

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            records.append(
                {
                    "text": text.strip(),
                    "filename": filename,
                    "page": page_number,
                }
            )

    return records


def extract_docx(file_bytes, filename):
    document = Document(io.BytesIO(file_bytes))
    text = "\n".join(
        paragraph.text.strip()
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    )

    if not text.strip():
        return []

    return [{"text": text, "filename": filename, "page": None}]


def extract_txt(file_bytes, filename):
    text = file_bytes.decode("utf-8", errors="ignore").strip()

    if not text:
        return []

    return [{"text": text, "filename": filename, "page": None}]


def extract_md(file_bytes, filename):
    text = file_bytes.decode("utf-8", errors="ignore").strip()

    if not text:
        return []

    return [{"text": text, "filename": filename, "page": None}]


def extract_document(file_bytes, filename):
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        return extract_pdf(file_bytes, filename)
    if suffix == ".docx":
        return extract_docx(file_bytes, filename)
    if suffix == ".txt":
        return extract_txt(file_bytes, filename)
    if suffix == ".md":
        return extract_md(file_bytes, filename)

    raise ValueError(f"Unsupported file type: {suffix}")


# -----------------------------
# Chunking
# -----------------------------
def chunk_text(text, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_OVERLAP):
    words = text.split()

    if not words:
        return []

    chunk_size = max(50, int(chunk_size))
    overlap = max(0, min(int(overlap), chunk_size - 1))
    step = max(1, chunk_size - overlap)

    chunks = []

    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]

        if not chunk_words:
            break

        chunks.append(" ".join(chunk_words))

        if start + chunk_size >= len(words):
            break

    return chunks


def create_chunks(records, chunk_size, overlap):
    chunks = []

    for record in records:
        pieces = chunk_text(record["text"], chunk_size, overlap)

        for number, piece in enumerate(pieces, start=1):
            chunks.append(
                {
                    "text": piece,
                    "filename": record["filename"],
                    "page": record["page"],
                    "chunk": number,
                }
            )

    return chunks


# -----------------------------
# Embeddings + FAISS
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(DEFAULT_EMBEDDING_MODEL)


def build_vector_store(chunks):
    model = load_embedding_model()
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index, embeddings


# -----------------------------
# Keyword search
# -----------------------------
def important_words(text):
    words = re.findall(r"\b[a-zA-Z0-9]{2,}\b", text.lower())
    return {word for word in words if word not in STOPWORDS}


def keyword_score(question, text):
    query_words = important_words(question)
    text_words = important_words(text)

    if not query_words:
        return 0.0

    matches = query_words.intersection(text_words)
    return len(matches) / len(query_words)


# -----------------------------
# Hybrid search
# -----------------------------
def hybrid_search(question, chunks, index, top_k=5):
    if not chunks:
        return []

    model = load_embedding_model()

    question_embedding = model.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    question_embedding = np.asarray(question_embedding, dtype="float32")

    search_k = min(max(top_k * 3, 10), len(chunks))
    semantic_scores, semantic_ids = index.search(
        question_embedding,
        search_k,
    )

    candidates = set()

    for item in semantic_ids[0]:
        if item >= 0:
            candidates.add(int(item))

    # Add chunks with keyword matches.
    for i, chunk in enumerate(chunks):
        if keyword_score(question, chunk["text"]) > 0:
            candidates.add(i)

    results = []

    semantic_map = {
        int(idx): float(score)
        for idx, score in zip(semantic_ids[0], semantic_scores[0])
        if idx >= 0
    }

    for idx in candidates:
        semantic_score = semantic_map.get(idx, 0.0)
        keyword = keyword_score(question, chunks[idx]["text"])

        hybrid = (0.75 * semantic_score) + (0.25 * keyword)

        result = dict(chunks[idx])
        result["semantic_score"] = semantic_score
        result["keyword_score"] = keyword
        result["hybrid_score"] = hybrid

        results.append(result)

    results.sort(key=lambda item: item["hybrid_score"], reverse=True)

    return results[:top_k]


# -----------------------------
# Google Drive
# -----------------------------
def drive_download(link):
    link = link.strip()

    if not link:
        raise ValueError("Please enter a Google Drive file or folder link.")

    temp_dir = Path("drive_downloads")
    temp_dir.mkdir(exist_ok=True)

    # Folder link
    if "/folders/" in link:
        downloaded = gdown.download_folder(
            url=link,
            output=str(temp_dir),
            quiet=True,
            use_cookies=False,
        )

        files = []

        for path in Path(temp_dir).rglob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(path)

        if not files:
            raise ValueError(
                "No supported PDF, DOCX, TXT, or MD files were found in the Drive folder."
            )

        return files

    # Single file link
    output_path = temp_dir / "drive_file"

    downloaded = gdown.download(
        url=link,
        output=str(output_path),
        fuzzy=True,
        quiet=True,
    )

    if not downloaded:
        raise ValueError(
            "Google Drive download failed. Make sure the file is accessible."
        )

    downloaded_path = Path(downloaded)

    # gdown may return the real filename.
    if downloaded_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            "The Google Drive file must be PDF, DOCX, TXT, or MD."
        )

    return [downloaded_path]


def load_drive_documents(link):
    paths = drive_download(link)
    records = []

    for path in paths:
        records.extend(
            extract_document(
                path.read_bytes(),
                path.name,
            )
        )

    return records


# -----------------------------
# Knowledge-base helpers
# -----------------------------
def document_signature(records):
    digest = hashlib.sha256()

    for record in sorted(
        records,
        key=lambda item: (
            item["filename"],
            item["page"] or 0,
            item["text"],
        ),
    ):
        digest.update(record["filename"].encode("utf-8"))
        digest.update(str(record["page"]).encode("utf-8"))
        digest.update(record["text"].encode("utf-8"))

    return digest.hexdigest()


def build_knowledge_base(records, chunk_size, overlap):
    chunks = create_chunks(records, chunk_size, overlap)

    if not chunks:
        raise ValueError("No readable text was found in the documents.")

    index, embeddings = build_vector_store(chunks)

    return {
        "records": records,
        "chunks": chunks,
        "index": index,
        "embeddings": embeddings,
        "signature": document_signature(records),
    }


# -----------------------------
# Gemini
# -----------------------------
def get_gemini_client():
    api_key = st.secrets.get(
        "GEMINI_API_KEY",
        os.getenv("GEMINI_API_KEY"),
    )

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is missing. Add it to Streamlit secrets."
        )

    return genai.Client(api_key=api_key)


def answer_question(question, retrieved_chunks):
    if not retrieved_chunks:
        return "That information is not available in the provided documents."

    context_parts = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        page_text = (
            f", page {chunk['page']}"
            if chunk.get("page") is not None
            else ""
        )

        context_parts.append(
            f"[Source {i}: {chunk['filename']}{page_text}]\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are an AI Document Assistant.

Answer the user's question ONLY using the document context below.
Do not use outside knowledge.
Do not invent facts, citations, page numbers, filenames, or sources.

If the answer is not contained in the provided context, reply exactly:
That information is not available in the provided documents.

Keep the answer clear and concise.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}
"""

    client = get_gemini_client()

    model_name = st.secrets.get(
        "GEMINI_MODEL",
        os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
    )

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )

    return response.text.strip()


# -----------------------------
# Session state
# -----------------------------
if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = None

if "processed_signature" not in st.session_state:
    st.session_state.processed_signature = None

if "documents" not in st.session_state:
    st.session_state.documents = []

if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    chunk_size = st.number_input(
        "Chunk size (words)",
        min_value=200,
        max_value=2000,
        value=DEFAULT_CHUNK_SIZE,
        step=100,
    )

    overlap = st.number_input(
        "Chunk overlap (words)",
        min_value=0,
        max_value=500,
        value=DEFAULT_OVERLAP,
        step=20,
    )

    top_k = st.slider(
        "Retrieved chunks",
        min_value=1,
        max_value=10,
        value=DEFAULT_TOP_K,
    )

    if st.button("Clear Knowledge Base"):
        st.session_state.knowledge_base = None
        st.session_state.processed_signature = None
        st.session_state.documents = []
        st.session_state.messages = []
        st.rerun()


# -----------------------------
# Main UI
# -----------------------------
st.title("📚 AI Document Assistant")
st.caption(
    "Upload documents or connect Google Drive, then ask questions using Gemini-powered RAG."
)

tab_local, tab_drive, tab_docs, tab_chat = st.tabs(
    ["📤 Local Upload", "☁️ Google Drive", "📄 Documents", "💬 Ask Questions"]
)


# -----------------------------
# Local upload
# -----------------------------
with tab_local:
    st.subheader("Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, TXT, or MD files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Process Uploaded Documents"):
        records = []

        with st.spinner("Extracting text and building embeddings..."):
            for uploaded_file in uploaded_files:
                records.extend(
                    extract_document(
                        uploaded_file.getvalue(),
                        uploaded_file.name,
                    )
                )

            if not records:
                st.error("No readable text was found.")
            else:
                signature = document_signature(records)

                if (
                    st.session_state.knowledge_base
                    and st.session_state.processed_signature == signature
                ):
                    st.success(
                        "These documents are already processed. Existing embeddings were reused."
                    )
                else:
                    st.session_state.knowledge_base = build_knowledge_base(
                        records,
                        chunk_size,
                        overlap,
                    )
                    st.session_state.processed_signature = signature
                    st.session_state.documents = records
                    st.session_state.messages = []

                    st.success(
                        f"Processed {len(records)} extracted sections and "
                        f"{len(st.session_state.knowledge_base['chunks'])} chunks."
                    )


# -----------------------------
# Google Drive
# -----------------------------
with tab_drive:
    st.subheader("Google Drive")

    drive_link = st.text_input(
        "Paste a public Google Drive file or folder link",
        value="https://drive.google.com/file/d/1KhtM3rMvzzG1aRDpzrnx8SXOBEh_Nwi-/view?usp=drivesdk",
    )

    st.info(
        "Supported Drive files: PDF, DOCX, TXT, and MD. "
        "The file or folder must be accessible to the app."
    )

    if st.button("Load from Google Drive"):
        try:
            with st.spinner("Downloading, extracting, and building embeddings..."):
                records = load_drive_documents(drive_link)

                signature = document_signature(records)

                if (
                    st.session_state.knowledge_base
                    and st.session_state.processed_signature == signature
                ):
                    st.success(
                        "These documents are already processed. Existing embeddings were reused."
                    )
                else:
                    st.session_state.knowledge_base = build_knowledge_base(
                        records,
                        chunk_size,
                        overlap,
                    )
                    st.session_state.processed_signature = signature
                    st.session_state.documents = records
                    st.session_state.messages = []

                    st.success(
                        f"Loaded {len(records)} extracted sections and "
                        f"{len(st.session_state.knowledge_base['chunks'])} chunks."
                    )

        except Exception as error:
            st.error(f"Google Drive error: {error}")


# -----------------------------
# Documents tab
# -----------------------------
with tab_docs:
    st.subheader("Extracted Document Information")

    kb = st.session_state.knowledge_base

    if not kb:
        st.info("Process a document first.")
    else:
        records = kb["records"]
        chunks = kb["chunks"]

        pages = sum(
            1
            for record in records
            if record.get("page") is not None
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Extracted sections", len(records))
        col2.metric("Chunks", len(chunks))
        col3.metric("Embedding vectors", len(kb["embeddings"]))

        filenames = sorted({record["filename"] for record in records})

        for filename in filenames:
            file_records = [
                record for record in records
                if record["filename"] == filename
            ]

            with st.expander(f"📄 {filename}"):
                st.write(f"Extracted sections/pages: {len(file_records)}")

                for record in file_records[:10]:
                    page_label = (
                        f"Page {record['page']}"
                        if record.get("page") is not None
                        else "No page number"
                    )

                    st.markdown(f"**{page_label}**")
                    st.write(record["text"][:1500])

        st.divider()
        st.subheader("Chunk Preview")

        for chunk in chunks[:10]:
            page_label = (
                f"Page {chunk['page']}"
                if chunk.get("page") is not None
                else "No page number"
            )

            st.markdown(
                f"**{chunk['filename']} — {page_label} — "
                f"Chunk {chunk['chunk']}**"
            )
            st.write(chunk["text"][:1200])


# -----------------------------
# Chat tab
# -----------------------------
with tab_chat:
    st.subheader("Ask Questions")

    kb = st.session_state.knowledge_base

    if not kb:
        st.info("Process documents before asking questions.")
    else:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

                if message["role"] == "assistant" and message.get("sources"):
                    st.markdown("#### Retrieved Sources")

                    for source in message["sources"]:
                        page_label = (
                            f"Page {source['page']}"
                            if source.get("page") is not None
                            else "Page not available"
                        )

                        with st.expander(
                            f"{source['filename']} — {page_label} "
                            f"(score: {source['hybrid_score']:.3f})"
                        ):
                            st.write(source["text"])

        question = st.chat_input("Ask something about your documents...")

        if question:
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": question,
                }
            )

            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("Searching documents and asking Gemini..."):
                    try:
                        retrieved = hybrid_search(
                            question,
                            kb["chunks"],
                            kb["index"],
                            top_k=top_k,
                        )

                        answer = answer_question(
                            question,
                            retrieved,
                        )

                        st.write(answer)

                        st.markdown("#### Retrieved Sources")

                        for source in retrieved:
                            page_label = (
                                f"Page {source['page']}"
                                if source.get("page") is not None
                                else "Page not available"
                            )

                            with st.expander(
                                f"{source['filename']} — {page_label} "
                                f"(score: {source['hybrid_score']:.3f})"
                            ):
                                st.write(source["text"])

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer,
                                "sources": retrieved,
                            }
                        )

                    except Exception as error:
                        error_message = f"Error: {error}"
                        st.error(error_message)

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": error_message,
                                "sources": [],
                            }
                        )
