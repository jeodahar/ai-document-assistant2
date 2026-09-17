# AI Document Assistant

A simple Streamlit RAG application that lets you upload documents or load supported files from Google Drive, search them using hybrid retrieval, and answer questions using the Gemini API.

## Features

- PDF, DOCX, TXT, and Markdown upload
- Separate extraction functions for each file type
- PDF page metadata preserved
- Word-based overlapping chunks
- Filename and page metadata preserved for chunks
- Sentence Transformers embeddings
- FAISS semantic vector search
- Keyword search
- Hybrid semantic + keyword ranking
- Embeddings reused through Streamlit session state
- Gemini API for context-only question answering
- Retrieved source chunks shown below every answer
- Google Drive file/folder loading
- No database or authentication required for this MVP
- Gemini API key stored in Streamlit secrets

## Project structure

```text
ai_document_assistant/
├── app.py
├── requirements.txt
├── readme.md
├── .gitignore
└── .streamlit/
    └── secrets.toml.example
```

## 1. Install locally

```bash
pip install -r requirements.txt
```

Run:

```bash
streamlit run app.py
```

## 2. Add Gemini API key

Create:

```text
.streamlit/secrets.toml
```

Add:

```toml
GEMINI_API_KEY = "your_gemini_api_key"
GEMINI_MODEL = "gemini-2.5-flash"
```

Do not upload your real `secrets.toml` to GitHub.

The app reads:

```python
st.secrets["GEMINI_API_KEY"]
```

with an environment-variable fallback.

## 3. Google Drive

Paste a Google Drive file or folder link in the Google Drive tab.

Supported file types:

- PDF
- DOCX
- TXT
- MD

The Drive resource must be accessible to the application. For a private Drive resource that requires your personal Google login, this simple `gdown` approach may not work.

## 4. How the RAG pipeline works

```text
Documents
   ↓
Text Extraction
   ↓
Chunks + Filename/Page Metadata
   ↓
Sentence Transformer Embeddings
   ↓
FAISS Vector Index
   ↓
User Question
   ↓
Question Embedding + Keyword Search
   ↓
Hybrid Ranking
   ↓
Top Relevant Chunks
   ↓
Gemini
   ↓
Answer + Retrieved Sources
```

## 5. Embedding reuse

Embeddings are created when documents are processed, not for every question.

The FAISS index, embeddings, chunks, and metadata are kept in `st.session_state.knowledge_base`.

The Sentence Transformer model is cached using `st.cache_resource`.

If the same documents are processed again during the session, the app uses the existing knowledge base when the document signature matches.

## 6. GitHub

Create a repository such as:

```text
ai-document-assistant
```

Upload:

- `app.py`
- `requirements.txt`
- `readme.md`
- `.gitignore`
- `.streamlit/secrets.toml.example`

Never upload:

```text
.streamlit/secrets.toml
```

## 7. Streamlit Community Cloud

1. Push the project to GitHub.
2. Open Streamlit Community Cloud.
3. Create a new app.
4. Select your GitHub repository.
5. Select the main branch.
6. Select `app.py`.
7. Deploy.
8. Open the app settings/secrets area.
9. Add:

```toml
GEMINI_API_KEY = "your_gemini_api_key"
GEMINI_MODEL = "gemini-2.5-flash"
```

10. Save/redeploy.

## 8. Testing checklist

Test:

1. Upload a PDF.
2. Confirm extracted sections/pages.
3. Confirm chunk count.
4. Confirm embedding vector count.
5. Ask a question whose answer exists in the document.
6. Check retrieved filename/page/chunk.
7. Ask an unrelated question.
8. Confirm the app responds that the information is not available.
9. Test DOCX, TXT, and MD.
10. Test a public Google Drive file or folder.

## Notes

- PDF page numbers are preserved when text extraction provides page-level content.
- DOCX/TXT/MD files do not automatically have reliable page numbers, so their page field is shown as unavailable.
- Restarting the Streamlit session clears the in-memory knowledge base and embeddings.
- The Gemini model can be changed through `GEMINI_MODEL` in Streamlit secrets.
