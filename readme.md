<div align="center">

# 📄 IntelliDoc
### AI-Powered Multilingual Document Analysis System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/LLM-Groq%20%7C%20LLaMA--3.3--70B-orange?style=for-the-badge)](https://groq.com)
[![FAISS](https://img.shields.io/badge/Vector%20Search-FAISS-green?style=for-the-badge)](https://github.com/facebookresearch/faiss)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](https://opensource.org/licenses/MIT)

**Upload PDFs or scanned images → Ask questions in any language → Get instant, accurate answers**

[Features](#-features) • [Architecture](#-system-architecture) • [Installation](#-installation) • [Usage](#-usage) • [Tech Stack](#-tech-stack) • [Project Structure](#-project-structure)

</div>

---

## 🌟 Overview

**IntelliDoc** is a full-stack AI application that transforms static documents into an interactive, multilingual knowledge base. Upload one or more PDFs or scanned images, and ask questions in natural language — including in Indian regional languages. The system automatically detects whether a document is digital or scanned, applies OCR where needed, chunks and indexes the content, and answers queries using a Groq-hosted LLaMA 3.3 70B model with RAG (Retrieval-Augmented Generation).

Built as a Final Year Major Project (B.Tech CSE, VTU 2026), IntelliDoc demonstrates production-level skills in LLM integration, vector search, OCR pipelines, parallel processing, and multilingual NLP.

---

## ✨ Features

### 🔍 Document Processing
- **Smart OCR Auto-Detection** — Automatically identifies scanned vs. digital pages; applies Tesseract OCR only where needed, cutting processing overhead significantly
- **Force OCR Mode** — Manual override for fully scanned or handwritten documents
- **Parallel Page Processing** — Multi-threaded PDF extraction using `ThreadPoolExecutor` with up to 4 workers for faster throughput
- **Multi-Format Support** — PDF, JPG, JPEG, PNG, BMP, TIFF

### 🌐 Multilingual Support (10+ Languages)
Supports input and output in: **English, Kannada, Hindi, Tamil, Telugu, Marathi, Malayalam, Gujarati, Bengali, Punjabi**

- Auto-detects the language of the question using `langdetect`
- Responds in the same language as the question by default
- Supports explicit translation requests ("translate this to Kannada")

### 💬 RAG-Based Question Answering
- TF-IDF vectorization with bigram support (up to 1000 features)
- FAISS `IndexFlatIP` (inner product / cosine similarity) for fast semantic retrieval
- Top-8 most relevant chunks retrieved per query with a similarity threshold of 0.1
- Context window capped at 4,500 characters for optimal LLM performance
- Powered by **LLaMA 3.3 70B Versatile** via Groq API (temperature 0.1 for factual accuracy)

### 📊 Document Intelligence
- **Table Extraction** — Detects and extracts structured tables from PDFs using PyMuPDF's `find_tables()`
- **Chart & Diagram Detection** — Uses OpenCV edge detection (Canny) with heuristics to classify embedded images as charts, flowcharts, or general images
- **Document Summarization** — Full-document summarization in any supported language

### 💾 Export & History
- Download full chat history as `.txt` or `.json`
- JSON export preserves source page references alongside each answer
- Session-based processing stats (pages/sec, characters extracted, chunks created)

### 🎨 UI / UX
- Professional dark-themed Streamlit interface with custom CSS
- Real-time processing progress indicators
- Expandable source references with page numbers displayed per answer
- Optional text-to-speech audio output via `gTTS`

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INTELLIDOC PIPELINE                       │
└─────────────────────────────────────────────────────────────────┘

  ┌──────────────┐     ┌─────────────────────────────────────────┐
  │  User Uploads │────▶│           DOCUMENT INGESTION            │
  │  PDF / Image  │     │                                         │
  └──────────────┘     │  1. Smart OCR Detection (PyMuPDF)       │
                        │     ├─ Digital page → direct extract    │
                        │     └─ Scanned page → Tesseract OCR     │
                        │  2. Parallel Processing (4 workers)     │
                        │  3. Table Extraction (PyMuPDF tables)   │
                        │  4. Chart Detection (OpenCV + Canny)    │
                        └──────────────┬──────────────────────────┘
                                       │
                                       ▼
                        ┌─────────────────────────────────────────┐
                        │            INDEXING PIPELINE            │
                        │                                         │
                        │  1. Smart Chunking                      │
                        │     (chunk=1500 chars, overlap=300)     │
                        │  2. TF-IDF Vectorization                │
                        │     (1000 features, 1-2 ngrams)         │
                        │  3. FAISS Index Build                   │
                        │     (IndexFlatIP / cosine similarity)   │
                        └──────────────┬──────────────────────────┘
                                       │
         ┌─────────────┐               ▼
         │  User Query  │────▶ ┌───────────────────────────────────┐
         │  (any lang)  │      │         RETRIEVAL & ANSWER        │
         └─────────────┘      │                                   │
                               │  1. Language Detection (langdetect)│
                               │  2. Query Embedding (TF-IDF)      │
                               │  3. Top-K Retrieval (FAISS, K=8)  │
                               │  4. Context Assembly (≤4500 chars)│
                               │  5. LLM Generation                │
                               │     (Groq / LLaMA-3.3-70B)       │
                               │  6. Response in detected language  │
                               └──────────────┬────────────────────┘
                                              │
                                              ▼
                               ┌─────────────────────────────────┐
                               │    STREAMLIT UI (Dark Theme)    │
                               │  Answer + Source Pages + Export │
                               └─────────────────────────────────┘
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.9 or higher
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed on your system
- A [Groq API key](https://console.groq.com/) (free tier available)

### Step 1 — Clone the Repository

```bash
git clone https://github.com/shakuntala-pawar/IntelliDoc-AI-Powered-Multilingual-Document-Analysis-System.git
cd IntelliDoc-AI-Powered-Multilingual-Document-Analysis-System
```

### Step 2 — Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Install Tesseract OCR

**Windows:**
Download and install from [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki).
Default path: `C:\Program Files\Tesseract-OCR\tesseract.exe`

For Indian language support, install the additional language packs (`kan`, `hin`, `tam`, `tel`, etc.) during setup.

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install tesseract-ocr
sudo apt-get install tesseract-ocr-kan tesseract-ocr-hin tesseract-ocr-tam tesseract-ocr-tel
```

**macOS:**
```bash
brew install tesseract
brew install tesseract-lang
```

### Step 4 — Configure Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
TESSDATA_PREFIX=C:\Program Files\Tesseract-OCR\tessdata
```

> On Linux/macOS, `TESSERACT_CMD` is typically `/usr/bin/tesseract` and `TESSDATA_PREFIX` is `/usr/share/tesseract-ocr/4.00/tessdata`.

### Step 5 — Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 🚀 Usage

### 1. Upload Documents
In the sidebar, upload one or more **PDF files** or **images** (JPG, PNG, BMP, TIFF).

### 2. Choose Processing Mode
- **Smart (Auto-detect):** Recommended. Applies OCR only to pages that need it.
- **Force OCR:** Use this for fully scanned documents or handwritten notes.

### 3. Enable Optional Features
- ✅ **Extract Tables** — Pulls structured tables from PDFs
- ✅ **Detect Charts/Diagrams** — Identifies visual elements using edge detection

### 4. Process
Click **🚀 Process Documents**. You'll see real-time progress with pages/sec metrics.

### 5. Ask Questions
Type your question in the chat box — in **any supported language**. Examples:

```
What are the main findings of this report?
ಈ ದಾಖಲೆಯ ಮುಖ್ಯ ಅಂಶಗಳು ಯಾವುವು?   (Kannada)
इस दस्तावेज़ का सारांश क्या है?          (Hindi)
Translate the key points to Tamil.
```

### 6. Export
Download your conversation as `.txt` or `.json` using the export buttons.

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| **Frontend** | Streamlit 1.29+ with custom CSS (dark theme) |
| **LLM** | LLaMA 3.3 70B Versatile via Groq API |
| **Vector Search** | FAISS (`IndexFlatIP` — cosine similarity) |
| **Embeddings** | Scikit-learn TF-IDF (1000 features, bigrams) |
| **PDF Processing** | PyMuPDF (fitz) |
| **OCR** | Tesseract OCR via pytesseract |
| **Image Processing** | OpenCV, Pillow |
| **Language Detection** | langdetect |
| **Parallelism** | Python `concurrent.futures.ThreadPoolExecutor` |
| **Data Handling** | Pandas, NumPy |
| **Text-to-Speech** | gTTS (optional) |
| **Config** | python-dotenv |

---

## 📁 Project Structure

```
IntelliDoc-AI-Powered-Multilingual-Document-Analysis-System/
│
├── app.py                  # Main application — all pipeline logic + Streamlit UI
├── htmlTemplates.py        # Custom HTML/CSS templates for chat message styling
├── requirements.txt        # Python dependencies
├── .python-version         # Python version pin
├── .gitignore
│
├── docs/
│   └── PDF-LangChain.jpg   # Architecture diagram
│
└── models/                 # Directory for any locally cached model artifacts
```

---

## 🔑 Key Technical Decisions

**Why TF-IDF + FAISS instead of sentence transformers?**
TF-IDF provides fast, lightweight embeddings without requiring a GPU or heavy model downloads, making the app runnable on standard hardware. FAISS with inner product similarity gives efficient nearest-neighbor search even at scale.

**Why Groq + LLaMA 3.3 70B?**
Groq's inference API delivers extremely low latency compared to other hosted LLM providers. LLaMA 3.3 70B offers strong multilingual reasoning, which is critical for Indian language support.

**Why smart OCR detection?**
Running Tesseract on every page is slow. By checking text content length before invoking OCR, the system skips OCR on digital pages entirely — significantly reducing processing time on mixed documents.

**Why parallel processing?**
PDF documents can have 50–100+ pages. Using `ThreadPoolExecutor` with 4 workers processes multiple pages concurrently, cutting end-to-end extraction time substantially.

---

## 🌍 Supported Languages

| Language | Code | OCR Support | QA & Translation |
|---|---|---|---|
| English | `en` | ✅ | ✅ |
| Kannada | `kn` | ✅ | ✅ |
| Hindi | `hi` | ✅ | ✅ |
| Tamil | `ta` | ✅ | ✅ |
| Telugu | `te` | ✅ | ✅ |
| Marathi | `mr` | ✅ | ✅ |
| Malayalam | `ml` | ✅ | ✅ |
| Gujarati | `gu` | ✅ | ✅ |
| Bengali | `bn` | ✅ | ✅ |
| Punjabi | `pa` | ✅ | ✅ |

---

## 📋 Requirements

```
streamlit>=1.29.0
numpy==1.26.4
faiss-cpu==1.8.0
scikit-learn>=1.3.0
pymupdf>=1.23.0
pillow>=10.0.0
pytesseract>=0.3.10
opencv-python>=4.8.0
groq>=0.11.0
langdetect>=1.0.9
python-dotenv>=1.0.0
pdf2image>=1.16.0

# Optional
gtts          # Text-to-speech output
```

---

## 🔧 Configuration Reference

| Variable | Description | Default |
|---|---|---|
| `GROQ_API_KEY` | Your Groq API key | — (required) |
| `TESSERACT_CMD` | Path to Tesseract executable | `C:\Program Files\Tesseract-OCR\tesseract.exe` |
| `TESSDATA_PREFIX` | Path to Tesseract language data | System default |

| Parameter | Value | Description |
|---|---|---|
| `OCR_LANGS` | `eng+kan+hin+tam+tel+mar+mal+guj+ben+pan` | Languages loaded for OCR |
| `TOP_K` | `8` | Number of chunks retrieved per query |
| `SIMILARITY_THRESHOLD` | `0.1` | Minimum cosine similarity for retrieval |
| `CHUNK_SIZE` | `1500` | Characters per text chunk |
| `CHUNK_OVERLAP` | `300` | Overlap between consecutive chunks |
| `MAX_CONTEXT_CHARS` | `4500` | Maximum context passed to LLM |
| `MAX_WORKERS` | `4` | Parallel threads for PDF processing |

---

## 📌 Known Limitations

- Tesseract OCR accuracy on low-resolution or heavily stylized scanned text may vary
- TF-IDF embeddings are not semantic — queries need to share vocabulary with document text for best retrieval results
- Very large documents (200+ pages) may take longer to process depending on system hardware
- Audio input requires additional setup (`SpeechRecognition`, `pyaudio`)

---

## 👩‍💻 Author

**Shakuntala Pawar**
B.Tech CSE — Visvesvaraya Technological University (VTU), 2026 | Top 5% of cohort

[![GitHub](https://img.shields.io/badge/GitHub-shakuntala--pawar-181717?style=flat&logo=github)](https://github.com/shakuntala-pawar)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Shakuntala%20Pawar-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/shakuntala-pawar-283a5632b/)

---

## 📄 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT) — feel free to use, modify, and distribute with attribution.

---

<div align="center">
<sub>Built with ❤️ using Python, Streamlit, FAISS, Groq, and Tesseract OCR</sub>
</div>
