
# #Completed Final Year - Major Project

# app.py — IntelliDoc: AI-Powered Multilingual Document Analysis System
import os
import streamlit as st
from dotenv import load_dotenv
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import numpy as np
import faiss
from groq import Groq
from langdetect import detect, DetectorFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import io
import json
from datetime import datetime
import pandas as pd
import cv2
import base64
import re

DetectorFactory.seed = 0

# -------------------- Load ENV --------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TESSERACT_CMD = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
TESSDATA_PREFIX = os.getenv("TESSDATA_PREFIX")

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
if TESSDATA_PREFIX:
    os.environ["TESSDATA_PREFIX"] = TESSDATA_PREFIX

# -------------------- Config --------------------
OCR_LANGS = "eng+kan+hin+tam+tel+mar+mal+guj+ben+pan"
TOP_K = 8
SIMILARITY_THRESHOLD = 0.1
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300
MAX_CONTEXT_CHARS = 4500
MAX_WORKERS = 4

# -------------------- Clients --------------------
groq_client = Groq(api_key=GROQ_API_KEY)

@st.cache_resource
def get_tfidf_vectorizer():
    return TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        min_df=1,
        stop_words=None
    )

# -------------------- Table Extraction --------------------
def extract_tables_from_pdf(pdf_bytes, doc_name):
    """Extract tables from PDF"""
    tables = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_num, page in enumerate(doc, start=1):
            try:
                # Extract tables using PyMuPDF
                page_tables = page.find_tables()
                if page_tables:
                    for table_idx, table in enumerate(page_tables.tables):
                        try:
                            table_data = table.extract()
                            if table_data:
                                df = pd.DataFrame(table_data[1:], columns=table_data[0] if table_data else None)
                                tables.append({
                                    'doc': doc_name,
                                    'page': page_num,
                                    'table_num': table_idx + 1,
                                    'data': df
                                })
                        except:
                            continue
            except:
                continue
        doc.close()
    except Exception as e:
        st.warning(f"Table extraction error: {e}")
    return tables

# -------------------- Chart/Diagram Detection --------------------
def detect_visual_elements(pdf_bytes, doc_name):
    """Detect charts, diagrams, flowcharts in PDF"""
    visuals = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_num, page in enumerate(doc, start=1):
            # Get images from page
            image_list = page.get_images()
            
            if image_list:
                for img_idx, img_info in enumerate(image_list):
                    try:
                        xref = img_info[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        
                        # Convert to PIL Image
                        img = Image.open(io.BytesIO(image_bytes))
                        
                        # Analyze image to detect if it's a chart/diagram
                        img_array = np.array(img.convert('RGB'))
                        
                        # Simple heuristics to detect charts/diagrams
                        # Check for lines, geometric shapes, etc.
                        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                        edges = cv2.Canny(gray, 50, 150)
                        
                        # Count edge pixels (charts have lots of edges)
                        edge_ratio = np.count_nonzero(edges) / edges.size
                        
                        visual_type = "Unknown"
                        if edge_ratio > 0.05:
                            # Try to determine type
                            if edge_ratio > 0.15:
                                visual_type = "Flowchart/Diagram"
                            elif edge_ratio > 0.08:
                                visual_type = "Chart/Graph"
                            else:
                                visual_type = "Image"
                        
                        visuals.append({
                            'doc': doc_name,
                            'page': page_num,
                            'type': visual_type,
                            'image': img,
                            'size': img.size
                        })
                    except:
                        continue
        doc.close()
    except Exception as e:
        st.warning(f"Visual detection error: {e}")
    return visuals

# -------------------- Smart OCR Detection --------------------
def needs_ocr(page):
    """Quickly determine if a page needs OCR"""
    text = page.get_text("text")
    if len(text.strip()) > 50:
        return False
    image_list = page.get_images()
    if len(image_list) > 0:
        return True
    return True

def extract_text_from_page_fast(page, page_num, use_ocr=False):
    """Fast text extraction with page number tracking"""
    if not use_ocr:
        text = page.get_text("text")
        if len(text.strip()) > 50:
            return text.strip(), page_num
    
    try:
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        
        text = pytesseract.image_to_string(
            img, 
            lang=OCR_LANGS,
            config='--psm 6 --oem 3'
        )
        return text.strip(), page_num
    except Exception as e:
        return "", page_num

# -------------------- Parallel PDF Processing --------------------
def process_page(page_info):
    """Process a single page"""
    page_num, page, doc_name, force_ocr = page_info
    
    try:
        use_ocr = force_ocr or needs_ocr(page)
        text, pg_num = extract_text_from_page_fast(page, page_num + 1, use_ocr=use_ocr)
        
        if text:
            return {
                'page_num': page_num,
                'actual_page': page_num + 1,
                'text': text,
                'doc_name': doc_name,
                'success': True
            }
        return {'success': False}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def extract_text_from_pdf_parallel(pdf_bytes, doc_name, force_ocr=False):
    """Extract text from PDF using parallel processing"""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(doc)
        
        page_tasks = [
            (i, doc[i], doc_name, force_ocr) 
            for i in range(total_pages)
        ]
        
        results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_page = {
                executor.submit(process_page, task): task[0] 
                for task in page_tasks
            }
            
            for future in as_completed(future_to_page):
                result = future.result()
                if result.get('success'):
                    results.append(result)
        
        doc.close()
        results.sort(key=lambda x: x['page_num'])
        
        full_text = ""
        for r in results:
            page_num = r['actual_page']
            text = r['text']
            full_text += f"\n\n[Page {page_num}]\n{text}"
        
        return full_text, len(results), total_pages
        
    except Exception as e:
        st.error(f"PDF processing error: {e}")
        return "", 0, 0

def extract_text_from_image_fast(image_file):
    """Fast image OCR"""
    try:
        img = Image.open(image_file)
        max_size = 2000
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        text = pytesseract.image_to_string(
            img,
            lang=OCR_LANGS,
            config='--psm 6 --oem 3'
        )
        return text.strip()
    except Exception as e:
        st.warning(f"Image OCR failed: {e}")
        return ""

# -------------------- Chunking --------------------
def chunk_text_smart(text, source_name, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Smart chunking with page tracking"""
    if not text or len(text.strip()) < 50:
        return [], []
    
    chunks, meta = [], []
    pages = text.split('[Page ')
    
    for page_section in pages:
        if not page_section.strip():
            continue
        
        page_num = None
        if ']' in page_section:
            try:
                page_num = int(page_section.split(']')[0])
                page_text = page_section.split(']', 1)[1]
            except:
                page_text = page_section
        else:
            page_text = page_section
        
        start = 0
        chunk_id = 0
        
        while start < len(page_text):
            end = start + chunk_size
            chunk = page_text[start:end].strip()
            
            if chunk:
                chunks.append(chunk)
                meta.append({
                    "source": source_name,
                    "page": page_num,
                    "chunk_id": chunk_id
                })
                chunk_id += 1
            
            start += chunk_size - overlap
    
    return chunks, meta

# -------------------- TF-IDF Embeddings --------------------
def embed_texts_tfidf(texts, vectorizer=None):
    """Fast TF-IDF embeddings"""
    try:
        if vectorizer is None:
            vectorizer = get_tfidf_vectorizer()
            matrix = vectorizer.fit_transform(texts)
        else:
            matrix = vectorizer.transform(texts)
        
        return matrix.toarray().astype("float32"), vectorizer
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return np.zeros((len(texts), 100), dtype="float32"), None

def build_faiss_index(emb_np):
    """Build FAISS index"""
    if emb_np is None or emb_np.shape[0] == 0:
        raise ValueError("Empty embeddings")
    
    norms = np.linalg.norm(emb_np, axis=1, keepdims=True)
    norms[norms == 0] = 1
    emb_np = emb_np / norms
    
    dim = emb_np.shape[1]
    idx = faiss.IndexFlatIP(dim)
    idx.add(emb_np)
    return idx

# -------------------- Language Utilities --------------------
LANG_MAP = {
    "kn": "Kannada", "hi": "Hindi", "ta": "Tamil", "te": "Telugu",
    "mr": "Marathi", "ml": "Malayalam", "gu": "Gujarati",
    "bn": "Bengali", "pa": "Punjabi", "en": "English"
}

LANG_CODES = {
    "english": "en", "kannada": "kn", "hindi": "hi", "tamil": "ta",
    "telugu": "te", "marathi": "mr", "malayalam": "ml", "gujarati": "gu",
    "bengali": "bn", "punjabi": "pa"
}

def detect_question_language(question):
    try:
        return detect(question)
    except:
        return "en"

def extract_target_language(question):
    q_lower = question.lower()
    for lang_name, code in LANG_CODES.items():
        if f"in {lang_name}" in q_lower or f"to {lang_name}" in q_lower:
            return code
    return None

# -------------------- Audio Functions --------------------
def get_audio_input():
    """Get audio input using HTML5 audio recorder"""
    audio_html = """
    <div style="text-align: center; padding: 20px;">
        <p style="color: #E0E0E0;">🎤 <strong>Voice Input Feature</strong></p>
        <p style="color: #888; font-size: 0.9em;">Note: Audio input requires microphone permissions.<br>
        Install speech_recognition and pyaudio packages for full functionality:<br>
        <code>pip install SpeechRecognition pyaudio</code></p>
    </div>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

def text_to_speech_button(text, lang='en'):
    """Create audio download button"""
    try:
        # Note: Install gtts package: pip install gtts
        from gtts import gTTS
        
        tts = gTTS(text=text, lang=lang, slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        
        audio_base64 = base64.b64encode(fp.read()).decode()
        audio_html = f"""
        <audio controls autoplay style="width: 100%;">
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)
        return True
    except ImportError:
        st.info("💡 Install gtts for audio output: `pip install gtts`")
        return False
    except Exception as e:
        st.warning(f"Audio generation error: {e}")
        return False

# -------------------- QA & Summarization --------------------
def ask_groq_with_context(context, question, target_lang=None):
    """Answer questions using context"""
    if target_lang:
        response_lang = LANG_MAP.get(target_lang, "the requested language")
    else:
        question_lang = detect_question_language(question)
        response_lang = LANG_MAP.get(question_lang, "English")
    
    is_translation = any(kw in question.lower() for kw in 
                         ["translate", "convert", "change to", "in kannada", "in hindi", 
                          "in telugu", "in marathi", "in tamil", "in english"])
    
    if is_translation:
        prompt = f"""You are a multilingual translator.

CONTEXT from documents:
{context}

USER REQUEST: {question}

Translate the relevant information to {response_lang}. Be accurate and preserve details.

Response in {response_lang}:"""
    else:
        prompt = f"""Answer this question using ONLY the CONTEXT provided.

CONTEXT:
{context}

QUESTION: {question}

Answer in {response_lang}. If not in context, say "I cannot find this in the documents."

Answer:"""
    
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1000
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

def summarize_document(text, target_lang="en"):
    """Generate summary"""
    if not text or len(text.strip()) < 50:
        return "Insufficient text to summarize."
    
    text_snippet = text[:8000]
    lang_name = LANG_MAP.get(target_lang, "English")
    
    prompt = f"""Summarize this document in {lang_name}. Include main topic, key points, and important details.

{text_snippet}

Summary:"""
    
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=800
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Summarization failed: {e}"

# -------------------- Download Functions --------------------
def create_chat_download(chat_history):
    """Create downloadable chat history"""
    chat_text = f"IntelliDoc Chat History\n"
    chat_text += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    chat_text += "="*80 + "\n\n"
    
    for item in chat_history:
        if len(item) == 2:
            role, message = item
            source_pages = None
        else:
            role, message, source_pages = item
        
        if role == "user":
            chat_text += f"USER:\n{message}\n\n"
        else:
            chat_text += f"INTELLIDOC:\n{message}\n"
            if source_pages:
                chat_text += f"Sources: {source_pages}\n"
            chat_text += "\n"
        chat_text += "-"*80 + "\n\n"
    
    return chat_text

def create_json_download(chat_history):
    """Create JSON format chat history"""
    conversations = []
    for item in chat_history:
        if len(item) == 2:
            role, message = item
            source_pages = None
        else:
            role, message, source_pages = item
        
        conversations.append({
            "role": role,
            "message": message,
            "sources": source_pages if role == "assistant" else None
        })
    
    chat_data = {
        "timestamp": datetime.now().isoformat(),
        "conversation": conversations
    }
    return json.dumps(chat_data, indent=2, ensure_ascii=False)

# -------------------- Custom CSS (Dark Theme) --------------------
# -------------------- Custom CSS (Professional Subtle Dark Theme) --------------------
def load_custom_css():
    st.markdown("""
    <style>
    /* Professional Dark theme */
    .stApp {
        background: #1e1e1e;
    }
    
    /* Header */
    .main-header {
        background: #ffffff;
        padding: 35px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        border-bottom: 2px solid #4a9eff;
    }
    
    .main-header h1 {
        color: #000000;
        font-size: 2.5em;
        font-weight: 700;
        margin: 0;
        letter-spacing: 1px;
    }
    
    .main-header p {
        color: #000000;
        font-size: 1.1em;
        margin: 10px 0 0 0;
        font-weight: 400;
    }
    
    /* Feature banner */
    .feature-banner {
        text-align: center;
        padding: 12px;
        background: #2d2d2d;
        border-radius: 8px;
        margin-bottom: 20px;
        border: 1px solid #3a3a3a;
    }
    
    .feature-banner p {
        color: #b0b0b0;
        font-size: 0.95em;
        margin: 0;
    }
    
    /* Chat messages */
    .user-message {
        background: #2d2d2d;
        color: #e0e0e0;
        padding: 18px;
        border-radius: 12px 12px 4px 12px;
        margin: 12px 0;
        border-left: 3px solid #4a9eff;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    
    .bot-message {
        background: #252525;
        color: #d0d0d0;
        padding: 18px;
        border-radius: 12px 12px 12px 4px;
        margin: 12px 0;
        border-left: 3px solid #6c63ff;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    
    .page-reference {
        display: inline-block;
        background: #4a9eff;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: 600;
        margin: 8px 4px 0 0;
        box-shadow: 0 2px 6px rgba(74,158,255,0.3);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #4a9eff 0%, #6c63ff 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s;
        box-shadow: 0 3px 10px rgba(74,158,255,0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(74,158,255,0.4);
        background: linear-gradient(135deg, #5aa3ff 0%, #7d73ff 100%);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #252525;
        border-right: 1px solid #3a3a3a;
    }
    
    [data-testid="stSidebar"] h3 {
        color: #e0e0e0;
    }
    
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
        color: #b0b0b0;
    }
    
    /* Success/Info boxes */
    .stSuccess {
        background-color: rgba(74,158,255,0.1);
        border-left: 4px solid #4a9eff;
        color: #4a9eff;
        border-radius: 4px;
    }
    
    .stInfo {
        background-color: rgba(108,99,255,0.1);
        border-left: 4px solid #6c63ff;
        color: #6c63ff;
        border-radius: 4px;
    }
    
    .stWarning {
        background-color: rgba(255,179,0,0.1);
        border-left: 4px solid #ffb300;
        color: #ffb300;
        border-radius: 4px;
    }
    
    .stError {
        background-color: rgba(255,82,82,0.1);
        border-left: 4px solid #ff5252;
        color: #ff5252;
        border-radius: 4px;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #2d2d2d;
        border: 2px dashed #4a9eff;
        border-radius: 10px;
        padding: 20px;
    }
    
    [data-testid="stFileUploader"] label {
        color: #b0b0b0;
    }
    
    /* Input boxes */
    .stTextInput>div>div>input {
        background: #2d2d2d;
        color: #e0e0e0;
        border: 1px solid #4a4a4a;
        border-radius: 8px;
        padding: 10px;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #4a9eff;
        box-shadow: 0 0 0 1px #4a9eff;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #2d2d2d;
        border-radius: 8px;
        color: #e0e0e0;
        font-weight: 600;
        border: 1px solid #3a3a3a;
    }
    
    .streamlit-expanderHeader:hover {
        background: #353535;
        border-color: #4a9eff;
    }
    
    /* Dataframe */
    .stDataFrame {
        background: #2d2d2d;
        border-radius: 8px;
    }
    
    /* Markdown text */
    .stMarkdown {
        color: #d0d0d0;
    }
    
    /* Checkbox */
    .stCheckbox {
        color: #b0b0b0;
    }
    
    /* Radio buttons */
    .stRadio > label {
        color: #e0e0e0;
    }
    
    .stRadio > div {
        color: #b0b0b0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #808080;
        padding: 25px;
        margin-top: 40px;
        border-top: 1px solid #3a3a3a;
    }
    
    .footer h3 {
        color: #e0e0e0;
        margin-bottom: 15px;
        font-size: 1.3em;
    }
    
    .footer a {
        color: #4a9eff;
        text-decoration: none;
        transition: color 0.3s;
    }
    
    .footer a:hover {
        color: #6c63ff;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1e1e1e;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #4a4a4a;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #5a5a5a;
    }
    
    /* Divider */
    hr {
        border-color: #3a3a3a;
    }
    
    /* Download buttons */
    .stDownloadButton>button {
        background: #2d2d2d;
        color: #4a9eff;
        border: 1px solid #4a9eff;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stDownloadButton>button:hover {
        background: #4a9eff;
        color: white;
        transform: translateY(-2px);
    }
    
    /* Feature grid */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 15px;
        margin: 20px 0;
    }
    
    .feature-card {
        background: #2d2d2d;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #3a3a3a;
        transition: all 0.3s;
    }
    
    .feature-card:hover {
        border-color: #4a9eff;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(74,158,255,0.2);
    }
    
    .feature-card strong {
        color: #4a9eff;
        display: block;
        margin-bottom: 5px;
    }
    
    .feature-card span {
        color: #909090;
        font-size: 0.9em;
    }
    </style>
    """, unsafe_allow_html=True)

# -------------------- Streamlit UI --------------------
st.set_page_config(
    page_title="IntelliDoc - AI Document Analysis",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
load_custom_css()

# Custom Header
# Custom Header
st.markdown("""
<div class="main-header">
    <h1>📄 IntelliDoc</h1>
    <p>AI-Powered Multilingual Document Analysis System</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="feature-banner">
    <p>
        ⚡ <strong>Features:</strong> Printed & Handwritten OCR | 10+ Languages | Real-Time Translation | 
        Table Extraction | Chart Detection | Audio I/O | 3x Faster Processing
    </p>
</div>
""", unsafe_allow_html=True)

# Session state
for key, val in [("chunks", []), ("meta", []), ("embeddings", None),
                 ("faiss_index", None), ("full_text", ""), ("chat_history", []),
                 ("file_info", {}), ("vectorizer", None), ("tables", []), 
                 ("visuals", [])]:
    if key not in st.session_state:
        st.session_state[key] = val

# Sidebar
st.sidebar.markdown("### 📂 Document Upload")

doc_mode = st.sidebar.radio(
    "Processing Mode:",
    ["🚀 Smart (Auto-detect)", "🔍 Force OCR (Scanned/Handwritten)"],
    help="Smart mode automatically detects which pages need OCR"
)

uploaded_pdfs = st.sidebar.file_uploader(
    "📑 PDF Files", 
    type="pdf", 
    accept_multiple_files=True,
    key="pdf_uploader"
)

uploaded_images = st.sidebar.file_uploader(
    "🖼️ Image Files", 
    type=["jpg", "jpeg", "png", "bmp", "tiff"],
    accept_multiple_files=True,
    key="img_uploader"
)

st.sidebar.markdown("### ⚙️ Advanced Options")
extract_tables = st.sidebar.checkbox("📊 Extract Tables", value=True)
detect_visuals = st.sidebar.checkbox("📈 Detect Charts/Diagrams", value=True)

process_btn = st.sidebar.button("🚀 Process Documents", type="primary", use_container_width=True)

# Processing
if process_btn:
    # Clear previous data
    for key in ["chunks", "meta", "embeddings", "faiss_index", "full_text", 
                "file_info", "chat_history", "tables", "visuals"]:
        if key in ["chunks", "meta", "file_info", "chat_history", "tables", "visuals"]:
            st.session_state[key].clear()
        else:
            st.session_state[key] = None if key != "full_text" else ""
    
    force_ocr = ("Force OCR" in doc_mode)
    total_start = time.time()
    
    with st.spinner("⚡ Processing documents with parallel OCR..."):
        # Process PDFs
        for pdf in uploaded_pdfs or []:
            start_time = time.time()
            st.info(f"📄 Processing: {pdf.name}")
            pdf_bytes = pdf.read()
            
            # Extract tables
            if extract_tables:
                with st.spinner(f"📊 Extracting tables from {pdf.name}..."):
                    tables = extract_tables_from_pdf(pdf_bytes, pdf.name)
                    if tables:
                        st.session_state.tables.extend(tables)
                        st.success(f"✅ Found {len(tables)} table(s)")
            
            # Detect visuals
            if detect_visuals:
                with st.spinner(f"📈 Detecting charts/diagrams in {pdf.name}..."):
                    visuals = detect_visual_elements(pdf_bytes, pdf.name)
                    if visuals:
                        st.session_state.visuals.extend(visuals)
                        st.success(f"✅ Found {len(visuals)} visual element(s)")
            
            # Extract text
            text, pages_processed, total_pages = extract_text_from_pdf_parallel(
                pdf_bytes, pdf.name, force_ocr=force_ocr
            )
            
            processing_time = time.time() - start_time
            
            if text:
                st.session_state.full_text += f"\n\n=== FILE: {pdf.name} ===\n{text}"
                ch, md = chunk_text_smart(text, pdf.name)
                st.session_state.chunks.extend(ch)
                st.session_state.meta.extend(md)
                
                st.session_state.file_info[pdf.name] = {
                    'chars': len(text),
                    'pages': total_pages,
                    'time': processing_time
                }
                
                speed = total_pages / processing_time if processing_time > 0 else 0
                st.success(
                    f"✅ {pdf.name}: {total_pages} pages in {processing_time:.1f}s "
                    f"({speed:.1f} pages/sec)"
                )
            else:
                st.warning(f"⚠️ Could not extract text from {pdf.name}")
        
        # Process Images
        for img in uploaded_images or []:
            start_time = time.time()
            st.info(f"🖼️ Processing: {img.name}")
            text = extract_text_from_image_fast(img)
            processing_time = time.time() - start_time
            
            if text:
              
                st.session_state.full_text += f"\n\n=== IMAGE: {img.name} ===\n{text}"
                ch, md = chunk_text_smart(text, img.name)
                st.session_state.chunks.extend(ch)
                st.session_state.meta.extend(md)
                
                st.session_state.file_info[img.name] = {
                    'chars': len(text),
                    'time': processing_time
                }
                st.success(f"✅ {img.name}: {len(text)} chars in {processing_time:.1f}s")
            else:
                st.warning(f"⚠️ Could not extract text from {img.name}")
        
        # Build embeddings
        if st.session_state.chunks:
            st.info(f"🔢 Creating search index for {len(st.session_state.chunks)} chunks...")
            embed_start = time.time()
            
            emb_np, vectorizer = embed_texts_tfidf(st.session_state.chunks)
            st.session_state.embeddings = emb_np
            st.session_state.vectorizer = vectorizer
            st.session_state.faiss_index = build_faiss_index(emb_np)
            
            embed_time = time.time() - embed_start
            total_time = time.time() - total_start
            
            st.success(
                f"✅ Search index created in {embed_time:.1f}s | "
                f"Total: {total_time:.1f}s"
            )
            
            # Show statistics
            st.sidebar.markdown("### 📊 Processing Stats")
            total_docs = len(st.session_state.file_info)
            total_chunks = len(st.session_state.chunks)
            st.sidebar.markdown(f"**Documents:** {total_docs} | **Chunks:** {total_chunks}")
            
            for fname, info in st.session_state.file_info.items():
                if 'pages' in info:
                    st.sidebar.text(f"📄 {fname[:25]}...: {info['pages']}p ({info['time']:.1f}s)")
                else:
                    st.sidebar.text(f"🖼️ {fname[:25]}...: ({info['time']:.1f}s)")
            
            if st.session_state.tables:
                st.sidebar.markdown(f"**Tables Found:** {len(st.session_state.tables)}")
            
            if st.session_state.visuals:
                st.sidebar.markdown(f"**Visuals Found:** {len(st.session_state.visuals)}")
        else:
            st.error("❌ No text extracted. Check file quality and OCR settings.")

# Display extracted tables
if st.session_state.tables:
    with st.expander(f"📊 Extracted Tables ({len(st.session_state.tables)} found)", expanded=False):
        for table_info in st.session_state.tables:
            st.markdown(f"**📄 {table_info['doc']} - Page {table_info['page']}, Table {table_info['table_num']}**")
            st.dataframe(table_info['data'], use_container_width=True)
            st.markdown("---")

# Display detected visuals
if st.session_state.visuals:
    with st.expander(f"📈 Detected Charts & Diagrams ({len(st.session_state.visuals)} found)", expanded=False):
        cols = st.columns(3)
        for idx, visual in enumerate(st.session_state.visuals):
            col = cols[idx % 3]
            with col:
                st.image(visual['image'], caption=f"{visual['doc']} - Page {visual['page']}\n{visual['type']}", use_container_width=True)

# Chat Interface
st.markdown("---")
st.markdown("### 💬 Chat with Your Documents")

if not st.session_state.faiss_index:
    st.info("👆 Upload and process documents using the sidebar first.")
else:
    # Audio input section
    with st.expander("🎤 Voice Input (Optional)", expanded=False):
        get_audio_input()
    
    # Text input
    user_input = st.text_input(
        "Your question:",
        placeholder="E.g., 'Summarize in Kannada', 'What dates are mentioned?', 'Translate to Hindi'",
        key="user_question"
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        ask_button = st.button("📤 Send Question", type="primary", use_container_width=True)
    with col2:
        audio_output = st.checkbox("🔊 Audio", value=False)
    
    with st.expander("💡 Example Questions", expanded=False):
        st.markdown("""
        **General Queries:**
        - "Summarize this document in Kannada"
        - "What is the main topic?"
        - "List all important dates mentioned"
        - "Explain this pdf in kannada"
        - "Give brief information ?"
        - "What content if given in the pdf?"
                    
        
        **Translation:**
        - "Translate the main points to Telugu"
        - "Convert this to Hindi"
        
        **Specific Information:**
        - "What are the names of people mentioned?"
        - "Extract all numerical values"
        - "What tables are present?"
        """)
    
    if ask_button and user_input:
        st.session_state.chat_history.append(("user", user_input))
        
        with st.spinner("🤔 Analyzing..."):
            is_summary = any(kw in user_input.lower() for kw in 
                           ["summary", "summarize", "overview", "brief", "main points"])
            
            target_lang = extract_target_language(user_input)
            if not target_lang:
                target_lang = detect_question_language(user_input)
            
            if is_summary:
                answer = summarize_document(st.session_state.full_text, target_lang)
                source_pages = "Summary from entire document"
            else:
                qvec, _ = embed_texts_tfidf(
                    [user_input], 
                    vectorizer=st.session_state.vectorizer
                )
                qvec = qvec.astype("float32")
                
                norm = np.linalg.norm(qvec)
                if norm > 0:
                    qvec = qvec / norm
                
                D, I = st.session_state.faiss_index.search(qvec, TOP_K)
                
                if len(D[0]) == 0 or D[0][0] < SIMILARITY_THRESHOLD:
                    answer = "I cannot find relevant information in the documents."
                    source_pages = "N/A"
                else:
                    context_pieces = []
                    total_chars = 0
                    source_pages_set = set()
                    
                    for idx, score in zip(I[0], D[0]):
                        if idx >= len(st.session_state.chunks):
                            continue
                        
                        chunk = st.session_state.chunks[idx]
                        source = st.session_state.meta[idx]['source']
                        page = st.session_state.meta[idx].get('page', '?')
                        
                        if page and page != '?':
                            source_pages_set.add(f"Page {page}")
                        
                        piece = f"[Source: {source}, Page: {page}]\n{chunk}"
                        
                        if total_chars + len(piece) > MAX_CONTEXT_CHARS:
                            break
                        
                        context_pieces.append(piece)
                        total_chars += len(piece)
                    
                    context = "\n\n---\n\n".join(context_pieces)
                    answer = ask_groq_with_context(context, user_input, target_lang)
                    
                    # Sort pages numerically
                    try:
                        sorted_pages = sorted(source_pages_set, key=lambda x: int(re.findall(r'\d+', x)[0]))
                        source_pages = " | ".join(sorted_pages) if sorted_pages else "N/A"
                    except:
                        source_pages = " | ".join(sorted(source_pages_set)) if source_pages_set else "N/A"
            
            # Add answer with source pages
            st.session_state.chat_history.append(("assistant", answer, source_pages))
            
            # Audio output if enabled
            if audio_output and answer:
                with st.spinner("🔊 Generating audio..."):
                    text_to_speech_button(answer, lang=target_lang)
    
    # Display chat with custom styling
    st.markdown("### 📜 Conversation History")
    
    if not st.session_state.chat_history:
        st.info("💭 No conversation yet. Ask your first question above!")
    
    for item in st.session_state.chat_history:
        if len(item) == 2:
            role, message = item
            source_pages = None
        else:
            role, message, source_pages = item
        
        if role == "user":
            st.markdown(
                f'<div class="user-message"><strong>👤 You:</strong><br><br>{message}</div>', 
                unsafe_allow_html=True
            )
        else:
            page_badges = ""
            if source_pages and source_pages not in ["N/A", "Summary from entire document"]:
                for page in source_pages.split(" | "):
                    page_badges += f'<span class="page-reference">📄 {page}</span>'
            elif source_pages == "Summary from entire document":
                page_badges = '<span class="page-reference">📄 Full Document</span>'
            
            st.markdown(
                f'<div class="bot-message"><strong>🤖 IntelliDoc:</strong><br>{page_badges}<br><br>{message}</div>', 
                unsafe_allow_html=True
            )
        st.markdown("<br>", unsafe_allow_html=True)
    
    # Download and Clear buttons
    if st.session_state.chat_history:
        st.markdown("---")
        col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
        
        with col1:
            chat_txt = create_chat_download(st.session_state.chat_history)
            st.download_button(
                label="📥 Download TXT",
                data=chat_txt,
                file_name=f"intellidoc_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col2:
            chat_json = create_json_download(st.session_state.chat_history)
            st.download_button(
                label="📥 Download JSON",
                data=chat_json,
                file_name=f"intellidoc_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col3:
            # Download full extracted text
            st.download_button(
                label="📄 Extracted Text",
                data=st.session_state.full_text,
                file_name=f"extracted_text_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col4:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history.clear()
                st.rerun()

# Footer
# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <h3>⚡ Performance & Features</h3>
    <div class="feature-grid">
        <div class="feature-card">
            <strong>🚀 Smart OCR</strong>
            <span>Auto-detects when to use OCR</span>
        </div>
        <div class="feature-card">
            <strong>⚡ 3x Faster</strong>
            <span>Parallel processing</span>
        </div>
        <div class="feature-card">
            <strong>🌐 10+ Languages</strong>
            <span>Multilingual support</span>
        </div>
        <div class="feature-card">
            <strong>📊 Smart Tables</strong>
            <span>Automatic extraction</span>
        </div>
        <div class="feature-card">
            <strong>📈 Chart Detection</strong>
            <span>Visual element recognition</span>
        </div>
        <div class="feature-card">
            <strong>🔊 Audio I/O</strong>
            <span>Voice input & output</span>
        </div>
    </div>
    <p style="margin-top: 20px; font-size: 0.9em;">
        <strong>IntelliDoc</strong> v1.0 | Built with Streamlit, PyMuPDF, Tesseract OCR, FAISS & Groq LLM<br>
        Supporting: English, Kannada, Hindi, Tamil, Telugu, Marathi, Malayalam, Gujarati, Bengali, Punjabi
    </p>
</div>
""", unsafe_allow_html=True)


























































































































































































































































































































































































































































































































































































#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# # app.py — IntelliDoc ULTRA-FAST Version (Speed Optimized)
# import os
# import streamlit as st
# from dotenv import load_dotenv
# import fitz  # PyMuPDF
# from PIL import Image
# import pytesseract
# import numpy as np
# import faiss
# from groq import Groq
# from langdetect import detect, DetectorFactory, LangDetectException
# from sklearn.feature_extraction.text import TfidfVectorizer
# from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
# import time
# import io
# import json
# from datetime import datetime
# import pandas as pd
# import cv2
# import re
# from typing import List, Dict, Tuple, Optional
# import logging
# from functools import lru_cache
# import hashlib
# import multiprocessing as mp
# from queue import Queue
# import threading

# DetectorFactory.seed = 0

# # -------------------- SPEED-OPTIMIZED Configuration --------------------
# logging.basicConfig(level=logging.WARNING)  # Reduced logging overhead
# logger = logging.getLogger(__name__)

# load_dotenv()
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# TESSERACT_CMD = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")

# pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# # SPEED OPTIMIZATIONS
# OCR_LANGS = "eng+kan+hin"  # Reduced languages for speed
# TOP_K = 15  # Reduced from 25
# SIMILARITY_THRESHOLD = 0.02  # Slightly lower for faster filtering
# CHUNK_SIZE = 2000  # Reduced for faster processing
# CHUNK_OVERLAP = 400  # Reduced overlap
# MAX_CONTEXT_CHARS = 12000  # Reduced context size
# MAX_WORKERS = min(16, mp.cpu_count() * 2)  # Aggressive parallelization
# MAX_OUTPUT_TOKENS = 3000  # Reduced token generation
# CACHE_SIZE = 512  # Increased cache

# # Use faster Groq model
# FAST_MODEL = "llama-3.1-70b-versatile"  # Faster than 3.3

# groq_client = Groq(api_key=GROQ_API_KEY)

# # -------------------- ULTRA-FAST CACHING SYSTEM --------------------
# class SpeedCache:
#     """Lightning-fast in-memory cache with LRU eviction"""
#     def __init__(self, maxsize=CACHE_SIZE):
#         self.cache = {}
#         self.access_times = {}
#         self.maxsize = maxsize
#         self.lock = threading.Lock()
    
#     def get(self, key: str):
#         with self.lock:
#             if key in self.cache:
#                 self.access_times[key] = time.time()
#                 return self.cache[key]
#             return None
    
#     def set(self, key: str, value):
#         with self.lock:
#             if len(self.cache) >= self.maxsize:
#                 # Remove oldest accessed item
#                 oldest = min(self.access_times.items(), key=lambda x: x[1])[0]
#                 del self.cache[oldest]
#                 del self.access_times[oldest]
            
#             self.cache[key] = value
#             self.access_times[key] = time.time()
    
#     def clear(self):
#         with self.lock:
#             self.cache.clear()
#             self.access_times.clear()

# # Global caches
# ocr_cache = SpeedCache(maxsize=1000)
# embedding_cache = SpeedCache(maxsize=500)
# text_cache = SpeedCache(maxsize=200)

# # -------------------- ULTRA-FAST OCR ENGINE --------------------
# class TurboOCR:
#     """Optimized OCR with aggressive caching and minimal preprocessing"""
    
#     @staticmethod
#     def _fast_preprocess(img_array: np.ndarray) -> np.ndarray:
#         """Single-pass optimized preprocessing"""
#         if len(img_array.shape) == 3:
#             gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
#         else:
#             gray = img_array
        
#         # Fast adaptive threshold (fastest method)
#         binary = cv2.adaptiveThreshold(
#             gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#             cv2.THRESH_BINARY, 11, 2
#         )
#         return binary
    
#     @staticmethod
#     def fast_ocr(image: Image.Image, lang: str = OCR_LANGS) -> str:
#         """Ultra-fast OCR with caching"""
#         # Generate cache key
#         img_bytes = image.tobytes()
#         cache_key = hashlib.md5(img_bytes).hexdigest()
        
#         # Check cache
#         cached = ocr_cache.get(cache_key)
#         if cached:
#             return cached
        
#         # Fast preprocessing
#         img_array = np.array(image)
#         processed = TurboOCR._fast_preprocess(img_array)
        
#         # Single OCR pass with optimal config
#         config = '--psm 6 --oem 3'  # Fastest reliable config
        
#         try:
#             text = pytesseract.image_to_string(processed, lang=lang, config=config)
#             result = text.strip()
            
#             # Cache result
#             ocr_cache.set(cache_key, result)
#             return result
#         except:
#             return ""

# turbo_ocr = TurboOCR()

# # -------------------- PARALLEL PDF PROCESSING --------------------
# def process_single_page_fast(args: Tuple) -> Dict:
#     """Ultra-fast single page processing"""
#     page_data, page_num, doc_name, force_ocr = args
    
#     try:
#         # Try text extraction first (fastest)
#         if not force_ocr:
#             text = page_data.get_text("text")
#             if len(text.strip()) > 30:
#                 return {
#                     'page': page_num,
#                     'text': text.strip(),
#                     'doc': doc_name,
#                     'success': True
#                 }
        
#         # Fast OCR fallback
#         mat = fitz.Matrix(2.0, 2.0)  # Reduced from 4.0 for speed
#         pix = page_data.get_pixmap(matrix=mat, alpha=False)
#         img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
#         text = turbo_ocr.fast_ocr(img)
        
#         return {
#             'page': page_num,
#             'text': text,
#             'doc': doc_name,
#             'success': True
#         }
#     except Exception as e:
#         logger.debug(f"Page {page_num} failed: {e}")
#         return {'success': False}

# def extract_text_from_pdf_turbo(pdf_bytes: bytes, doc_name: str, 
#                                 force_ocr: bool = False) -> Tuple[str, int, int]:
#     """ULTRA-FAST parallel PDF processing"""
    
#     # Check cache
#     cache_key = f"pdf_{hashlib.md5(pdf_bytes).hexdigest()}"
#     cached = text_cache.get(cache_key)
#     if cached:
#         return cached
    
#     try:
#         doc = fitz.open(stream=pdf_bytes, filetype="pdf")
#         total_pages = len(doc)
        
#         # Prepare all pages
#         page_args = [(doc[i], i+1, doc_name, force_ocr) for i in range(total_pages)]
        
#         results = []
        
#         # Use ProcessPoolExecutor for true parallel processing
#         with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
#             futures = [executor.submit(process_single_page_fast, args) for args in page_args]
            
#             for future in as_completed(futures):
#                 result = future.result()
#                 if result.get('success'):
#                     results.append(result)
        
#         doc.close()
        
#         # Sort by page number
#         results.sort(key=lambda x: x['page'])
        
#         # Combine text
#         full_text = "\n\n".join([f"[Page {r['page']}]\n{r['text']}" for r in results])
        
#         result_tuple = (full_text, len(results), total_pages)
        
#         # Cache result
#         text_cache.set(cache_key, result_tuple)
        
#         return result_tuple
    
#     except Exception as e:
#         logger.error(f"PDF processing error: {e}")
#         return "", 0, 0

# def extract_text_from_image_turbo(image_file) -> str:
#     """Ultra-fast image OCR"""
#     try:
#         img = Image.open(image_file)
        
#         # Smart resize for speed
#         max_dim = 2000  # Reduced from 4000
#         if max(img.size) > max_dim:
#             ratio = max_dim / max(img.size)
#             new_size = tuple(int(dim * ratio) for dim in img.size)
#             img = img.resize(new_size, Image.Resampling.BILINEAR)  # Faster than LANCZOS
        
#         return turbo_ocr.fast_ocr(img)
#     except Exception as e:
#         logger.error(f"Image OCR error: {e}")
#         return ""

# # -------------------- FAST CHUNKING --------------------
# @lru_cache(maxsize=256)
# def chunk_text_fast(text: str, source_name: str) -> Tuple[List[str], List[Dict]]:
#     """Fast chunking with minimal overhead"""
#     if not text or len(text.strip()) < 30:
#         return [], []
    
#     chunks, meta = [], []
    
#     # Fast page splitting
#     pages = re.split(r'\[Page\s+(\d+)\]', text)
    
#     current_page = 1
#     for i, segment in enumerate(pages):
#         if not segment.strip():
#             continue
        
#         if segment.strip().isdigit():
#             current_page = int(segment)
#             continue
        
#         page_text = segment.strip()
        
#         # Fast chunking - no complex sentence splitting
#         if len(page_text) <= CHUNK_SIZE:
#             chunks.append(page_text)
#             meta.append({"source": source_name, "page": current_page})
#         else:
#             # Simple chunking by character count
#             for start in range(0, len(page_text), CHUNK_SIZE - CHUNK_OVERLAP):
#                 chunk = page_text[start:start + CHUNK_SIZE]
#                 if chunk.strip():
#                     chunks.append(chunk.strip())
#                     meta.append({"source": source_name, "page": current_page})
    
#     return chunks, meta

# # -------------------- FAST EMBEDDINGS --------------------
# @st.cache_resource
# def get_fast_vectorizer():
#     """Optimized vectorizer for speed"""
#     return TfidfVectorizer(
#         max_features=2000,  # Reduced from 3000
#         ngram_range=(1, 2),  # Reduced from (1,4)
#         min_df=1,
#         max_df=0.95,
#         token_pattern=r'(?u)\b\w+\b'
#     )

# def embed_texts_fast(texts: List[str], vectorizer=None) -> Tuple[np.ndarray, any]:
#     """Fast embedding with caching"""
    
#     # Generate cache key
#     texts_hash = hashlib.md5("".join(texts[:5]).encode()).hexdigest()
#     cache_key = f"embed_{texts_hash}_{len(texts)}"
    
#     cached = embedding_cache.get(cache_key)
#     if cached and vectorizer is not None:
#         return cached
    
#     try:
#         if vectorizer is None:
#             vectorizer = get_fast_vectorizer()
#             matrix = vectorizer.fit_transform(texts)
#         else:
#             matrix = vectorizer.transform(texts)
        
#         result = matrix.toarray().astype("float32"), vectorizer
        
#         if vectorizer is not None:
#             embedding_cache.set(cache_key, result)
        
#         return result
#     except Exception as e:
#         logger.error(f"Embedding error: {e}")
#         return np.zeros((len(texts), 100), dtype="float32"), None

# def build_faiss_index_fast(emb_np: np.ndarray):
#     """Fast FAISS index building"""
#     if emb_np is None or emb_np.shape[0] == 0:
#         raise ValueError("Empty embeddings")
    
#     # Simple normalization
#     norms = np.linalg.norm(emb_np, axis=1, keepdims=True)
#     norms[norms == 0] = 1
#     emb_np = emb_np / norms
    
#     dim = emb_np.shape[1]
    
#     # Use FlatIP for small datasets (fastest)
#     if emb_np.shape[0] < 5000:
#         index = faiss.IndexFlatIP(dim)
#         index.add(emb_np)
#     else:
#         # IVF for larger datasets
#         nlist = min(100, emb_np.shape[0] // 50)
#         quantizer = faiss.IndexFlatIP(dim)
#         index = faiss.IndexIVFFlat(quantizer, dim, nlist)
#         index.train(emb_np)
#         index.add(emb_np)
#         index.nprobe = 5  # Reduced for speed
    
#     return index

# # -------------------- FAST RETRIEVAL --------------------
# def retrieve_context_fast(query: str, chunks: List[str], metadata: List[Dict],
#                          index, vectorizer, top_k: int = TOP_K) -> Tuple[str, List[Dict]]:
#     """Lightning-fast retrieval"""
#     if not chunks or index is None:
#         return "", []
    
#     try:
#         # Embed query
#         q_vec = vectorizer.transform([query]).toarray().astype("float32")
        
#         # Normalize
#         norm = np.linalg.norm(q_vec)
#         if norm > 0:
#             q_vec = q_vec / norm
        
#         # Fast search
#         D, I = index.search(q_vec, min(top_k, len(chunks)))
        
#         # Quick filtering and context building
#         context_parts = []
#         selected_meta = []
        
#         for idx, score in zip(I[0], D[0]):
#             if idx < len(chunks) and score >= SIMILARITY_THRESHOLD:
#                 context_parts.append(chunks[idx])
#                 selected_meta.append(metadata[idx])
        
#         # Fast join
#         full_context = "\n\n".join(context_parts[:top_k])
        
#         # Fast truncation
#         if len(full_context) > MAX_CONTEXT_CHARS:
#             full_context = full_context[:MAX_CONTEXT_CHARS]
        
#         return full_context, selected_meta[:top_k]
    
#     except Exception as e:
#         logger.error(f"Retrieval error: {e}")
#         return "", []

# # -------------------- FAST LLM GENERATION --------------------
# def generate_answer_fast(query: str, context: str) -> str:
#     """Fast answer generation with minimal prompt"""
    
#     system_prompt = """You are IntelliDoc. Answer based ONLY on the provided context. Be concise and accurate. Cite page numbers."""
    
#     messages = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"}
#     ]
    
#     try:
#         response = groq_client.chat.completions.create(
#             model=FAST_MODEL,
#             messages=messages,
#             temperature=0.1,
#             max_tokens=MAX_OUTPUT_TOKENS,
#             top_p=0.9,
#             stream=False
#         )
#         return response.choices[0].message.content.strip()
#     except Exception as e:
#         logger.error(f"LLM error: {e}")
#         return "Error generating response. Please try again."

# # -------------------- UTILITIES --------------------
# @lru_cache(maxsize=128)
# def detect_language_fast(text: str) -> str:
#     """Fast language detection"""
#     try:
#         return detect(text[:200])
#     except:
#         return "en"

# def format_page_refs_fast(metadata: List[Dict]) -> str:
#     """Fast page reference formatting"""
#     if not metadata:
#         return ""
    
#     sources = {}
#     for m in metadata:
#         src = m.get('source', 'Unknown')
#         pg = m.get('page')
#         if src not in sources:
#             sources[src] = set()
#         if pg:
#             sources[src].add(pg)
    
#     refs = []
#     for src, pages in sources.items():
#         page_list = sorted(list(pages))[:5]  # Limit to 5 pages
#         refs.append(f"📄 {src} (p.{','.join(map(str, page_list))})")
    
#     return " | ".join(refs)

# # -------------------- STREAMLIT UI (SPEED OPTIMIZED) --------------------
# def init_session_state():
#     """Initialize session state"""
#     defaults = {
#         'doc_texts': {},
#         'all_chunks': [],
#         'all_metadata': [],
#         'vectorizer': None,
#         'faiss_index': None,
#         'chat_history': [],
#         'processing': False
#     }
    
#     for key, value in defaults.items():
#         if key not in st.session_state:
#             st.session_state[key] = value

# def main():
#     st.set_page_config(
#         page_title="IntelliDoc Turbo",
#         page_icon="⚡",
#         layout="wide"
#     )
    
#     init_session_state()
    
#     # Minimal CSS
#     st.markdown("""
#     <style>
#     .main-header {font-size: 2.5rem; font-weight: 700; color: #FF5722; text-align: center;}
#     .speed-badge {background: #FF5722; color: white; padding: 5px 15px; border-radius: 20px; font-weight: bold;}
#     </style>
#     """, unsafe_allow_html=True)
    
#     # Header
#     col1, col2, col3 = st.columns([1, 2, 1])
#     with col2:
#         st.markdown('<div class="main-header">⚡ IntelliDoc TURBO</div>', unsafe_allow_html=True)
#         st.markdown('<center><span class="speed-badge">10X FASTER</span></center>', unsafe_allow_html=True)
    
#     # Sidebar
#     with st.sidebar:
#         st.header("📁 Upload")
        
#         uploaded_files = st.file_uploader(
#             "Drop PDF/Images here",
#             type=["pdf", "png", "jpg", "jpeg"],
#             accept_multiple_files=True
#         )
        
#         force_ocr = st.checkbox("Force OCR", value=False)
        
#         if st.button("⚡ PROCESS", type="primary", use_container_width=True):
#             if uploaded_files:
#                 process_documents_turbo(uploaded_files, force_ocr)
#             else:
#                 st.warning("Upload files first!")
        
#         if st.session_state.all_chunks:
#             st.divider()
#             st.metric("📄 Documents", len(st.session_state.doc_texts))
#             st.metric("📑 Chunks", len(st.session_state.all_chunks))
            
#             if st.button("🗑️ Clear All"):
#                 for key in ['doc_texts', 'all_chunks', 'all_metadata', 'chat_history']:
#                     st.session_state[key] = [] if 'history' in key or 'chunks' in key or 'metadata' in key else {}
#                 st.session_state.faiss_index = None
#                 ocr_cache.clear()
#                 text_cache.clear()
#                 embedding_cache.clear()
#                 st.rerun()
    
#     # Main Chat Interface
#     st.header("💬 Chat")
    
#     # Query Input
#     query = st.text_area(
#         "Ask your question:",
#         height=100,
#         placeholder="What is this document about?"
#     )
    
#     col1, col2 = st.columns([3, 1])
#     with col1:
#         ask_button = st.button("🚀 Ask", type="primary", use_container_width=True)
#     with col2:
#         if st.button("Clear Chat", use_container_width=True):
#             st.session_state.chat_history = []
#             st.rerun()
    
#     # Process Query
#     if ask_button and query:
#         if not st.session_state.all_chunks:
#             st.error("⚠️ Process documents first!")
#         else:
#             start_time = time.time()
            
#             with st.spinner("⚡ Processing..."):
#                 # Retrieve
#                 context, meta = retrieve_context_fast(
#                     query,
#                     st.session_state.all_chunks,
#                     st.session_state.all_metadata,
#                     st.session_state.faiss_index,
#                     st.session_state.vectorizer
#                 )
                
#                 if context:
#                     # Generate
#                     answer = generate_answer_fast(query, context)
                    
#                     elapsed = time.time() - start_time
                    
#                     # Display
#                     st.success(f"✅ Answered in {elapsed:.2f}s")
#                     st.markdown("### 💡 Answer")
#                     st.markdown(answer)
                    
#                     # Sources
#                     if meta:
#                         st.info(f"📚 {format_page_refs_fast(meta)}")
                    
#                     # Save history
#                     st.session_state.chat_history.append({
#                         'query': query,
#                         'answer': answer,
#                         'time': elapsed
#                     })
#                 else:
#                     st.warning("No relevant content found.")
    
#     # Chat History
#     if st.session_state.chat_history:
#         st.divider()
#         st.subheader("📜 History")
        
#         for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
#             with st.expander(f"Q: {chat['query'][:50]}... ({chat['time']:.1f}s)"):
#                 st.markdown(f"**Q:** {chat['query']}")
#                 st.markdown(f"**A:** {chat['answer']}")

# def process_documents_turbo(uploaded_files, force_ocr: bool):
#     """Ultra-fast document processing"""
    
#     if st.session_state.processing:
#         return
    
#     st.session_state.processing = True
    
#     progress = st.progress(0)
#     status = st.empty()
    
#     start_time = time.time()
    
#     try:
#         # Clear previous
#         st.session_state.doc_texts = {}
#         st.session_state.all_chunks = []
#         st.session_state.all_metadata = []
        
#         total = len(uploaded_files)
        
#         # Process all files in parallel
#         def process_file(uploaded_file):
#             file_bytes = uploaded_file.read()
#             file_name = uploaded_file.name
            
#             if file_name.lower().endswith('.pdf'):
#                 text, pages_ok, total_pages = extract_text_from_pdf_turbo(
#                     file_bytes, file_name, force_ocr
#                 )
#             else:
#                 text = extract_text_from_image_turbo(io.BytesIO(file_bytes))
            
#             return file_name, text
        
#         # Parallel file processing
#         with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#             futures = [executor.submit(process_file, f) for f in uploaded_files]
            
#             for idx, future in enumerate(as_completed(futures)):
#                 file_name, text = future.result()
                
#                 progress.progress((idx + 1) / total)
#                 status.text(f"Processing {file_name}...")
                
#                 if text and len(text.strip()) > 30:
#                     st.session_state.doc_texts[file_name] = text
                    
#                     # Chunk
#                     chunks, meta = chunk_text_fast(text, file_name)
#                     st.session_state.all_chunks.extend(chunks)
#                     st.session_state.all_metadata.extend(meta)
        
#         # Build index
#         if st.session_state.all_chunks:
#             status.text("Building search index...")
            
#             embeddings, vectorizer = embed_texts_fast(st.session_state.all_chunks)
#             st.session_state.vectorizer = vectorizer
            
#             index = build_faiss_index_fast(embeddings)
#             st.session_state.faiss_index = index
            
#             elapsed = time.time() - start_time
            
#             progress.progress(1.0)
            
#             st.success(f"""
#             ⚡ **COMPLETED in {elapsed:.1f}s**
#             - Files: {len(st.session_state.doc_texts)}
#             - Chunks: {len(st.session_state.all_chunks)}
#             - Speed: {len(st.session_state.all_chunks)/elapsed:.1f} chunks/sec
#             """)
#         else:
#             st.warning("No text extracted.")
    
#     except Exception as e:
#         st.error(f"Error: {str(e)}")
    
#     finally:
#         st.session_state.processing = False
#         progress.empty()
#         status.empty()

# if __name__ == "__main__":
#     main()







#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000

# # app.py — IntelliDoc: AI-Powered Multilingual Document Analysis System
# import os
# import streamlit as st
# from dotenv import load_dotenv
# import fitz  # PyMuPDF
# from PIL import Image
# import pytesseract
# import numpy as np
# import faiss
# from groq import Groq
# from langdetect import detect, DetectorFactory
# from sklearn.feature_extraction.text import TfidfVectorizer
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import time
# import io
# import json
# from datetime import datetime
# import pandas as pd
# import cv2
# import base64
# import re

# DetectorFactory.seed = 0

# # -------------------- Load ENV --------------------
# load_dotenv()
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# TESSERACT_CMD = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
# TESSDATA_PREFIX = os.getenv("TESSDATA_PREFIX")

# pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
# if TESSDATA_PREFIX:
#     os.environ["TESSDATA_PREFIX"] = TESSDATA_PREFIX

#     # --- INSERT THIS CODE FOR DEBUGGING ---
# print(f"DEBUG: TESSERACT_CMD resolved to: {pytesseract.pytesseract.tesseract_cmd}")
# print(f"DEBUG: TESSDATA_PREFIX resolved to: {os.environ.get('TESSDATA_PREFIX')}")
# # --- END DEBUGGING CODE ---

# # -------------------- Config --------------------
# OCR_LANGS = "eng+kan+hin+tam+tel+mar+mal+guj+ben+pan"
# TOP_K = 8
# SIMILARITY_THRESHOLD = 0.1
# CHUNK_SIZE = 1500
# CHUNK_OVERLAP = 300
# MAX_CONTEXT_CHARS = 4500
# MAX_WORKERS = 4

# # -------------------- Clients --------------------
# groq_client = Groq(api_key=GROQ_API_KEY)

# @st.cache_resource
# def get_tfidf_vectorizer():
#     return TfidfVectorizer(
#         max_features=1000,
#         ngram_range=(1, 2),
#         min_df=1,
#         stop_words=None
#     )

# # -------------------- Table Extraction --------------------
# def extract_tables_from_pdf(pdf_bytes, doc_name):
#     """Extract tables from PDF"""
#     tables = []
#     try:
#         doc = fitz.open(stream=pdf_bytes, filetype="pdf")
#         for page_num, page in enumerate(doc, start=1):
#             try:
#                 # Extract tables using PyMuPDF
#                 page_tables = page.find_tables()
#                 if page_tables:
#                     for table_idx, table in enumerate(page_tables.tables):
#                         try:
#                             table_data = table.extract()
#                             if table_data:
#                                 df = pd.DataFrame(table_data[1:], columns=table_data[0] if table_data else None)
#                                 tables.append({
#                                     'doc': doc_name,
#                                     'page': page_num,
#                                     'table_num': table_idx + 1,
#                                     'data': df
#                                 })
#                         except:
#                             continue
#             except:
#                 continue
#         doc.close()
#     except Exception as e:
#         st.warning(f"Table extraction error: {e}")
#     return tables

# # -------------------- Chart/Diagram Detection --------------------
# def detect_visual_elements(pdf_bytes, doc_name):
#     """Detect charts, diagrams, flowcharts in PDF"""
#     visuals = []
#     try:
#         doc = fitz.open(stream=pdf_bytes, filetype="pdf")
#         for page_num, page in enumerate(doc, start=1):
#             # Get images from page
#             image_list = page.get_images()
            
#             if image_list:
#                 for img_idx, img_info in enumerate(image_list):
#                     try:
#                         xref = img_info[0]
#                         base_image = doc.extract_image(xref)
#                         image_bytes = base_image["image"]
                        
#                         # Convert to PIL Image
#                         img = Image.open(io.BytesIO(image_bytes))
                        
#                         # Analyze image to detect if it's a chart/diagram
#                         img_array = np.array(img.convert('RGB'))
                        
#                         # Simple heuristics to detect charts/diagrams
#                         # Check for lines, geometric shapes, etc.
#                         gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
#                         edges = cv2.Canny(gray, 50, 150)
                        
#                         # Count edge pixels (charts have lots of edges)
#                         edge_ratio = np.count_nonzero(edges) / edges.size
                        
#                         visual_type = "Unknown"
#                         if edge_ratio > 0.05:
#                             # Try to determine type
#                             if edge_ratio > 0.15:
#                                 visual_type = "Flowchart/Diagram"
#                             elif edge_ratio > 0.08:
#                                 visual_type = "Chart/Graph"
#                             else:
#                                 visual_type = "Image"
                        
#                         visuals.append({
#                             'doc': doc_name,
#                             'page': page_num,
#                             'type': visual_type,
#                             'image': img,
#                             'size': img.size
#                         })
#                     except:
#                         continue
#         doc.close()
#     except Exception as e:
#         st.warning(f"Visual detection error: {e}")
#     return visuals

# # -------------------- Smart OCR Detection --------------------
# def needs_ocr(page):
#     """Quickly determine if a page needs OCR"""
#     text = page.get_text("text")
#     if len(text.strip()) > 50:
#         return False
#     image_list = page.get_images()
#     if len(image_list) > 0:
#         return True
#     return True

# def extract_text_from_page_fast(page, page_num, use_ocr=False):
#     """Fast text extraction with page number tracking"""
#     if not use_ocr:
#         text = page.get_text("text")
#         if len(text.strip()) > 50:
#             return text.strip(), page_num
    
#     try:
#         mat = fitz.Matrix(2.0, 2.0)
#         pix = page.get_pixmap(matrix=mat, alpha=False)
#         img_data = pix.tobytes("png")
#         img = Image.open(io.BytesIO(img_data))
        
#         text = pytesseract.image_to_string(
#             img, 
#             lang=OCR_LANGS,
#             config='--psm 6 --oem 3'
#         )
#         return text.strip(), page_num
#     except Exception as e:
#         return "", page_num

# # -------------------- Parallel PDF Processing --------------------
# def process_page(page_info):
#     """Process a single page"""
#     page_num, page, doc_name, force_ocr = page_info
    
#     try:
#         use_ocr = force_ocr or needs_ocr(page)
#         text, pg_num = extract_text_from_page_fast(page, page_num + 1, use_ocr=use_ocr)
        
#         if text:
#             return {
#                 'page_num': page_num,
#                 'actual_page': page_num + 1,
#                 'text': text,
#                 'doc_name': doc_name,
#                 'success': True
#             }
#         return {'success': False}
#     except Exception as e:
#         return {'success': False, 'error': str(e)}

# def extract_text_from_pdf_parallel(pdf_bytes, doc_name, force_ocr=False):
#     """Extract text from PDF using parallel processing"""
#     try:
#         doc = fitz.open(stream=pdf_bytes, filetype="pdf")
#         total_pages = len(doc)
        
#         page_tasks = [
#             (i, doc[i], doc_name, force_ocr) 
#             for i in range(total_pages)
#         ]
        
#         results = []
#         with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#             future_to_page = {
#                 executor.submit(process_page, task): task[0] 
#                 for task in page_tasks
#             }
            
#             for future in as_completed(future_to_page):
#                 result = future.result()
#                 if result.get('success'):
#                     results.append(result)
        
#         doc.close()
#         results.sort(key=lambda x: x['page_num'])
        
#         full_text = ""
#         for r in results:
#             page_num = r['actual_page']
#             text = r['text']
#             full_text += f"\n\n[Page {page_num}]\n{text}"
        
#         return full_text, len(results), total_pages
        
#     except Exception as e:
#         st.error(f"PDF processing error: {e}")
#         return "", 0, 0

# def extract_text_from_image_fast(image_file):
#     """Fast image OCR"""
#     try:
#         img = Image.open(image_file)
#         max_size = 2000
#         if max(img.size) > max_size:
#             ratio = max_size / max(img.size)
#             new_size = tuple(int(dim * ratio) for dim in img.size)
#             img = img.resize(new_size, Image.Resampling.LANCZOS)
        
#         text = pytesseract.image_to_string(
#             img,
#             lang=OCR_LANGS,
#             config='--psm 6 --oem 3'
#         )
#         return text.strip()
#     except Exception as e:
#         st.warning(f"Image OCR failed: {e}")
#         return ""

# # -------------------- Chunking --------------------
# def chunk_text_smart(text, source_name, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
#     """Smart chunking with page tracking"""
#     if not text or len(text.strip()) < 50:
#         return [], []
    
#     chunks, meta = [], []
#     pages = text.split('[Page ')
    
#     for page_section in pages:
#         if not page_section.strip():
#             continue
        
#         page_num = None
#         if ']' in page_section:
#             try:
#                 page_num = int(page_section.split(']')[0])
#                 page_text = page_section.split(']', 1)[1]
#             except:
#                 page_text = page_section
#         else:
#             page_text = page_section
        
#         start = 0
#         chunk_id = 0
        
#         while start < len(page_text):
#             end = start + chunk_size
#             chunk = page_text[start:end].strip()
            
#             if chunk:
#                 chunks.append(chunk)
#                 meta.append({
#                     "source": source_name,
#                     "page": page_num,
#                     "chunk_id": chunk_id
#                 })
#                 chunk_id += 1
            
#             start += chunk_size - overlap
    
#     return chunks, meta

# # -------------------- TF-IDF Embeddings --------------------
# def embed_texts_tfidf(texts, vectorizer=None):
#     """Fast TF-IDF embeddings"""
#     try:
#         if vectorizer is None:
#             vectorizer = get_tfidf_vectorizer()
#             matrix = vectorizer.fit_transform(texts)
#         else:
#             matrix = vectorizer.transform(texts)
        
#         return matrix.toarray().astype("float32"), vectorizer
#     except Exception as e:
#         st.error(f"Embedding error: {e}")
#         return np.zeros((len(texts), 100), dtype="float32"), None

# def build_faiss_index(emb_np):
#     """Build FAISS index"""
#     if emb_np is None or emb_np.shape[0] == 0:
#         raise ValueError("Empty embeddings")
    
#     norms = np.linalg.norm(emb_np, axis=1, keepdims=True)
#     norms[norms == 0] = 1
#     emb_np = emb_np / norms
    
#     dim = emb_np.shape[1]
#     idx = faiss.IndexFlatIP(dim)
#     idx.add(emb_np)
#     return idx

# # -------------------- Language Utilities --------------------
# LANG_MAP = {
#     "kn": "Kannada", "hi": "Hindi", "ta": "Tamil", "te": "Telugu",
#     "mr": "Marathi", "ml": "Malayalam", "gu": "Gujarati",
#     "bn": "Bengali", "pa": "Punjabi", "en": "English"
# }

# LANG_CODES = {
#     "english": "en", "kannada": "kn", "hindi": "hi", "tamil": "ta",
#     "telugu": "te", "marathi": "mr", "malayalam": "ml", "gujarati": "gu",
#     "bengali": "bn", "punjabi": "pa"
# }

# #...................................................................................

# def detect_language_robust(text: str) -> str:
#     """Robust language detection with fallback"""
#     try:
#         return detect(text)
#     except LangDetectException:
#         # Fallback: check for script-specific characters
#         if re.search(r'[\u0C80-\u0CFF]', text):  # Kannada
#             return "kn"
#         elif re.search(r'[\u0900-\u097F]', text):  # Hindi
#             return "hi"
#         elif re.search(r'[\u0C00-\u0C7F]', text):  # Telugu
#             return "te"
#         elif re.search(r'[\u0B80-\u0BFF]', text):  # Tamil
#             return "ta"
#         return "en"

# def extract_target_language(question: str) -> Optional[str]:
#     """Extract target language from query"""
#     q_lower = question.lower()
#     for lang_name, code in LANG_CODES.items():
#         if f"in {lang_name}" in q_lower or f"to {lang_name}" in q_lower:
#             return code
#     return None

# # -------------------- Hallucination Reduction & Fact-Checking --------------------
# class FactualityEnhancer:
#     """Reduce hallucinations and improve factuality"""

#     @staticmethod
#     def verify_response(response: str, context: str) -> Tuple[str, float]:
#         """
#         Verify response against context
#         Returns: (verified_response, confidence_score)
#         """
#         # Check if response contains information not in context
#         response_words = set(re.findall(r'\w+', response.lower()))
#         context_words = set(re.findall(r'\w+', context.lower()))

#         # Calculate overlap
#         overlap = len(response_words.intersection(context_words))
#         total = len(response_words)

#         confidence = overlap / total if total > 0 else 0.0

#         # Flag suspicious content
#         suspicious_phrases = [
#             "i think", "probably", "might be", "could be",
#             "it seems", "appears to be", "likely"
#         ]

#         contains_speculation = any(phrase in response.lower() for phrase in suspicious_phrases)

#         if contains_speculation and confidence < 0.5:
#             verified = "Based on the available context, I cannot provide a definitive answer to this question. The information may not be present in the uploaded documents."
#             return verified, 0.3

#         return response, confidence

#     @staticmethod
#     def add_citations(response: str, metadata: List[Dict]) -> str:
#         """Add source citations to response"""
#         if not metadata:
#             return response

#         pages = set(m.get('page') for m in metadata if m.get('page'))
#         if pages:
#             citation = f"\n\n[Sources: Pages {', '.join(map(str, sorted(pages)))}]"
#             return response + citation

#         return response

# fact_checker = FactualityEnhancer()

# # -------------------- Advanced LLM Interaction with Anti-Hallucination --------------------
# def call_llm_with_retry(prompt: str, max_retries: int = 3) -> str:
#     """Call LLM with retry logic"""
#     for attempt in range(max_retries):
#         try:
#             resp = groq_client.chat.completions.create(
#                 model="llama-3.3-70b-versatile",
#                 messages=[{"role": "user", "content": prompt}],
#                 temperature=0.05,  # Very low for factuality
#                 max_tokens=MAX_OUTPUT_TOKENS,
#                 top_p=0.9
#             )
#             return resp.choices[0].message.content.strip()
#         except Exception as e:
#             # logger.warning(f"LLM attempt {attempt + 1} failed: {e}") # Use logger if available
#             print(f"LLM attempt {attempt + 1} failed: {e}")
#             if attempt < max_retries - 1:
#                 time.sleep(2 ** attempt)  # Exponential backoff
#             else:
#                 raise
#     return "Error: Unable to generate response after multiple attempts."

# def ask_groq_production(context: str, question: str, target_lang: Optional[str] = None,
#                         metadata: List[Dict] = None) -> str:
#     """Production-grade QA with anti-hallucination measures"""

#     if target_lang:
#         response_lang = LANG_MAP.get(target_lang, "the requested language")
#     else:
#         question_lang = detect_language_robust(question)
#         response_lang = LANG_MAP.get(question_lang, "English")

#     is_translation = any(kw in question.lower() for kw in
#                          ["translate", "convert", "change to"])

#     if is_translation:
#         prompt = f"""You are a professional translator. Translate the following content to {response_lang}.
# IMPORTANT INSTRUCTIONS:

# Translate ONLY the information present in the CONTEXT
# Preserve ALL details: numbers, names, dates, measurements
# Maintain formatting and structure
# Do NOT add any information not in the context
# If context is insufficient, clearly state what's missing
# CONTEXT:
# {context}
# USER REQUEST: {question}
# Translation in {response_lang}:"""
#     else:
#         prompt = f"""You are a factual document analyst. Answer based ONLY on the provided CONTEXT.
# CRITICAL ANTI-HALLUCINATION RULES:

# Answer using ONLY information explicitly stated in the CONTEXT
# If information is NOT in the context, respond: "This information is not found in the provided documents."
# Do NOT infer, guess, or add information beyond what's explicitly stated
# Quote specific details from context when possible
# If partially available, state exactly what IS found and what ISN'T
# Never use phrases like "probably", "might be", "I think" - only state facts from context
# Cite page numbers when available
# CONTEXT:
# {context}
# QUESTION: {question}
# INSTRUCTIONS FOR RESPONSE:

# Answer in {response_lang}
# Be comprehensive but ONLY use information from context
# Include ALL relevant details (numbers, names, dates, etc.)
# Structure answer clearly
# If answer requires information not in context, explicitly state what's missing
# Factual answer in {response_lang}:"""

#     try:
#         response = call_llm_with_retry(prompt)

#         # Verify factuality
#         verified_response, confidence = fact_checker.verify_response(response, context)

#         # Add citations if confidence is good
#         if confidence > 0.6 and metadata:
#             verified_response = fact_checker.add_citations(verified_response, metadata)
#             return verified_response
#     except Exception as e:
#         # logger.error(f"LLM error: {e}") # Use logger if available
#         print(f"LLM error: {e}")
#         return f"Error generating response: {str(e)}"

# def summarize_complete_production(text: str, target_lang: str = "en") -> str:
#     """Production-grade complete summarization"""
#     if not text or len(text.strip()) < 50:
#         return "Insufficient text to summarize."

#     lang_name = LANG_MAP.get(target_lang, "English")

#     # Process in chunks if needed
#     if len(text) <= MAX_SUMMARY_CHARS:
#         prompt = f"""Provide a COMPREHENSIVE summary of this document in {lang_name}.
# MANDATORY REQUIREMENTS:

# Include ALL names, IDs, reference numbers
# Include ALL dates, times, and time periods
# Include ALL numerical data (scores, amounts, grades, measurements)
# Include ALL categories, subjects, sections with their details
# Include ALL results, outcomes, conclusions, status
# Include organizational/institutional information
# Main topics and key points
# Do NOT omit ANY information
# Document:
# {text}
# Complete detailed summary in {lang_name}:"""
#         return call_llm_with_retry(prompt)

#     # Multi-pass for large documents
#     chunks = [text[i:i+MAX_SUMMARY_CHARS] for i in range(0, len(text), MAX_SUMMARY_CHARS)]
#     chunk_summaries = []
#     for idx, chunk in enumerate(chunks):
#         prompt = f"""Summarize Part {idx+1} of {len(chunks)} in {lang_name}.
# CRITICAL: Preserve ALL specific details including names, IDs, numbers, dates, scores, grades.
# Text:
# {chunk}
# Detailed summary:"""

#         try:
#             summary = call_llm_with_retry(prompt)
#             chunk_summaries.append(summary)
#         except:
#             continue

#     # Combine
#     combined = "\n\n".join(chunk_summaries)
#     final_prompt = f"""Combine these section summaries into ONE COMPLETE summary in {lang_name}.
# CRITICAL: Include ALL information without omitting any details.
# Summaries:
# {combined}
# Final comprehensive summary:"""

#     return call_llm_with_retry(final_prompt)

# def extract_all_information(text: str, target_lang: str = "en") -> str:
#     """Extract every piece of information"""
#     lang_name = LANG_MAP.get(target_lang, "English")

#     sections = [text[i:i+20000] for i in range(0, len(text), 20000)]
#     all_info = []
#     for section in sections:
#         prompt = f"""Extract and list EVERY piece of information from this section in {lang_name}.
# Create a comprehensive structured list:
# 📋 ALL names (people, places, organizations)
# 📅 ALL dates, times, periods
# 🔢 ALL numbers, scores, amounts, measurements
# 📊 ALL categories, subjects with values
# ✅ ALL results, grades, outcomes
# 🎯 ALL purposes, goals, objectives
# 📍 ALL locations, addresses, contacts
# 🆔 ALL IDs, codes, reference numbers
# 📝 ALL additional details
# Section:
# {section}
# Complete extraction:"""

#         try:
#             info = call_llm_with_retry(prompt)
#             all_info.append(info)
#         except:
#             continue

#     combined = "\n\n".join(all_info)
#     if len(sections) > 1:
#         final_prompt = f"""Organize all extracted information in {lang_name}.
# Remove duplicates. Present clearly. Do NOT omit information.
# Data:
# {combined}
# Organized complete information:"""

#         return call_llm_with_retry(final_prompt)
#     return combined

# # -------------------- Audio Output --------------------
# def text_to_speech_enhanced(text: str, lang: str = 'en') -> Optional[io.BytesIO]:
#     """Enhanced TTS with language support"""
#     try:
#         from gtts import gTTS

#         # Map language codes
#         tts_lang = lang if lang in ['en', 'hi', 'kn', 'ta', 'te', 'ml', 'mr', 'gu', 'bn', 'pa'] else 'en'

#         tts = gTTS(text=text[:500], lang=tts_lang, slow=False)  # Limit to 500 chars for speed
#         fp = io.BytesIO()
#         tts.write_to_fp(fp)
#         fp.seek(0)
#         return fp
#     except ImportError:
#         # logger.warning("gtts not installed") # Use logger if available
#         print("WARNING: gtts not installed")
#         return None
#     except Exception as e:
#         # logger.error(f"TTS error: {e}") # Use logger if available
#         print(f"TTS error: {e}")
#         return None

# def create_audio_player(audio_fp: io.BytesIO) -> str:
#     """Create HTML audio player"""
#     audio_bytes = audio_fp.read()
#     b64 = base64.b64encode(audio_bytes).decode()
#     return f'<audio controls autoplay style="width: 100%;"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'

# # -------------------- Download Functions --------------------
# def create_chat_download(chat_history: List) -> str:
#     """Create downloadable chat history"""
#     chat_text = f"IntelliDoc Chat History - Enhanced Version\n"
#     chat_text += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
#     chat_text += "="*80 + "\n\n"

#     for item in chat_history:
#         if len(item) == 2:
#             role, message = item
#             source_pages = None
#         else:
#             role, message, source_pages = item

#         if role == "user":
#             chat_text += f"USER:\n{message}\n\n"
#         else:
#             chat_text += f"INTELLIDOC:\n{message}\n"
#             if source_pages:
#                 chat_text += f"Sources: {source_pages}\n"
#             chat_text += "\n"
#         chat_text += "-"*80 + "\n\n"
#     return chat_text

# def create_json_download(chat_history: List) -> str:
#     """Create JSON format chat history"""
#     conversations = []
#     for item in chat_history:
#         if len(item) == 2:
#             role, message = item
#             source_pages = None
#         else:
#             role, message, source_pages = item

#         conversations.append({
#             "role": role,
#             "message": message,
#             "sources": source_pages if role == "assistant" else None,
#             "timestamp": datetime.now().isoformat()
#         })
#     chat_data = {
#         "version": "IntelliDoc Enhanced v2.0",
#         "timestamp": datetime.now().isoformat(),
#         "conversation": conversations
#     }
#     return json.dumps(chat_data, indent=2, ensure_ascii=False)

# #.....................................................................................

# def detect_question_language(question):
#     try:
#         return detect(question)
#     except:
#         return "en"

# def extract_target_language(question):
#     q_lower = question.lower()
#     for lang_name, code in LANG_CODES.items():
#         if f"in {lang_name}" in q_lower or f"to {lang_name}" in q_lower:
#             return code
#     return None

# # -------------------- Audio Functions --------------------
# def get_audio_input():
#     """Get audio input using HTML5 audio recorder"""
#     audio_html = """
#     <div style="text-align: center; padding: 20px;">
#         <p style="color: #E0E0E0;">🎤 <strong>Voice Input Feature</strong></p>
#         <p style="color: #888; font-size: 0.9em;">Note: Audio input requires microphone permissions.<br>
#         Install speech_recognition and pyaudio packages for full functionality:<br>
#         <code>pip install SpeechRecognition pyaudio</code></p>
#     </div>
#     """
#     st.markdown(audio_html, unsafe_allow_html=True)

# def text_to_speech_button(text, lang='en'):
#     """Create audio download button"""
#     try:
#         # Note: Install gtts package: pip install gtts
#         from gtts import gTTS
        
#         tts = gTTS(text=text, lang=lang, slow=False)
#         fp = io.BytesIO()
#         tts.write_to_fp(fp)
#         fp.seek(0)
        
#         audio_base64 = base64.b64encode(fp.read()).decode()
#         audio_html = f"""
#         <audio controls autoplay style="width: 100%;">
#             <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
#         </audio>
#         """
#         st.markdown(audio_html, unsafe_allow_html=True)
#         return True
#     except ImportError:
#         st.info("💡 Install gtts for audio output: `pip install gtts`")
#         return False
#     except Exception as e:
#         st.warning(f"Audio generation error: {e}")
#         return False

# # -------------------- QA & Summarization --------------------
# def ask_groq_with_context(context, question, target_lang=None):
#     """Answer questions using context"""
#     if target_lang:
#         response_lang = LANG_MAP.get(target_lang, "the requested language")
#     else:
#         question_lang = detect_question_language(question)
#         response_lang = LANG_MAP.get(question_lang, "English")
    
#     is_translation = any(kw in question.lower() for kw in 
#                          ["translate", "convert", "change to", "in kannada", "in hindi", 
#                           "in telugu", "in marathi", "in tamil", "in english"])
    
#     if is_translation:
#         prompt = f"""You are a multilingual translator.

# CONTEXT from documents:
# {context}

# USER REQUEST: {question}

# Translate the relevant information to {response_lang}. Be accurate and preserve details.

# Response in {response_lang}:"""
#     else:
#         prompt = f"""Answer this question using ONLY the CONTEXT provided.

# CONTEXT:
# {context}

# QUESTION: {question}

# Answer in {response_lang}. If not in context, say "I cannot find this in the documents."

# Answer:"""
    
#     try:
#         resp = groq_client.chat.completions.create(
#             model="llama-3.3-70b-versatile",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.1,
#             max_tokens=1000
#         )
#         return resp.choices[0].message.content.strip()
#     except Exception as e:
#         return f"Error: {e}"

# def summarize_document(text, target_lang="en"):
#     """Generate summary"""
#     if not text or len(text.strip()) < 50:
#         return "Insufficient text to summarize."
    
#     text_snippet = text[:8000]
#     lang_name = LANG_MAP.get(target_lang, "English")
    
#     prompt = f"""Summarize this document in {lang_name}. Include main topic, key points, and important details.

# {text_snippet}

# Summary:"""
    
#     try:
#         resp = groq_client.chat.completions.create(
#             model="llama-3.3-70b-versatile",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.2,
#             max_tokens=800
#         )
#         return resp.choices[0].message.content.strip()
#     except Exception as e:
#         return f"Summarization failed: {e}"

# # -------------------- Download Functions --------------------
# def create_chat_download(chat_history):
#     """Create downloadable chat history"""
#     chat_text = f"IntelliDoc Chat History\n"
#     chat_text += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
#     chat_text += "="*80 + "\n\n"
    
#     for item in chat_history:
#         if len(item) == 2:
#             role, message = item
#             source_pages = None
#         else:
#             role, message, source_pages = item
        
#         if role == "user":
#             chat_text += f"USER:\n{message}\n\n"
#         else:
#             chat_text += f"INTELLIDOC:\n{message}\n"
#             if source_pages:
#                 chat_text += f"Sources: {source_pages}\n"
#             chat_text += "\n"
#         chat_text += "-"*80 + "\n\n"
    
#     return chat_text

# def create_json_download(chat_history):
#     """Create JSON format chat history"""
#     conversations = []
#     for item in chat_history:
#         if len(item) == 2:
#             role, message = item
#             source_pages = None
#         else:
#             role, message, source_pages = item
        
#         conversations.append({
#             "role": role,
#             "message": message,
#             "sources": source_pages if role == "assistant" else None
#         })
    
#     chat_data = {
#         "timestamp": datetime.now().isoformat(),
#         "conversation": conversations
#     }
#     return json.dumps(chat_data, indent=2, ensure_ascii=False)

# # -------------------- Custom CSS (Dark Theme) --------------------
# # -------------------- Custom CSS (Professional Subtle Dark Theme) --------------------
# def load_custom_css():
#     st.markdown("""
#     <style>
#     /* Professional Dark theme */
#     .stApp {
#         background: #1e1e1e;
#     }
    
#     /* Header */
#     .main-header {
#         background: #ffffff;
#         padding: 35px;
#         border-radius: 12px;
#         text-align: center;
#         margin-bottom: 25px;
#         box-shadow: 0 4px 15px rgba(0,0,0,0.3);
#         border-bottom: 2px solid #4a9eff;
#     }
    
#     .main-header h1 {
#         color: #000000;
#         font-size: 2.5em;
#         font-weight: 700;
#         margin: 0;
#         letter-spacing: 1px;
#     }
    
#     .main-header p {
#         color: #000000;
#         font-size: 1.1em;
#         margin: 10px 0 0 0;
#         font-weight: 400;
#     }
    
#     /* Feature banner */
#     .feature-banner {
#         text-align: center;
#         padding: 12px;
#         background: #2d2d2d;
#         border-radius: 8px;
#         margin-bottom: 20px;
#         border: 1px solid #3a3a3a;
#     }
    
#     .feature-banner p {
#         color: #b0b0b0;
#         font-size: 0.95em;
#         margin: 0;
#     }
    
#     /* Chat messages */
#     .user-message {
#         background: #2d2d2d;
#         color: #e0e0e0;
#         padding: 18px;
#         border-radius: 12px 12px 4px 12px;
#         margin: 12px 0;
#         border-left: 3px solid #4a9eff;
#         box-shadow: 0 2px 8px rgba(0,0,0,0.2);
#     }
    
#     .bot-message {
#         background: #252525;
#         color: #d0d0d0;
#         padding: 18px;
#         border-radius: 12px 12px 12px 4px;
#         margin: 12px 0;
#         border-left: 3px solid #6c63ff;
#         box-shadow: 0 2px 8px rgba(0,0,0,0.2);
#     }
    
#     .page-reference {
#         display: inline-block;
#         background: #4a9eff;
#         color: white;
#         padding: 4px 12px;
#         border-radius: 12px;
#         font-size: 0.8em;
#         font-weight: 600;
#         margin: 8px 4px 0 0;
#         box-shadow: 0 2px 6px rgba(74,158,255,0.3);
#     }
    
#     /* Buttons */
#     .stButton>button {
#         background: linear-gradient(135deg, #4a9eff 0%, #6c63ff 100%);
#         color: white;
#         border: none;
#         border-radius: 8px;
#         padding: 10px 24px;
#         font-weight: 600;
#         transition: all 0.3s;
#         box-shadow: 0 3px 10px rgba(74,158,255,0.3);
#     }
    
#     .stButton>button:hover {
#         transform: translateY(-2px);
#         box-shadow: 0 5px 15px rgba(74,158,255,0.4);
#         background: linear-gradient(135deg, #5aa3ff 0%, #7d73ff 100%);
#     }
    
#     /* Sidebar */
#     [data-testid="stSidebar"] {
#         background: #252525;
#         border-right: 1px solid #3a3a3a;
#     }
    
#     [data-testid="stSidebar"] h3 {
#         color: #e0e0e0;
#     }
    
#     [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
#         color: #b0b0b0;
#     }
    
#     /* Success/Info boxes */
#     .stSuccess {
#         background-color: rgba(74,158,255,0.1);
#         border-left: 4px solid #4a9eff;
#         color: #4a9eff;
#         border-radius: 4px;
#     }
    
#     .stInfo {
#         background-color: rgba(108,99,255,0.1);
#         border-left: 4px solid #6c63ff;
#         color: #6c63ff;
#         border-radius: 4px;
#     }
    
#     .stWarning {
#         background-color: rgba(255,179,0,0.1);
#         border-left: 4px solid #ffb300;
#         color: #ffb300;
#         border-radius: 4px;
#     }
    
#     .stError {
#         background-color: rgba(255,82,82,0.1);
#         border-left: 4px solid #ff5252;
#         color: #ff5252;
#         border-radius: 4px;
#     }
    
#     /* File uploader */
#     [data-testid="stFileUploader"] {
#         background: #2d2d2d;
#         border: 2px dashed #4a9eff;
#         border-radius: 10px;
#         padding: 20px;
#     }
    
#     [data-testid="stFileUploader"] label {
#         color: #b0b0b0;
#     }
    
#     /* Input boxes */
#     .stTextInput>div>div>input {
#         background: #2d2d2d;
#         color: #e0e0e0;
#         border: 1px solid #4a4a4a;
#         border-radius: 8px;
#         padding: 10px;
#     }
    
#     .stTextInput>div>div>input:focus {
#         border-color: #4a9eff;
#         box-shadow: 0 0 0 1px #4a9eff;
#     }
    
#     /* Expander */
#     .streamlit-expanderHeader {
#         background: #2d2d2d;
#         border-radius: 8px;
#         color: #e0e0e0;
#         font-weight: 600;
#         border: 1px solid #3a3a3a;
#     }
    
#     .streamlit-expanderHeader:hover {
#         background: #353535;
#         border-color: #4a9eff;
#     }
    
#     /* Dataframe */
#     .stDataFrame {
#         background: #2d2d2d;
#         border-radius: 8px;
#     }
    
#     /* Markdown text */
#     .stMarkdown {
#         color: #d0d0d0;
#     }
    
#     /* Checkbox */
#     .stCheckbox {
#         color: #b0b0b0;
#     }
    
#     /* Radio buttons */
#     .stRadio > label {
#         color: #e0e0e0;
#     }
    
#     .stRadio > div {
#         color: #b0b0b0;
#     }
    
#     /* Footer */
#     .footer {
#         text-align: center;
#         color: #808080;
#         padding: 25px;
#         margin-top: 40px;
#         border-top: 1px solid #3a3a3a;
#     }
    
#     .footer h3 {
#         color: #e0e0e0;
#         margin-bottom: 15px;
#         font-size: 1.3em;
#     }
    
#     .footer a {
#         color: #4a9eff;
#         text-decoration: none;
#         transition: color 0.3s;
#     }
    
#     .footer a:hover {
#         color: #6c63ff;
#     }
    
#     /* Scrollbar */
#     ::-webkit-scrollbar {
#         width: 10px;
#         height: 10px;
#     }
    
#     ::-webkit-scrollbar-track {
#         background: #1e1e1e;
#     }
    
#     ::-webkit-scrollbar-thumb {
#         background: #4a4a4a;
#         border-radius: 5px;
#     }
    
#     ::-webkit-scrollbar-thumb:hover {
#         background: #5a5a5a;
#     }
    
#     /* Divider */
#     hr {
#         border-color: #3a3a3a;
#     }
    
#     /* Download buttons */
#     .stDownloadButton>button {
#         background: #2d2d2d;
#         color: #4a9eff;
#         border: 1px solid #4a9eff;
#         border-radius: 8px;
#         padding: 10px 20px;
#         font-weight: 600;
#         transition: all 0.3s;
#     }
    
#     .stDownloadButton>button:hover {
#         background: #4a9eff;
#         color: white;
#         transform: translateY(-2px);
#     }
    
#     /* Feature grid */
#     .feature-grid {
#         display: grid;
#         grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
#         gap: 15px;
#         margin: 20px 0;
#     }
    
#     .feature-card {
#         background: #2d2d2d;
#         padding: 15px;
#         border-radius: 8px;
#         border: 1px solid #3a3a3a;
#         transition: all 0.3s;
#     }
    
#     .feature-card:hover {
#         border-color: #4a9eff;
#         transform: translateY(-2px);
#         box-shadow: 0 4px 12px rgba(74,158,255,0.2);
#     }
    
#     .feature-card strong {
#         color: #4a9eff;
#         display: block;
#         margin-bottom: 5px;
#     }
    
#     .feature-card span {
#         color: #909090;
#         font-size: 0.9em;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# # -------------------- Streamlit UI --------------------
# st.set_page_config(
#     page_title="IntelliDoc - AI Document Analysis",
#     page_icon="📄",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Load custom CSS
# load_custom_css()

# # Custom Header
# # Custom Header
# st.markdown("""
# <div class="main-header">
#     <h1>📄 IntelliDoc</h1>
#     <p>AI-Powered Multilingual Document Analysis System</p>
# </div>
# """, unsafe_allow_html=True)

# st.markdown("""
# <div class="feature-banner">
#     <p>
#         ⚡ <strong>Features:</strong> Printed & Handwritten OCR | 10+ Languages | Real-Time Translation | 
#         Table Extraction | Chart Detection | Audio I/O | 3x Faster Processing
#     </p>
# </div>
# """, unsafe_allow_html=True)

# # Session state
# for key, val in [("chunks", []), ("meta", []), ("embeddings", None),
#                  ("faiss_index", None), ("full_text", ""), ("chat_history", []),
#                  ("file_info", {}), ("vectorizer", None), ("tables", []), 
#                  ("visuals", [])]:
#     if key not in st.session_state:
#         st.session_state[key] = val

# # Sidebar
# st.sidebar.markdown("### 📂 Document Upload")

# doc_mode = st.sidebar.radio(
#     "Processing Mode:",
#     ["🚀 Smart (Auto-detect)", "🔍 Force OCR (Scanned/Handwritten)"],
#     help="Smart mode automatically detects which pages need OCR"
# )

# uploaded_pdfs = st.sidebar.file_uploader(
#     "📑 PDF Files", 
#     type="pdf", 
#     accept_multiple_files=True,
#     key="pdf_uploader"
# )

# uploaded_images = st.sidebar.file_uploader(
#     "🖼️ Image Files", 
#     type=["jpg", "jpeg", "png", "bmp", "tiff"],
#     accept_multiple_files=True,
#     key="img_uploader"
# )

# st.sidebar.markdown("### ⚙️ Advanced Options")
# extract_tables = st.sidebar.checkbox("📊 Extract Tables", value=True)
# detect_visuals = st.sidebar.checkbox("📈 Detect Charts/Diagrams", value=True)

# process_btn = st.sidebar.button("🚀 Process Documents", type="primary", use_container_width=True)

# # Processing
# if process_btn:
#     # Clear previous data
#     for key in ["chunks", "meta", "embeddings", "faiss_index", "full_text", 
#                 "file_info", "chat_history", "tables", "visuals"]:
#         if key in ["chunks", "meta", "file_info", "chat_history", "tables", "visuals"]:
#             st.session_state[key].clear()
#         else:
#             st.session_state[key] = None if key != "full_text" else ""
    
#     force_ocr = ("Force OCR" in doc_mode)
#     total_start = time.time()
    
#     with st.spinner("⚡ Processing documents with parallel OCR..."):
#         # Process PDFs
#         for pdf in uploaded_pdfs or []:
#             start_time = time.time()
#             st.info(f"📄 Processing: {pdf.name}")
#             pdf_bytes = pdf.read()
            
#             # Extract tables
#             if extract_tables:
#                 with st.spinner(f"📊 Extracting tables from {pdf.name}..."):
#                     tables = extract_tables_from_pdf(pdf_bytes, pdf.name)
#                     if tables:
#                         st.session_state.tables.extend(tables)
#                         st.success(f"✅ Found {len(tables)} table(s)")
            
#             # Detect visuals
#             if detect_visuals:
#                 with st.spinner(f"📈 Detecting charts/diagrams in {pdf.name}..."):
#                     visuals = detect_visual_elements(pdf_bytes, pdf.name)
#                     if visuals:
#                         st.session_state.visuals.extend(visuals)
#                         st.success(f"✅ Found {len(visuals)} visual element(s)")
            
#             # Extract text
#             text, pages_processed, total_pages = extract_text_from_pdf_parallel(
#                 pdf_bytes, pdf.name, force_ocr=force_ocr
#             )
            
#             processing_time = time.time() - start_time
            
#             if text:
#                 st.session_state.full_text += f"\n\n=== FILE: {pdf.name} ===\n{text}"
#                 ch, md = chunk_text_smart(text, pdf.name)
#                 st.session_state.chunks.extend(ch)
#                 st.session_state.meta.extend(md)
                
#                 st.session_state.file_info[pdf.name] = {
#                     'chars': len(text),
#                     'pages': total_pages,
#                     'time': processing_time
#                 }
                
#                 speed = total_pages / processing_time if processing_time > 0 else 0
#                 st.success(
#                     f"✅ {pdf.name}: {total_pages} pages in {processing_time:.1f}s "
#                     f"({speed:.1f} pages/sec)"
#                 )
#             else:
#                 st.warning(f"⚠️ Could not extract text from {pdf.name}")
        
#         # Process Images
#         for img in uploaded_images or []:
#             start_time = time.time()
#             st.info(f"🖼️ Processing: {img.name}")
#             text = extract_text_from_image_fast(img)
#             processing_time = time.time() - start_time
            
#             if text:
              
#                 st.session_state.full_text += f"\n\n=== IMAGE: {img.name} ===\n{text}"
#                 ch, md = chunk_text_smart(text, img.name)
#                 st.session_state.chunks.extend(ch)
#                 st.session_state.meta.extend(md)
                
#                 st.session_state.file_info[img.name] = {
#                     'chars': len(text),
#                     'time': processing_time
#                 }
#                 st.success(f"✅ {img.name}: {len(text)} chars in {processing_time:.1f}s")
#             else:
#                 st.warning(f"⚠️ Could not extract text from {img.name}")
        
#         # Build embeddings
#         if st.session_state.chunks:
#             st.info(f"🔢 Creating search index for {len(st.session_state.chunks)} chunks...")
#             embed_start = time.time()
            
#             emb_np, vectorizer = embed_texts_tfidf(st.session_state.chunks)
#             st.session_state.embeddings = emb_np
#             st.session_state.vectorizer = vectorizer
#             st.session_state.faiss_index = build_faiss_index(emb_np)
            
#             embed_time = time.time() - embed_start
#             total_time = time.time() - total_start
            
#             st.success(
#                 f"✅ Search index created in {embed_time:.1f}s | "
#                 f"Total: {total_time:.1f}s"
#             )
            
#             # Show statistics
#             st.sidebar.markdown("### 📊 Processing Stats")
#             total_docs = len(st.session_state.file_info)
#             total_chunks = len(st.session_state.chunks)
#             st.sidebar.markdown(f"**Documents:** {total_docs} | **Chunks:** {total_chunks}")
            
#             for fname, info in st.session_state.file_info.items():
#                 if 'pages' in info:
#                     st.sidebar.text(f"📄 {fname[:25]}...: {info['pages']}p ({info['time']:.1f}s)")
#                 else:
#                     st.sidebar.text(f"🖼️ {fname[:25]}...: ({info['time']:.1f}s)")
            
#             if st.session_state.tables:
#                 st.sidebar.markdown(f"**Tables Found:** {len(st.session_state.tables)}")
            
#             if st.session_state.visuals:
#                 st.sidebar.markdown(f"**Visuals Found:** {len(st.session_state.visuals)}")
#         else:
#             st.error("❌ No text extracted. Check file quality and OCR settings.")

# # Display extracted tables
# if st.session_state.tables:
#     with st.expander(f"📊 Extracted Tables ({len(st.session_state.tables)} found)", expanded=False):
#         for table_info in st.session_state.tables:
#             st.markdown(f"**📄 {table_info['doc']} - Page {table_info['page']}, Table {table_info['table_num']}**")
#             st.dataframe(table_info['data'], use_container_width=True)
#             st.markdown("---")

# # Display detected visuals
# if st.session_state.visuals:
#     with st.expander(f"📈 Detected Charts & Diagrams ({len(st.session_state.visuals)} found)", expanded=False):
#         cols = st.columns(3)
#         for idx, visual in enumerate(st.session_state.visuals):
#             col = cols[idx % 3]
#             with col:
#                 st.image(visual['image'], caption=f"{visual['doc']} - Page {visual['page']}\n{visual['type']}", use_container_width=True)

# # Chat Interface
# st.markdown("---")
# st.markdown("### 💬 Chat with Your Documents")

# if not st.session_state.faiss_index:
#     st.info("👆 Upload and process documents using the sidebar first.")
# else:
#     # Audio input section
#     with st.expander("🎤 Voice Input (Optional)", expanded=False):
#         get_audio_input()
    
#     # Text input
#     user_input = st.text_input(
#         "Your question:",
#         placeholder="E.g., 'Summarize in Kannada', 'What dates are mentioned?', 'Translate to Hindi'",
#         key="user_question"
#     )
    
#     col1, col2 = st.columns([3, 1])
#     with col1:
#         ask_button = st.button("📤 Send Question", type="primary", use_container_width=True)
#     with col2:
#         audio_output = st.checkbox("🔊 Audio", value=False)
    
#     with st.expander("💡 Example Questions", expanded=False):
#         st.markdown("""
#         **General Queries:**
#         - "Summarize this document in Kannada"
#         - "What is the main topic?"
#         - "List all important dates mentioned"
        
#         **Translation:**
#         - "Translate the main points to Telugu"
#         - "Convert this to Hindi"
        
#         **Specific Information:**
#         - "What are the names of people mentioned?"
#         - "Extract all numerical values"
#         - "What tables are present?"
#         """)
    
#     if ask_button and user_input:
#         st.session_state.chat_history.append(("user", user_input))
        
#         with st.spinner("🤔 Analyzing..."):
#             is_summary = any(kw in user_input.lower() for kw in 
#                            ["summary", "summarize", "overview", "brief", "main points"])
            
#             target_lang = extract_target_language(user_input)
#             if not target_lang:
#                 target_lang = detect_question_language(user_input)
            
#             if is_summary:
#                 answer = summarize_document(st.session_state.full_text, target_lang)
#                 source_pages = "Summary from entire document"
#             else:
#                 qvec, _ = embed_texts_tfidf(
#                     [user_input], 
#                     vectorizer=st.session_state.vectorizer
#                 )
#                 qvec = qvec.astype("float32")
                
#                 norm = np.linalg.norm(qvec)
#                 if norm > 0:
#                     qvec = qvec / norm
                
#                 D, I = st.session_state.faiss_index.search(qvec, TOP_K)
                
#                 if len(D[0]) == 0 or D[0][0] < SIMILARITY_THRESHOLD:
#                     answer = "I cannot find relevant information in the documents."
#                     source_pages = "N/A"
#                 else:
#                     context_pieces = []
#                     total_chars = 0
#                     source_pages_set = set()
                    
#                     for idx, score in zip(I[0], D[0]):
#                         if idx >= len(st.session_state.chunks):
#                             continue
                        
#                         chunk = st.session_state.chunks[idx]
#                         source = st.session_state.meta[idx]['source']
#                         page = st.session_state.meta[idx].get('page', '?')
                        
#                         if page and page != '?':
#                             source_pages_set.add(f"Page {page}")
                        
#                         piece = f"[Source: {source}, Page: {page}]\n{chunk}"
                        
#                         if total_chars + len(piece) > MAX_CONTEXT_CHARS:
#                             break
                        
#                         context_pieces.append(piece)
#                         total_chars += len(piece)
                    
#                     context = "\n\n---\n\n".join(context_pieces)
#                     answer = ask_groq_with_context(context, user_input, target_lang)
                    
#                     # Sort pages numerically
#                     try:
#                         sorted_pages = sorted(source_pages_set, key=lambda x: int(re.findall(r'\d+', x)[0]))
#                         source_pages = " | ".join(sorted_pages) if sorted_pages else "N/A"
#                     except:
#                         source_pages = " | ".join(sorted(source_pages_set)) if source_pages_set else "N/A"
            
#             # Add answer with source pages
#             st.session_state.chat_history.append(("assistant", answer, source_pages))
            
#             # Audio output if enabled
#             if audio_output and answer:
#                 with st.spinner("🔊 Generating audio..."):
#                     text_to_speech_button(answer, lang=target_lang)
    
#     # Display chat with custom styling
#     st.markdown("### 📜 Conversation History")
    
#     if not st.session_state.chat_history:
#         st.info("💭 No conversation yet. Ask your first question above!")
    
#     for item in st.session_state.chat_history:
#         if len(item) == 2:
#             role, message = item
#             source_pages = None
#         else:
#             role, message, source_pages = item
        
#         if role == "user":
#             st.markdown(
#                 f'<div class="user-message"><strong>👤 You:</strong><br><br>{message}</div>', 
#                 unsafe_allow_html=True
#             )
#         else:
#             page_badges = ""
#             if source_pages and source_pages not in ["N/A", "Summary from entire document"]:
#                 for page in source_pages.split(" | "):
#                     page_badges += f'<span class="page-reference">📄 {page}</span>'
#             elif source_pages == "Summary from entire document":
#                 page_badges = '<span class="page-reference">📄 Full Document</span>'
            
#             st.markdown(
#                 f'<div class="bot-message"><strong>🤖 IntelliDoc:</strong><br>{page_badges}<br><br>{message}</div>', 
#                 unsafe_allow_html=True
#             )
#         st.markdown("<br>", unsafe_allow_html=True)
    
#     # Download and Clear buttons
#     if st.session_state.chat_history:
#         st.markdown("---")
#         col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
        
#         with col1:
#             chat_txt = create_chat_download(st.session_state.chat_history)
#             st.download_button(
#                 label="📥 Download TXT",
#                 data=chat_txt,
#                 file_name=f"intellidoc_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
#                 mime="text/plain",
#                 use_container_width=True
#             )
        
#         with col2:
#             chat_json = create_json_download(st.session_state.chat_history)
#             st.download_button(
#                 label="📥 Download JSON",
#                 data=chat_json,
#                 file_name=f"intellidoc_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
#                 mime="application/json",
#                 use_container_width=True
#             )
        
#         with col3:
#             # Download full extracted text
#             st.download_button(
#                 label="📄 Extracted Text",
#                 data=st.session_state.full_text,
#                 file_name=f"extracted_text_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
#                 mime="text/plain",
#                 use_container_width=True
#             )
        
#         with col4:
#             if st.button("🗑️ Clear Chat", use_container_width=True):
#                 st.session_state.chat_history.clear()
#                 st.rerun()

# # Footer
# # Footer
# st.markdown("---")
# st.markdown("""
# <div class="footer">
#     <h3>⚡ Performance & Features</h3>
#     <div class="feature-grid">
#         <div class="feature-card">
#             <strong>🚀 Smart OCR</strong>
#             <span>Auto-detects when to use OCR</span>
#         </div>
#         <div class="feature-card">
#             <strong>⚡ 3x Faster</strong>
#             <span>Parallel processing</span>
#         </div>
#         <div class="feature-card">
#             <strong>🌐 10+ Languages</strong>
#             <span>Multilingual support</span>
#         </div>
#         <div class="feature-card">
#             <strong>📊 Smart Tables</strong>
#             <span>Automatic extraction</span>
#         </div>
#         <div class="feature-card">
#             <strong>📈 Chart Detection</strong>
#             <span>Visual element recognition</span>
#         </div>
#         <div class="feature-card">
#             <strong>🔊 Audio I/O</strong>
#             <span>Voice input & output</span>
#         </div>
#     </div>
#     <p style="margin-top: 20px; font-size: 0.9em;">
#         <strong>IntelliDoc</strong> v1.0 | Built with Streamlit, PyMuPDF, Tesseract OCR, FAISS & Groq LLM<br>
#         Supporting: English, Kannada, Hindi, Tamil, Telugu, Marathi, Malayalam, Gujarati, Bengali, Punjabi
#     </p>
# </div>
# """, unsafe_allow_html=True)














































































































# ===================================================================================================================================
# # Final - code 


# #app.py — Ultra-Fast Multilingual PDF Chatbot with Parallel Processing
# import os
# import streamlit as st
# from dotenv import load_dotenv
# import fitz  # PyMuPDF
# from PIL import Image
# import pytesseract
# import numpy as np
# import faiss
# from groq import Groq
# from langdetect import detect, DetectorFactory
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import time
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# DetectorFactory.seed = 0

# # -------------------- Load ENV --------------------
# load_dotenv()
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# TESSERACT_CMD = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
# TESSDATA_PREFIX = os.getenv("TESSDATA_PREFIX")

# pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
# if TESSDATA_PREFIX:
#     os.environ["TESSDATA_PREFIX"] = TESSDATA_PREFIX

# # -------------------- Config --------------------
# OCR_LANGS = "eng+kan+hin+tam+tel+mar+mal+guj+ben+pan"
# TOP_K = 8
# SIMILARITY_THRESHOLD = 0.1
# CHUNK_SIZE = 1500
# CHUNK_OVERLAP = 300
# MAX_CONTEXT_CHARS = 4500
# MAX_WORKERS = 4  # Parallel processing threads

# # -------------------- Clients --------------------
# groq_client = Groq(api_key=GROQ_API_KEY)

# # TF-IDF vectorizer for embeddings (no ML libraries needed)
# @st.cache_resource
# def get_tfidf_vectorizer():
#     return TfidfVectorizer(
#         max_features=1000,
#         ngram_range=(1, 2),
#         min_df=1,
#         stop_words=None  # Keep all words for multilingual support
#     )

# # -------------------- Smart OCR Detection --------------------
# def needs_ocr(page):
#     """Quickly determine if a page needs OCR"""
#     # Try to extract text
#     text = page.get_text("text")
    
#     # Check if page has extractable text
#     if len(text.strip()) > 50:
#         return False
    
#     # Check if page has images (likely scanned)
#     image_list = page.get_images()
#     if len(image_list) > 0:
#         return True
    
#     return True  # Default to OCR if uncertain


# def extract_text_from_page_fast(page, use_ocr=False):
#     """Fast text extraction with optional OCR"""
#     if not use_ocr:
#         # Try native extraction first (FAST)
#         text = page.get_text("text")
#         if len(text.strip()) > 50:
#             return text.strip()
    
#     # Use PyMuPDF's integrated OCR (FASTER than pdf2image + pytesseract)
#     try:
#         # Get page as pixmap at optimal resolution
#         mat = fitz.Matrix(2.0, 2.0)  # 2x zoom = ~144 DPI (good balance)
#         pix = page.get_pixmap(matrix=mat, alpha=False)
        
#         # Convert to PIL Image
#         img_data = pix.tobytes("png")
#         img = Image.open(io.BytesIO(img_data))
        
#         # Fast OCR with optimized config
#         text = pytesseract.image_to_string(
#             img, 
#             lang=OCR_LANGS,
#             config='--psm 6 --oem 3'  # Fast mode
#         )
        
#         return text.strip()
#     except Exception as e:
#         st.warning(f"OCR failed on page: {e}")
#         return ""


# # -------------------- Parallel PDF Processing --------------------
# def process_page(page_info):
#     """Process a single page (for parallel execution)"""
#     page_num, page, doc_name, force_ocr = page_info
    
#     try:
#         # Determine if OCR is needed
#         use_ocr = force_ocr or needs_ocr(page)
        
#         # Extract text
#         text = extract_text_from_page_fast(page, use_ocr=use_ocr)
        
#         if text:
#             return {
#                 'page_num': page_num,
#                 'text': f"\n\n[Page {page_num + 1}]\n{text}",
#                 'doc_name': doc_name,
#                 'success': True
#             }
#         return {'success': False}
#     except Exception as e:
#         return {'success': False, 'error': str(e)}


# def extract_text_from_pdf_parallel(pdf_bytes, doc_name, force_ocr=False):
#     """Extract text from PDF using parallel processing"""
#     try:
#         doc = fitz.open(stream=pdf_bytes, filetype="pdf")
#         total_pages = len(doc)
        
#         # Create page tasks
#         page_tasks = [
#             (i, doc[i], doc_name, force_ocr) 
#             for i in range(total_pages)
#         ]
        
#         # Process pages in parallel
#         results = []
#         with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#             future_to_page = {
#                 executor.submit(process_page, task): task[0] 
#                 for task in page_tasks
#             }
            
#             for future in as_completed(future_to_page):
#                 result = future.result()
#                 if result.get('success'):
#                     results.append(result)
        
#         doc.close()
        
#         # Sort by page number and combine
#         results.sort(key=lambda x: x['page_num'])
#         full_text = ''.join([r['text'] for r in results])
        
#         return full_text, len(results), total_pages
        
#     except Exception as e:
#         st.error(f"PDF processing error: {e}")
#         return "", 0, 0


# def extract_text_from_image_fast(image_file):
#     """Fast image OCR"""
#     try:
#         img = Image.open(image_file)
        
#         # Resize if too large (faster processing)
#         max_size = 2000
#         if max(img.size) > max_size:
#             ratio = max_size / max(img.size)
#             new_size = tuple(int(dim * ratio) for dim in img.size)
#             img = img.resize(new_size, Image.Resampling.LANCZOS)
        
#         # Fast OCR
#         text = pytesseract.image_to_string(
#             img,
#             lang=OCR_LANGS,
#             config='--psm 6 --oem 3'
#         )
        
#         return text.strip()
#     except Exception as e:
#         st.warning(f"Image OCR failed: {e}")
#         return ""


# # -------------------- Chunking --------------------
# def chunk_text_smart(text, source_name, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
#     """Smart chunking that preserves context"""
#     if not text or len(text.strip()) < 50:
#         return [], []
    
#     chunks, meta = [], []
    
#     # Split by pages first
#     pages = text.split('[Page ')
    
#     for page_section in pages:
#         if not page_section.strip():
#             continue
        
#         # Extract page number if present
#         page_num = None
#         if ']' in page_section:
#             try:
#                 page_num = int(page_section.split(']')[0])
#                 page_text = page_section.split(']', 1)[1]
#             except:
#                 page_text = page_section
#         else:
#             page_text = page_section
        
#         # Chunk the page text
#         start = 0
#         chunk_id = 0
        
#         while start < len(page_text):
#             end = start + chunk_size
#             chunk = page_text[start:end].strip()
            
#             if chunk:
#                 chunks.append(chunk)
#                 meta.append({
#                     "source": source_name,
#                     "page": page_num,
#                     "chunk_id": chunk_id
#                 })
#                 chunk_id += 1
            
#             start += chunk_size - overlap
    
#     return chunks, meta


# # -------------------- TF-IDF Embeddings (No PyTorch needed) --------------------
# def embed_texts_tfidf(texts, vectorizer=None, existing_matrix=None):
#     """Fast TF-IDF embeddings without ML libraries"""
#     try:
#         if vectorizer is None:
#             vectorizer = get_tfidf_vectorizer()
#             matrix = vectorizer.fit_transform(texts)
#         else:
#             matrix = vectorizer.transform(texts)
        
#         return matrix.toarray().astype("float32"), vectorizer
#     except Exception as e:
#         st.error(f"Embedding error: {e}")
#         return np.zeros((len(texts), 100), dtype="float32"), None


# def build_faiss_index(emb_np):
#     """Build FAISS index"""
#     if emb_np is None or emb_np.shape[0] == 0:
#         raise ValueError("Empty embeddings")
    
#     # Normalize for cosine similarity
#     norms = np.linalg.norm(emb_np, axis=1, keepdims=True)
#     norms[norms == 0] = 1  # Avoid division by zero
#     emb_np = emb_np / norms
    
#     dim = emb_np.shape[1]
#     idx = faiss.IndexFlatIP(dim)
#     idx.add(emb_np)
#     return idx


# # -------------------- Language Utilities --------------------
# LANG_MAP = {
#     "kn": "Kannada", "hi": "Hindi", "ta": "Tamil", "te": "Telugu",
#     "mr": "Marathi", "ml": "Malayalam", "gu": "Gujarati",
#     "bn": "Bengali", "pa": "Punjabi", "en": "English"
# }

# LANG_CODES = {
#     "english": "en", "kannada": "kn", "hindi": "hi", "tamil": "ta",
#     "telugu": "te", "marathi": "mr", "malayalam": "ml", "gujarati": "gu",
#     "bengali": "bn", "punjabi": "pa"
# }


# def detect_question_language(question):
#     try:
#         return detect(question)
#     except:
#         return "en"


# def extract_target_language(question):
#     q_lower = question.lower()
#     for lang_name, code in LANG_CODES.items():
#         if f"in {lang_name}" in q_lower or f"to {lang_name}" in q_lower:
#             return code
#     return None


# # -------------------- QA & Summarization --------------------
# def ask_groq_with_context(context, question, target_lang=None):
#     """Answer questions using context"""
#     if target_lang:
#         response_lang = LANG_MAP.get(target_lang, "the requested language")
#     else:
#         question_lang = detect_question_language(question)
#         response_lang = LANG_MAP.get(question_lang, "English")
    
#     is_translation = any(kw in question.lower() for kw in 
#                          ["translate", "convert", "change to", "in kannada", "in hindi", 
#                           "in telugu", "in marathi", "in tamil", "in english"])
    
#     if is_translation:
#         prompt = f"""You are a multilingual translator.

# CONTEXT from documents:
# {context}

# USER REQUEST: {question}

# Translate the relevant information to {response_lang}. Be accurate and preserve details.

# Response in {response_lang}:"""
#     else:
#         prompt = f"""Answer this question using ONLY the CONTEXT provided.

# CONTEXT:
# {context}

# QUESTION: {question}

# Answer in {response_lang}. If not in context, say "I cannot find this in the documents."

# Answer:"""
    
#     try:
#         resp = groq_client.chat.completions.create(
#             model="llama-3.3-70b-versatile",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.1,
#             max_tokens=1000
#         )
#         return resp.choices[0].message.content.strip()
#     except Exception as e:
#         return f"Error: {e}"


# def summarize_document(text, target_lang="en"):
#     """Generate summary"""
#     if not text or len(text.strip()) < 50:
#         return "Insufficient text to summarize."
    
#     text_snippet = text[:8000]
#     lang_name = LANG_MAP.get(target_lang, "English")
    
#     prompt = f"""Summarize this document in {lang_name}. Include main topic, key points, and important details.

# {text_snippet}

# Summary:"""
    
#     try:
#         resp = groq_client.chat.completions.create(
#             model="llama-3.3-70b-versatile",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.2,
#             max_tokens=800
#         )
#         return resp.choices[0].message.content.strip()
#     except Exception as e:
#         return f"Summarization failed: {e}"


# # -------------------- Streamlit UI --------------------
# import io  # Add this import

# st.set_page_config(page_title="Ultra-Fast PDF Chatbot", layout="wide")

# st.title("⚡ IntelliDoc: AI-Powered Multilingual Document Analysis System")
# st.markdown("""
# **Optimized for Speed** - Processes scanned PDFs up to **3x faster** using parallel processing!

# **Features:** Printed & Handwritten | Multi-language | Translation | Q&A | Summaries
# """)

# # Session state
# for key, val in [("chunks", []), ("meta", []), ("embeddings", None),
#                  ("faiss_index", None), ("full_text", ""), ("chat_history", []),
#                  ("file_info", {}), ("processing_stats", {}), ("vectorizer", None)]:
#     if key not in st.session_state:
#         st.session_state[key] = val

# # Sidebar
# st.sidebar.header("📂 Upload Documents")

# doc_mode = st.sidebar.radio(
#     "Processing Mode:",
#     ["Smart (Auto-detect)", "Force OCR (Scanned/Handwritten)"],
#     help="Smart mode is faster for mixed documents"
# )

# uploaded_pdfs = st.sidebar.file_uploader(
#     "PDF Files", 
#     type="pdf", 
#     accept_multiple_files=True,
#     key="pdf_uploader"
# )

# uploaded_images = st.sidebar.file_uploader(
#     "Image Files", 
#     type=["jpg", "jpeg", "png", "bmp", "tiff"],
#     accept_multiple_files=True,
#     key="img_uploader"
# )

# process_btn = st.sidebar.button("🚀 Process Documents", type="primary")

# # Processing
# if process_btn:
#     st.session_state.chunks.clear()
#     st.session_state.meta.clear()
#     st.session_state.embeddings = None
#     st.session_state.faiss_index = None
#     st.session_state.full_text = ""
#     st.session_state.file_info.clear()
#     st.session_state.chat_history.clear()
#     st.session_state.processing_stats.clear()
    
#     force_ocr = (doc_mode == "Force OCR (Scanned/Handwritten)")
    
#     total_start = time.time()
    
#     with st.spinner("⚡ Processing documents with parallel OCR..."):
#         # Process PDFs
#         for pdf in uploaded_pdfs or []:
#             start_time = time.time()
            
#             st.info(f"📄 Processing: {pdf.name}")
#             pdf_bytes = pdf.read()
            
#             text, pages_processed, total_pages = extract_text_from_pdf_parallel(
#                 pdf_bytes, 
#                 pdf.name,
#                 force_ocr=force_ocr
#             )
            
#             processing_time = time.time() - start_time
            
#             if text:
#                 st.session_state.full_text += f"\n\n=== FILE: {pdf.name} ===\n{text}"
#                 ch, md = chunk_text_smart(text, pdf.name)
#                 st.session_state.chunks.extend(ch)
#                 st.session_state.meta.extend(md)
                
#                 st.session_state.file_info[pdf.name] = {
#                     'chars': len(text),
#                     'pages': total_pages,
#                     'time': processing_time
#                 }
                
#                 speed = total_pages / processing_time if processing_time > 0 else 0
#                 st.success(
#                     f"✅ {pdf.name}: {total_pages} pages in {processing_time:.1f}s "
#                     f"({speed:.1f} pages/sec)"
#                 )
#             else:
#                 st.warning(f"⚠️ Could not extract text from {pdf.name}")
        
#         # Process Images
#         for img in uploaded_images or []:
#             start_time = time.time()
            
#             st.info(f"🖼️ Processing: {img.name}")
#             text = extract_text_from_image_fast(img)
            
#             processing_time = time.time() - start_time
            
#             if text:
#                 st.session_state.full_text += f"\n\n=== IMAGE: {img.name} ===\n{text}"
#                 ch, md = chunk_text_smart(text, img.name)
#                 st.session_state.chunks.extend(ch)
#                 st.session_state.meta.extend(md)
                
#                 st.session_state.file_info[img.name] = {
#                     'chars': len(text),
#                     'time': processing_time
#                 }
                
#                 st.success(f"✅ {img.name}: {len(text)} chars in {processing_time:.1f}s")
#             else:
#                 st.warning(f"⚠️ Could not extract text from {img.name}")
        
#         # Build embeddings
#         if st.session_state.chunks:
#             st.info(f"🔢 Creating search index for {len(st.session_state.chunks)} chunks...")
#             embed_start = time.time()
            
#             emb_np, vectorizer = embed_texts_tfidf(st.session_state.chunks)
#             st.session_state.embeddings = emb_np
#             st.session_state.vectorizer = vectorizer
#             st.session_state.faiss_index = build_faiss_index(emb_np)
            
#             embed_time = time.time() - embed_start
#             total_time = time.time() - total_start
            
#             st.success(
#                 f"✅ Search index created in {embed_time:.1f}s | "
#                 f"Total time: {total_time:.1f}s"
#             )
            
#             # Show statistics
#             st.sidebar.markdown("### 📊 Processing Stats")
#             for fname, info in st.session_state.file_info.items():
#                 if 'pages' in info:
#                     st.sidebar.text(
#                         f"• {fname}: {info['pages']} pages "
#                         f"({info['time']:.1f}s)"
#                     )
#                 else:
#                     st.sidebar.text(
#                         f"• {fname}: {info['chars']} chars "
#                         f"({info['time']:.1f}s)"
#                     )
#         else:
#             st.error("❌ No text extracted. Check file quality and OCR settings.")

# # Chat Interface
# st.markdown("---")
# st.subheader("💬 Chat with Your Documents")

# if not st.session_state.faiss_index:
#     st.info("👆 Upload and process documents first.")
# else:
#     col1, col2 = st.columns([5, 1])
    
#     with col1:
#         user_input = st.text_input(
#             "Your question:",
#             placeholder="E.g., 'What is the main topic?' or 'Translate to Hindi'",
#             key="user_question"
#         )
    
#     with col2:
#         ask_button = st.button("Send", type="primary")
    
#     with st.expander("💡 Example Questions"):
#         st.markdown("""
#         - "Summarize this document in Kannada"
#         - "What dates are mentioned?"
#         - "Translate the main points to Telugu"
#         - "List all important names"
#         - "What is the document about?"
#         """)
    
#     if ask_button and user_input:
#         st.session_state.chat_history.append(("user", user_input))
        
#         with st.spinner("🤔 Thinking..."):
#             is_summary = any(kw in user_input.lower() for kw in 
#                            ["summary", "summarize", "overview", "brief", "main points"])
            
#             target_lang = extract_target_language(user_input)
#             if not target_lang:
#                 target_lang = detect_question_language(user_input)
            
#             if is_summary:
#                 answer = summarize_document(st.session_state.full_text, target_lang)
#             else:
#                 # Convert question to TF-IDF vector
#                 qvec, _ = embed_texts_tfidf(
#                     [user_input], 
#                     vectorizer=st.session_state.vectorizer
#                 )
#                 qvec = qvec.astype("float32")
                
#                 # Normalize query vector
#                 norm = np.linalg.norm(qvec)
#                 if norm > 0:
#                     qvec = qvec / norm
                
#                 D, I = st.session_state.faiss_index.search(qvec, TOP_K)
                
#                 if len(D[0]) == 0 or D[0][0] < SIMILARITY_THRESHOLD:
#                     answer = "I cannot find relevant information in the documents."
#                 else:
#                     context_pieces = []
#                     total_chars = 0
                    
#                     for idx, score in zip(I[0], D[0]):
#                         if idx >= len(st.session_state.chunks):
#                             continue
                        
#                         chunk = st.session_state.chunks[idx]
#                         source = st.session_state.meta[idx]['source']
#                         page = st.session_state.meta[idx].get('page', '?')
                        
#                         piece = f"[Source: {source}, Page: {page}]\n{chunk}"
                        
#                         if total_chars + len(piece) > MAX_CONTEXT_CHARS:
#                             break
                        
#                         context_pieces.append(piece)
#                         total_chars += len(piece)
                    
#                     context = "\n\n---\n\n".join(context_pieces)
#                     answer = ask_groq_with_context(context, user_input, target_lang)
            
#             st.session_state.chat_history.append(("assistant", answer))
    
#     # Display chat
#     st.markdown("### 📜 Conversation")
#     for role, message in st.session_state.chat_history:
#         if role == "user":
#             st.markdown(f"**👤 You:** {message}")
#         else:
#             st.markdown(f"**🤖 Bot:** {message}")
#         st.markdown("---")
    
#     if st.button("🗑️ Clear Chat"):
#         st.session_state.chat_history.clear()
#         st.rerun()

# # Footer
# st.markdown("---")
# st.markdown("""
# ### ⚡ Performance Tips:
# - **Smart Mode**: Automatically detects which pages need OCR (faster)
# - **Force OCR**: Use for handwritten or poor-quality scans (slower but thorough)
# - **Parallel Processing**: Processes multiple pages simultaneously (3x faster)
# - **High-Resolution Scans**: 300 DPI recommended for best results
# """)





 




















#===============================================================================================================================



