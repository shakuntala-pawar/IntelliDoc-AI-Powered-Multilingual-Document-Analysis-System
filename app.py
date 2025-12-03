# # app.py — Multilingual PDF Chatbot (EasyOCR + Tesseract fallback, FAISS embeddings, Groq LLM)
# import os
# import tempfile
# import streamlit as st
# from dotenv import load_dotenv
# import fitz  # PyMuPDF
# from pdf2image import convert_from_bytes
# from PIL import Image
# import pytesseract
# import easyocr
# from sentence_transformers import SentenceTransformer
# import numpy as np
# import faiss
# from groq import Groq
# from langdetect import detect, DetectorFactory

# # make langdetect deterministic
# DetectorFactory.seed = 0

# # -------------------- Load env/config --------------------
# load_dotenv()
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# TESSERACT_CMD = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
# TESSDATA_PREFIX = os.getenv("TESSDATA_PREFIX")  # optional
# POPPLER_PATH = os.getenv("POPPLER_PATH")        # optional for pdf2image on Windows

# # configure Tesseract path if provided
# pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
# if TESSDATA_PREFIX:
#     os.environ["TESSDATA_PREFIX"] = TESSDATA_PREFIX

# # -------------------- Config knobs --------------------
# OCR_LANG_LIST = ["en","kn","hi","ta","te","mr","ml","gu","bn","pa"]  # used by EasyOCR reader
# OCR_LANGS_TESSERACT = "+".join(["eng","kan","hin","tam","tel","mar","mal","guj","ben","pan"])  # for pytesseract
# EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# EMBED_BATCH = 64
# TOP_K = 5
# SIMILARITY_THRESHOLD = 0.20
# SUMMARY_CHUNK_CHARS = 2500
# MAX_CONTEXT_CHARS = 3500
# SUMMARIES_TO_COMBINE = 6

# # -------------------- Initialize models & clients --------------------
# groq_client = Groq(api_key=GROQ_API_KEY)
# embedder = SentenceTransformer(EMBED_MODEL)

# # try to create EasyOCR reader (some platforms may need GPU disabled)
# try:
#     easyocr_reader = easyocr.Reader(OCR_LANG_LIST, gpu=False)
# except Exception:
#     easyocr_reader = None

# # -------------------- Helper functions: extraction --------------------
# def extract_text_with_pymupdf_bytes(pdf_bytes):
#     """Extract text from PDF if it contains embedded text (fast & preferred)."""
#     text = ""
#     try:
#         pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
#         for pageno, page in enumerate(pdf, start=1):
#             page_text = page.get_text("text")
#             if page_text and page_text.strip():
#                 text += f"\n\n[Page {pageno}]\n" + page_text
#     except Exception:
#         return ""
#     return text.strip()

# def extract_text_with_easyocr_bytes(pdf_bytes):
#     """Convert PDF to images and run EasyOCR on each page. Returns concatenated text."""
#     text = ""
#     convert_kwargs = {}
#     if POPPLER_PATH:
#         convert_kwargs["poppler_path"] = POPPLER_PATH
#     try:
#         images = convert_from_bytes(pdf_bytes, **convert_kwargs)
#     except Exception:
#         return ""
#     for i, img in enumerate(images, start=1):
#         try:
#             if easyocr_reader:
#                 results = easyocr_reader.readtext(np.asarray(img), detail=0)
#                 page_text = "\n".join(results).strip()
#             else:
#                 page_text = ""
#         except Exception:
#             page_text = ""
#         if not page_text:
#             # fallback to pytesseract on this image
#             try:
#                 page_text = pytesseract.image_to_string(img, lang=OCR_LANGS_TESSERACT)
#             except Exception:
#                 page_text = ""
#         if page_text and page_text.strip():
#             text += f"\n\n[Page {i}]\n" + page_text
#     return text.strip()

# def extract_text_with_tesseract_bytes(pdf_bytes):
#     """Fallback: convert PDF to images then use pytesseract."""
#     text = ""
#     convert_kwargs = {}
#     if POPPLER_PATH:
#         convert_kwargs["poppler_path"] = POPPLER_PATH
#     try:
#         images = convert_from_bytes(pdf_bytes, **convert_kwargs)
#     except Exception:
#         return ""
#     for i, img in enumerate(images, start=1):
#         try:
#             page_text = pytesseract.image_to_string(img, lang=OCR_LANGS_TESSERACT)
#         except Exception:
#             page_text = ""
#         if page_text and page_text.strip():
#             text += f"\n\n[Page {i}]\n" + page_text
#     return text.strip()

# def extract_text_from_pdf_file(file_obj):
#     """Try PyMuPDF first; if insufficient, try EasyOCR -> pytesseract."""
#     pdf_bytes = file_obj.read()
#     # try text extraction first
#     text = extract_text_with_pymupdf_bytes(pdf_bytes)
#     if len(text.strip()) >= 60:
#         return text
#     # try EasyOCR (preferred for scanned regional languages)
#     text_eo = extract_text_with_easyocr_bytes(pdf_bytes)
#     if len(text_eo.strip()) >= 40:
#         return text_eo
#     # fallback to pytesseract
#     text_t = extract_text_with_tesseract_bytes(pdf_bytes)
#     return text_t or ""

# def extract_text_from_image_file(image_file):
#     """OCR an image file using EasyOCR primary, then pytesseract fallback."""
#     try:
#         img = Image.open(image_file)
#     except Exception:
#         return ""
#     page_text = ""
#     if easyocr_reader:
#         try:
#             results = easyocr_reader.readtext(np.asarray(img), detail=0)
#             page_text = "\n".join(results).strip()
#         except Exception:
#             page_text = ""
#     if not page_text:
#         try:
#             page_text = pytesseract.image_to_string(img, lang=OCR_LANGS_TESSERACT)
#         except Exception:
#             page_text = ""
#     return page_text or ""

# # -------------------- Helpers: chunking & embeddings --------------------
# def chunk_text_keep_meta(text, source_name, chunk_chars=2000, overlap_chars=300):
#     if not text:
#         return [], []
#     chunks, metas = [], []
#     start = 0
#     L = len(text)
#     while start < L:
#         end = start + chunk_chars
#         chunk = text[start:end].strip()
#         if chunk:
#             chunks.append(chunk)
#             metas.append({"source": source_name})
#         start += chunk_chars - overlap_chars
#     return chunks, metas

# def embed_texts_in_batches(texts):
#     if not texts:
#         return np.zeros((0, embedder.get_sentence_embedding_dimension()), dtype="float32")
#     vectors = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False, batch_size=EMBED_BATCH)
#     return np.asarray(vectors, dtype="float32")

# def build_faiss_index(emb_np):
#     if emb_np is None or emb_np.shape[0] == 0:
#         raise ValueError("Empty embeddings")
#     faiss.normalize_L2(emb_np)
#     dim = emb_np.shape[1]
#     idx = faiss.IndexFlatIP(dim)
#     idx.add(emb_np.astype("float32"))
#     return idx

# # -------------------- Helpers: Groq calls & summarization --------------------
# LANG_MAP = {
#     "kn":"Kannada","hi":"Hindi","ta":"Tamil","te":"Telugu","mr":"Marathi",
#     "ml":"Malayalam","gu":"Gujarati","bn":"Bengali","pa":"Punjabi","en":"English"
# }

# def ask_groq_with_context(context, question, lang_code="en", require_quotes=True, highlights=False, max_tokens=700):
#     lang_name = LANG_MAP.get(lang_code, "the same language as the question")
#     quote_instr = "When giving numeric/date/name facts, quote the exact excerpt from CONTEXT and include source." if require_quotes else ""
#     highlight_instr = "Also provide key highlights and bullet-list important parts." if highlights else ""
#     prompt = f"""
# You are a careful PDF-QA assistant. Use ONLY the CONTEXT below to answer the QUESTION. Do NOT invent facts.
# If the answer cannot be found in the CONTEXT, reply exactly: "This question is not related to the uploaded PDF."
# Answer concisely in {lang_name}.

# {quote_instr}
# {highlight_instr}

# --- CONTEXT START ---
# {context}
# --- CONTEXT END ---

# QUESTION:
# {question}

# Answer:
# """
#     try:
#         resp = groq_client.chat.completions.create(
#             model="llama-3.1-8b-instant",
#             messages=[{"role":"user","content": prompt}],
#             temperature=0.0,
#             max_tokens=max_tokens
#         )
#         return resp.choices[0].message.content.strip()
#     except Exception as e:
#         # bubble up friendly message
#         return f"LLM error: {getattr(e,'args',e)}"

# def summarize_large_text_iteratively(full_text, lang_code="en"):
#     if not full_text or len(full_text.strip()) < 30:
#         return "No extractable text to summarize from the uploaded documents."
#     # split into char chunks
#     parts = []
#     L = len(full_text)
#     step = SUMMARY_CHUNK_CHARS
#     overlap = 400
#     i = 0
#     while i < L:
#         parts.append(full_text[i:i+step])
#         i += step - overlap

#     # summarize each part
#     part_summaries = []
#     for p in parts:
#         prompt = f"Summarize this passage in up to 200 words in {LANG_MAP.get(lang_code,'the same language')}. Do NOT add facts.\n\nPassage:\n{p}\n\nSummary:"
#         try:
#             resp = groq_client.chat.completions.create(
#                 model="llama-3.1-8b-instant",
#                 messages=[{"role":"user","content": prompt}],
#                 max_tokens=350,
#                 temperature=0.0
#             )
#             s = resp.choices[0].message.content.strip()
#         except Exception:
#             s = ""
#         if s:
#             part_summaries.append(s)

#     # combine groups to avoid huge payloads
#     while len(part_summaries) > SUMMARIES_TO_COMBINE:
#         new_summaries = []
#         for j in range(0, len(part_summaries), SUMMARIES_TO_COMBINE):
#             group = "\n\n".join(part_summaries[j:j+SUMMARIES_TO_COMBINE])
#             prompt = f"Combine and condense these summaries into one concise summary (max 200 words):\n\n{group}\n\nCombined summary:"
#             try:
#                 resp = groq_client.chat.completions.create(
#                     model="llama-3.1-8b-instant",
#                     messages=[{"role":"user","content": prompt}],
#                     max_tokens=400,
#                     temperature=0.0
#                 )
#                 ns = resp.choices[0].message.content.strip()
#             except Exception:
#                 ns = ""
#             if ns:
#                 new_summaries.append(ns)
#         part_summaries = new_summaries

#     # final combine
#     final_text = "\n\n".join(part_summaries)
#     prompt = f"Produce a final formal summary (3-6 short sentences) and then a brief friendly summary for a general user. Text:\n\n{final_text}\n\nOutput:"
#     try:
#         resp = groq_client.chat.completions.create(
#             model="llama-3.1-8b-instant",
#             messages=[{"role":"user","content": prompt}],
#             max_tokens=600,
#             temperature=0.0
#         )
#         return resp.choices[0].message.content.strip()
#     except Exception:
#         return "Sorry — the LLM couldn't produce a final summary."

# # -------------------- Streamlit UI --------------------
# st.set_page_config(page_title="Multilingual PDF Chatbot", page_icon="📚", layout="wide")
# st.title("📄 Multilingual PDF Chatbot — EasyOCR + Tesseract fallback + FAISS + Groq")
# st.markdown("**Hello!** I can read uploaded PDFs or images (scanned or text) in many Indian languages and answer or summarize based ONLY on the documents. How can I assist you today?")

# # session state defaults
# if "chunks" not in st.session_state:
#     st.session_state.chunks = []
# if "meta" not in st.session_state:
#     st.session_state.meta = []
# if "embeddings" not in st.session_state:
#     st.session_state.embeddings = None
# if "faiss_index" not in st.session_state:
#     st.session_state.faiss_index = None
# if "full_text" not in st.session_state:
#     st.session_state.full_text = ""
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # Sidebar: uploads
# st.sidebar.header("Upload PDFs or images")
# uploaded_pdfs = st.sidebar.file_uploader("Upload PDF files (one or more)", accept_multiple_files=True, type=["pdf"])
# uploaded_images = st.sidebar.file_uploader("Upload scanned images (jpg/png/jpeg) — optional", accept_multiple_files=True, type=["png","jpg","jpeg"])
# process_btn = st.sidebar.button("Process uploads")

# if process_btn:
#     # reset
#     st.session_state.chunks = []
#     st.session_state.meta = []
#     st.session_state.embeddings = None
#     st.session_state.faiss_index = None
#     st.session_state.full_text = ""
#     st.success("Started processing uploads — OCR & embeddings may take a moment.")

#     # process PDFs
#     for pdf in uploaded_pdfs or []:
#         st.sidebar.write(f"Processing PDF: {pdf.name}")
#         text = extract_text_from_pdf_file(pdf)
#         if not text or len(text.strip()) < 20:
#             st.sidebar.write(f" → Very little text extracted from {pdf.name}.")
#         st.session_state.full_text += f"\n\n[File: {pdf.name}]\n" + (text or "")
#         ch, md = chunk_text_keep_meta(text or "", source_name=pdf.name)
#         for c,m in zip(ch, md):
#             st.session_state.chunks.append(c)
#             st.session_state.meta.append(m)

#     # images (optional)
#     for imgf in uploaded_images or []:
#         st.sidebar.write(f"Processing Image: {imgf.name}")
#         txt = extract_text_from_image_file(imgf)
#         st.session_state.full_text += f"\n\n[Image: {imgf.name}]\n" + (txt or "")
#         ch, md = chunk_text_keep_meta(txt or "", source_name=imgf.name)
#         for c,m in zip(ch, md):
#             st.session_state.chunks.append(c)
#             st.session_state.meta.append(m)

#     # build embeddings and faiss (if chunks exist)
#     if st.session_state.chunks:
#         st.sidebar.info("Computing embeddings (this can take a while for many chunks)...")
#         emb_np = embed_texts_in_batches(st.session_state.chunks)
#         if emb_np.shape[0] == 0:
#             st.sidebar.error("Embeddings failed. No vectors produced.")
#         else:
#             st.session_state.embeddings = emb_np
#             st.session_state.faiss_index = build_faiss_index(emb_np.copy())
#             st.sidebar.success(f"Vector store created with {emb_np.shape[0]} chunks.")
#     else:
#         st.sidebar.warning("No chunks extracted. Try clearer scans or ensure tessdata/EasyOCR support installed.")

# st.markdown("---")
# st.subheader("Chat — ask questions about the uploaded documents")

# if not st.session_state.faiss_index:
#     st.info("Please upload & process PDFs/images (sidebar) to enable chat.")
# else:
#     # use text_input with a key; on_click handler will process and clear
#     if "question_input" not in st.session_state:
#         st.session_state.question_input = ""

#     def on_ask():
#         q = st.session_state.get("question_input", "").strip()
#         if not q:
#             return
#         st.session_state.chat_history.append(("user", q))
#         # detect language
#         try:
#             user_lang = detect(q)
#         except Exception:
#             user_lang = "en"

#         # determine if user asked for a broad explanation / summary
#         q_lower = q.lower()
#         summary_triggers = ["explain", "summary", "what is in", "what's in", "brief", "give an overview",
#                             "describe this document", "what does this pdf say", "explain this pdf",
#                             "tell me about this pdf", "what information", "summarize"]
#         is_summary = any(t in q_lower for t in summary_triggers)

#         if is_summary:
#             full_txt = st.session_state.full_text
#             if not full_txt or len(full_txt.strip()) < 20:
#                 bot_ans = "I couldn't find substantial text to summarize from the uploaded documents. Try clearer scans or ensure language packs are installed."
#             else:
#                 bot_ans = summarize_large_text_iteratively(full_txt, lang_code=user_lang)
#         else:
#             # QA flow: embed question, search top chunks
#             qvec = embedder.encode([q], convert_to_numpy=True).astype("float32")
#             faiss.normalize_L2(qvec)
#             D, I = st.session_state.faiss_index.search(qvec, TOP_K)
#             top_scores = D[0] if D.shape[0] else []
#             top_idx = I[0] if I.shape[0] else []

#             if len(top_scores) == 0 or top_scores[0] < SIMILARITY_THRESHOLD:
#                 # If retrieval low, we still offer a summarized explanation if user explicitly asked earlier,
#                 # but for general QA we must avoid hallucination.
#                 bot_ans = "This question is not related to the uploaded PDF."
#             else:
#                 # assemble limited context (respect MAX_CONTEXT_CHARS to avoid payload errors)
#                 pieces = []
#                 total_chars = 0
#                 for idx in top_idx:
#                     if idx >= len(st.session_state.chunks):
#                         continue
#                     md = st.session_state.meta[idx]
#                     chunk = st.session_state.chunks[idx]
#                     piece = f"[Source: {md.get('source','unknown')}]\n{chunk}"
#                     if total_chars + len(piece) > MAX_CONTEXT_CHARS:
#                         break
#                     pieces.append(piece)
#                     total_chars += len(piece)
#                 context = "\n\n---\n\n".join(pieces)
#                 # call Groq with context
#                 bot_ans = ask_groq_with_context(context, q, lang_code=user_lang, require_quotes=True, highlights=False)

#         st.session_state.chat_history.append(("bot", bot_ans))
#         # clear the input for next queries safely inside the callback
#         st.session_state["question_input"] = ""

#     st.text_input("Ask your question (any language):", key="question_input")
#     st.button("Ask", on_click=on_ask)

#     # show history
#     st.markdown("### Conversation")
#     for role, text in st.session_state.chat_history:
#         if role == "user":
#             st.markdown(f"**You:** {text}")
#         else:
#             st.markdown(f"**Bot:** {text}")
#         st.markdown("---")

# st.markdown(
#     """
# **Tips & Notes**
# - EasyOCR generally gives better results for Indic scripts (Kannada/Tamil/Telugu/Malayalam etc.). If you get poor OCR, try higher-resolution (300 DPI) scans.
# - If Groq returns token/payload errors for very large documents, use the summary/explain request which triggers iterative summarization.
# - Tune SIMILARITY_THRESHOLD, MAX_CONTEXT_CHARS and SUMMARY_CHUNK_CHARS near the top of this file for your setup.
# - You can add/remove languages by editing OCR_LANG_LIST and OCR_LANGS_TESSERACT.
# """
# )



#========================================================================================================================
# #1


# # app.py — Multilingual PDF Chatbot (Groq + FAISS + OCR + iterative summarization)
# import os
# import math
# import tempfile
# import streamlit as st
# from dotenv import load_dotenv
# import fitz  # PyMuPDF
# from pdf2image import convert_from_bytes
# from PIL import Image
# import pytesseract

# from sentence_transformers import SentenceTransformer
# import numpy as np
# import faiss
# from groq import Groq

# from langdetect import detect, DetectorFactory
# DetectorFactory.seed = 0



# # -------------------- Load ENV --------------------
# load_dotenv()
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# TESSERACT_CMD = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
# TESSDATA_PREFIX = os.getenv("TESSDATA_PREFIX")  # optional
# POPPLER_PATH = os.getenv("POPPLER_PATH")        # optional (pdf2image on Windows)

# # configure tesseract
# pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
# if TESSDATA_PREFIX:
#     os.environ["TESSDATA_PREFIX"] = TESSDATA_PREFIX

# # -------------------- Config --------------------
# OCR_LANGS = "eng+kan+hin+tam+tel+mar+mal+guj+ben+pan"  # languages to support in OCR
# EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# EMBED_BATCH = 64
# TOP_K = 5
# SIMILARITY_THRESHOLD = 0.20   # tune if too strict / lenient
# SUMMARY_CHUNK_CHARS = 3000    # characters per chunk for iterative summarization
# MAX_CONTEXT_CHARS = 3500      # limit context sent in single LLM call to avoid TPM limits
# SUMMARIES_TO_COMBINE = 6      # when doc large, combine N chunk summaries then summarize again

# # -------------------- Clients / models --------------------
# groq_client = Groq(api_key=GROQ_API_KEY)
# embedder = SentenceTransformer(EMBED_MODEL)

# # -------------------- Helpers: extraction --------------------
# def extract_text_with_pymupdf_bytes(pdf_bytes):
#     text = ""
#     try:
#         pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
#         for pageno, page in enumerate(pdf, start=1):
#             page_text = page.get_text("text")
#             if page_text and page_text.strip():
#                 text += f"\n\n[Page {pageno}]\n" + page_text
#     except Exception:
#         return ""
#     return text.strip()

# def extract_text_with_ocr_bytes(pdf_bytes):
#     text = ""
#     conv_kwargs = {}
#     if POPPLER_PATH:
#         conv_kwargs["poppler_path"] = POPPLER_PATH
#     try:
#         images = convert_from_bytes(pdf_bytes, **conv_kwargs)
#     except Exception:
#         return ""
#     for i, img in enumerate(images, start=1):
#         try:
#             page_text = pytesseract.image_to_string(img, lang=OCR_LANGS)
#             if page_text and page_text.strip():
#                 text += f"\n\n[Page {i}]\n" + page_text
#         except Exception:
#             continue
#     return text.strip()

# def extract_text_from_pdf_file(file_obj):
#     pdf_bytes = file_obj.read()
#     text = extract_text_with_pymupdf_bytes(pdf_bytes)
#     if len(text.strip()) < 50:
#         ocr_text = extract_text_with_ocr_bytes(pdf_bytes)
#         if len(ocr_text.strip()) > 20:
#             text = ocr_text
#     return text

# def extract_text_from_image_file(image_file):
#     try:
#         img = Image.open(image_file)
#         text = pytesseract.image_to_string(img, lang=OCR_LANGS)
#         return text.strip()
#     except Exception:
#         return ""

# # -------------------- Helpers: chunking --------------------
# def chunk_text_keep_meta(text, source_name, chunk_chars=2000, overlap_chars=300):
#     if not text:
#         return [], []
#     text = text.strip()
#     chunks = []
#     meta = []
#     start = 0
#     L = len(text)
#     while start < L:
#         end = start + chunk_chars
#         chunk = text[start:end].strip()
#         if chunk:
#             chunks.append(chunk)
#             meta.append({"source": source_name})
#         start += chunk_chars - overlap_chars
#     return chunks, meta

# # -------------------- Helpers: embeddings & faiss --------------------
# def embed_texts_in_batches(texts):
#     if not texts:
#         return np.zeros((0, embedder.get_sentence_embedding_dimension()), dtype="float32")
#     vectors = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False, batch_size=EMBED_BATCH)
#     return np.asarray(vectors, dtype="float32")

# def build_faiss_index(emb_np):
#     if emb_np is None or emb_np.shape[0] == 0:
#         raise ValueError("Empty embeddings")
#     faiss.normalize_L2(emb_np)
#     dim = emb_np.shape[1]
#     idx = faiss.IndexFlatIP(dim)
#     idx.add(emb_np.astype("float32"))
#     return idx

# # -------------------- Helpers: Groq LLM utilities --------------------
# LANG_MAP = {
#     "kn":"Kannada","hi":"Hindi","ta":"Tamil","te":"Telugu","mr":"Marathi",
#     "ml":"Malayalam","gu":"Gujarati","bn":"Bengali","pa":"Punjabi","en":"English"
# }

# def ask_groq_with_context(context, question, lang_code="en", require_quotes=True, highlights=False):
#     lang_name = LANG_MAP.get(lang_code, "the same language as the question")
#     quote_instr = "When giving numeric/date/name facts, quote the exact excerpt from CONTEXT and include source." if require_quotes else ""
#     highlight_instr = "Also provide key highlights and bullet-list important parts." if highlights else ""

#     prompt = f"""
# You are a careful PDF-QA assistant. Use ONLY the CONTEXT below to answer the QUESTION. Do NOT invent facts.
# If the answer cannot be found in the CONTEXT, reply exactly: "This question is not related to the uploaded PDF."
# Answer concisely in {lang_name}.

# {quote_instr}
# {highlight_instr}

# --- CONTEXT START ---
# {context}
# --- CONTEXT END ---

# QUESTION:
# {question}

# Answer:
# """
#     resp = groq_client.chat.completions.create(
#         model="llama-3.1-8b-instant",
#         messages=[{"role":"user","content": prompt}],
#         temperature=0.0,
#         max_tokens=700
#     )
#     try:
#         return resp.choices[0].message.content.strip()
#     except Exception:
#         return "Sorry — LLM failed to return an answer."

# # Iterative summarization to avoid payload errors (chunk -> summarize -> combine)
# def summarize_large_text_iteratively(full_text, lang_code="en"):
#     # first split into char chunks
#     if not full_text or len(full_text.strip()) < 30:
#         return "No extractable text to summarize."

#     parts = []
#     L = len(full_text)
#     step = SUMMARY_CHUNK_CHARS
#     overlap = 400
#     i = 0
#     while i < L:
#         part = full_text[i:i+step]
#         parts.append(part)
#         i += step - overlap

#     # Summarize each part
#     part_summaries = []
#     for p in parts:
#         prompt = f"""
# You are a helpful summarizer. Summarize the following passage in up to 200 words in {LANG_MAP.get(lang_code,'the same language')}.
# Do NOT add any facts not present.
# Passage:
# {p}
# Summary:
# """
#         resp = groq_client.chat.completions.create(
#             model="llama-3.1-8b-instant",
#             messages=[{"role":"user","content":prompt}],
#             max_tokens=350,
#             temperature=0.0
#         )
#         try:
#             s = resp.choices[0].message.content.strip()
#         except Exception:
#             s = ""
#         if s:
#             part_summaries.append(s)

#     # If many part_summaries, combine in groups to avoid token overload
#     while len(part_summaries) > SUMMARIES_TO_COMBINE:
#         new_summaries = []
#         for j in range(0, len(part_summaries), SUMMARIES_TO_COMBINE):
#             group = "\n\n".join(part_summaries[j:j+SUMMARIES_TO_COMBINE])
#             prompt = f"Combine and condense these summaries into one concise summary (max 200 words):\n\n{group}\n\nCombined summary:"
#             resp = groq_client.chat.completions.create(
#                 model="llama-3.1-8b-instant",
#                 messages=[{"role":"user","content":prompt}],
#                 max_tokens=400,
#                 temperature=0.0
#             )
#             try:
#                 ns = resp.choices[0].message.content.strip()
#             except Exception:
#                 ns = ""
#             if ns:
#                 new_summaries.append(ns)
#         part_summaries = new_summaries

#     # Final combine
#     final_text = "\n\n".join(part_summaries)
#     prompt = f"Produce a final formal summary (3-6 short sentences) and then a simple brief summary for a general user. Text:\n\n{final_text}\n\nOutput:"
#     resp = groq_client.chat.completions.create(
#         model="llama-3.1-8b-instant",
#         messages=[{"role":"user","content":prompt}],
#         max_tokens=600,
#         temperature=0.0
#     )
#     try:
#         return resp.choices[0].message.content.strip()
#     except Exception:
#         return "Sorry — could not produce final summary."

# # -------------------- Streamlit UI --------------------
# st.set_page_config(page_title="Multilingual PDF Chatbot", layout="wide")
# st.title("📄 Multilingual PDF Chatbot — OCR, Multilingual, Groq + FAISS")
# st.markdown("**Hello!** I can read your uploaded PDFs or images (scanned or text), extract multilingual text, and answer or summarize based ONLY on the document. How can I help you today?")

# # session-state defaults
# if "chunks" not in st.session_state:
#     st.session_state.chunks = []
# if "meta" not in st.session_state:
#     st.session_state.meta = []
# if "embeddings" not in st.session_state:
#     st.session_state.embeddings = None
# if "faiss_index" not in st.session_state:
#     st.session_state.faiss_index = None
# if "full_text" not in st.session_state:
#     st.session_state.full_text = ""
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # Sidebar: upload PDFs & images
# st.sidebar.header("Upload PDFs or images")
# uploaded_pdfs = st.sidebar.file_uploader("Upload PDF files (one or more)", accept_multiple_files=True, type=["pdf"])
# uploaded_images = st.sidebar.file_uploader("Upload scanned images (jpg/png) — optional", accept_multiple_files=True, type=["png","jpg","jpeg"])
# process_btn = st.sidebar.button("Process uploads")

# # PROCESS callback implemented inline when button pressed
# if process_btn:
#     # reset previous
#     st.session_state.chunks = []
#     st.session_state.meta = []
#     st.session_state.embeddings = None
#     st.session_state.faiss_index = None
#     st.session_state.full_text = ""
#     st.success("Started processing uploads — this may take a moment for OCR & embeddings.")

#     # PDFs
#     for pdf in uploaded_pdfs or []:
#         st.sidebar.write(f"Extracting: {pdf.name} ...")
#         text = extract_text_from_pdf_file(pdf)
#         if not text or len(text.strip()) < 20:
#             st.sidebar.write(f" → Very little text extracted from {pdf.name}.")
#         st.session_state.full_text += f"\n\n[File: {pdf.name}]\n{(text or '')}"
#         ch, md = chunk_text_keep_meta(text or "", pdf.name)
#         for c,m in zip(ch, md):
#             st.session_state.chunks.append(c)
#             st.session_state.meta.append(m)

#     # Images
#     for imgf in uploaded_images or []:
#         st.sidebar.write(f"OCR image: {imgf.name} ...")
#         txt = extract_text_from_image_file(imgf)
#         st.session_state.full_text += f"\n\n[Image: {imgf.name}]\n{txt}"
#         ch, md = chunk_text_keep_meta(txt or "", imgf.name)
#         for c,m in zip(ch, md):
#             st.session_state.chunks.append(c)
#             st.session_state.meta.append(m)

#     # build embeddings & FAISS if we have chunks
#     if st.session_state.chunks:
#         st.sidebar.info("Computing embeddings...")
#         emb_np = embed_texts_in_batches(st.session_state.chunks)
#         if emb_np.shape[0] == 0:
#             st.sidebar.error("Embeddings failed. No vectors created.")
#         else:
#             st.session_state.embeddings = emb_np
#             st.session_state.faiss_index = build_faiss_index(emb_np.copy())
#             st.sidebar.success(f"Created vector store ({emb_np.shape[0]} chunks).")
#     else:
#         st.sidebar.warning("No chunks extracted from uploads. Try increasing image quality / installing tessdata for languages.")

# # Chat UI
# st.markdown("---")
# st.subheader("Chat — ask questions about the uploaded documents")

# if not st.session_state.faiss_index:
#     st.info("Please upload & process PDFs or images from the sidebar to enable chat.")
# else:
#     # persistent text_input key is "question_input"
#     if "question_input" not in st.session_state:
#         st.session_state["question_input"] = ""

#     # We'll use on_click callback to process (so we can safely clear the widget)
#     def on_ask():
#         q = st.session_state.get("question_input", "").strip()
#         if not q:
#             return
#         st.session_state.chat_history.append(("user", q))
#         # detect language
#         try:
#             user_lang = detect(q)
#         except Exception:
#             user_lang = "en"

#         # If user asks for broad explanation / summary, run iterative summarization over full_text
#         q_lower = q.lower()
#         summary_triggers = ["explain", "summary", "what is in", "what's in", "brief", "give an overview", "describe this document", "what does this pdf say", "explain this pdf", "tell me about this pdf"]
#         is_summary = any(t in q_lower for t in summary_triggers)

#         if is_summary:
#             # use iterative summarization to avoid payload limits
#             full_txt = st.session_state.full_text
#             if not full_txt or len(full_txt.strip()) < 20:
#                 bot_ans = "I couldn't find substantial text in the uploaded documents to summarize. Try re-uploading a clearer scan or ensure tessdata for the document language is installed."
#             else:
#                 bot_ans = summarize_large_text_iteratively(full_txt, lang_code=user_lang)
#         else:
#             # QA path (retrieve top chunks)
#             qvec = embedder.encode([q], convert_to_numpy=True).astype("float32")
#             faiss.normalize_L2(qvec)
#             D, I = st.session_state.faiss_index.search(qvec, TOP_K)
#             top_scores = D[0]
#             top_idx = I[0]
#             if len(top_scores) == 0 or top_scores[0] < SIMILARITY_THRESHOLD:
#                 bot_ans = "This question is not related to the uploaded PDF."
#             else:
#                 # assemble limited context (respect MAX_CONTEXT_CHARS)
#                 pieces = []
#                 total_chars = 0
#                 for idx in top_idx:
#                     if idx >= len(st.session_state.chunks):
#                         continue
#                     md = st.session_state.meta[idx]
#                     chunk = st.session_state.chunks[idx]
#                     piece = f"[Source: {md.get('source','unknown')}]\n{chunk}"
#                     if total_chars + len(piece) > MAX_CONTEXT_CHARS:
#                         break
#                     pieces.append(piece)
#                     total_chars += len(piece)
#                 context = "\n\n---\n\n".join(pieces)
#                 bot_ans = ask_groq_with_context(context, q, lang_code=user_lang, require_quotes=True, highlights=False)

#         # append
#         st.session_state.chat_history.append(("bot", bot_ans))
#         # clear input for next query
#         st.session_state["question_input"] = ""

#     # show input & ask button (on_click uses callback)
#     st.text_input("Ask your question (any language):", key="question_input")
#     st.button("Ask", on_click=on_ask)

#     # Display chat history
#     st.markdown("### Conversation")
#     for role, text in st.session_state.chat_history:
#         if role == "user":
#             st.markdown(f"**You:** {text}")
#         else:
#             st.markdown(f"**Bot:** {text}")
#         st.markdown("---")

# st.markdown(
#     """
# **Tips**
# - For better OCR: upload high-resolution scans (300 DPI) and ensure the appropriate `*.traineddata` files exist in your Tesseract `tessdata` folder.
# - If Groq complains about token limits for very large uploads, use the "Explain / Summary" feature which uses iterative summarization.
# - Want the answer in English or another language? Ask: "Please answer in English" or "Translate the summary to English".
# """
# )







#================================================================================================================================















# import os
# import tempfile
# import streamlit as st
# from dotenv import load_dotenv
# import fitz
# from pdf2image import convert_from_bytes
# import pytesseract

# from sentence_transformers import SentenceTransformer
# import numpy as np
# import faiss
# from groq import Groq

# from langdetect import detect, DetectorFactory
# DetectorFactory.seed = 0

# # -------------------- Load ENV --------------------
# load_dotenv()
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# TESSERACT_CMD = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
# TESSDATA_PREFIX = os.getenv("TESSDATA_PREFIX")
# POPPLER_PATH = os.getenv("POPPLER_PATH")

# pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
# if TESSDATA_PREFIX:
#     os.environ["TESSDATA_PREFIX"] = TESSDATA_PREFIX

# # -------------------- Config --------------------
# OCR_LANGS = "eng+kan+hin+tam+tel+mar+mal+guj+ben+pan"
# EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# SIMILARITY_THRESHOLD = 0.20
# TOP_K = 5

# embedder = SentenceTransformer(EMBED_MODEL)
# groq_client = Groq(api_key=GROQ_API_KEY)

# # ------------------ TEXT EXTRACTION ------------------
# def extract_text_pymupdf(pdf_bytes):
#     text = ""
#     try:
#         pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
#         for pg, page in enumerate(pdf, start=1):
#             t = page.get_text("text")
#             if t.strip():
#                 text += f"\n\n[Page {pg}]\n{t}"
#     except:
#         return ""
#     return text.strip()

# def extract_text_ocr(pdf_bytes):
#     text = ""
#     kwargs = {}
#     if POPPLER_PATH:
#         kwargs["poppler_path"] = POPPLER_PATH

#     try:
#         images = convert_from_bytes(pdf_bytes, **kwargs)
#     except:
#         return ""

#     for i, img in enumerate(images, start=1):
#         try:
#             t = pytesseract.image_to_string(img, lang=OCR_LANGS)
#             if t.strip():
#                 text += f"\n\n[Page {i}]\n{t}"
#         except:
#             continue
#     return text.strip()

# def extract_pdf_text(file_obj):
#     pdf_bytes = file_obj.read()
#     text = extract_text_pymupdf(pdf_bytes)

#     if len(text.strip()) < 50:
#         ocr_text = extract_text_ocr(pdf_bytes)
#         if len(ocr_text.strip()) > 20:
#             return ocr_text

#     return text

# # --------------- CHUNKING ------------------
# def chunk_text(text, src, chunk_size=500, overlap=80):
#     words = text.split()
#     chunks, meta = [], []
#     i = 0
#     while i < len(words):
#         c = " ".join(words[i:i+chunk_size]).strip()
#         if len(c) > 20:
#             chunks.append(c)
#             meta.append({"source": src})
#         i += chunk_size - overlap
#     return chunks, meta

# # ------------------ EMBEDDINGS + FAISS ------------------
# def build_faiss(vectors):
#     faiss.normalize_L2(vectors)
#     idx = faiss.IndexFlatIP(vectors.shape[1])
#     idx.add(vectors.astype("float32"))
#     return idx

# # --------------- GROQ LLM ------------------
# def llm_answer(context, question, lang="en", override_pdf_context=False):
#     language_map = {
#         "kn": "Kannada", "hi": "Hindi", "ta": "Tamil", "te": "Telugu",
#         "mr": "Marathi", "ml": "Malayalam", "gu": "Gujarati",
#         "bn": "Bengali", "pa": "Punjabi", "en": "English"
#     }
#     lang_name = language_map.get(lang, "the same language")

#     # special case: user wants full PDF explanation
#     if override_pdf_context:
#         system_prompt = f"""
# You are a PDF explanation assistant.
# Explain the PDF content in {lang_name}. Summarize, highlight key points, and give important details.
# Use ONLY the full extracted PDF text given below.

# --- PDF CONTENT START ---
# {context}
# --- PDF CONTENT END ---
# """
#     else:
#         system_prompt = f"""
# You are a PDF-QA assistant. Use ONLY the provided context to answer.
# If the answer is not found in the context, say: "This question is not related to the uploaded PDF."
# Respond in {lang_name}.

# --- CONTEXT START ---
# {context}
# --- CONTEXT END ---
# """

#     resp = groq_client.chat.completions.create(
#         model="llama-3.1-8b-instant",
#         messages=[{"role": "user", "content": system_prompt + "\n\nQuestion:\n" + question}],
#         temperature=0.0,
#         max_tokens=700
#     )
#     return resp.choices[0].message.content.strip()

# # -------------------- STREAMLIT UI --------------------
# st.set_page_config(page_title="Multilingual PDF Chatbot", layout="wide")
# st.title("📚 Multilingual PDF Chatbot (Groq + OCR + Vector Search)")

# # Session state
# for key, default in {
#     "chunks": [], "meta": [], "emb": None, "faiss": None,
#     "full_text": "", "chat": []
# }.items():
#     if key not in st.session_state:
#         st.session_state[key] = default

# # Sidebar
# st.sidebar.header("Upload PDFs")
# files = st.sidebar.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])
# process = st.sidebar.button("Process PDFs")

# if process:
#     st.session_state["chunks"] = []
#     st.session_state["meta"] = []
#     st.session_state["full_text"] = ""
#     st.session_state["faiss"] = None

#     for f in files:
#         st.sidebar.write(f"🔄 Extracting: {f.name}")
#         text = extract_pdf_text(f)
#         st.session_state["full_text"] += "\n\n" + text

#         ch, mt = chunk_text(text, f.name)
#         st.session_state["chunks"].extend(ch)
#         st.session_state["meta"].extend(mt)

#     # Embeddings
#     if st.session_state["chunks"]:
#         vectors = embedder.encode(st.session_state["chunks"], convert_to_numpy=True)
#         vectors = np.array(vectors, dtype="float32")
#         st.session_state["emb"] = vectors
#         st.session_state["faiss"] = build_faiss(vectors)
#         st.sidebar.success("✔ PDFs processed successfully!")
#     else:
#         st.sidebar.error("No text extracted.")

# st.markdown("---")
# st.subheader("Chat")

# if not st.session_state["faiss"]:
#     st.info("Upload and process PDFs first.")
# else:
#     q = st.text_input("Ask a question:", key="ask_box")
#     ask_btn = st.button("Ask")

#     if ask_btn and q.strip():
#         st.session_state["chat"].append(("user", q))

#         try:
#             lang = detect(q)
#         except:
#             lang = "en"

#         # Detect full PDF request
#         explain_trigger = any([
#             "explain" in q.lower(),
#             "summary" in q.lower(),
#             "what is in this pdf" in q.lower(),
#             "brief" in q.lower()
#         ])

#         if explain_trigger:
#             answer = llm_answer(
#                 context=st.session_state["full_text"],
#                 question=q,
#                 lang=lang,
#                 override_pdf_context=True
#             )
#         else:
#             # vector search
#             qvec = embedder.encode([q], convert_to_numpy=True).astype("float32")
#             faiss.normalize_L2(qvec)
#             D, I = st.session_state["faiss"].search(qvec, TOP_K)

#             if D[0][0] < SIMILARITY_THRESHOLD:
#                 answer = "This question is not related to the uploaded PDF."
#             else:
#                 ctx = "\n\n---\n\n".join(
#                     f"[{st.session_state['meta'][i]['source']}]\n{st.session_state['chunks'][i]}"
#                     for i in I[0]
#                 )
#                 answer = llm_answer(ctx, q, lang)

#         st.session_state["chat"].append(("bot", answer))

#         # Auto-clear input
#         st.session_state["ask_box"] = ""

#     # Display chat
#     for r, t in st.session_state["chat"]:
#         st.markdown(f"**{'You' if r=='user' else 'Bot'}:** {t}")
#         st.markdown("---")





















#=========================================================================================================================================================================

#  2




import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
import fitz  # PyMuPDF
from pdf2image import convert_from_bytes
import pytesseract

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from groq import Groq

from langdetect import detect, DetectorFactory

# Stable language detection (deterministic)
DetectorFactory.seed = 0

# -------------------- Load env --------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TESSERACT_CMD = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
TESSDATA_PREFIX = os.getenv("TESSDATA_PREFIX")  # optional
POPPLER_PATH = os.getenv("POPPLER_PATH")  # optional, used by pdf2image on Windows if needed

# set up Tesseract
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
if TESSDATA_PREFIX:
    os.environ["TESSDATA_PREFIX"] = TESSDATA_PREFIX

# -------------------- Config --------------------
OCR_LANGS = "eng+kan+hin+tam+tel+mar+mal+guj+ben+pan"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_BATCH_SIZE = 64
TOP_K = 5
SIMILARITY_THRESHOLD = 0.25  # inner product threshold after normalization (tweakable)

# -------------------- Clients / Models --------------------
groq_client = Groq(api_key=GROQ_API_KEY)
embedder = SentenceTransformer(EMBED_MODEL)

# -------------------- Helpers --------------------
def extract_text_with_pymupdf_bytes(pdf_bytes):
    """Try text extraction via PyMuPDF (fast for text PDFs)."""
    text = ""
    try:
        pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
        for pageno, page in enumerate(pdf, start=1):
            page_text = page.get_text("text")
            if page_text:
                text += f"\n\n[Page {pageno}]\n" + page_text
    except Exception:
        return ""
    return text.strip()


def extract_text_with_ocr_bytes(pdf_bytes):
    """Convert to images and run pytesseract OCR (slower, but works for scanned)."""
    text = ""
    convert_kwargs = {}
    if POPPLER_PATH:
        convert_kwargs["poppler_path"] = POPPLER_PATH
    images = convert_from_bytes(pdf_bytes, **convert_kwargs)
    for i, img in enumerate(images, start=1):
        try:
            page_text = pytesseract.image_to_string(img, lang=OCR_LANGS)
            if page_text and page_text.strip():
                text += f"\n\n[Page {i}]\n" + page_text
        except Exception as e:
            # continue on OCR errors for some pages
            continue
    return text.strip()


def extract_pdf_text(file_obj):
    """Try PyMuPDF first; if insufficient text -> fallback to OCR"""
    pdf_bytes = file_obj.read()
    text = extract_text_with_pymupdf_bytes(pdf_bytes)
    # if text is tiny/empty, fallback to OCR
    if len(text.strip()) < 50:
        ocr_text = extract_text_with_ocr_bytes(pdf_bytes)
        if len(ocr_text.strip()) > 20:
            text = ocr_text
    return text


def chunk_text_keep_meta(text, source_name, chunk_size_words=500, chunk_overlap=50):
    """Chunk text into overlapping word-chunks, keep metadata (source, optional page header)."""
    words = text.split()
    chunks = []
    metadatas = []
    i = 0
    n = len(words)
    while i < n:
        chunk_words = words[i:i+chunk_size_words]
        chunk_text = " ".join(chunk_words).strip()
        if chunk_text:
            chunks.append(chunk_text)
            metadatas.append({"source": source_name})
        i += chunk_size_words - chunk_overlap
    return chunks, metadatas


def build_faiss_index(embeddings_np):
    """Build an FAISS index for cosine similarity using IndexFlatIP and normalized vectors."""
    if embeddings_np is None or embeddings_np.shape[0] == 0:
        raise ValueError("Empty embeddings provided.")
    # normalize vectors (L2) so inner product == cosine similarity
    faiss.normalize_L2(embeddings_np)
    dim = embeddings_np.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_np.astype("float32"))
    return index


def embed_texts_in_batches(texts):
    """Use sentence-transformers to encode and return numpy array."""
    if not texts:
        return np.zeros((0, embedder.get_sentence_embedding_dimension()), dtype="float32")
    vectors = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False, batch_size=EMBED_BATCH_SIZE)
    return np.asarray(vectors, dtype="float32")


def ask_groq_with_context(context, question, lang_code="en"):
    """Call Groq chat with a strict prompt that asks to use context and avoid hallucination."""
    # instruct the model to answer in the detected language
    language_map = {
        "kn": "Kannada", "hi": "Hindi", "ta": "Tamil", "te": "Telugu", "mr": "Marathi",
        "ml": "Malayalam", "gu": "Gujarati", "bn": "Bengali", "pa": "Punjabi", "en": "English"
    }
    lang_name = language_map.get(lang_code, "the same language as the question")
    prompt = f"""
You are a careful PDF-QA assistant. Use only the CONTEXT passages below to answer the QUESTION.
If the answer cannot be found in the provided context, reply exactly: "This question is not related to the uploaded PDF."
Answer concisely and in {lang_name}.

--- CONTEXT START ---
{context}
--- CONTEXT END ---

QUESTION:
{question}

Answer:
"""
    resp = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role":"user", "content": prompt}],
        temperature=0.0,
        max_tokens=600
    )
    # Groq API returns choices
    try:
        return resp.choices[0].message.content.strip()
    except Exception:
        # fallback
        return "Sorry, I couldn't get an answer from the LLM."


# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Multilingual PDF Chatbot", page_icon="📚", layout="wide")
st.title("📄 Multilingual PDF Chatbot — Local embeddings + Groq")

# session state initializations
if "docs_chunks" not in st.session_state:
    st.session_state.docs_chunks = []        # list of chunk strings
if "docs_meta" not in st.session_state:
    st.session_state.docs_meta = []          # list of metadata dicts aligned with chunks
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None       # numpy array
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []       # list of (role, text)
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []    # filenames processed

st.sidebar.header("Upload PDFs")
uploaded_files = st.sidebar.file_uploader("Upload one or more PDFs (text or scanned)", accept_multiple_files=True, type=["pdf"])

process_btn = st.sidebar.button("Process PDFs", key="process_pdfs_btn")

if process_btn:
    if not uploaded_files:
        st.sidebar.error("Please upload at least one PDF.")
    else:
        # Reset indexes & buffers
        st.session_state.docs_chunks = []
        st.session_state.docs_meta = []
        st.session_state.embeddings = None
        st.session_state.faiss_index = None
        st.session_state.processed_files = []

        with st.sidebar.expander("Processing...", expanded=True):
            for pdf_f in uploaded_files:
                fname = pdf_f.name
                st.write(f"Processing: {fname} ...")
                text = extract_pdf_text(pdf_f)
                if not text or len(text.strip()) < 20:
                    st.write(" → no text detected, OCR fallback used (or empty).")
                chunks, metas = chunk_text_keep_meta(text, source_name=fname, chunk_size_words=500, chunk_overlap=80)
                # filter out very short chunks
                filtered = [(c,m) for c,m in zip(chunks, metas) if len(c.strip()) > 20]
                if not filtered:
                    st.write(f"Warning: no usable text extracted from {fname}")
                    continue
                for c,m in filtered:
                    st.session_state.docs_chunks.append(c)
                    st.session_state.docs_meta.append(m)
                st.session_state.processed_files.append(fname)
                st.write(f" → {len(filtered)} chunks added from {fname}")

        # create embeddings (if any chunks)
        if st.session_state.docs_chunks:
            st.sidebar.info("Computing embeddings (this can take a moment)...")
            embeddings_np = embed_texts_in_batches(st.session_state.docs_chunks)
            if embeddings_np.shape[0] == 0:
                st.sidebar.error("No embeddings produced. Check PDF extraction / model.")
            else:
                st.session_state.embeddings = embeddings_np
                st.session_state.faiss_index = build_faiss_index(embeddings_np.copy())
                st.sidebar.success(f"Vector store created with {embeddings_np.shape[0]} vectors.")
        else:
            st.sidebar.error("No chunks were extracted from uploaded PDFs.")

st.markdown("---")
st.subheader("Chat")

if not st.session_state.faiss_index:
    st.info("Upload and Process PDFs (sidebar) to enable chat.")
else:
    # persistent chat input
    user_input = st.text_input("Ask a question (in any language):", key="main_question_box")
    ask = st.button("Ask", key="ask_btn")

    if ask and user_input and user_input.strip():
        # remember user message
        st.session_state.chat_history.append(("user", user_input.strip()))

        # detect question language (best-effort)
        try:
            user_lang = detect(user_input)
        except Exception:
            user_lang = "en"

        # embed question and search
        qvec = embedder.encode([user_input], convert_to_numpy=True)
        qvec = np.asarray(qvec, dtype="float32")
        faiss.normalize_L2(qvec)
        D, I = st.session_state.faiss_index.search(qvec, TOP_K)  # D: inner products

        top_scores = D[0] if D.shape[0] else []
        top_idx = I[0] if I.shape[0] else []

        if len(top_scores) == 0 or top_scores[0] < SIMILARITY_THRESHOLD:
            bot_response = "This question is not related to the uploaded PDF."
        else:
            # assemble context from top chunks (include metadata)
            context_pieces = []
            for idx in top_idx:
                if idx < len(st.session_state.docs_chunks):
                    md = st.session_state.docs_meta[idx]
                    chunk = st.session_state.docs_chunks[idx]
                    context_pieces.append(f"[Source: {md.get('source','unknown')}]\n{chunk}")
            context = "\n\n---\n\n".join(context_pieces)
            bot_response = ask_groq_with_context(context, user_input, lang_code=user_lang)

        st.session_state.chat_history.append(("bot", bot_response))

    # Display chat history (most recent last)
    st.markdown("### Conversation")
    for role, text in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"**You:** {text}")
        else:
            st.markdown(f"**Bot:** {text}")
        st.markdown("---")

# Simple footer / tips
st.markdown(
    """
**Tips & Notes**
- For scanned PDFs ensure Tesseract traineddata for the languages you need are installed in your `tessdata` folder.
- If OCR results are poor, try increasing image DPI or check Tesseract language packs.
- Similarity threshold may be tuned (currently set to reduce hallucinations).
"""
)





# #===================================================================================================================================
