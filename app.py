import os
import logging
from pathlib import Path

# Neutralize torch classes inspection to prevent Streamlit file watcher warning
try:
    import torch
    torch.classes.__path__ = []
except ImportError:
    pass

import streamlit as st
import json
import re
from PIL import Image

import config
from core.pdf_processor import PDFProcessor
from core.evidence_store import EvidenceStore
from core.orchestrator import RequestOrchestrator
from core.annotation_engine import AnnotationEngine
from core.debug_panel import DebugPanel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Streamlit Page Configuration
st.set_page_config(
    page_title="MEP Drawing Interpretation System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Rich Aesthetics)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Outfit:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        background: linear-gradient(135deg, #FF6B6B 0%, #4D96FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #8A9Aad;
        margin-bottom: 1.5rem;
    }
    
    .card-container {
        background-color: #1E293B;
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #334155;
        margin-bottom: 1rem;
    }
    
    .insight-card {
        background-color: #0F172A;
        border-left: 5px solid #38BDF8;
        border-radius: 6px;
        padding: 1rem;
        margin-bottom: 1rem;
        border-top: 1px solid #1E293B;
        border-right: 1px solid #1E293B;
        border-bottom: 1px solid #1E293B;
    }

    .insight-header {
        font-family: 'Outfit', sans-serif;
        color: #38BDF8;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .badge {
        display: inline-block;
        padding: 0.25em 0.6em;
        font-size: 75%;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.375rem;
        margin-right: 0.5rem;
    }
    
    .badge-high {
        background-color: #059669;
        color: #ECFDF5;
    }
    
    .badge-medium {
        background-color: #D97706;
        color: #FEF3C7;
    }
    
    .badge-low {
        background-color: #DC2626;
        color: #FEF2F2;
    }
    
    .badge-review {
        background-color: #B91C1C;
        color: #FFF5F5;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .human-review-banner {
        background-color: #7F1D1D;
        border: 2px solid #FE2525;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        color: #FFF5F5;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .answer-banner {
        background-color: #1E293B;
        border: 2px solid #38BDF8;
        border-radius: 8px;
        padding: 1.5rem;
        color: #F8FAFC;
        margin-bottom: 1rem;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# App Title
st.markdown("<h1 class='main-title'>MEP Drawing Interpreter</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Evidence Retrieval & Reasoning System for Blueprints</p>", unsafe_allow_html=True)

# Initialize Session State
if "repo" not in st.session_state:
    st.session_state.repo = None
if "store" not in st.session_state:
    st.session_state.store = None
if "last_processed_file" not in st.session_state:
    st.session_state.last_processed_file = None
if "result" not in st.session_state:
    st.session_state.result = None
if "session_memory" not in st.session_state:
    from core.session_memory import SessionMemory
    st.session_state.session_memory = SessionMemory()

# Sidebar Content
with st.sidebar:
    st.markdown("## 📥 Ingestion & Controls")
    
    # PDF File Uploader
    uploaded_file = st.file_uploader("Upload MEP Drawing (PDF)", type=["pdf"])
    
    if uploaded_file is not None:
        if st.session_state.last_processed_file != uploaded_file.name:
            with st.spinner("Processing PDF, rendering pages, and running OCR/Layout..."):
                temp_pdf_path = config.UPLOADS_DIR / uploaded_file.name
                with open(temp_pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                try:
                    repo = PDFProcessor.process_pdf(temp_pdf_path)
                    st.session_state.repo = repo
                    st.session_state.store = EvidenceStore(repo)
                    st.session_state.session_memory.clear()
                    st.session_state.last_processed_file = uploaded_file.name
                    st.session_state.result = None  # Clear previous results
                    st.success("Drawing uploaded and processed successfully!")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")
                    logger.exception("Ingestion failed")

    # If document loaded, show selection parameters
    if st.session_state.repo is not None:
        repo = st.session_state.repo
        st.markdown("---")
        st.markdown("## ⚙️ Sheet Settings")
        
        # Select page
        page_nums = [p.page_number for p in repo.pages]
        selected_page_num = st.selectbox("Select Page / Sheet", page_nums)
        
        page_entry = next(p for p in repo.pages if p.page_number == selected_page_num)
        
        # Display Page thumbnail preview
        st.markdown("### Sheet Thumbnail")
        if os.path.exists(page_entry.preview_path):
            st.image(page_entry.preview_path, caption=f"Page {selected_page_num} Preview", use_column_width=True)

        st.markdown("---")
        st.markdown("## 🧠 Reasoning Mode")
        mode_label = st.radio(
            "Select Reasoning Mode",
            ["Current Sheet", "Smart Retrieval", "Full Document"],
            index=1,
            help="Current Sheet uses only the selected page. Smart Retrieval automatically ranks and finds relevant pages. Full Document scans all pages."
        )
        reasoning_mode = "smart"
        if mode_label == "Current Sheet":
            reasoning_mode = "sheet"
        elif mode_label == "Full Document":
            reasoning_mode = "full"
            st.warning("Full document mode increases context and latency but improves cross-sheet understanding.")

        st.markdown("---")
        st.markdown("## ⚙️ Reasoning Settings")
        enable_smart = st.checkbox("Enable Smart Retrieval", value=True)
        enable_full = st.checkbox("Enable Full Document Context", value=True)
        enable_cross = st.checkbox("Enable Cross-Sheet Reasoning", value=True)
        enable_gemini_ver = st.checkbox("Enable Gemini Verification", value=True)
        enable_local_ver = st.checkbox("Enable Local Verification", value=True)
        enable_counting_ver = st.checkbox("Enable Counting Verification", value=True)
        
        settings_dict = {
            "enable_smart_retrieval": enable_smart,
            "enable_full_document": enable_full,
            "enable_cross_sheet": enable_cross,
            "enable_gemini_verification": enable_gemini_ver,
            "enable_local_verification": enable_local_ver,
            "enable_counting_verification": enable_counting_ver
        }

        st.markdown("---")
        st.markdown("## 🛡️ Debrief Settings")
        enable_debrief = st.checkbox("Enable Explainability Debrief Mode", value=False)
            
        # Collapsible Repository JSON viewer
        with st.expander("Inspect Repository Metadata (JSON)"):
            json_preview = {
                "page_number": page_entry.page_number,
                "dimensions_pts": {"width": page_entry.dimensions.width, "height": page_entry.dimensions.height},
                "page_hash": page_entry.page_hash,
                "layout_regions_count": len(page_entry.layout_regions),
                "ocr_elements_count": len(page_entry.ocr_elements),
                "metadata": page_entry.metadata
            }
            st.json(json_preview)
    else:
        st.info("Upload an MEP drawing PDF in the sidebar to get started.")

# Main Application Panel
if st.session_state.repo is not None and st.session_state.store is not None:
    repo = st.session_state.repo
    store = st.session_state.store
    page_entry = next(p for p in repo.pages if p.page_number == selected_page_num)

    # 1. Document Summary Banner
    st.markdown("### Document & Sheet Summary")
    
    total_ocr_elements = len(page_entry.ocr_elements)
    layout_regions_list = [r.region_type for r in page_entry.layout_regions]
    detected_regions_str = ", ".join([r.replace("_", " ").title() for r in layout_regions_list])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class='card-container'>
            <div class='metric-label'>Total Pages</div>
            <div class='metric-value'>{len(repo.pages)}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='card-container'>
            <div class='metric-label'>Dimensions (Pts)</div>
            <div class='metric-value'>{int(page_entry.dimensions.width)}x{int(page_entry.dimensions.height)}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class='card-container'>
            <div class='metric-label'>OCR Elements</div>
            <div class='metric-value'>{total_ocr_elements}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class='card-container'>
            <div class='metric-label'>Detected Regions</div>
            <div class='metric-value' style='font-size: 1.1rem; padding-top: 0.6rem;'>{detected_regions_str if detected_regions_str else 'None'}</div>
        </div>
        """, unsafe_allow_html=True)

    # =========================================================================
    # V2 UPGRADE: Drawing Insight Panel
    # =========================================================================
    st.markdown("### 📊 Drawing Insight Panel")
    
    # Automatically generate stats
    notes_region_valid = any(r.region_type == "notes" for r in page_entry.layout_regions)
    legends_region_valid = any(r.region_type == "legend" for r in page_entry.layout_regions)
    
    # Parse unique equipment tags from OCR
    text_corpus = " ".join([el.text for el in page_entry.ocr_elements])
    equipment_tags = sorted(list(set(re.findall(r"\b(FCU-\d+|VAV-\d+|RTU-\d+|CD-\d+|RG-\d+|TG-\d+|FD-\d+|M-\d+)\b", text_corpus))))[:10]
    
    # Recommend potential question categories
    rec_categories = []
    if legends_region_valid:
        rec_categories.append("Legend Symbol Verifications (e.g. 'What does CD-1 mean?')")
    if notes_region_valid:
        rec_categories.append("Drawing Specification Checks (e.g. 'What notes apply to duct lining?')")
    if equipment_tags:
        rec_categories.append("Equipment Data lookups (e.g. 'What is flow rate of FCU-1?')")
    rec_categories.append("Counting items (e.g. 'How many diffusers are in the drawing area?')")

    ins_col1, ins_col2 = st.columns(2)
    with ins_col1:
        st.markdown(f"""
        <div class='insight-card'>
            <div class='insight-header'>🔍 Automatically Parsed Blueprint Elements</div>
            - <strong>Detected Notes Table</strong>: {"✅ YES (region coordinates locked)" if notes_region_valid else "❌ NO (vector fallback)"}<br>
            - <strong>Detected Legend Symbols</strong>: {"✅ YES (symbol bounds locked)" if legends_region_valid else "❌ NO (vector fallback)"}<br>
            - <strong>Detected Layout Margins</strong>: Title Block, Legend, Notes, Drawing Area<br>
            - <strong>Parsed Unique Equipment Tags</strong>: {", ".join(equipment_tags) if equipment_tags else "None explicitly matches CAD formats"}
        </div>
        """, unsafe_allow_html=True)
    with ins_col2:
        st.markdown("<div class='insight-card'><div class='insight-header'>💡 Potential Question Categories</div>" + 
                    "".join([f"- {cat}<br>" for cat in rec_categories]) + "</div>", unsafe_allow_html=True)

    # Document Intelligence Panel
    st.markdown("### 🔍 Document Intelligence Memory")
    doc_intel = getattr(repo, "doc_intelligence", {})
    if doc_intel:
        with st.expander("Expand Document Intelligence Details", expanded=False):
            tab_intel1, tab_intel2, tab_intel3, tab_intel4 = st.tabs([
                "📄 Layout Pages", 
                "🏷️ Discovered Equipment Tags", 
                "🔣 Symbols & Abbreviations",
                "📝 Code Specifications"
            ])
            with tab_intel1:
                st.write(f"- **Legend Sheets**: `{doc_intel.get('legend_pages', [])}`")
                st.write(f"- **General Notes Sheets**: `{doc_intel.get('notes_pages', [])}`")
                st.write(f"- **Equipment Schedule Sheets**: `{doc_intel.get('schedule_pages', [])}`")
                st.write(f"- **Title Block Sheets**: `{doc_intel.get('title_block_pages', [])}`")
            with tab_intel2:
                st.write(doc_intel.get("equipment_tags", []))
            with tab_intel3:
                col_intel_sym, col_intel_abb = st.columns(2)
                with col_intel_sym:
                    st.markdown("**Discovered Legend Symbols:**")
                    st.write(doc_intel.get("symbols", []))
                with col_intel_abb:
                    st.markdown("**Discovered Abbreviations:**")
                    st.write(doc_intel.get("abbreviations", []))
            with tab_intel4:
                st.write(doc_intel.get("specifications", []))
    else:
        st.info("No Document Intelligence data found for this repository.")

    # Visual Mode Toggle
    st.markdown("### Visual Grounding Modes")
    col_toggle1, col_toggle2 = st.columns(2)
    with col_toggle1:
        visualize_layouts = st.checkbox("Overlay Detected Layout Regions (Legend, Title Block, Notes)", value=False)

    # 3. Question Answering Form
    st.markdown("### Ask a Question about this Sheet")
    question_placeholder = "e.g., What is the drawing scale in the title block? or How many supply diffusers are in the drawing area?"
    user_question = st.text_input("Enter your question here:", placeholder=question_placeholder)
    
    if st.button("Interpret Sheet", type="primary"):
        if not user_question.strip():
            st.warning("Please enter a question first.")
        else:
            # Run Orchestrator
            orchestrator = RequestOrchestrator()
            with st.spinner("Executing interpretation pipeline (Analysis -> Context Retrieval -> Model Verification -> Confidence)..."):
                try:
                    result = orchestrator.interpret_question(
                        store, 
                        selected_page_num, 
                        user_question,
                        reasoning_mode=reasoning_mode,
                        settings=settings_dict,
                        session_memory=st.session_state.session_memory
                    )
                    st.session_state.result = result
                except Exception as e:
                    st.error(f"Interpretation Pipeline Error: {e}")
                    logger.exception("Pipeline Error")

    # 4. Results Panel
    if st.session_state.result is not None:
        res = st.session_state.result
        
        answer_text = res.get("answer", "")
        if isinstance(answer_text, str) and answer_text.strip().startswith("{"):
            try:
                parsed = json.loads(answer_text)
                answer_text = (
                    parsed.get("answer") or
                    parsed.get("reasoning_summary") or
                    parsed.get("reasoning") or
                    answer_text
                )
            except Exception:
                pass
        # Strip any remaining JSON if parsing failed
        if isinstance(answer_text, str) and answer_text.strip().startswith("{"):
            answer_text = re.sub(r'[{}":]', ' ', answer_text).strip()

        st.markdown("---")
        st.markdown("## 🏁 Interpretation Output")
        
        # Display Answer Box 
        st.markdown(f"""
        <div class='answer-banner'>
            <div style='font-size: 0.9rem; color: #94A3B8; margin-bottom: 0.5rem; text-transform: uppercase;'>Stated Answer</div>
            <strong>Answer:</strong> {answer_text}
        </div>
        """, unsafe_allow_html=True)

        # Warning Badge for Human Review (V2: Never suppress answers, display banner alongside)
        if res["review_required"]:
            st.markdown(f"""
            <div class='human-review-banner'>
                ⚠️ HUMAN REVIEW REQUIRED<br>
                <span style='font-size: 0.9rem; font-weight: normal;'>
                    The system could not verify the answer with high confidence (disagreement score or OCR matches are low). Please verify the drawing details manually.
                </span>
            </div>
            """, unsafe_allow_html=True)
            
        # Display Metrics / Audit trail
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        
        conf_label = res["confidence_label"]
        conf_score = int(res["confidence_score"] * 100)
        if conf_label == "High":
            badge_class = "badge-high"
        elif conf_label == "Medium":
            badge_class = "badge-medium"
        elif conf_label == "Low":
            badge_class = "badge-low"
        else:
            badge_class = "badge-review"
            
        with m_col1:
            st.markdown(f"""
            <div class='card-container'>
                <div class='metric-label'>Confidence Rating</div>
                <div style='margin-top: 0.5rem;'><span class='badge {badge_class}' style='font-size: 1rem;'>{conf_label} ({conf_score}%)</span></div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col2:
            st.markdown(f"""
            <div class='card-container'>
                <div class='metric-label'>Active Model</div>
                <div class='metric-value' style='font-size: 1.2rem; padding-top: 0.4rem;'>{res['audit_trail']['model_used']}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col3:
            st.markdown(f"""
            <div class='card-container'>
                <div class='metric-label'>OCR Grounding</div>
                <div class='metric-value' style='font-size: 1.2rem; padding-top: 0.4rem;'>
                    {'Verified' if res['confidence_breakdown'].get('ocr_match_score', 0.0) >= 0.5 or res['evidence_found'] else 'Not Found'}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col4:
            fallback_status = "Yes" if res["audit_trail"]["fallback_triggered"] else "No"
            verification_status = "Yes" if res["audit_trail"]["verified"] else "No"
            st.markdown(f"""
            <div class='card-container'>
                <div class='metric-label'>Pipeline Flags</div>
                <div style='font-size: 0.85rem; color: #E2E8F0; padding-top: 0.3rem;'>
                    Fallback Triggered: <strong>{fallback_status}</strong><br>
                    Verifier Agreement: <strong>{verification_status}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Drawing display
        st.markdown("### Drawing Canvas & Grounded Evidence")
        
        img_to_show = None
        if visualize_layouts:
            layout_annotated_name = f"{repo.file_hash}_p{selected_page_num}_layout.png"
            layout_annotated_path = config.ANNOTATIONS_DIR / layout_annotated_name
            if not layout_annotated_path.exists():
                AnnotationEngine.annotate_layout_regions(
                    image_path=Path(page_entry.image_path),
                    layout_regions=page_entry.layout_regions,
                    output_path=layout_annotated_path
                )
            img_to_show = str(layout_annotated_path.resolve())
            caption_text = f"Sheet {selected_page_num} layout boundaries overlay"
        elif res["annotated_image_path"] and not res["review_required"]:
            img_to_show = res["annotated_image_path"]
            primary_img_page = res.get("primary_image_page", selected_page_num)
            caption_text = f"Sheet {primary_img_page} visual grounding highlights (Answer Location)"
        else:
            img_to_show = page_entry.image_path
            caption_text = f"Sheet {selected_page_num} Base High-Resolution Image"
            
        if img_to_show and os.path.exists(img_to_show):
            st.image(img_to_show, caption=caption_text, use_column_width=True)
            primary_img_page = res.get("primary_image_page")
            if primary_img_page is not None and primary_img_page != selected_page_num:
                st.info(f"Answer found on Sheet {primary_img_page}")
        else:
            st.error("Grounded image could not be located on disk.")

        # V2 UPGRADE: Debug Panel (Renders 5 Tabs)
        DebugPanel.render_debug_panel(res)

        # V2 UPGRADE: Debrief Mode (Explainability View)
        if enable_debrief:
            st.markdown("---")
            DebugPanel.render_debrief_mode(res)
