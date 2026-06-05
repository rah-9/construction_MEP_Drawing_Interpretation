import streamlit as st
import json
import re
from typing import Dict, Any

class DebugPanel:
    @staticmethod
    def render_debug_panel(result: Dict[str, Any]) -> None:
        """
        Renders the Streamlit Debug View containing 5 tabs:
        1. Answer: Stated answer and reasoning.
        2. Evidence: Retrieved OCR context text block.
        3. Models: Outputs from NVIDIA Model, Gemini, Qwen, and fallback chain logs.
        4. Validation: OCR Match, Bbox Validation, Agreement, and Review reasons.
        5. Confidence Breakdown: Weighted scores table and density contributions.
        """
        st.markdown("## 🔍 Debug Control Board")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "💬 Answer", 
            "📄 Evidence", 
            "🤖 Models", 
            "✅ Validation", 
            "📊 Confidence Breakdown"
        ])
        
        with tab1:
            st.markdown("### Stated Model Answer")
            answer_text = result.get("answer", "")
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
            
            st.write(answer_text)
            st.markdown("### Reasoning Summary")
            st.info(result["reasoning_summary"])

        with tab2:
            st.markdown("### Retrieved Context Text")
            st.text_area("OCR Context Evidence Block", result["retrieved_context"], height=300)

        with tab3:
            st.markdown("### Multi-Model Output Logs")
            v_results = result.get("model_verifier_results", {})
            st.write(f"**Final Answer Provider**: `{result['audit_trail']['model_used']}`")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Tier 1: NVIDIA Model**")
                st.code(v_results.get("nvidia_output", "N/A"))
            with col2:
                st.markdown("**Tier 2: Gemini 2.5 Flash**")
                st.code(v_results.get("gemini_output", "N/A"))
            with col3:
                st.markdown("**Tier 3: Local VLM**")
                st.code(v_results.get("qwen_output", "N/A"))

            st.markdown("### Fallback Chain Audit")
            st.write(f"- Fallback Triggered: `{result['audit_trail']['fallback_triggered']}`")
            st.write(f"- Multi-Model Verification Executed: `{result['audit_trail']['verified']}`")

        with tab4:
            st.markdown("### Validation Metrics & Rules")
            bd = result.get("confidence_breakdown", {})
            v_results = result.get("model_verifier_results", {})
            
            val_col1, val_col2, val_col3 = st.columns(3)
            with val_col1:
                st.metric("OCR Match / Support Score", f"{bd.get('ocr_match_score', 0.0) * 100:.0f}%")
            with val_col2:
                st.metric("BBox Grounding Valid", "Yes" if bd.get("has_bbox", False) else "No")
            with val_col3:
                st.metric("Agreement Score", f"{v_results.get('agreement_score', 1.0) * 100:.0f}%")
            
            st.write(f"**Question Intent**: `{result.get('question_analysis', {}).get('question_type', 'N/A').upper()}`")
            st.write(f"**Retrieved Page Contexts**: `{result.get('retrieved_pages', [])}` (Mode: `{result.get('reasoning_mode', 'smart')}`)")
            
            # Display Page Classifications
            classifications = result.get("page_classifications", [])
            if classifications:
                st.markdown("#### 📄 Page Classifications")
                cols = st.columns(min(4, len(classifications)))
                for idx, c in enumerate(classifications):
                    col_idx = idx % min(4, len(classifications))
                    emoji = "🗺️" if c["page_type"] == "legend" else ("📝" if c["page_type"] == "notes" else ("📊" if c["page_type"] == "schedule" else ("📐" if c["page_type"] == "drawing" else "📁")))
                    cols[col_idx].markdown(f"**Page {c['page']}** → {emoji} `{c['page_type'].upper()}`")

            # Display Page Ranking Details
            scores = result.get("ranked_pages_scores", [])
            if scores:
                st.markdown("#### 📈 Page Ranking Scores & Matches")
                for s in scores[:5]:  # show top 5
                    st.markdown(f"- **Page {s['page']}** | Score: `{s['score']:.2f}` (OCR: `{s.get('ocr_match', 0.0):.2f}`, Tag: `{s.get('tag_match', 0.0):.1f}`, Intent: `{s.get('intent_match', 0.0):.1f}`)")
                    if s.get("ocr_keyword_matches") or s.get("equipment_matches") or s.get("symbol_matches"):
                        st.markdown(f"  * Keywords: `{s.get('ocr_keyword_matches', [])}`")
                        st.markdown(f"  * Tags: `{s.get('equipment_matches', [])}`")
                        st.markdown(f"  * Symbols: `{s.get('symbol_matches', [])}`")

            # Counting Audit Panel
            q_analysis = result.get("question_analysis", {})
            counting_results = result.get("counting_results")
            if q_analysis.get("question_type") == "counting" and counting_results:
                st.markdown("#### 🧮 Counting Audit Consensus")
                audit_col1, audit_col2, audit_col3, audit_col4, audit_col5 = st.columns(5)
                with audit_col1:
                    st.metric("Target Object", str(counting_results.get("target_object", "N/A")))
                with audit_col2:
                    st.metric("OCR Count", str(counting_results.get("ocr_count", 0)))
                with audit_col3:
                    st.metric("Visual Count", str(counting_results.get("visual_count", 0)))
                with audit_col4:
                    st.metric("Verifier Count", str(counting_results.get("verifier_count", 0)))
                with audit_col5:
                    st.metric("Final Count", str(counting_results.get("count", 0)))
                
                resolved_syms = counting_results.get("resolved_symbols", [])
                if resolved_syms:
                    st.write(f"**Resolved Symbols/Abbreviations from Legend**: `{resolved_syms}`")

            st.write(f"**Audit Message**: `{result['audit_trail']['validation_msg']}`")
            st.write(f"**Disagreement Log**: `{v_results.get('disagreement_reason', 'N/A')}`")

        with tab5:
            st.markdown("### Evidence-Based Confidence Score Details")
            bd = result.get("confidence_breakdown", {})
            if not bd:
                st.info("No breakdown metadata available.")
                return

            st.write(f"**Combined Confidence Formula**:")
            st.latex(r"\text{Confidence} = 0.30 \cdot \text{OCR} + 0.25 \cdot \text{Agreement} + 0.20 \cdot \text{BBox} + 0.15 \cdot \text{Density} + 0.10 \cdot \text{ModelConf}")

            # Draw breakdown table
            data = [
                {"Factor": "OCR Match", "Weight": "30%", "Input Score": f"{bd['ocr_match_score']:.2f}", "Contribution": f"+{bd['ocr_contribution']:.3f}"},
                {"Factor": "Model Agreement", "Weight": "25%", "Input Score": f"{bd['agreement_score']:.2f}", "Contribution": f"+{bd['agreement_contribution']:.3f}"},
                {"Factor": "Bounding Box Presence", "Weight": "20%", "Input Score": "1.00" if bd["has_bbox"] else "0.00", "Contribution": f"+{bd['bbox_contribution']:.3f}"},
                {"Factor": "Evidence OCR Density", "Weight": "15%", "Input Score": f"{bd['density_score']:.2f} ({bd['ocr_elements_count']} words)", "Contribution": f"+{bd['density_contribution']:.3f}"},
                {"Factor": "Raw Model Confidence", "Weight": "10%", "Input Score": f"{bd['model_confidence']:.2f}", "Contribution": f"+{bd['model_confidence_contribution']:.3f}"},
            ]
            st.table(data)
            st.metric("Final Calculated Confidence Score", f"{result['confidence_score'] * 100:.0f}%", help="Sum of all weighted contributions.")

            # Confidence Audit Log
            audit = result.get("confidence_audit")
            if audit:
                st.markdown("#### 🛡️ Confidence Consistency Audit")
                st.json(audit)
                
                st.markdown("**Adjustments Check & Rules Applied:**")
                if audit.get("review_required"):
                    st.warning("⚠️ **Review Required** is True → Confidence score capped at `0.69` max.")
                if not audit.get("evidence_found"):
                    st.warning("⚠️ **Evidence Found** is False → Confidence score capped at `0.59` max.")
                if audit.get("review_required") and not audit.get("evidence_found"):
                    st.warning("⚠️ **Review Required** is True and **Evidence Found** is False → Confidence score capped at `0.49` max.")
                if audit.get("bbox_score", 1.0) == 0.0:
                    st.info("ℹ️ **BBox Score** is 0.0 → Bounding box not detected in response.")

    @staticmethod
    def render_debrief_mode(result: Dict[str, Any]) -> None:
        """Renders technical explainability debrief logs for design presentation."""
        st.markdown("## 🛡️ Technical Debrief Audit Trail")
        
        with st.container():
            st.markdown("### 1. Intent & Question Classification")
            q_analysis = result.get("question_analysis", {})
            st.write(f"- **Detected Question Type**: `{q_analysis.get('question_type', 'N/A').upper()}`")
            st.write(f"- **Retrieval Strategy**: {q_analysis.get('retrieval_strategy', 'N/A')}")
            st.write(f"- **Reasoning Strategy**: {q_analysis.get('reasoning_strategy', 'N/A')}")
            
            st.markdown("### 2. Context Retrieval Pipeline")
            st.write(f"- **Retrieved OCR Word Tokens Count**: `{result.get('confidence_breakdown', {}).get('ocr_elements_count', 0)}`")
            
            st.markdown("### 3. Model Orchestration Audit")
            st.write(f"- **Primary Model Queried**: `{result['audit_trail']['model_used']}`")
            st.write(f"- **Cloud API Fallback Triggered**: `{result['audit_trail']['fallback_triggered']}`")
            
            st.markdown("### 4. Verification & Agreement scoring")
            v_results = result.get("model_verifier_results", {})
            st.write(f"- **Cross-Model Agreement Score**: `{v_results.get('agreement_score', 1.0)}`")
            st.write(f"- **Verification Log**: {v_results.get('verification_summary', 'N/A')}")
            
            st.markdown("### 5. Final Decision Parameters")
            st.write(f"- **Is Human Review Required Flag**: `{result['review_required']}`")
            st.write(f"- **Reasoning Summary**: {result['reasoning_summary']}")
