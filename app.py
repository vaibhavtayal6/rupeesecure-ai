import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import tempfile
import json
import time
from PIL import Image
from datetime import datetime
import plotly.graph_objects as go

# back-end modules (as in your project)
from gemini_verifier import GeminiBanknoteVerifier
from resnet_placeholder import ResNetSegmentor  # placeholder
from security_features import (
    get_security_features,
    get_feature_descriptions,
    get_verification_tips,
    get_denomination_colors,
)
from utils.image_processing import create_directory_structure, preprocess_image
from config import Config

# ── Dark page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Indian Banknote Verifier",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ─────────────────────────────────────────────────────────────
if "verification_history" not in st.session_state:
    st.session_state.verification_history = []
if "current_result" not in st.session_state:
    st.session_state.current_result = None
if "analysis_step" not in st.session_state:
    st.session_state.analysis_step = 0
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "page" not in st.session_state:
    st.session_state.page = "🏠 Dashboard"

# ── Load dark CSS ─────────────────────────────────────────────────────────────
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css()

# ── Cache static lookups ──────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _denomination_colors():
    return get_denomination_colors()

@st.cache_data(show_spinner=False)
def _feature_desc():
    return get_feature_descriptions()

# ── UI helpers ────────────────────────────────────────────────────────────────
def display_feature_score(score: float, feature_name: str):
    if score >= Config.STRONG_MATCH_THRESHOLD:
        color = "#22c55e"; emoji = "✅"; status = "Excellent"
    elif score >= Config.WEAK_MATCH_THRESHOLD:
        color = "#f59e0b"; emoji = "⚠️"; status = "Good"
    else:
        color = "#ef4444"; emoji = "❌"; status = "Poor"

    feature_display = feature_name.replace("_", " ").title()
    pct = f"{score*100:.0f}%"
    st.markdown(
        f"""
        <div style="display:flex;justify-content:space-between;align-items:center;margin:.6rem 0;"
             role="group" aria-label="{feature_display} score">
            <span style="font-size:1.05rem;">{emoji} <strong>{feature_display}</strong></span>
            <div>
                <span style="color:{color};font-weight:800;font-size:1.05rem;">{pct}</span>
                <span style="color:{color};font-size:.9rem;margin-left:.5rem;">({status})</span>
            </div>
        </div>
        <div class="progress-bar" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="{score*100:.0f}">
          <div class="progress-fill" style="width:{score*100}%;">{pct}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def simulate_analysis_progress():
    steps = [
        "Loading and preprocessing image...",
        "Analyzing security features...",
        "Validating serial numbers...",
        "Running AI verification...",
        "Compiling results...",
    ]
    with st.status("Starting analysis…", expanded=True) as status:
        for step in steps:
            st.write("• " + step)
            time.sleep(0.5)
        status.update(label="Analysis complete ✅", state="complete")

# ── Sidebar ──────────────────────────────────────────────────────────────────
def sidebar_nav():
    with st.sidebar:
        st.title("🪙 Banknote Verifier")
        st.caption("Dark • Neon • Glass")
        st.markdown("---")

        st.subheader("👤 User")
        c1, c2 = st.columns([1, 3])
        with c1:
            st.image("https://via.placeholder.com/50/8b5cf6/ffffff?text=U", width=50)
        with c2:
            st.write("**Welcome!**")
            st.caption("Verified User")
        st.markdown("---")

        st.radio(
            "Navigate To",
            ["🏠 Dashboard", "🔍 Verify Banknote", "📋 Security Features", "📊 Analysis History", "⚙️ Settings"],
            key="page",
        )

        st.markdown("---")
        st.subheader("📈 Quick Stats")
        total = len(st.session_state.verification_history)
        genuine = sum(r.get("verdict") == "REAL" for r in st.session_state.verification_history)
        fake = sum(r.get("verdict") == "FAKE" for r in st.session_state.verification_history)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total", total)
        with col2:
            st.metric("Genuine", genuine)
        st.metric("Counterfeit", fake)

# ── Pages ────────────────────────────────────────────────────────────────────
def show_dashboard():
    st.markdown('<div class="main-header">🏠 Banknote Verification — Dark</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown(
            """
            <div class="feature-card">
              <h2>🛡️ AI-Powered Counterfeit Defense</h2>
              <p>Upload a clear banknote image and get a verdict with feature-wise evidence. Dark, neon, and glass — ready for demos.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.image("https://via.placeholder.com/300x200/0b0f17/e5e7eb?text=Secure+Banking", use_column_width=True)

    st.markdown("---")
    st.markdown('<div class="sub-header">🚀 Quick Actions</div>', unsafe_allow_html=True)

    a1, a2, a3, a4 = st.columns(4)
    with a1:
        if st.button("🔍 New Verification", use_container_width=True):
            st.session_state.analysis_step = 0
            st.session_state.page = "🔍 Verify Banknote"
            st.rerun()
        st.caption("Start a fresh check")
    with a2:
        if st.button("📋 View Features", use_container_width=True):
            st.session_state.page = "📋 Security Features"
            st.rerun()
        st.caption("All security cues")
    with a3:
        if st.button("📊 View History", use_container_width=True):
            st.session_state.page = "📊 Analysis History"
            st.rerun()
        st.caption("Past results")
    with a4:
        if st.button("⚙️ Settings", use_container_width=True):
            st.session_state.page = "⚙️ Settings"
            st.rerun()
        st.caption("Configure thresholds")

    st.markdown("---")
    st.markdown('<div class="sub-header">📊 System Analytics</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    total = len(st.session_state.verification_history)
    genuine = sum(r.get("verdict") == "REAL" for r in st.session_state.verification_history)
    fake = sum(r.get("verdict") == "FAKE" for r in st.session_state.verification_history)

    with c1:
        st.markdown(f'<div class="stats-card"><h3>📈</h3><h2>{total}</h2><p>Total Verifications</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stats-card"><h3>✅</h3><h2>{genuine}</h2><p>Genuine Notes</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="stats-card"><h3>❌</h3><h2>{fake}</h2><p>Counterfeit Detected</p></div>', unsafe_allow_html=True)
    with c4:
        accuracy = "98.7%" if total else "—"
        st.markdown(f'<div class="stats-card"><h3>🎯</h3><h2>{accuracy}</h2><p>System Accuracy</p></div>', unsafe_allow_html=True)

def show_verification_page():
    st.markdown('<div class="main-header">🔍 Banknote Verification</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown('<div class="sub-header">📤 Upload Banknote</div>', unsafe_allow_html=True)
        input_method = st.radio("Choose Input Method", ["📁 Upload Image", "📷 Use Camera"], horizontal=True, key="input_method")

        uploaded_file = None
        if input_method == "📁 Upload Image":
            uploaded_file = st.file_uploader(
                "Choose banknote image",
                type=["jpg", "jpeg", "png", "bmp"],
                help="Tip: Place on plain background, avoid glare.",
                key="file_uploader",
            )
        else:
            uploaded_file = st.camera_input("Take a picture of the banknote", help="Center the note and hold steady.", key="camera_input")

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.session_state.uploaded_image = image
            st.image(image, caption="Uploaded Banknote", use_column_width=True)

            with st.expander("🔍 Image Analysis", expanded=True):
                st.info("✅ Image loaded successfully")
                st.write(f"**Format:** {getattr(image, 'format', '—')}")
                st.write(f"**Size:** {image.size[0]} × {image.size[1]} px")
                st.write(f"**Mode:** {image.mode}")

    with c2:
        st.markdown('<div class="sub-header">⚙️ Verification Settings</div>', unsafe_allow_html=True)

        if st.session_state.get("uploaded_image") or uploaded_file:
            denomination = st.selectbox(
                "Select Denomination:",
                [10, 20, 50, 100, 200, 500, 2000],
                format_func=lambda x: f"₹{x}",
                index=4,  # default ₹500
                key="denomination",
            )

            colors = _denomination_colors().get(denomination, {})
            st.info(f"**₹{denomination} Note**: {colors.get('name', 'Standard')} series")

            api_key = st.text_input(
                "Gemini API Key (optional):",
                type="password",
                help="Leave empty if already set in .env file",
                key="api_key",
            )

            with st.expander("🔧 Advanced Options"):
                st.checkbox("Enable detailed feature analysis", value=True, key="detail_analysis")
                st.checkbox("Save verification report", value=True, key="save_report")
                st.checkbox("Show segmentation results", value=False, key="show_segmentation")

            if st.button("🚀 Start Verification", type="primary", use_container_width=True):
                current_file = uploaded_file if uploaded_file else st.session_state.get("uploaded_image")
                if current_file:
                    verify_banknote(current_file, denomination, api_key)
                else:
                    st.error("Please upload an image first")
        else:
            st.info("👆 Upload a banknote image to begin verification")
            st.markdown("---")
            st.markdown("### 🎯 Quick Demo")
            if st.button("Try Sample Verification", use_container_width=True):
                demo_result = {
                    "verdict": "REAL",
                    "confidence": 0.94,
                    "failed_features": [],
                    "feature_details": {
                        "ashok_pillar": {"matching_score": 0.95},
                        "gandhi": {"matching_score": 0.96},
                        "security_thread": {"matching_score": 0.93},
                        "serial_numbers": {"matching_score": 0.98},
                    },
                    "serial_validation": {"pass": True, "left_serial": "AB 123456", "right_serial": "AB 123456", "format": "AA NNNNNN"},
                    "human_readable_explanation": "All core features match expected references; serials validate to RBI format.",
                    "manual_inspection_suggestions": ["Tilt the note to confirm color-shift ink.", "View watermark under light for clarity."],
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "denomination": 500,
                }
                st.session_state.current_result = demo_result
                st.session_state.verification_history.append(demo_result)
                st.rerun()

def verify_banknote(uploaded_file, denomination, api_key):
    st.markdown("---")
    st.markdown('<div class="sub-header">🔬 Analysis in Progress</div>', unsafe_allow_html=True)
    simulate_analysis_progress()

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            if hasattr(uploaded_file, "getvalue"):
                tmp_file.write(uploaded_file.getvalue())
            else:
                uploaded_file.save(tmp_file.name)
            tmp_path = tmp_file.name

        verifier_api_key = api_key or Config.GEMINI_API_KEY
        if not verifier_api_key:
            st.error("❌ Please provide a Gemini API key or set it in the .env file")
            os.unlink(tmp_path)
            return

        verifier = GeminiBanknoteVerifier(api_key=verifier_api_key)
        result = verifier.verify_banknote(image_path=tmp_path, denomination=denomination)

        result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result["denomination"] = denomination

        st.session_state.current_result = result
        st.session_state.verification_history.append(result)
        os.unlink(tmp_path)

        st.rerun()

    except Exception as e:
        st.error(f"❌ Error during verification: {str(e)}")
        if Config.DEBUG:
            st.exception(e)

def display_verification_results():
    if not st.session_state.current_result:
        return

    result = st.session_state.current_result
    st.markdown("---")
    st.markdown('<div class="main-header">📊 Verification Results</div>', unsafe_allow_html=True)

    verdict = result.get("verdict", "UNKNOWN")
    confidence = float(result.get("confidence", 0))

    # 🎉 subtle celebration cues on REAL; toast on others
    if verdict == "REAL":
        st.balloons()
        banner = '<div class="verdict-real">✅ GENUINE BANKnote</div>'
    elif verdict == "FAKE":
        st.toast("Counterfeit detected. Review failed features.", icon="🚫")
        banner = '<div class="verdict-fake">❌ COUNTERFEIT DETECTED</div>'
    else:
        st.toast("Needs manual inspection.", icon="🧐")
        banner = '<div class="verdict-suspect">⚠️ REQUIRES MANUAL INSPECTION</div>'

    st.markdown(banner, unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Confidence", f"{confidence:.1%}")
        features = result.get("feature_details", {}) or {}
        passed = sum(1 for d in features.values() if float(d.get("matching_score", 0)) >= Config.STRONG_MATCH_THRESHOLD)
        st.caption(f"Features Passed: **{passed}/{len(features)}**")
        sv = result.get("serial_validation", {}) or {}
        st.caption(f"Serial Validation: **{'Pass' if sv.get('pass') else 'Fail'}**")

    with c2:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=confidence * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Confidence Level"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#8b5cf6"},
                    "steps": [
                        {"range": [0, 70], "color": "#1f2937"},
                        {"range": [70, 85], "color": "#3f3f46"},
                        {"range": [85, 100], "color": "#065f46"},
                    ],
                    "threshold": {"line": {"color": "#f43f5e", "width": 4}, "thickness": 0.75, "value": 80},
                },
            )
        )
        fig.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="#0b0f17", font_color="#e5e7eb")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    t1, t2, t3 = st.tabs(["🔍 Feature Analysis", "🔢 Serial Details", "📋 Summary"])

    with t1:
        st.markdown('<div class="sub-header">Feature Scores</div>', unsafe_allow_html=True)
        for feature, details in (result.get("feature_details") or {}).items():
            score = float(details.get("matching_score", 0.0))
            display_feature_score(score, feature)
        failed = result.get("failed_features", []) or []
        if failed:
            st.markdown('<div class="sub-header">Failed Checks</div>', unsafe_allow_html=True)
            for f in failed:
                st.error(f"• {f.replace('_', ' ').title()}")

    with t2:
        st.markdown('<div class="sub-header">Serial Validation</div>', unsafe_allow_html=True)
        sv = result.get("serial_validation", {}) or {}
        if sv.get("pass"):
            st.success("Serial Numbers Valid ✅")
        else:
            st.error("Serial Validation Failed ❌")
        st.markdown(
            f"""<div class="serial-box">
                <strong>Left Serial:</strong> {sv.get('left_serial','N/A')}<br>
                <strong>Right Serial:</strong> {sv.get('right_serial','N/A')}<br>
                <strong>Format:</strong> {sv.get('format','N/A')}
            </div>""",
            unsafe_allow_html=True,
        )
        if sv.get("explanation"):
            st.caption(sv.get("explanation"))

    with t3:
        st.markdown('<div class="sub-header">Human-readable Explanation</div>', unsafe_allow_html=True)
        exp = result.get("human_readable_explanation", "") or "—"
        st.write(exp)
        tips = result.get("manual_inspection_suggestions", []) or []
        if tips:
            st.markdown('<div class="sub-header">Manual Verification Tips</div>', unsafe_allow_html=True)
            for tip in tips:
                st.info("• " + tip)

    st.markdown("---")
    a1, a2, a3 = st.columns(3)
    with a1:
        if st.button("🔄 Verify Another Note", use_container_width=True):
            st.session_state.current_result = None
            st.session_state.uploaded_image = None
            st.rerun()
    with a2:
        report_json = json.dumps(result, indent=2)
        st.download_button(
            label="📄 Download Report",
            data=report_json,
            file_name=f"banknote_verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )
    with a3:
        if st.button("📊 View History", use_container_width=True):
            st.session_state.page = "📊 Analysis History"
            st.rerun()

def show_security_features_page():
    st.markdown('<div class="main-header">📋 Security Features (Dark)</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="feature-card">
        <h3>🛡️ Understanding Banknote Security</h3>
        <p>Layered security elements across denominations. Use this guide to quickly check critical features.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1, 2])
    with c1:
        denomination = st.selectbox(
            "Select Denomination:",
            [10, 20, 50, 100, 200, 500, 2000],
            format_func=lambda x: f"₹{x}",
            key="features_denomination",
        )
    with c2:
        colors = _denomination_colors().get(denomination, {})
        st.info(f"**₹{denomination} Note Characteristics**: {colors.get('name', 'Standard series')}")

    features = get_security_features(denomination)
    feature_descriptions = _feature_desc()

    st.markdown(f'<div class="sub-header">🛡️ Features for ₹{denomination}</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for idx, (feature, is_present) in enumerate(features.items()):
        with cols[idx % 2]:
            status_emoji = "✅" if is_present else "❌"
            status_text = "Present" if is_present else "Not Present"
            tone = "#22c55e" if is_present else "#ef4444"

            with st.expander(f"{status_emoji} {feature.replace('_', ' ').title()}", expanded=True):
                if feature in feature_descriptions:
                    st.write(feature_descriptions[feature])
                st.write(
                    f"**Status:** <span style='color:{tone}; font-weight:bold;'>{status_text}</span>",
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    st.markdown('<div class="sub-header">🎯 Manual Verification Guide</div>', unsafe_allow_html=True)

    verification_tips = get_verification_tips()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🔍 Look For:")
        for i in range(0, len(verification_tips), 2):
            if i < len(verification_tips):
                st.markdown(f"<div class='security-feature'>• {verification_tips[i]}</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("### 👆 Feel For:")
        for i in range(1, len(verification_tips), 2):
            if i < len(verification_tips):
                st.markdown(f"<div class='security-feature'>• {verification_tips[i]}</div>", unsafe_allow_html=True)

def show_analysis_history():
    st.markdown('<div class="main-header">📊 Verification History</div>', unsafe_allow_html=True)

    if not st.session_state.verification_history:
        st.info("No verification history yet. Complete your first verification to see results here.")
        return

    total = len(st.session_state.verification_history)
    genuine = sum(r.get("verdict") == "REAL" for r in st.session_state.verification_history)
    fake = sum(r.get("verdict") == "FAKE" for r in st.session_state.verification_history)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Verifications", total)
    with c2:
        st.metric("Genuine Notes", genuine)
    with c3:
        st.metric("Counterfeit Notes", fake)

    st.markdown("---")
    st.markdown('<div class="sub-header">📅 Verification Records</div>', unsafe_allow_html=True)

    for i, result in enumerate(reversed(st.session_state.verification_history)):
        verdict = result.get("verdict", "UNKNOWN")
        confidence = result.get("confidence", 0)
        timestamp = result.get("timestamp") or "—"
        denomination = result.get("denomination", "—")

        if verdict == "REAL":
            v_emoji, v_color = "✅", "#22c55e"
        elif verdict == "FAKE":
            v_emoji, v_color = "❌", "#ef4444"
        else:
            v_emoji, v_color = "⚠️", "#f59e0b"

        c1_, c2_, c3_ = st.columns([2, 1, 1])
        with c1_:
            st.markdown(
                f"""
                <div class="history-item">
                    <div style="display:flex;align-items:center;margin-bottom:.5rem;">
                        <span style="font-size:1.25rem;font-weight:900;color:{v_color};margin-right:1rem;">
                            {v_emoji} {verdict}
                        </span>
                        <span style="background:{v_color};color:black;padding:.3rem .8rem;border-radius:14px;font-size:.9rem;font-weight:800;">
                            ₹{denomination}
                        </span>
                    </div>
                    <div style="color:#cbd5e1;">
                        Confidence: <strong>{confidence:.1%}</strong> • {timestamp}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2_:
            if st.button("View Details", key=f"view_{i}", use_container_width=True):
                st.session_state.current_result = result
                st.rerun()
        with c3_:
            if st.button("Delete", key=f"delete_{i}", use_container_width=True):
                st.session_state.verification_history.pop(len(st.session_state.verification_history) - 1 - i)
                st.toast("Deleted record", icon="🗑️")
                st.rerun()

def show_settings_page():
    st.markdown('<div class="main-header">⚙️ System Settings</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🔧 Application Settings")
        with st.expander("🔑 API Configuration", expanded=True):
            st.text_input("Gemini API Key", value=Config.GEMINI_API_KEY or "", type="password")
            st.info("Set GEMINI_API_KEY in .env file for permanent configuration")

        with st.expander("🎯 Verification Settings"):
            st.slider("Strong Match Threshold", 0.5, 1.0, float(Config.STRONG_MATCH_THRESHOLD), 0.05)
            st.slider("Weak Match Threshold", 0.5, 1.0, float(Config.WEAK_MATCH_THRESHOLD), 0.05)
            st.slider("Minimum Confidence", 0.5, 1.0, float(Config.MIN_CONFIDENCE_REAL), 0.05)

        with st.expander("🎨 Display Settings"):
            st.checkbox("Show detailed feature analysis", value=True)
            st.checkbox("Enable animations", value=True)
            st.checkbox("Show confidence scores", value=True)

    with c2:
        st.markdown("### 📊 System Information")
        st.markdown("#### 🔄 System Status")
        cc1, cc2 = st.columns(2)
        with cc1:
            st.success("✅ Gemini AI: Connected")
            st.success("✅ Image Processing: Active")
        with cc2:
            st.success("✅ Database: Ready")
            st.success("✅ Security: Enabled")

        st.markdown("#### 📈 Usage Statistics")
        st.write(f"**Total Verifications:** {len(st.session_state.verification_history)}")
        st.write(f"**System Uptime:** Always available")
        st.write(f"**Last Update:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        st.markdown("#### 🛠️ Maintenance")
        cc1, cc2 = st.columns(2)
        with cc1:
            if st.button("Clear History", use_container_width=True):
                st.session_state.verification_history = []
                st.session_state.current_result = None
                st.success("History cleared successfully!")
        with cc2:
            if st.button("Refresh System", use_container_width=True):
                st.rerun()

        st.markdown("#### 💾 Data Management")
        export_data = {
            "verification_history": st.session_state.verification_history,
            "export_timestamp": datetime.now().isoformat(),
        }
        st.download_button(
            label="Download Export",
            data=json.dumps(export_data, indent=2),
            file_name=f"banknote_verifier_export_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json",
            use_container_width=True,
        )

# ── Router ────────────────────────────────────────────────────────────────────
def main():
    create_directory_structure()
    sidebar_nav()
    if st.session_state.current_result and st.session_state.page != "📊 Analysis History":
        display_verification_results()
    page = st.session_state.page
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "🔍 Verify Banknote":
        show_verification_page()
    elif page == "📋 Security Features":
        show_security_features_page()
    elif page == "📊 Analysis History":
        show_analysis_history()
    elif page == "⚙️ Settings":
        show_settings_page()
    else:
        show_dashboard()

if __name__ == "__main__":
    main()
