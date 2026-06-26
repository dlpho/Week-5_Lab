"""
Streamlit frontend — Student Handbook Chatbot

app.py owns only UI concerns:
  - page config, CSS, layout
  - session state for chat history and handbook status
  - calling engine.initialize_handbook() once at startup
  - rendering the sidebar reference panel
  - driving the chat input → SSE stream → write_stream loop

All PDF path resolution, ingestion, RAG, memory, and guardrails
live in core/engine.py.
"""

import os
import requests
import streamlit as st

from core.engine import initialize_handbook

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
BACKEND_URL     = os.environ.get("BACKEND_URL", "http://localhost:8000")
STREAM_ENDPOINT = f"{BACKEND_URL}/chat/stream"

# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Handbook Chatbot",
    page_icon="📚",
    layout="wide",
)

st.markdown("""
<style>
/* ── 3-dot bounce loading animation ── */
.dot-flashing {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 10px 4px;
}
.dot-flashing span {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: #9ca3af;
    animation: dotBounce 1.2s infinite ease-in-out;
}
.dot-flashing span:nth-child(2) { animation-delay: 0.2s; }
.dot-flashing span:nth-child(3) { animation-delay: 0.4s; }

@keyframes dotBounce {
    0%, 80%, 100% { transform: translateY(0);   opacity: 0.4; }
    40%            { transform: translateY(-8px); opacity: 1;   }
}

[data-testid="stSidebarContent"] h1 { margin-top: 0.25rem; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Handbook initialisation — once per session, delegated entirely to engine
# --------------------------------------------------------------------------
if "handbook_status" not in st.session_state:
    with st.spinner("Loading school handbook…"):
        st.session_state.handbook_status = initialize_handbook()

status = st.session_state.handbook_status

# --------------------------------------------------------------------------
# Sidebar — status badge + ground-truth quick reference
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("📚 Reference")
    st.caption("Ground-truth excerpts from the student handbook.")
    if status["ok"]:
        chunks = status.get("chunks", "?")
        path   = status.get("path", "")
        st.success(f"✅ Handbook loaded successfully")
    else:
        st.error(
            f"❌ Handbook not loaded\n\n"
            f"**Reason:** {status.get('error')}\n\n"
            "**Looked in:**\n"
            + "\n".join(f"- `{p}`" for p in status.get("searched", []))
        )
    st.divider()
    with st.expander("🏫 Governance & Values", expanded=False):
        st.markdown("""
**Mission**  
Oakridge Academy is dedicated to fostering intellectual curiosity, critical thinking, and moral
integrity. We prepare diverse student populations to become responsible global citizens and lifelong
learners through rigorous academics, comprehensive arts programs, and competitive athletics.

**Vision**  
Our vision is to be a benchmark institution where tradition meets innovation, cultivating leaders
who approach the world with empathy, resilience, and outstanding scholarly capability.

**Core Values**  
Academic Excellence · Integrity · Respect · Stewardship
        """)
    with st.expander("📊 Grading Framework & GPA Scale", expanded=False):
        st.markdown("""
| Grade | Range     | GPA Equivalent | Academic Standing  |
|-------|-----------|----------------|--------------------|
| A     | 93 – 100  | 4.00           | Excellent          |
| B     | 85 – 92   | 3.00           | Above Average      |
| C     | 75 – 84   | 2.00           | Satisfactory       |
| D     | 65 – 74   | 1.00           | Passing / At Risk  |
| F     | Below 65  | 0.00           | Failing            |

Minimum passing grade: **65%**  
Academic probation triggered below **2.00** GPA.
        """)
    with st.expander("📋 Key Policies", expanded=False):
        st.markdown("""
- **Attendance**: max **10 unexcused absences** per semester — exceeding this automatically voids academic credits for that term regardless of current grade
- **Academic Integrity**: plagiarism or unauthorized AI use → zero on assessment (first offence); repeat offences → formal suspension hearing before the Honor Council
- **The Oakridge Pledge**: "As a member of the Oakridge Academy community, I pledge to live
honorably, to refrain from lying, cheating, or stealing, and to actively defend the physical,
digital, and intellectual spaces of my school from disrespect and harm."
        """)
    with st.expander("🎽 Uniform Policy", expanded=False):
        st.markdown("""
**Monday – Thursday**  
Navy blue blazer (official school crest), tailored khaki trousers or institutional plaid pleated skirt, solid white collared dress shirt, dark leather dress shoes.

**Friday (Spirit Days)**  
Approved Oakridge polo shirt paired with neat denim jeans (no distressing, holes, or visible frayed patches).
        """)
    with st.expander("📜 Code of Conduct", expanded=False):
        st.markdown("""
- Uphold the **Oakridge Pledge** at all times
- **Electronic devices**: phones, wearables, and personal laptops must be stored away during instructional blocks unless teacher grants permission
- **Bullying / hazing / cyber-harassment**: immediate external suspension pending expulsion assessment
- **Vandalism**: full financial restitution + community service
- **Academic dishonesty**: zero on assessment (1st offence); repeat offences → Honor Council suspension hearing
        """)
    with st.expander("🏥 Operations & Emergency", expanded=False):
        st.markdown("""
- **Health suite**: continuously staffed by a registered nurse
- **Prescription drugs & OTC medication**: stored in the locked Health Suite (documentation required)
- **Emergency inhalers / Epi-Pens**: student self-carry authorized with dual doctor & parent waiver
- **Emergency drills**: fire, severe weather, and security drills occur **monthly**
- **Evacuation maps**: posted near the entry threshold of every room
        """)
        
    st.divider()
    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --------------------------------------------------------------------------
# Main area
# --------------------------------------------------------------------------
st.title("Student Handbook Chatbot")
st.caption("Ask me anything about school policies, grades, uniform, attendance, and more.")

if not status["ok"]:
    st.error(
        "The handbook could not be loaded so answers may be incomplete. "
        "Check the sidebar for details."
    )

# --------------------------------------------------------------------------
# Chat state + history replay
# --------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --------------------------------------------------------------------------
# Chat input + streaming response
# --------------------------------------------------------------------------
if prompt := st.chat_input("Ask a question about school policies…"):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        # Show 3-dot animation while waiting for the first SSE chunk
        loading_slot = st.empty()
        loading_slot.markdown(
            '<div class="dot-flashing">'
            '<span></span><span></span><span></span>'
            '</div>',
            unsafe_allow_html=True,
        )

        def stream_from_api():
            first_chunk = True
            try:
                with requests.post(
                    STREAM_ENDPOINT,
                    json={"message": prompt},
                    stream=True,
                    timeout=120,
                ) as response:
                    response.raise_for_status()
                    for raw_line in response.iter_lines():
                        if not raw_line:
                            continue
                        decoded = raw_line.decode("utf-8")
                        if decoded.startswith("data: "):
                            chunk = decoded[6:].replace("\\n", "\n")
                            if first_chunk:
                                loading_slot.empty()
                                first_chunk = False
                            yield chunk

            except requests.exceptions.ConnectionError:
                loading_slot.empty()
                yield (
                    "🚨 **Cannot reach the backend.**  "
                    "Make sure `uvicorn api:app --port 8000` is running."
                )
            except requests.exceptions.Timeout:
                loading_slot.empty()
                yield "🚨 **Request timed out.** The model may be overloaded — try again."
            except requests.exceptions.ChunkedEncodingError:
                loading_slot.empty()
            except Exception as exc:
                loading_slot.empty()
                yield f"🚨 Unexpected error: {exc}"

        full_response = st.write_stream(stream_from_api())

    st.session_state.messages.append({"role": "assistant", "content": full_response or ""})