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
    with st.spinner("Loading student handbook…"):
        st.session_state.handbook_status = initialize_handbook()

status = st.session_state.handbook_status

# --------------------------------------------------------------------------
# Sidebar — status badge + ground-truth quick reference
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("📚 Quick Reference")
    st.caption("Ground-truth excerpts from the student handbook.")

    if status["ok"]:
        chunks = status.get("chunks", "?")
        path   = status.get("path", "")
        label  = "cached index" if path == "cached" else f"{chunks} chunks indexed"
        st.success(f"✅ Handbook ready — {label}")
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
To develop principled, capable, and globally competitive learners.

**Vision**  
A school community committed to excellence, integrity, and service.

**Core Values**  
Excellence · Integrity · Respect · Responsibility · Service
        """)

    with st.expander("📊 Grading Framework", expanded=False):
        st.markdown("""
| Grade | Range |
|-------|-------|
| A     | 93 – 100 |
| B     | 85 – 92  |
| C     | 78 – 84  |
| D     | 70 – 77  |
| F     | Below 70 |

Minimum passing grade: **70%**  
Academic probation triggered below **75% average**.
        """)

    with st.expander("📋 Key Policies", expanded=False):
        st.markdown("""
- **Attendance**: max 3 absences per subject per term
- **Tardiness**: 3 lates = 1 absence
- **Academic Probation**: < 75% cumulative average
- **Appeals**: within 5 school days of grade release
- **Tutoring**: Mon–Fri, 3:00–5:00 PM at the Learning Hub
        """)

    with st.expander("🎽 Uniform Policy", expanded=False):
        st.markdown("""
**Monday – Thursday**  
White polo shirt (school logo), khaki slacks/skirt, black shoes.

**Friday**  
PE uniform or house colour shirt (as scheduled).

**Violations**  
1st offence: verbal warning.  
2nd offence: written notice to parent/guardian.
        """)

    with st.expander("📜 Code of Conduct", expanded=False):
        st.markdown("""
- Uphold the **Oakridge Pledge** at all times
- No mobile phones during class without teacher permission
- Bullying → immediate suspension
- Academic dishonesty → zero on assessment + probation
        """)

    with st.expander("🏥 Operations & Emergency", expanded=False):
        st.markdown("""
- **Clinic hours**: 7:00 AM – 5:00 PM
- **Emergency drill**: last Friday of every month
- **Evacuation routes**: posted on every classroom door
- **Contact**: registrar@school.edu · +63 2 8XXX XXXX
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