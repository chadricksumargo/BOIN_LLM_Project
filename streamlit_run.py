import os
os.environ["RPY2_CFFI_MODE"] = "ABI"

import api_key
from google import genai
from google.genai import types
import pathlib
import re
import string
from rpy2.robjects import r
import base64

import streamlit as st

PASSWORD = "making cancer history"

def check_password():
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    if not st.session_state.password_correct:
        pwd = st.text_input("Enter password to access:", type="password")
        if pwd:
            if pwd == PASSWORD:
                st.session_state.password_correct = True
                st.experimental_rerun()
            else:
                st.error("❌ Password incorrect")
        return False
    else:
        return True

if not check_password():
    st.stop()

client = genai.Client(api_key=api_key.api_key)

def get_boundary(target=0.3, ncohort=10, cohortsize=3):
    r_code = f"""
    suppressMessages(library(BOIN))
    bound <- get.boundary(target={target}, ncohort={ncohort}, cohortsize={cohortsize})
    tbl <- bound$boundary.tab
    colnames(tbl) <- c('Patients Treated', 'Escalate if ≤', 'De-escalate if ≥', 'Stay if =', 'Eliminate if ≥')
    as.character(capture.output(print(tbl, row.names=FALSE)))
    """
    return "\n".join(r(r_code))

def get_oc(ptrue, target=0.3, ncohort=10, cohortsize=3):
    ptrue_str = ",".join(map(str, ptrue))
    r_code = f"""
    suppressMessages(library(BOIN))
    result <- get.oc(target={target}, p.true=c({ptrue_str}), ncohort={ncohort}, cohortsize={cohortsize})
    output <- capture.output({{
      cat("Percentage of selecting each dose as MTD:\n")
      print(round(result$MTD.select.percent, 1))
      cat("\\nPercentage of patients treated at each dose level:\n")
      print(round(result$n.pat.percent, 1))
      cat("\\nPercentage of toxicity at each dose level:\n")
      print(round(result$tox.percent, 1))
    }})
    as.character(output)
    """
    return "\n".join(r(r_code))

def select_mtd(ntox, npts, target=0.3):
    ntox_str = ",".join(map(str, ntox))
    npts_str = ",".join(map(str, npts))
    r_code = f"""
    suppressMessages(library(BOIN))
    result <- select.mtd(target={target}, ntox=c({ntox_str}), npts=c({npts_str}))
    cat("Selected MTD: Dose Level", result, "\\n")
    """
    return "\n".join(r(r_code))

system_instruction = """
BOIN API GUIDELINES

Introduction:
You are a BOIN dose-finding expert trained on MD Anderson's BOIN R Package, powered by Gemini, and launched via Streamlit.
Your role is to assist users in designing BOIN clinical trials by providing accurate, clear, and concise information based on the BOIN R Package functionalities.
You must read and rely on the following four foundational documents before responding:
1. Real Guidelines for BOIN api LLM_
2. BOIN Tutorial
3. Overview of BOIN
4. BOIN for Phase 1 Clinical Trials

A) Core workflow (user-facing flow)
Collect inputs for BOIN (target toxicity rate, number of dose levels, cohort size, max sample size, elimination/overdose control rule settings, etc.). If anything is missing, politely ask the user to provide it.
Any numerical output should only come from the BOIN R Package (always use corresponding functions) for maximum accuracy.
Sanity-check inputs: if any parameter looks unreasonable, recommend more reasonable values and ask for confirmation.
Compute and display BOIN decision boundaries (full decision table like the Shiny app and must always contain the four decision rules: escalate / de-escalate / stay / eliminate).
Since BOIN decision boundaries are the most important output to clinicians/users, it is imperative that boundaries are calculated thrice for the most accurate result before sending to the user.
Ask the user to confirm the decision boundaries and rules (not the parameters) exactly once before proceeding.
Ask which dose level to start at (default is 1).
Run the simulation / operating characteristics only after the user has seen and confirmed the escalation/de-escalation boundaries.
Select the MTD using isotonic regression at the end of the simulated/actual trial conduct.
Provide a Trial Protocol (concise, standardized), reflecting boundaries, stopping/elimination rules, start dose, and MTD selection.

B) Parameter & defaults policy
Missing inputs: If any parameters are missing for functions like get.boundary, get.oc, or select.mtd, politely ask the user to provide them.
Complete inputs: If the user supplies all required parameters, format them clearly so code can run (no code in the reply itself; just crisp, runnable parameter blocks).
Reasonableness: If values seem far off expected ranges, suggest reasonable alternatives and request confirmation.
Prohibition: If the user insists on using unreasonable parameters and ranges, strictly prohibit them from proceeding to boundaries and trial results. Explain that the user cannot use any BOIN R Package function until all parameters are in reasonable ranges.
Target toxicity rate: Default guidance is 0.25 for “low-toxicity” settings and 0.30 for “higher-toxicity” settings.

C) Output & confirmation rules
Boundary confirmation: Before any results beyond boundaries, ask the user to confirm the decision boundaries/rules once.
Decision boundaries: Always provide both the four decision rules (e.g., “escalate if ≤ …, de-escalate if ≥ …”) and the full decision table akin to the BOIN Shiny app.
Table format rule: Table’s format should always be consistent (looks the same) from session to session, user to user, and chat to chat.
Simulation gating: Show boundaries first, then ask if the user wants to proceed with simulation.
MTD selection: Use isotonic regression to select the MTD.
Protocol: After running simulation, deliver a Trial Protocol summary.
Style: Stay on topic, have a natural conversation, answer clearly, ask follow-up questions when necessary, do not include code in responses, and always provide tables wherever feasible.

Quick checklist (in order):
Gather + validate inputs
Compute & show full decision table + four rules
One-time confirmation of boundaries
Ask for start dose (default 1)
Offer to run simulation / get.oc
Select MTD via isotonic regression
Output Trial Protocol (tabular, concise)
"""

st.set_page_config(page_title="BOIN + Gemini Assistant", layout="wide")
st.title("🤖 BOIN Clinical Trial Design Assistant")
st.markdown("Ask anything about BOIN trial design. When you're done, just type `'end'`.")

if "chats" not in st.session_state:
    st.session_state.chats = {"Chat 1": {"history": [], "confirmed_boundaries": False}}
if "current_chat" not in st.session_state:
    st.session_state.current_chat = "Chat 1"

confirm_words = {"yes", "y", "confirmed", "confirm", "proceed", "yea", "ye", "yeah", "agree", "u can go", "u can proceed", "u can continue", "go ahead", "continue", "proceed with the boundaries", "proceed with the rules"}

def normalize(text):
    return text.lower().translate(str.maketrans('', '', string.punctuation))

def img_to_base64(path):
    with open(path, "rb") as img_file:
        b64 = base64.b64encode(img_file.read()).decode()
    return b64

img_b64 = img_to_base64("MD Anderson Logo.png")

st.sidebar.markdown(
    f"""
    <a href="https://www.mdanderson.org/" target="_blank" style="display: inline-block;">
      <img src="data:image/png;base64,{img_b64}" width="175" style="margin-bottom: 10px; margin-right: 10px;" />
    </a>
    """,
    unsafe_allow_html=True
)

st.sidebar.title("BOIN + Gemini Assistant")
chat_names = list(st.session_state.chats.keys())
chat_names.append("➕ New Chat")
selected_chat = st.sidebar.selectbox("Select Chat", chat_names, index=chat_names.index(st.session_state.current_chat) if st.session_state.current_chat in chat_names else 0)

if selected_chat == "➕ New Chat":
    new_chat_name = f"Chat {len(st.session_state.chats)+1}"
    st.session_state.chats[new_chat_name] = {"history": [], "confirmed_boundaries": False}
    st.session_state.current_chat = new_chat_name
else:
    st.session_state.current_chat = selected_chat

chat_data = st.session_state.chats[st.session_state.current_chat]

for msg in chat_data["history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

userInput = st.chat_input("Ask a question about BOIN design...")

if userInput:
    chat_data["history"].append({"role": "user", "content": userInput})
    with st.chat_message("user"):
        st.markdown(userInput)

    if userInput.strip().lower() == 'end':
        st.success("This session has ended. Goodbye!")
    else:
        user_message = normalize(userInput.strip())
        words = set(user_message.split())
        if words.intersection(confirm_words):
            chat_data["confirmed_boundaries"] = True

        dynamic_system_instruction = system_instruction
        if not chat_data["confirmed_boundaries"]:
            dynamic_system_instruction += "\nPlease confirm the BOIN decision boundaries before proceeding. Reply with 'yes' to confirm."

        from google.generativeai.types import Part

        pdf_paths = [
            "Real Guidelines for BOIN api LLM_.pdf",
            "BOIN Tutorial.pdf",
            "Overview of BOIN.pdf",
            "BOIN for Phase 1 Clinical Trials.pdf"
        ]

        pdf_parts = []
        for path in pdf_paths:
            try:
                with open(path, "rb") as f:
                    pdf_parts.append(Part.from_bytes(f.read(), mime_type="application/pdf"))
            except Exception as e:
                st.error(f"Failed to read PDF {path}: {e}")

        contents = pdf_parts + [
            f"[chat history start]: {''.join([m['content'] for m in chat_data['history'] if m['role'] == 'user'])}[chat history end]\nUser: {userInput}"
        ]

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(system_instruction=dynamic_system_instruction),
            contents=contents
        )

        gemini_text = response.text.strip()

        chat_data["history"].append({"role": "assistant", "content": gemini_text})
        with st.chat_message("assistant"):
            st.markdown(gemini_text)

        try:
            if "target" in gemini_text and "ncohort" in gemini_text and "cohortsize" in gemini_text:
                match = re.search(r'target\s*=\s*([0-9.]+).*?ncohort\s*=\s*(\d+).*?cohortsize\s*=\s*(\d+)', gemini_text, re.DOTALL)
                if match:
                    target = float(match[1])
                    ncohort = int(match[2])
                    cohortsize = int(match[3])
                    output = get_boundary(target, ncohort, cohortsize)
                    st.code(output, language="text")

            elif "ptrue" in gemini_text:
                ptrue_match = re.findall(r'[0-9]*\.?[0-9]+', gemini_text)
                if len(ptrue_match) >= 3:
                    ptrue = list(map(float, ptrue_match))
                    output = get_oc(ptrue)
                    st.code(output, language="text")

            elif "ntox" in gemini_text and "npts" in gemini_text:
                ntox = re.findall(r'ntox\s*=\s*\[(.*?)\]', gemini_text)
                npts = re.findall(r'npts\s*=\s*\[(.*?)\]', gemini_text)
                if ntox and npts:
                    ntox_vals = list(map(int, ntox[0].split(',')))
                    npts_vals = list(map(int, npts[0].split(',')))
                    output = select_mtd(ntox_vals, npts_vals)
                    st.code(output, language="text")

        except Exception as e:
            st.error(f"Error during BOIN function execution: {e}")