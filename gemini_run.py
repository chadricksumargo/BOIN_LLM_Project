import os
os.environ["RPY2_CFFI_MODE"] = "ABI"
import api_key
from google import genai
from google.genai import types
import pathlib
import json
from rpy2.robjects import r

client = genai.Client(api_key=api_key.api_key)

filepath = pathlib.Path('Real Guidelines for BOIN api LLM_.pdf')

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
      cat("\nPercentage of patients treated at each dose level:\n")
      print(round(result$n.pat.percent, 1))
      cat("\nPercentage of toxicity at each dose level:\n")
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
    cat("Selected MTD: Dose Level", result, "\n")
    """
    return "\n".join(r(r_code))

system_instruction = """
You are a BOIN dose-finding expert trained on MD Anderson's BOIN R Package. Always stay on topic. Do not include code in your response. Ask follow up questions after your response.
Your job is to help users with BOIN clinical trial design. Answer clearly.
If the user provides all required parameters for a function (like get.boundary, get.oc, or select.mtd), you may format them clearly so code can run.
Otherwise, just continue the conversation naturally and ask follow-up questions if needed.
If any parameters are missing, politely ask the user to provide them.
If any parameters seem way off than what is expected, politely recommend more reasonable values and ask for confirmation.
Target toxicity rate is usually either 0.25 (low toxicity) or 0.3 (high toxicity).
Very important: when providing BOIN decision boundaries, correctly and accurately calculate both full decision table similar to the Shiny app and the four decision rules (for example: escalate if ≤ 0.236, de-escalate if ≥ 0.359, stay if..., eliminate if...).
Before diving into the results, always ask the user to confirm the BOIN decision boundaries and rules given, not the parameters.
In the summary section, always provide results in table format whenever possible.
"""

print("Welcome! Ask me anything about the BOIN clinical trial design, I am at your service.")
history = ''

while True:
    userInput = input("User: ")
    if userInput.strip().lower() == 'end':
        print("This session has ended. Goodbye!")
        break

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(system_instruction=system_instruction),
        contents=[
            types.Part.from_bytes(
                data=filepath.read_bytes(),
                mime_type='application/pdf',
            ),
            f"[chat history start]: {history}[chat history end]\nUser: {userInput}"
        ]
    )

    gemini_text = response.text.strip()
    print("\nGemini:")
    print(gemini_text)

    try:
        if "target" in gemini_text and "ncohort" in gemini_text and "cohortsize" in gemini_text:
            match = re.search(r'target\s*=\s*([0-9.]+).*?ncohort\s*=\s*(\d+).*?cohortsize\s*=\s*(\d+)', gemini_text, re.DOTALL)
            if match:
                output = get_boundary(float(match[1]), int(match[2]), int(match[3]))
                print("\n" + output)

        elif "ptrue" in gemini_text:
            ptrue_match = re.findall(r'[0-9]*\.?[0-9]+', gemini_text)
            if len(ptrue_match) >= 3:
                ptrue = list(map(float, ptrue_match))
                output = get_oc(ptrue)
                print("\n" + output)

        elif "ntox" in gemini_text and "npts" in gemini_text:
            ntox = re.findall(r'ntox\s*=\s*\[(.*?)\]', gemini_text)
            npts = re.findall(r'npts\s*=\s*\[(.*?)\]', gemini_text)
            if ntox and npts:
                ntox_vals = list(map(int, ntox[0].split(',')))
                npts_vals = list(map(int, npts[0].split(',')))
                output = select_mtd(ntox_vals, npts_vals)
                print("\n" + output)

    except Exception:
        pass

    history += f"User: {userInput}\nGemini: {gemini_text}\n"