import os
import re
import json
import fitz  # Assuming PyMuPDF is installed
import google.generativeai as genai


# NOTE: The chromadb and sentence_transformers imports have been removed
# as they introduce unnecessary complexity and risk of fragmentation for
# structured quote documents like these.

# ============================================================
# 1. PDF EXTRACTION (multi-line table rows + container awareness) - KEPT AS IS
# ============================================================

def extract_clean_text(pdf_path):
    """
    Extract clean text from PDF with proper table row merging and container awareness.
    This function uses PyMuPDF (fitz).
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF {pdf_path}: {e}")
        return "", []

    final_text = ""
    all_lines = []

    container_pattern = re.compile(r"20'STD|40'STD|40HC", re.I)

    for page_num, page in enumerate(doc):
        blocks = page.get_text("blocks")
        merged_lines = []
        current_line = ""
        current_container = ""

        # Simplified text merging logic
        for b in blocks:
            txt = b[4].strip()
            if not txt:
                continue

            # Attempt to retain structural context
            txt = txt.replace('\n', ' ')

            merged_lines.append(txt)
            all_lines.append(txt)

        final_text += f"\n[PAGE {page_num + 1}]\n" + "\n".join(merged_lines)

    doc.close()
    return final_text, all_lines


# ============================================================
# Helper: Clean place tokens / remove RAMP / R A I L / PORT / TERMINAL words
# ============================================================

def _remove_non_city_tokens_from_text(text: str) -> str:
    """
    Remove tokens like RAMP, R A I L, R AI L (with spaces), PORT, TERMINAL, YARD, WHSE
    while preserving city/state/country strings and punctuation.
    This performs a conservative replacement so we don't alter numeric rates.
    """
    if not text:
        return text

    # common tokens to remove (allow spaced variants like R A I L or R AI L)
    tokens = [
        r"\bR\s*A\s*M\s*P\b",
        r"\bR\s*A\s*I\s*L\b",
        r"\bRAMP\b",
        r"\bRAIL\b",
        r"\bPORT\b",
        r"\bTERMINAL\b",
        r"\bYARD\b",
        r"\bWHSE\b",
        r"\bR\s*A\s*M\b",  # defensive
        r"\bR\s*A\b",      # defensive
    ]

    cleaned = text
    for t in tokens:
        cleaned = re.sub(t, "", cleaned, flags=re.I)

    # collapse multiple spaces while keeping commas and punctuation
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+,", ",", cleaned)
    cleaned = re.sub(r",\s{2,}", ", ", cleaned)
    return cleaned.strip()


def clean_origin_destination(token: str) -> str:
    """
    Clean a single origin/destination token by removing non-city words and
    trimming to a reasonable city,state,country-like part if possible.
    """
    if not token:
        return token
    # Remove bracketed/parenthetical items and separators, then remove tokens
    t = re.sub(r"[\(\)\[\]]+", " ", token)
    t = _remove_non_city_tokens_from_text(t)
    # collapse whitespace and trim punctuation
    t = re.sub(r"\s{2,}", " ", t).strip(" ,;-")
    return t


# ============================================================
# 2. HELPER: Export Surcharge Line Detection - MODIFIED (exclude overweight)
# ============================================================

def _is_overweight_line(line: str) -> bool:
    """
    Returns True for overweight / weight-range related lines that must be ignored.
    """
    if not line:
        return False
    patterns = [
        r"Overweight Additional",               # explicit header
        r"TON container gross weight",          # phrase found in overweight rows
        r"\bBetween\s+\d",                      # "Between 20..."
        r"\bcontainer gross weight\b",
        r"\bTON\b",
        r"Rate is different depending on weight",
    ]
    return any(re.search(p, line, re.I) for p in patterns)


def extract_export_table_lines(all_lines):
    """
    Returns lines that belong to the ExportSurcharges table.
    If table doesn't exist, returns empty list.

    Modified to ignore overweight-related rows (so they never reach the LLM).
    """
    export_start = -1
    export_lines = []

    for i, l in enumerate(all_lines):
        if re.search(r"Export Surcharges|Export Charges", l, re.I):
            export_start = i + 1
            break

    if export_start >= 0:
        # Check for the end of the surcharge block
        for l in all_lines[export_start:]:
            if not l.strip() or re.search(r"Freight|Local|Import|Quotation number|Validity|Valid from|Valid to", l, re.I):
                break
            # skip overweight / weight-based lines explicitly
            if _is_overweight_line(l):
                # intentionally ignore these lines
                continue
            export_lines.append(l.strip())

    return export_lines


# ============================================================
# 3. LLM EXTRACTION FUNCTION (Requesting JSON) - MINOR IMPROVEMENTS, SAME FORMAT
# ============================================================

def extract_rates_with_gemini(file_name, full_text_for_llm, api_key):
    """
    Calls the Gemini model to extract structured data (JSON) from one PDF's text.
    """
    print(f"-> Sending {file_name} to Gemini...")

    # sanitize the prompt text to remove RAMP / R A I L / PORT tokens that are not city names
    sanitized_full_text = _remove_non_city_tokens_from_text(full_text_for_llm)

    # ------------------ PROMPT DEFINITION ------------------
    prompt = f"""
    You are a logistics pricing extraction AI.
    The following text is the contents of ONE single shipping quote document, which may contain MULTIPLE services (routes).
    Your task is to extract all pricing and logistics fields for EVERY container type of EVERY service into a list of JSON objects.

    Extract rates EXACTLY as shown in the text. DO NOT assume, infer, estimate, or invent ANY value.

    =========================================================
    OUTPUT FORMAT — STRICT JSON ARRAY ONLY
    =========================================================

    Output must be a single JSON array (list of objects). Each object must strictly contain these keys:

    [
        {{
            "ShippingLine": "...",
            "QuotationNumber": "...",
            "Origin": "...",
            "Destination": "...",
            "ViaPort": "N/A or port name",
            "Commodity": "...",
            "ContainerType": "20'STD | 40'STD | 40'HC",
            "ValidFrom": "...",
            "ValidTo": "...",
            "BasicFreightUSD": "number",
            "ExportSurcharges": {{ "USD": "number", "details": "itemized list of charges/currencies" }},
            "FreightSurcharges": {{ "USD": "number", "details": "itemized list of charges/currencies" }},
            "ImportSurcharges": {{ "INR": "number or N/A", "details": "itemized list of charges/currencies" }},
            "LumpsumCharges": "itemized list of charges/currencies",
            "TotalUSD": "Total of all USD components (BasicFreight + ExportSurcharges + USD parts of FreightSurcharges/ImportSurcharges)"

        }},
        // ... one object for every container type of every service ...
    ]

    Rules:
    - **Each service (route) must result in three JSON objects** (20'STD, 40'STD, 40HC).
    - **ExportSurcharges / FreightSurcharges:** Put the *total USD amount* for that charge type in the main key (e.g., BasicFreightUSD) and itemize the components in the 'details' sub-key.
    - **If a surcharge table is missing:** Use 0 for the USD number and "N/A" for details.
    - **ViaPort:** Use "N/A" if not found.
    -**DONOT PICK UP OTHER NAMES THAN CITY NAMES - CHICAGO, IL, US R AMP, R AI L IN THIS CASE PICK UP ONLY 'CHICAGO, IL, US' AND DONOT INCLUDE PORT RAIL OR RAMP FOR ALL ORIGIN,DESTINATION
    PDF CONTENT BELOW
    -**Do Not Read the Charges - 'Overweight Additional On Olf' - We don't need them
    =========================================================

    {sanitized_full_text}

    Return ONLY the valid JSON array. Nothing else.
    """

    # ------------------ LLM CALL ------------------
    try:
        genai.configure(api_key=api_key)

        system_instruction = """
        You are a meticulous logistics agent specialized in extracting pricing data from shipping line PDF quotes.
        Your primary goal is to produce perfectly valid, well-structured JSON output.
        Always calculate totals correctly and strictly adhere to the requested schema.
        Do NOT include any surrounding text, headers, or markdown (like ```json). ONLY the raw JSON array.
        """

        model_gemini = genai.GenerativeModel(
            model_name="gemini-2.5-flash-lite",
            system_instruction=system_instruction,
        )

        response = model_gemini.generate_content(prompt)

        # Attempt to clean and parse the response text
        json_text = ""
        if response and hasattr(response, "text") and response.text:
            json_text = response.text.strip()
        else:
            print(f"!!! Empty response from Gemini for {file_name}")
            return []

        # Remove common markdown fence if the model incorrectly added it
        if json_text.startswith("```json"):
            json_text = json_text[7:]
        if json_text.startswith("```"):
            # remove starting triple backticks
            json_text = json_text.lstrip("`")
        if json_text.endswith("```"):
            json_text = json_text[:-3]

        # Some models may add stray leading/trailing whitespace or non-json text, try to locate first '['
        first_bracket = json_text.find('[')
        if first_bracket > 0:
            json_text = json_text[first_bracket:]

        extracted_data = json.loads(json_text)
        print(f"<- Successfully extracted {len(extracted_data)} records from {file_name}.")
        return extracted_data

    except json.JSONDecodeError as e:
        # attempt to print a helpful snippet
        snippet = ""
        try:
            snippet = response.text[:1000]
        except Exception:
            snippet = "<no response text available>"
        print(f"!!! Error parsing JSON for {file_name}: {e}")
        print(f"Received text (snippet): {snippet}...")
        return []
    except Exception as e:
        print(f"!!! Error processing {file_name} with Gemini: {e}")
        return []


# ============================================================
# 4. MAIN EXECUTION PIPELINE
# ============================================================

def process_all_quotes(folder_path, gemini_api_key):
    """
    Main function to loop through all PDF files and extract data into a single list of dictionaries.
    """
    all_extracted_data = []

    if not os.path.exists(folder_path):
        print(f"Error: Folder path not found: {folder_path}")
        return all_extracted_data

    for file_name in os.listdir(folder_path):
        if not file_name.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(folder_path, file_name)
        print(f"\n=== Starting Extraction for {file_name} ===")

        # 1. Extract Text
        pdf_text, all_lines = extract_clean_text(pdf_path)

        if not pdf_text:
            continue

        # sanitize pdf_text to remove RAMP/RAIL/PORT tokens (so origins/destinations are cleaner)
        pdf_text_sanitized = _remove_non_city_tokens_from_text(pdf_text)

        # 2. Inject Export Surcharge Context
        export_lines = extract_export_table_lines(all_lines)
        export_text = "\n[EXPORT SURCHARGES TABLE]\n" + ("\n".join(export_lines) if export_lines else "0")
        full_text_for_llm = pdf_text_sanitized + "\n" + export_text

        # 3. Call LLM for Structured JSON
        file_data = extract_rates_with_gemini(file_name, full_text_for_llm, gemini_api_key)

        # 4. Consolidate Results
        # post-clean origins/destinations in returned objects if present
        cleaned_file_data = []
        for obj in file_data:
            try:
                if isinstance(obj, dict):
                    if "Origin" in obj and obj["Origin"]:
                        obj["Origin"] = clean_origin_destination(obj["Origin"])
                    if "Destination" in obj and obj["Destination"]:
                        obj["Destination"] = clean_origin_destination(obj["Destination"])
                cleaned_file_data.append(obj)
            except Exception:
                cleaned_file_data.append(obj)

        all_extracted_data.extend(cleaned_file_data)

    return all_extracted_data


# ============================================================
# 5. EXECUTION BLOCK
# ============================================================

if __name__ == "__main__":
    # --- CONFIGURE THESE VALUES ---
    # NOTE: Replace with your actual folder path and API key
    FOLDER_PATH = r"\QUOTES"
    # IMPORTANT: Never hardcode your API key in production code. Use environment variables.
    GEMINI_API_KEY = "YOUR API KEY"

    # Run the processing pipeline
    final_data_list = process_all_quotes(FOLDER_PATH, GEMINI_API_KEY)

    print("\n\n===== FINAL CONSOLIDATED DATA (Python List of Dicts) =====")
    print(f"Total records extracted: {len(final_data_list)}")

    # Example: Print the first few records for review
    print(json.dumps(final_data_list[:3], indent=4))

    JSON_OUTPUT_PATH = "consolidated_freight_quotes.json"

    try:
        # Open the file for writing
        with open(JSON_OUTPUT_PATH, "w", encoding="utf-8") as f:
            # Use json.dump to write the list of dictionaries to the file
            # indent=4 ensures the file is saved in a human-readable, formatted way
            json.dump(final_data_list, f, indent=4)

        print(f"\n✅ Successfully saved {len(final_data_list)} records to {JSON_OUTPUT_PATH}")

    except Exception as e:
        print(f"\n❌ Error saving file to {JSON_OUTPUT_PATH}: {e}")

        # Optional: Display a quick view as previously requested


    def display_quick_list(data_list):
        print(f"\n--- Quick Summary View ({len(data_list)} Records) ---")
        keys_to_check = ["QuotationNumber", "Origin", "Destination", "ContainerType", "TotalUSD"]
        for i, item in enumerate(data_list[:5]):  # Only show the first 5 records
            summary = f"Record {i + 1}: "
            for key in keys_to_check:
                value = item.get(key, "MISSING")
                summary += f"{key}: {value} | "
            print(summary.strip(" | "))


    display_quick_list(final_data_list)

    # --- NEXT STEPS FOR LONG-TERM STORAGE ---
    # Now you can easily load this list into a database, e.g.:
    # import pandas as pd
    # df = pd.DataFrame(final_data_list)
    # df.to_sql('freight_quotes', your_database_engine, if_exists='append', index=False)
