import pandas as pd
import re
import spacy
import contractions
from collections import Counter
from spacy.symbols import ORTH
from spacy.matcher import Matcher
from spacy.pipeline import EntityRuler
from spacy.tokens import Span
from tqdm import tqdm

MONTHS = ["january","february","march","april","may","june",
            "july","august","september","october","november","december",
            "jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]

WEEKDAYS = ["monday","tuesday","wednesday","thursday", "friday","saturday","sunday"]

def cleaning_process(df, dataset_type):

  df['expanded_text'] = df['expanded_text'].replace({r'\r\n': ' ', r'\n': ' ', r'\r': r' '}, regex=True)
  df['expanded_text'] = df['expanded_text'].str.replace(r'[^\x00-\x7F]+', '', regex=True)
  df['expanded_text'] = df['expanded_text'].str.replace(r'[\x00-\x1f\x7f-\x9f\xad]', '', regex=True)
  df['expanded_text'] = df['expanded_text'].str.replace(u'\xa0', ' ', regex=False)

  df['expanded_text'] = df['expanded_text'].str.replace(r"(a\.|p\.)\s(m\.)", r"\1\2", regex=True)

  # Normalize the period, apostrophy, backward and forward slash, and at symbol for emails
  df['expanded_text'] = df['expanded_text'].str.replace(r'(?<=\w)\s*\.\s*(?=\w)', '.', regex=True)
  df['expanded_text'] = df['expanded_text'].str.replace(r'(?<=\w)\s*\'\s*(?=\w)', "'", regex=True)
  df['expanded_text'] = df['expanded_text'].str.replace(r'(?<=\w)\s*\\\s*(?=\w)', r'\\', regex=True)
  # df['expanded_text'] = df['expanded_text'].str.replace(r'(?<=\w)\s*/\s*(?=\w)', '/', regex=True)
  df['expanded_text'] = df['expanded_text'].str.replace(r'(?<=\w)\s*@\s*(?=\w)', '@', regex=True)
  df['expanded_text'] = df['expanded_text'].str.replace(r'(?<=\w)\s*\_\s*(?=\w)', '_', regex=True)
  df['expanded_text'] = df['expanded_text'].str.replace(r'(?<=\w)\s*\-\s*(?=\w)', '-', regex=True)
  df['expanded_text'] = df['expanded_text'].str.replace(r'(?<=\w)\s*\+\s*(?=\w)', '+', regex=True)

  # Normalize the period, comma, colon, forward and backward slash for Digits
  df['expanded_text'] = df['expanded_text'].str.replace(r'(?<=\d)\s*\.\s*(?=\d)', '.', regex=True)
  df['expanded_text'] = df['expanded_text'].str.replace(r'(?<=\d)\s*,\s*(?=\d)', ',', regex=True)
  df['expanded_text'] = df['expanded_text'].str.replace(r'(?<=\d)\s*:\s*(?=\d)', ':', regex=True)
  df['expanded_text'] = df['expanded_text'].str.replace(r'(?<=\d)\s*\\\s*(?=\d)', r'\\', regex=True)
  df['expanded_text'] = df['expanded_text'].str.replace(r'(?<=\d)\s*/\s*(?=\d)', '/', regex=True)
  df['expanded_text'] = df['expanded_text'].str.replace(r'(?<=\d)\s*%', '%', regex=True)

  df['expanded_text'] = df['expanded_text'].str.replace(r'\$+', '$', regex=True)
  df['expanded_text'] = df['expanded_text'].str.replace(r'@+', '@', regex=True)
  df['expanded_text'] = df['expanded_text'].str.replace(r':+', ':', regex=True)
  df['expanded_text'] = df['expanded_text'].str.replace(r'\.+', '.', regex=True)
  df['expanded_text'] = df['expanded_text'].str.replace(r',+', ',', regex=True)
  df['expanded_text'] = df['expanded_text'].str.replace(r'\*+', '*', regex=True)
  df['expanded_text'] = df['expanded_text'].str.replace(r'\-+', '-', regex=True)
  df['expanded_text'] = df['expanded_text'].str.replace(r'\++', '+', regex=True)
  df['expanded_text'] = df['expanded_text'].str.replace(r'_+', '_', regex=True)
  df['expanded_text'] = df['expanded_text'].str.replace(r'=+', '=', regex=True)
  df['expanded_text'] = df['expanded_text'].str.replace(r'\\+', r'\\', regex=True)
  df['expanded_text'] = df['expanded_text'].str.replace(r'//+', r'//', regex=True)
  df['expanded_text'] = df['expanded_text'].str.replace(r'\<+', r'<', regex=True)
  df['expanded_text'] = df['expanded_text'].str.replace(r'\>+', r'>', regex=True)
  df['expanded_text'] = df['expanded_text'].str.replace(r'%+', r'%', regex=True)
  df['expanded_text'] = df['expanded_text'].str.replace(r'#+', r'#', regex=True)

  df['expanded_text'] = df['expanded_text'].str.replace(r'(?<=[0-9a-zA-Z])\s*/\s*(?=[0-9a-zA-Z])', '/', regex=True)

  # Exception for apostrophy with numbers
  df['expanded_text'] = df['expanded_text'].str.replace(r'(\d+)[\'"]([a-zA-Z]+)', '\1 \2\3', regex=True)
  df['expanded_text'] = df['expanded_text'].str.replace(r'([a-zA-Z]+)[\'"](\d+)', '\1\2 \3', regex=True)

  # Resolve the overlaping word with number that detects as WEBNER and without
  # changing the email itself.
  df['expanded_text'] = df['expanded_text'].str.replace(r'(?<![^\s/])([a-zA-Z]+)\.(\d+)(?![^\s]*@)', r'\1 . \2', regex=True) # YES
  df['expanded_text'] = df['expanded_text'].str.replace(r'(?<![^\s/])([\d,]+)\.([a-zA-Z]+)(?![^\s]*@)', r'\1 . \2', regex=True) # YES

  # Words that connect with digits containing comma
  df['expanded_text'] = df['expanded_text'].str.replace(r'(?<![^\s/])([a-zA-Z,]+)(\d+(?:,\d+)*)(?![^\s]*@|\.\w)', r'\1 \2', regex=True) # YES
  df['expanded_text'] = df['expanded_text'].str.replace(r'(?<!^\S)(\d+(?:,\d+)*)([\.,])([a-zA-Z]+)(?![^\s]*@|\.\w)', r'\1 \2 \3', regex=True) # YES

  # Resolve the http spaces
  # http : / / www.etc.com
  # http : / www .
  df['expanded_text'] = df['expanded_text'].str.replace(r'((?:https?://)?[^\s]+?)(_+)(?!\w)', r'\1 \2', regex=True)
  df['expanded_text'] = df['expanded_text'].str.replace(r'(\S+)(https?)', r'\1 \2', regex=True)
  df['expanded_text'] = df['expanded_text'].str.replace(r'(https?):///\s+', r'\1://', regex=True)
  df['expanded_text'] = df['expanded_text'].str.replace(r'(https?)\s*:\s*/\s*/\s*(\S+)', r'\1://\2', regex=True)
  df['expanded_text'] = df['expanded_text'].str.replace(r'(https?)\s*/\s*/\s*:\s*(\S+)', r'\1://\2', regex=True)
  df['expanded_text'] = df['expanded_text'].str.replace(r'(https?)\s*/\s*/\s*(\S+)', r'\1://\2', regex=True)
  df['expanded_text'] = df['expanded_text'].str.replace(r'(https?)://www\s+\.?', r'\1://www.', regex=True)
  df['expanded_text'] = df['expanded_text'].str.replace(r'(https?)://www\.\s+', r'\1://www.', regex=True)

  # URLs without the https: in front
  df['expanded_text'] = df['expanded_text'].str.replace(r'(?<!\S)(www)\s*\.\s*', r'\1.', regex=True)
  df['expanded_text'] = df['expanded_text'].str.replace(r"(?<=\S)(https?)", r' \1', regex=True)
  
  df['expanded_text'] = df['expanded_text'].str.replace(r'(?<!\S)([a-zA-Z]+)\.([a-zA-Z]+)(?![^\s]*@|\.\w)', r'\1 . \2', regex=True) # YES

  # marketingtoyou@126 . com
  df['expanded_text'] = df['expanded_text'].str.replace(r'(?<!\S)([a-zA-Z]+)@(\d+)(?![^\s]*@|\.\w)', r'\1 . \2', regex=True) # YES
  df['expanded_text'] = df['expanded_text'].str.replace(r'(\S+@\S+)\s+\.\s+(\S+)', r'\1.\2', regex=True) # YES

  # Solving the multiple forward slash with spaces for URLs
  df['expanded_text'] = df['expanded_text'].str.replace(
    r'(https?://www\.[a-zA-Z0-9]+)((?:\s*\.\s*[a-zA-Z0-9]+)+)(\.[a-zA-Z0-9\.]*)?',
    lambda m: f"{m.group(1)}{m.group(2).replace(' ', '')}{m.group(3) or ''}", 
  regex=True)

  df['expanded_text'] = df['expanded_text'].str.replace(
    r'(https?://www\.[a-zA-Z0-9]+)((?:\s*[a-zA-Z0-9]+)+)(\.[a-zA-Z0-9\.]*)?',
    lambda m: f"{m.group(1)}{m.group(2).replace(' ', '')}{m.group(3) or ''}",
  regex=True)

  df['expanded_text'] = df['expanded_text'].str.replace(
    r'([a-zA-Z0-9]+(?:\s*/\s*[a-zA-Z0-9]+)+)',
    lambda m: '/'.join(p.strip() for p in m.group(1).split('/')),
  regex=True)

  df['expanded_text'] = df['expanded_text'].str.replace(
    r'([a-zA-Z]+(?:\s*\.\s*(?!https?\b)[a-zA-Z]+)+)',
    lambda m: '.'.join(p.strip() for p in m.group(1).split('.')),
  regex=True)  

  df['expanded_text'] = df['expanded_text'].str.replace(
    r'(https?://www\.[a-zA-Z0-9]+)\s*\.\s*([a-zA-Z0-9]+)(/[a-zA-Z0-9/]*)?', r'\1.\2\3', regex=True)

  df['expanded_text'] = df['expanded_text'].str.replace(
    r'\s+([a-zA-Z]+)\.([a-zA-Z]+)\s+(?![^\s]*@|\.\w)', r' \1 . \2 ', regex=True)

  # Quotation issue on URL
  df['expanded_text'] = df['expanded_text'].str.replace(r'^"([^"]*)"$', r'\1', regex=True)
  df['expanded_text'] = df['expanded_text'].str.replace(r'"([^"]+)"', r'\1', regex=True)

  # Quotation issue on start and end of words
  df['expanded_text'] = df['expanded_text'].str.replace(r"(?<!\w)'([^']+)'(?!\w)", r'\1', regex=True)
  
  # Special Characters
  df['expanded_text'] = df['expanded_text'].str.replace(r'¤\s*', '', regex=True)

  # Clean up multiple spaces
  df['expanded_text'] = df['expanded_text'].str.replace(r'\s{2,}', ' ', regex=True)

  return df

# Function to expand contractions
def expand_contractions(df):
    def replace_contraction(text):
        return contractions.fix(text)

    # Apply the library's fix method to the column
    df['expanded_text'] = df['expanded_text'].apply(replace_contraction)

    return df
    
def missed_entities(nlp):

  ruler = nlp.add_pipe("entity_ruler", before="ner", config={"overwrite_ents": True})

  patterns = []

  # ---------------- DATE (Regex-First) ----------------
  patterns += [
      # Full dates as a single token regex (High precision)
      {"label": "DATE_EXTRA", "pattern": [{"TEXT": {"REGEX": r"^\d{1,2}[/-:]\d{1,2}[/-:]\d{2,4}$"}}]},
      # may 31 2001 or may 31, 2001 or may , 2001
      {"label": "DATE_EXTRA", "pattern": [{"LOWER": {"IN": MONTHS}}, {"TEXT": ",", "OP": "?"}, {"IS_DIGIT": True}]},
      {"label": "DATE_EXTRA", "pattern": [{"LOWER": {"IN": MONTHS}}, {"IS_DIGIT": True}, {"IS_DIGIT": True}]},
      {"label": "DATE_EXTRA", "pattern": [{"LOWER": {"IN": MONTHS}}, {"IS_DIGIT": True}, {"TEXT": ",", "OP": "?"}, {"IS_DIGIT": True}]},
      # Natural language dates
      {"label": "DATE_EXTRA", "pattern": [{"IS_DIGIT": True}, {"LOWER": {"IN": MONTHS}}, {"IS_DIGIT": True}]},
      {"label": "DATE_EXTRA", "pattern": [{"LOWER": {"IN": MONTHS}}, {"IS_DIGIT": True}]},
      {"label": "DATE_EXTRA", "pattern": [{"LOWER": {"IN": WEEKDAYS}}, {"LOWER": {"IN": MONTHS}}, {"IS_DIGIT": True}]},
      # monday , november 26, 2001
      {"label": "DATE_EXTRA", "pattern": [
        {"LOWER": {"IN": WEEKDAYS}}, 
        {"TEXT": ",", "OP": "?"}, 
        {"LOWER": {"IN": MONTHS}}, 
        {"IS_DIGIT": True}, 
        {"TEXT": ",", "OP": "?"},
        {"IS_DIGIT": True}]},

      # Thu, 22 Aug 2002
      {"label": "DATE_EXTRA", "pattern": [
        {"LOWER": {"IN": WEEKDAYS}}, 
        {"TEXT": ",", "OP": "?"}, 
        {"IS_DIGIT": True}, 
        {"LOWER": {"IN": MONTHS}}, 
        {"IS_DIGIT": True}]},
  
      # april 21-23, 2001
      {"label": "DATE_EXTRA", "pattern": [ 
        {"LOWER": {"IN": MONTHS}}, 
        {"IS_DIGIT": True},
        {"TEXT": "-", "OP": "?"},
        {"IS_DIGIT": True},
        {"TEXT": ",", "OP": "?"},
        {"IS_DIGIT": True}]},

      # april 7
      {"label": "DATE_EXTRA", "pattern": [ 
        {"LOWER": {"IN": MONTHS}}, 
        {"IS_DIGIT": True},
        {"TEXT": "-", "OP": "?"},
        {"IS_DIGIT": True},
        {"TEXT": ",", "OP": "?"},
        {"IS_DIGIT": True}]}
  ]

  # ---------------- TIME (Standardized) ----------------
  patterns += [
      # 08:30 or 08:30:15
      {"label": "TIME_EXTRA", "pattern": [{"TEXT": {"REGEX": r"^\d{1,2}:\d{1,2}(:\d{1,2})?$"}}]},
      # 08:30 am
      {"label": "TIME_EXTRA", "pattern": [
          {"TEXT": {"REGEX": r"^\d{1,2}:\d{2}$"}}, 
          {"LOWER": {"REGEX": r"^(a\.?m\.?|p\.?m\.?)$"}}
      ]},

      # 6 pm/am
      {"label": "TIME_EXTRA", "pattern": [
          {"TEXT": {"REGEX": r"^\d{1,2}$"}}, 
          {"LOWER": {"REGEX": r"^(a\.?m\.?|p\.?m\.?)$"}}
      ]}
  ]

  # ---------------- MONEY (Symbol + Value) ----------------
  patterns += [
      # $100 or €50.50
      {"label": "MONEY_EXTRA", "pattern": [{"TEXT": {"REGEX": r"^[€£$¥]\d+(?:[.,]\d+)?$"}}]},
      {"label": "MONEY_EXTRA", "pattern": [{"TEXT": {"REGEX": r"\$\s?\d+([\.,]\d+)?"}}]},
      # 100 dollars
      {"label": "MONEY_EXTRA", "pattern": [
          {"LIKE_NUM": True}, 
          {"LOWER": {"IN": ["dollars", "euro", "euros", "pounds", "yen"]}}
      ]},
      {"label": "MONEY_EXTRA", 
      "pattern": [
        {"IS_CURRENCY": True},
        {"LIKE_NUM": True}]},
  ]

  # ---------------- PHONE (Variations) ----------------
  # WARNING THE PHONE FIX!
  patterns += [
    # 713 - 521 - 4535
    {"label": "PHONE_EXTRA", 
    "pattern": [
      {"IS_DIGIT": True}, 
      {"TEXT": "-", "OP": "?"},
      {"IS_DIGIT": True}, 
      {"TEXT": "-", "OP": "?"},
      {"IS_DIGIT": True}]},
    # 42/521/4535213
    {"label": "PHONE_EXTRA", 
    "pattern": [
      {"IS_DIGIT": True}, 
      {"TEXT": "/", "OP": "?"},
      {"IS_DIGIT": True, "LENGTH": 3}, 
      {"TEXT": "/", "OP": "?"},
      {"IS_DIGIT": True, "LENGTH": 3}]},
    # Standard 123-456-7890 or (123) 456-7890 or ( 65 ) 772 - 6571
    {"label": "PHONE_EXTRA", 
    "pattern": [
      {"TEXT": "(", "OP": "?"}, 
      {"IS_DIGIT": True}, 
      {"TEXT": ")", "OP": "?"}, 
      {"IS_DIGIT": True},
      {"TEXT": {"REGEX": "-"}}, 
      {"IS_DIGIT": True}]},
    # (123) 456 7890
    {"label": "PHONE_EXTRA", 
    "pattern": [
      {"TEXT": "(", "OP": "?"}, 
      {"IS_DIGIT": True}, 
      {"TEXT": ")", "OP": "?"}, 
      {"IS_DIGIT": True}, 
      {"IS_DIGIT": True}]},
    # 47 - 77645616
    {"label": "PHONE_EXTRA", 
    "pattern": [
      {"IS_DIGIT": True},
      {"TEXT": {"REGEX": "-"}}, 
      {"IS_DIGIT": True, "LENGTH": 8}]},
    # 1 - 317 - 726 - 1482
    {"label": "PHONE_EXTRA", 
    "pattern": [
      {"IS_DIGIT": True},
      {"TEXT": {"REGEX": "-"}}, 
      {"IS_DIGIT": True},
      {"TEXT": {"REGEX": "-"}}, 
      {"IS_DIGIT": True},
      {"TEXT": {"REGEX": "-"}}, 
      {"IS_DIGIT": True},
      ]},
    
    # 10 digit strings
    {"label": "PHONE_EXTRA", "pattern": [{"TEXT": {"REGEX": r"^\d{10}$"}}]}
  ]

  # ---------------- EMAIL / WEB (Built-in Flags) ----------------
  patterns += [
    # {"label": "EMAIL_EXTRA", "pattern": [{"TEXT": {"REGEX": r'[\w\.-]+@[\w\.-]+\.\w+'}}]},
    {"label": "EMAIL", "pattern": [{"LIKE_EMAIL": True}]},
  ]

  patterns += [
    {"label": "WEB", "pattern": [{"LIKE_URL": True}]}
  ]

  # ---------------- PERCENTNER ----------------
  patterns += [
    {"label": "PERCENT_EXTRA", "pattern": [{"TEXT": {"REGEX": r"^\d+(\.\d+)?\s*%$"}}]}
  ]

  # ---------------- NUMNER ----------------
  patterns += [
    {"label": "CARDINAL_EXTRA", "pattern": [{"TEXT": {"REGEX": r"^(?:\d+|(?=.*\d)[A-Za-z\d]+)$"}}]},
    {"label": "CARDINAL_EXTRA", "pattern": [{"TEXT": {"REGEX": r"^(?:\d+|(?=.*\d))$"}}]},
    {"label": "CARDINAL", "pattern": [{"LIKE_NUM": True}]}
  ]

  ruler.add_patterns(patterns)

  return nlp

LABEL_REPLACEMENTS = {
    "PERSON":"PERNER",
    "ORG":"ORGNER",
    "GPE":"GPENER",
    "LOC":"LOCNER",
    "PRODUCT":"PRODUCTNER",
    "DATE":"DATENER", "DATE_EXTRA":"DATENER",
    "TIME":"TIMENER", "TIME_EXTRA":"TIMENER",
    "MONEY":"MONEYNER", "MONEY_EXTRA":"MONEYNER",
    "PHONE_EXTRA":"PHONENER",
    "PERCENT":"PERCENTNER", "PERCENT_EXTRA": "PERCENT",
    "EVENT":"EVENTNER",
    "NORP":"NORPNER",
    "LANGUAGE":"LANGNER",
    "CARDINAL":"NUMNER", "ORDINAL":"NUMNER", "QUANTITY":"NUMNER", "CARDINAL_EXTRA":"NUMNER",
    "EMAIL":"EMAILNER",
    "PHONE":"PHONENER",
    "WEB":"WEBNER",
}

# 16 labels in total
LABEL_PRIORITY = {
    "EMAILNER": 10,      # Most specific — never let email be overwritten
    "PHONENER": 9,       # Phone patterns are distinct
    "WEBNER": 8,         # URLs are specific
    "DATENER": 7,        # Dates (multi-token, structured)
    "TIMENER": 6,        # Times (can overlap with dates, date wins)
    "MONEYNER": 5,       # $100 more specific than 100
    "PERCENTNER": 4,     # 50% more specific than 50
    "PERNER": 3,         # Person names (SpaCy NER)
    "ORGNER": 3,         # Organizations
    "GPENER": 3,         # Geopolitical entities
    "LOCNER": 3,         # Locations
    "PRODUCTNER": 2,     # Products
    "EVENTNER": 2,       # Events
    "NORPNER": 2,        # Nationalities/religions/political groups
    "LANGNER": 2,        # Languages
    "NUMNER": 1,         # Least specific — pure digits fallback
}

def resolve_overlaps(spans_with_labels, label_priority):
    spans_with_labels.sort(key=lambda x: (x[0][0], -(x[0][1] - x[0][0])))

    kept = []

    for (s, e), lab in spans_with_labels:
        new_p = label_priority.get(lab, 0)
        new_len = e - s

        overlaps = []
        should_add = True

        # First pass: check all overlaps
        for (ks, ke), klab in kept:
            if not (e <= ks or s >= ke):  # overlap
                old_p = label_priority.get(klab, 0)
                old_len = ke - ks

                if (new_p > old_p) or (new_p == old_p and new_len > old_len):
                    overlaps.append(((ks, ke), klab))  # candidate to remove
                else:
                    should_add = False
                    break  # new loses → stop early

        if should_add:
            # Remove all overlapped spans that lost
            kept = [k for k in kept if k not in overlaps]
            kept.append(((s, e), lab))

    kept.sort(key=lambda x: x[0][0])
    return kept

def token_replacement(texts, nlp, label_replacements, label_priority, batch_size=32):
  processed_texts = []

  for doc in tqdm(nlp.pipe(texts, batch_size=batch_size), total=len(texts)):

    spans = []

    # ---- ENTITY EXTRACTION ----
    for ent in doc.ents:
      if ent.label_ in label_replacements:
        spans.append(((ent.start, ent.end), label_replacements[ent.label_]))

    for tok in doc:
      if tok.like_email:
        spans.append(((tok.i, tok.i + 1), "EMAILNER"))
        continue

      if tok.like_url:
        spans.append(((tok.i, tok.i + 1), "WEBNER"))
        continue

      if tok.ent_type_ != "":
          continue

      # ---- ALL NUMBERS ----
      text = tok.text
      if text.isdigit():
          spans.append(((tok.i, tok.i + 1), "NUMNER"))

    # ---- RESOLVE ----
    spans = resolve_overlaps(spans, label_priority)

    # ---- BUILD OUTPUT ----
    new_tokens = []
    i = 0
    si = 0

    while i < len(doc):
      if si < len(spans) and i == spans[si][0][0]:
        (_, end), lab = spans[si]
        new_tokens.append(lab)
        i = end
        si += 1
      else:
        new_tokens.append(doc[i].text)
        i += 1

    processed_texts.append(" ".join(new_tokens))

  return processed_texts


# - Start -
def load_and_preprocess(file_path, output_path, matcher_enabled, chunk_size=1_000):

  # Incasae if the file doesn't exist
  if not isinstance(output_path, str) or output_path is None:
      print("You didn't specify an output file path")
      return

  # Define the final export path
  variant = "NER" if matcher_enabled else "Cleaned"

  final_export_path = f"drive/My Drive/Dissertation/Datasets/{variant}/{output_path}/{output_path}_{variant}.csv"

  # Initialize spaCy if matcher is enabled
  nlp = spacy.load("en_core_web_lg") if matcher_enabled else None
  nlp = missed_entities(nlp) if matcher_enabled else None

  # Check if we are dealing with a CSV file for chunking
  if isinstance(file_path, str) and file_path.endswith('.csv'):
    
    with open(file_path, 'r', encoding='utf-8') as f:
      total_rows = sum(1 for _ in f) - 1  # subtract header that is text and label

    total_chunks = (total_rows // chunk_size) + 1
    
    reader = pd.read_csv(file_path, chunksize=chunk_size)

    # Creating a progress bar
    progress_bar = tqdm(reader, total=total_chunks, 
    desc=f"Processing {output_path}", unit="chunk", dynamic_ncols=True)
    
    first_chunk = True
    for chunk in progress_bar: 
      # Process the individual chunk

      tqdm.write("\n\n--- Processing New Chunk ---")
      processed_chunk = process_data_frame(chunk, output_path, matcher_enabled, nlp)
      
      # Write to CSV: 'w' for the first chunk, 'a' for subsequent
      mode = 'w' if first_chunk else 'a'
      header = True if first_chunk else False
      processed_chunk.to_csv(final_export_path, index=False, mode=mode, header=header)
      
      progress_bar.set_postfix({"rows_saved": len(chunk)})
      first_chunk = False
          
  else:
    if isinstance(file_path, str):
        df = pd.DataFrame({'text': [file_path], 'label': [0]})
    elif isinstance(file_path, list):
        df = pd.DataFrame({'text': file_path, 'label': [0] * len(file_path)})
    else:
        df = file_path.copy(deep=True)
        
    df_final = process_data_frame(df, output_path, matcher_enabled, nlp)
    df_final.to_csv(final_export_path, index=False)
    return df_final

def process_data_frame(df, dataset_type, matcher_enabled, nlp):
  df['expanded_text'] = df['text']
  
  # 1. Regex Cleaning
  df = cleaning_process(df, dataset_type=None)
  
  # 2. Expanding Contractions
  df = expand_contractions(df)
  
  # 3. NER / Token Replacement (EntityRuler-based)
  if matcher_enabled:
      texts = df["expanded_text"].fillna("").astype(str).tolist()
      
      processed_texts = token_replacement(
        texts,
        nlp,
        LABEL_REPLACEMENTS,
        LABEL_PRIORITY
      )
      
      df["processed_text"] = processed_texts #df['expanded_text'].apply(abstract_text)
  else:
      df["processed_text"] = df['expanded_text']
      
  return df