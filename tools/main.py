import os
import re
import tempfile
import subprocess
import json
import hashlib
from datasets import load_dataset
from tqdm import tqdm
from rich.console import Console
from rich.markdown import Markdown

# --- NLP DEPENDENCIES ---
try:
    import spacy
    HAS_SPACY = True
    print("Loading spaCy NLP model (de_core_news_sm)...")
    # Disable parser and NER for speed; we only need the tagger for POS (Nouns)
    nlp = spacy.load("de_core_news_sm", disable=["parser", "ner"])
except ImportError:
    HAS_SPACY = False
    print("WARNING: 'spacy' not installed. Run: pip install spacy && python -m spacy download de_core_news_sm")

try:
    import pyphen
    HAS_PYPHEN = True
    # Load the German hyphenation dictionary
    dic = pyphen.Pyphen(lang='de_DE')
except ImportError:
    HAS_PYPHEN = False
    print("WARNING: 'pyphen' not installed. Compound splitting is gracefully disabled.")
    print("Run: pip install pyphen")

def redact_pii(text):
    """
    Basic Regex-based PII anonymizer. 
    Replaces common sensitive patterns with placeholder tokens.
    """
    # 1. Emails
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    text = re.sub(email_pattern, '<EMAIL>', text)
    
    # 2. Phone Numbers (Basic international & German format catching)
    phone_pattern = r'(?:\+?[0-9]{1,3}[ \-\.]?)?(?:\(0\)[ \-\.]?)?[0-9]{3,5}[ \-\.]?[0-9]{4,8}'
    text = re.sub(phone_pattern, lambda m: '<PHONE>' if len(re.sub(r'\D', '', m.group(0))) >= 8 else m.group(0), text)
    
    # 3. IBANs (German / European)
    iban_pattern = r'[A-Z]{2}[0-9]{2}(?:[ ]?[0-9]{4}){4}(?:[ ]?[0-9]{1,2})?'
    text = re.sub(iban_pattern, '<IBAN>', text)
    
    return text

def split_german_compounds(text, min_length=12):
    """
    Uses spaCy to identify long nouns and Pyphen to split them.
    Approximates compound splitting by finding the valid syllable 
    break closest to the middle of the word.
    """
    if not HAS_SPACY:
        return text
        
    doc = nlp(text)
    processed_tokens = []
    
    for token in doc:
        ws = token.whitespace_
        
        if HAS_PYPHEN and token.pos_ in ["NOUN", "PROPN"] and len(token.text) >= min_length:
            # Get all valid grammatical split positions
            positions = dic.positions(token.text)
            
            if positions:
                # Filter out splits that create tiny fragments (e.g. at least 4 chars long)
                valid_positions = [p for p in positions if p >= 4 and (len(token.text) - p) >= 4]
                
                if valid_positions:
                    # Heuristic: Pick the split point closest to the middle of the word
                    middle = len(token.text) / 2
                    best_split = min(valid_positions, key=lambda p: abs(p - middle))
                    
                    head = token.text[:best_split]
                    tail = token.text[best_split:]
                    
                    # Inject the structural <JOIN> token
                    processed_tokens.append(f"{head} <JOIN> {tail}{ws}")
                    continue
                
        # Keep words exactly as they are if they aren't split
        processed_tokens.append(token.text_with_ws)
        
    return "".join(processed_tokens)

def edit_in_terminal(text):
    """
    Opens the text in the user's default terminal editor for manual cleaning.
    """
    editor = os.environ.get('EDITOR', 'nano')
    
    with tempfile.NamedTemporaryFile(mode='w+', suffix=".txt", delete=False, encoding='utf-8') as tf:
        tf.write(text)
        tf.flush()
        temp_path = tf.name
    
    subprocess.call([editor, temp_path])
    
    with open(temp_path, 'r', encoding='utf-8') as tf:
        edited_text = tf.read().strip()
        
    os.remove(temp_path)
    return edited_text

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def generate_doc_id(text, subset):
    """Generates a short, deterministic hash for the document ID."""
    hash_input = f"{subset}-{text[:50]}".encode('utf-8')
    return hashlib.sha256(hash_input).hexdigest()[:12]

def extract_split_track(dataset_name, subset, split, txt_output, meta_output, column="text", max_docs=None, interactive=False):
    """
    Streams a Hugging Face dataset, cleans it, allows interactive editing, 
    and writes it to a split-track format (Corpus text + JSONL registry).
    """
    print(f"Opening stream for {dataset_name} (Subset: {subset})...")
    dataset = load_dataset(dataset_name, subset, split=split, streaming=True)
    
    print(f"Extracting Corpus to {txt_output}")
    print(f"Extracting Registry to {meta_output}")
    
    console = Console() if interactive else None
    count = 0
    
    with open(txt_output, "w", encoding="utf-8") as f_txt, \
         open(meta_output, "w", encoding="utf-8") as f_meta:
         
        pbar = tqdm(desc="Writing documents", total=max_docs) if not interactive else None
        
        for entry in dataset:
            text = entry.get(column, "").strip()
            title = entry.get("title", "No Title")
            url = entry.get("url", "No URL")
            
            if not text:
                continue
                
            # --- 1. AUTOMATED CLEANING & NLP PROCESSING ---
            text = redact_pii(text)
            
            # Apply our NLP compound splitting
            text = split_german_compounds(text, min_length=12)
            
            # --- 2. INTERACTIVE MANUAL CLEANING ---
            if interactive:
                clear_screen()
                print(f"=== Document {count + 1} | Title: {title} ===")
                
                preview_len = 800
                preview_text = text[:preview_len]
                if len(text) > preview_len:
                    preview_text += "...\n\n**[TEXT TRUNCATED FOR PREVIEW]**"
                
                console.print(Markdown(preview_text))
                print("=" * 30)
                
                print("\nOptions:")
                print("  [A]ccept   - Save document as is")
                print("  [E]dit     - Open in terminal editor (nano/vim)")
                print("  [R]eject   - Skip this document entirely")
                print("  [S]top UI  - Auto-accept the rest without asking")
                print("  [Q]uit     - Stop extraction entirely")
                
                choice = input("\nYour choice [a/e/r/s/q]: ").strip().lower()
                
                if choice == 'q':
                    print("Aborting extraction early.")
                    break
                elif choice == 'r':
                    continue 
                elif choice == 'e':
                    text = edit_in_terminal(text)
                    if not text:
                        continue
                elif choice == 's':
                    interactive = False
                    print("Interactive mode disabled. Auto-processing the rest...")
                    pbar = tqdm(desc="Writing documents", total=max_docs, initial=count)
            
            # --- 3. THE SPLIT TRACK WRITER ---
            doc_id = entry.get("id", generate_doc_id(text, subset))
            
            f_txt.write(f"### DOC_ID: {doc_id} ###\n")
            f_txt.write(text + "\n\n")
            
            metadata_record = {
                "doc_id": doc_id,
                "source_dataset": dataset_name,
                "subset": subset,
                "title": title,
                "url": url
            }
            f_meta.write(json.dumps(metadata_record, ensure_ascii=False) + "\n")
            
            count += 1
            if pbar: pbar.update(1)
            if max_docs and count >= max_docs: break
                
        if pbar: pbar.close()
                
    print(f"\nSuccess! Wrote {count} documents.")
    print(f" -> Corpus   : {txt_output}")
    print(f" -> Registry : {meta_output}")

if __name__ == "__main__":
    dataset_name = "wikimedia/wikipedia"
    subset = "20231101.de" 
    split = "train"
    
    txt_output = "german_large_corpus.txt"
    meta_output = "german_corpus_registry.jsonl"
    
    # Set interactive=True to enable manual data inspection/editing!
    # You will see the <JOIN> tokens injected live in the terminal output.
    extract_split_track(
        dataset_name=dataset_name, 
        subset=subset, 
        split=split, 
        txt_output=txt_output,
        meta_output=meta_output,
        max_docs=1_000_000,
        interactive=False
    )
