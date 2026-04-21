import json
import asyncio
import random
import re
import os
from collections import Counter
from openai import AsyncOpenAI
import spacy
from langdetect import detect, DetectorFactory

# ==========================================
# 1. Initialization & Setup
# ==========================================

# Ensure consistent language detection
DetectorFactory.seed = 0 

# Load the small German NLP model for lemmatization
print("Loading spaCy NLP model (de_core_news_sm)...")
try:
    nlp = spacy.load("de_core_news_sm")
except OSError:
    print("Model not found. Please install it using: python -m spacy download de_core_news_sm")
    exit(1)

# Point the OpenAI client to your local Dockerized llama-server
client = AsyncOpenAI(
    base_url="https://tellme-gemma.dev.easybits.tech/v1",
    api_key="sk-no-key-needed" # Local server ignores this
)

OUTPUT_FILE = "tinystories_de_clean.jsonl"

# ==========================================
# 2. Vocabulary & Feature Definitions
# ==========================================

NOUNS = [
    # Animals
    "Katze", "Hund", "Vogel", "Bär", "Maus", "Frosch", "Löwe", "Tiger", "Elefant", "Affe", 
    "Hase", "Kuh", "Schwein", "Pferd", "Schaf", "Ziege", "Huhn", "Ente", "Eule", "Fuchs", 
    "Wolf", "Igel", "Eichhörnchen", "Schmetterling", "Biene", "Fisch", "Delfin", "Schildkröte",
    # Nature
    "Baum", "Blume", "Sonne", "Mond", "Stern", "Wolke", "Regen", "Schnee", "Wind", "Fluss", 
    "Berg", "Meer", "Strand", "Wald", "Wiese", "Höhle", "Insel", "Stein", "Sand",
    # Objects & Places
    "Auto", "Haus", "Buch", "Ball", "Schiff", "Zug", "Schuh", "Bett", "Stuhl", "Tisch", 
    "Tür", "Fenster", "Spiegel", "Uhr", "Bild", "Topf", "Teller", "Tasse", "Schrank", "Schlüssel",
    "Schloss", "Brücke", "Straße", "Stadt", "Dorf", "Bauernhof", "Garten", "Schule",
    # Fantasy & People
    "Mädchen", "Junge", "König", "Prinzessin", "Ritter", "Drache", "Geist", "Hexe", "Zauberer", 
    "Zwerg", "Riese", "Fee", "Einhorn", "Pirat", "Schatz", "Freund", "Oma", "Opa"
]

VERBS = [
    # Movement
    "rennen", "springen", "fliegen", "schwimmen", "tanzen", "klettern", "laufen", "gehen", 
    "kriechen", "hüpfen", "fallen", "stolpern", "rutschen", "reiten", "fahren", "tauchen",
    # Actions
    "spielen", "schlafen", "essen", "suchen", "finden", "verstecken", "helfen", "lesen", 
    "schreiben", "malen", "zeichnen", "bauen", "werfen", "fangen", "tragen", "ziehen", 
    "schieben", "öffnen", "schließen", "drehen", "verlieren", "gewinnen", "waschen", "kochen",
    # Emotions & Senses
    "weinen", "lachen", "singen", "hören", "sehen", "riechen", "schmecken", "fühlen", 
    "denken", "träumen", "hoffen", "wünschen", "lächeln", "erschrecken", "freuen",
    # Interactions
    "rufen", "flüstern", "schreien", "streicheln", "umarmen", "küssen", "teilen", "schenken", 
    "danken", "fragen", "antworten", "zaubern"
]

ADJECTIVES = [
    # Size & Shape
    "groß", "klein", "dick", "dünn", "breit", "schmal", "hoch", "tief", "lang", "kurz", 
    "rund", "eckig", "spitz", "flach",
    # Colors
    "rot", "blau", "gelb", "grün", "schwarz", "weiß", "bunt", "farblos", "hell", "dunkel",
    # Emotions & Traits
    "glücklich", "traurig", "böse", "lieb", "müde", "wach", "mutig", "feige", "klug", "dumm", 
    "lustig", "ängstlich", "überrascht", "freundlich", "neugierig", "schüchtern", "stolz",
    # Physical states
    "schnell", "langsam", "kalt", "warm", "heiß", "kühl", "nass", "trocken", "schwer", "leicht", 
    "hart", "weich", "laut", "leise", "sauber", "schmutzig", "voll", "leer", "kaputt", "neu", "alt",
    # Tastes & others
    "süß", "sauer", "schön", "hässlich", "lecker", "stark", "schwach", "magisch", "geheimnisvoll"
]

FEATURES = [
    "einen Dialog", 
    "ein glückliches Ende", 
    "ein schlechtes Ende", 
    "eine Moral", 
    "einen unerwarteten Twist",
    "eine unerwartete Wende",
    "kurze Sätze",
    "eine magische Entdeckung",
    "ein sprechendes Tier",
    "ein kleines Geheimnis",
    "ein lustiges Missverständnis",
    "ein mutiges Tier",
    "eine Lektion über Freundschaft",
    "ein verlorener Gegenstand, der wiedergefunden wird",
    "eine Reise an einen neuen Ort",
    "eine schwierige Aufgabe, die gelöst wird",
    "ein Traum, der wahr wird",
    "drei wiederkehrende Ereignisse",
    "eine Frage, die am Ende beantwortet wird"
]

# ==========================================
# 3. Failsafe & State Recovery
# ==========================================

def get_existing_story_count(filepath):
    """
    Reads the existing dataset file to determine how many valid stories 
    have already been generated. This allows safe resumption.
    """
    if not os.path.exists(filepath):
        return 0

    valid_count = 0
    needs_newline_fix = False

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
                valid_count += 1
            except json.JSONDecodeError:
                print(f"Warning: Found corrupted JSON on line {idx + 1}. Ignoring it for count.")
        
        # Check if the last character of the file is a newline. 
        # If not, the next append will corrupt the JSON.
        if lines and not lines[-1].endswith('\n'):
            needs_newline_fix = True

    if needs_newline_fix:
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write('\n')

    return valid_count

# ==========================================
# 4. Prompt Generation
# ==========================================

def generate_prompt():
    """Samples vocabulary and constructs the system/user messages."""
    n = random.sample(NOUNS, 2)
    v = random.choice(VERBS)
    a = random.choice(ADJECTIVES)
    f = random.choice(FEATURES)
    
    # Strict system instruction to prevent language drift
    system_instruction = (
        "Du bist ein hilfreicher Assistent, der Geschichten für kleine Kinder (3-4 Jahre alt) schreibt. "
        "Achte zwingend auf korrekte deutsche Grammatik (z.B. 'das Mädchen', nicht 'der Mädchen'). "
        "Verwende einfache Sprache und kurze Sätze."
    )
    
    few_shot_example = """
Beispiel:
Wörter: [Hund, Wald, rennen, glücklich]
Feature: ein glückliches Ende
Geschichte: Es war einmal ein kleiner Hund. Er hieß Bello. Bello war sehr glücklich. Eines Tages ging Bello in den Wald. Er wollte spielen. Plötzlich fing er an zu rennen. Er rannte und rannte durch die grünen Bäume. Am Ende fand er einen großen Knochen. Er nahm den Knochen mit nach Hause. Das war ein glückliches Ende.
"""
    user_prompt = (
        f"{few_shot_example}\n"
        f"Schreibe jetzt eine neue kurze Kindergeschichte. \n"
        f"Wörter: [{n[0]}, {n[1]}, {v}, {a}]\n"
        f"Feature: {f}\n"
        f"Geschichte:"
    )
    
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_prompt}
    ]
    
    metadata = {
        "words": [n[0], n[1], v, a], 
        "feature": f
    }
    
    return messages, metadata

# ==========================================
# 5. Evaluation Heuristics
# ==========================================

def check_language(text):
    """Ensures the model didn't start speaking English."""
    try:
        return detect(text) == 'de'
    except:
        return False

def check_repetition(text, max_ngram_ratio=0.4):
    """Catches models stuck in a generative loop."""
    words = text.split()
    if len(words) < 10:
        return False
    
    # Check for excessive 3-gram repetition
    trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
    if not trigrams:
        return True
        
    most_common_trigram, count = Counter(trigrams).most_common(1)[0]
    
    # If the same 3 words make up too much of the story, reject it
    if count / len(trigrams) > max_ngram_ratio:
        return False
    return True

def check_vocabulary(text, required_words):
    """Uses spaCy to check for lemmas (e.g., Baum -> Bäume is accepted)."""
    doc = nlp(text)
    
    # Extract the base form (lemma) of every word in the generated story, lowercased
    story_lemmas = {token.lemma_.lower() for token in doc}
    
    # Check if all required prompt words (also lemmatized) are in the story
    for word in required_words:
        target_lemma = nlp(word)[0].lemma_.lower()
        if target_lemma not in story_lemmas:
            return False
    return True

def check_feature(text, feature):
    """Basic heuristic checks for specific structural features."""
    if feature == "einen Dialog":
        # Look for typical German or English quotation marks
        if not re.search(r'[„"»\'].+[”"«\']', text):
            return False
    return True

# ==========================================
# 6. Main Generation Loop
# ==========================================

async def generate_single_story_async(max_retries=3):
    for attempt in range(max_retries):
        messages, metadata = generate_prompt()
        words = metadata["words"]
        feature = metadata["feature"]
        
        try:
            # Non-blocking call to the server
            response = await client.chat.completions.create(
                model="qwen3-0.6b-german", 
                messages=messages,
                max_tokens=512,
                temperature=0.8,
                top_p=0.9,
            )
            story_text = response.choices[0].message.content.strip()
            story_text = story_text.split('Geschichte:')[-1].strip()
            
            # Evaluate
            if (30 <= len(story_text.split()) <= 250 and
                check_language(story_text) and
                check_repetition(story_text) and
                check_vocabulary(story_text, words) and
                check_feature(story_text, feature)):
                
                return {"story": story_text, "metadata": metadata}
                
        except Exception as e:
            # For AI researchers: Handle potential local server connection resets
            await asyncio.sleep(1 + attempt) # Exponential backoff
            
    return None # Return None if it failed all retries

# ==========================================
# 7. Batch Orchestrator
# ==========================================

async def generate_dataset_batched(num_target_stories=1000, batch_size=1):
    print("Checking existing dataset for recovery...")
    successful_generations = get_existing_story_count(OUTPUT_FILE)
    
    if successful_generations >= num_target_stories:
        print(f"Target of {num_target_stories} already reached! (Found {successful_generations} in {OUTPUT_FILE})")
        return
        
    print(f"Resuming from {successful_generations} stories. Need {num_target_stories - successful_generations} more.\n")
    
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        while successful_generations < num_target_stories:
            
            # Calculate how many stories we still need to hit the target
            current_batch_size = min(batch_size, num_target_stories - successful_generations)
            
            print(f"Dispatching batch of {current_batch_size} concurrent requests...")
            
            # Fire off X requests concurrently
            tasks = [generate_single_story_async(10) for _ in range(current_batch_size)]
            results = await asyncio.gather(*tasks)
            
            # Process the results as they come back
            batch_success = 0
            for result in results:
                if result is not None:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    successful_generations += 1
                    batch_success += 1
            
            f.flush() # Ensure it's written to disk in case of crash
            print(f"Batch complete. Yield: {batch_success}/{current_batch_size} | Progress: [{successful_generations}/{num_target_stories}]")

# ==========================================
# 8. Execution Entry Point
# ==========================================

if __name__ == "__main__":
    try:
        # Since you use a local dockerized llama-server, set batch_size 
        # to match your server's max concurrent connection capacity (-np flag)
        asyncio.run(generate_dataset_batched(num_target_stories=100000, batch_size=1))
    except KeyboardInterrupt:
        print("\n[!] Gracefully exiting. Generation paused.")
        print(f"Run the script again to resume where you left off.")
