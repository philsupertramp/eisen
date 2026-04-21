from transformers import AutoTokenizer
import re

# 1. Load your custom tokenizer 
# (Replace with the path to your local tokenizer files)
# tokenizer = AutoTokenizer.from_pretrained("./my_4096_tokenizer")

# For the sake of this script, let's pretend we loaded yours. 
# If you haven't wrapped it in HuggingFace yet, you can also use:
# from tokenizers import Tokenizer; tokenizer = Tokenizer.from_file("tokenizer.json")

def analyze_fertility(tokenizer, text):
    # Clean punctuation to get actual words
    clean_text = re.sub(r'[^\w\säöüßÄÖÜ]', '', text)
    words = clean_text.split()
    
    total_words = len(words)
    total_tokens = 0
    
    shred_stats = []

    for word in words:
        # Tokenize the individual word
        # Note: If using tokenizers.Tokenizer directly, use tokenizer.encode(word).tokens
        tokens = tokenizer.tokenize(word) 
        token_count = len(tokens)
        total_tokens += token_count
        
        shred_stats.append({
            "word": word,
            "token_count": token_count,
            "tokens": tokens
        })
        
    # Calculate overall fertility
    fertility_rate = total_tokens / total_words
    
    print("=== Tokenizer Diagnostics ===")
    print(f"Total Words:  {total_words}")
    print(f"Total Tokens: {total_tokens}")
    print(f"Fertility Rate: {fertility_rate:.2f} tokens/word\n")
    
    # Sort to find the most shredded words
    shred_stats.sort(key=lambda x: x["token_count"], reverse=True)
    
    print("=== Top 10 Most Shredded Words ===")
    for stat in shred_stats[:10]:
        if stat["token_count"] > 1:
            print(f"[{stat['token_count']} tokens] {stat['word'].ljust(15)} -> {stat['tokens']}")

if __name__ == "__main__":
    # A sample German text combining simple TinyStories vocab and some compounds
    sample_text = """
    Es war einmal ein kleines Mädchen namens Lily. Sie ging durch den großen Tannenwald. 
    Plötzlich sah sie ein schnelles Feuerwehrauto und einen bunten Schmetterling. 
    Die Unheilfulness der Situation war beängstigend, aber sie fand einen Apfelbaum 
    und aß glücklich einen Schokoladenkeks.
    """
    
    # Run the diagnostic
    analyze_fertility(tokenizer, sample_text)
