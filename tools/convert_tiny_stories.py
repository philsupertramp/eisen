import json
from tqdm import tqdm


with open('./tinystories_de_clean.jsonl', 'r') as f:
    stories = [json.loads(l.strip()) for l in f.readlines()]

count = 0

with open('german_tiny_story_corpus.txt', 'w') as f:
    for story in tqdm(stories, desc="Writing documents", total=len(stories)):
        text = story.get('story')
        f.write(f'<story>\n{text}\n</story>\n\n')
        count += 1

print(f"\nSuccess! Wrote {count} cleaned documents to ./german_tiny_story_corpus.txt")


