import os
from striprtf.striprtf import rtf_to_text
from transformers import MarianMTModel, MarianTokenizer
import sacrebleu

try:
    with open('input.rtf', 'r', encoding='utf-8') as f:
        input_rtf = f.read()
except UnicodeDecodeError:
    with open('input.rtf', 'r', encoding='mac_roman') as f:
        input_rtf = f.read()

try:
    with open('output.rtf', 'r', encoding='utf-8') as f:
        output_rtf = f.read()
except UnicodeDecodeError:
    with open('output.rtf', 'r', encoding='mac_roman') as f:
        output_rtf = f.read()

input_text = rtf_to_text(input_rtf)
output_text = rtf_to_text(output_rtf)

input_sentences = [line.strip() for line in input_text.split('\n') if line.strip() and not line.startswith('#')]
reference_sentences = [line.strip() for line in output_text.split('\n') if line.strip() and not line.startswith('#')]

print(f"Loaded {len(input_sentences)} input sentences and {len(reference_sentences)} reference sentences.")

# Load model
model_name = "Helsinki-NLP/opus-mt-bn-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Translate
print("Translating...")
translated_sentences = []
batch_size = 16
for i in range(0, len(input_sentences), batch_size):
    batch = input_sentences[i:i+batch_size]
    tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
    generated = model.generate(**tokens)
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    translated_sentences.extend(decoded)

with open('output2.txt', 'w', encoding='utf-8') as f:
    for sent in translated_sentences:
        f.write(sent + '\n')

print("=" * 60)
print(f"First Bengali Statement : {input_sentences[0]}")
print(f"Generated Output        : {translated_sentences[0]}")
if reference_sentences:
    print(f"Reference Output        : {reference_sentences[0]}")
print("=" * 60)

# BLEU score
if reference_sentences:
    min_len = min(len(translated_sentences), len(reference_sentences))
    hypotheses = translated_sentences[:min_len]
    references = [reference_sentences[:min_len]]

    bleu = sacrebleu.corpus_bleu(hypotheses, references)
    print(f"BLEU score                 : {bleu.score:.2f}")
    print(f"Full SacreBLEU output      : {bleu}")
else:
    print("Warning: Could not parse reference sentences cleanly for BLEU evaluation.")
