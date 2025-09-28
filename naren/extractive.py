from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
import torch
import nltk
# from inltk.inltk import tokenize
from indicnlp.tokenize import sentence_tokenize


# Download nltk's sentence tokenizer for Hindi (if needed)
# nltk.download('punkt', download_dir=r"c:\users\naren\nltk_data", )
# nltk.download('punkt_tab', download_dir=r"c:\users\naren\nltk_data", )

# Load models
bert_model_name = "bert-base-multilingual-cased"  # replace with suitable Hindi BERT fine-tuned for importance scoring
summarization_model_name = "l3cube-pune/hindi-bart-summary"

# Load tokenizer and models for scoring
# For simplicity, using BERT for sentence importance scoring (you might need a fine-tuned model)
tokenizer_bert = AutoTokenizer.from_pretrained(bert_model_name)
model_bert = AutoModelForSequenceClassification.from_pretrained(bert_model_name)


# Input Hindi text
text = "यह आपकी हिंदी में लंबा टेक्स्ट हो सकता है जिसे आप संक्षेप में प्रस्तुत करना चाहते हैं।"
text = """
भारत एक महान देश है। यह विश्व की सबसे पुरानी सभ्यताओं में से एक है। 
यहां की संस्कृति बहुत विविध और समृद्ध है। 
भारत में विभिन्न धर्मों और भाषाओं के लोग रहते हैं। 
महात्मा गांधी, जिन्हें राष्ट्रपिता कहा जाता है, ने भारत को आज़ादी दिलाई।
"""

# Step 1: Load transcript from your uploaded notepad file
file_path = r"naren\meeting1-hindi.txt"   # <-- change to your actual filename
with open(file_path, "r", encoding="utf-8") as f:
    transcript = f.read()

text = transcript
# Step 1: Sentence segmentation
# sentences = nltk.sent_tokenize(text, language='hindi')
sentences = sentence_tokenize.sentence_split(text, lang='hi') #tokenize in hindi


# Step 2: Score sentences
sentence_scores = []
sentences = list(set(sentences))
for sentence in sentences:
    inputs = tokenizer_bert(sentence, return_tensors="pt", truncation=True, max_length=512)
    outputs = model_bert(**inputs)
    # score = torch.sigmoid(outputs.logits[0]).item()
    # score = torch.sigmoid(outputs.logits)
    importance_prob = torch.softmax(outputs.logits, dim=1)[0][1]
    score = importance_prob.item()
    sentence_scores.append((sentence, score))

# Step 3: Select top 40% sentences
topn = round(0.4 * len(sentence_scores))
top_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:topn]  # select top 3
important_text = " ".join([s[0] for s in top_sentences])

print("Extractive Summary:")
print("==================================")
print(important_text)

'''
# Load summarization model
tokenizer_sum = AutoTokenizer.from_pretrained(summarization_model_name)
model_sum = AutoModelForSeq2SeqLM.from_pretrained(summarization_model_name)

# Step 4: Generate a summary from selected sentences
inputs = tokenizer_sum.encode(important_text, return_tensors="pt", max_length=1024, truncation=True)
summary_ids = model_sum.generate(
    inputs,
    max_length=200,
    min_length=50,
    length_penalty=2.0,
    num_beams=4,
    early_stopping=True
)
final_summary = tokenizer_sum.decode(summary_ids[0], skip_special_tokens=True)

print("Final Extractive + Abstractive Summary:\n", final_summary)
'''