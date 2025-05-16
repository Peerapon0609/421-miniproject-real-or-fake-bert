import streamlit as st
import pandas as pd
import torch
from transformers import BertForSequenceClassification
import pickle

# 0. กำหนด device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. โหลด BERT model และ tokenizer จากไฟล์พิกเคิล
@st.cache_resource
def load_model_and_data():
    with open("bert_fakenews_model_state_dict.pkl", "rb") as f_mod:
        state_dict = pickle.load(f_mod)

    with open("bert_fakenews_tokenizer.pkl", "rb") as f_tok:
        tokenizer = pickle.load(f_tok)

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, tokenizer

model, tokenizer = load_model_and_data()

# 2. สร้าง UI
st.title('🧠 Fake/Real News Detector (BERT)')

st.header('📝 Predict a News Article')
title_input = st.text_input('Enter the news title:')
text_input  = st.text_area('Enter the news body text:')

if st.button('Predict'):
    if title_input.strip() and text_input.strip():
        content = f"{title_input} {text_input}"
        inputs = tokenizer(
            content,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=-1).item()
            prob = torch.softmax(logits, dim=-1)[0][pred].item()

        label_text = 'Real' if pred == 0 else 'Fake'
        st.subheader('🔍 Prediction Result:')
        st.success(f'📰 This news is: **{label_text}**')
        st.write(f'📊 Confidence: `{prob:.2f}`')
    else:
        st.warning('Please enter both a title and body text.')
