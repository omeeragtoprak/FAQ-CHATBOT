#direkt cevap veren chatbot 30.07.2024 multilingual base embedding model
from flask import Flask, render_template, request, jsonify
import json
import ollama
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

def load_faq(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['sorucevap']

def create_embeddings(model, questions):
    return model.encode(questions)

def find_most_similar_questions(user_question, embeddings, model, top_n=5, threshold=0.4):
    user_embedding = model.encode([user_question])
    similarities = cosine_similarity(user_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_n]
    top_similarities = similarities[top_indices]
    
    result = []
    for idx, sim in zip(top_indices, top_similarities):
        if sim > threshold:
            result.append((idx, sim))
    
    return result

def ask_mistral(question, context, faq_items):
    SYSTEM_PROMPT = """Sen bir müşteri hizmetleri temsilcisisin. Sana verilen bağlamı ve FAQ öğelerini kullanarak soruyu doğru ve öz, soruya uygun sade bir şekilde yanıtla. not:Bireysel Emeklilik Sistemi kısaltımı bes olabilir. Cevapta "bağlantıya tıklayın" gibi ibareler varsa "url" sini yaz. Yoksa "url" yazmana gerek yok.
    Sana verilen en iyi 5 FAQ öğelerini analiz et ve sorulan soruyla ilgili cevabı belgeyi tarayıp kendin cevapla. Soruyla ilgili belgede çok benzer cevaplar varsa kendin de belgenin içinden ek cevap üretebilirsin, yoksa ek cevap üretmene de gerek yok.
    Eğer sorulan soru verilen bağlamla ilgili değilse, 
    bunu nazikçe belirt ve yardımcı olabileceğin başka bir konu olup olmadığını sor."""

    faq_context = "\n".join([f"Soru {i+1}: {item['question']}\nCevap {i+1}: {item['answer']}\nBenzerlik: {sim:.2f}" 
                             for i, (item, sim) in enumerate(faq_items)])

    response = ollama.chat(
        model='gemma2',
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user", 
                "content": f"Bağlam: {context}\n\nFAQ Öğeleri:\n{faq_context}\n\nKullanıcı Sorusu: {question}"
            }
        ]
    )
    return response["message"]["content"]


faq_list = load_faq("soru_cevap.json")
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
questions = [item['question'] for item in faq_list]
question_embeddings = create_embeddings(model, questions)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_question = request.json['question']
    similar_questions = find_most_similar_questions(user_question, question_embeddings, model)
    
    if similar_questions:
        context = "En benzer sorular:\n" + "\n".join([f"{faq_list[idx]['question']} (Benzerlik: {sim:.2f})" for idx, sim in similar_questions])
        faq_items = [(faq_list[idx], sim) for idx, sim in similar_questions]
        answer = ask_mistral(user_question, context, faq_items)
    else:
        answer = "Üzgünüm, sorunuzla ilgili doğrudan bir bilgi bulamadım. Başka bir konuda yardımcı olabilir miyim?"
    
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)