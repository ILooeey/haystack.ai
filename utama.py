from flask import Flask, render_template, request, redirect, flash, session
import os
import pdfplumber
from haystack import Document, Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers import InMemoryBM25Retriever
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.utils import Secret
import secrets
import uuid
import time

document_id = str(uuid.uuid4())

app = Flask(__name__, template_folder='E:/templates')
app.secret_key = secrets.token_hex(16)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Haystack
document_store = InMemoryDocumentStore()
retriever = InMemoryBM25Retriever(document_store=document_store)

# RAG Propmt
prompt_template = """
Given these documents, answer the question.
Documents:
{% for doc in documents %}
    {{ doc.content }}
{% endfor %}
Question: {{question}}
Answer:
"""
prompt_builder = PromptBuilder(template=prompt_template)
llm = OpenAIGenerator(api_key=Secret.from_token("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"))

# Pipeline RAG
rag_pipeline = Pipeline()
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("llm", llm)
rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "llm")

# Flask
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files["file"]
        if file and file.filename.endswith(".pdf"):
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)

            text = ""
            with pdfplumber.open(filename) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n\n"
            
            document_store.write_documents([Document(content=text, meta={"source": filename, "id": document_id})])

            flash('File berhasil diunggah dan diproses.')
            return redirect("/ask_question")

    return render_template("index.html")

# Route QnA
@app.route("/ask_question", methods=["GET", "POST"])
def ask_question():
    if "chat_history" not in session:
        session["chat_history"] = [] 
    if request.method == "POST":
        question = request.form["question"]
        start_time = time.time()

        results = rag_pipeline.run(
            {
                "retriever": {"query": question},
                "prompt_builder": {"question": question},
            }
        )

        end_time = time.time()
        response_time = end_time - start_time

        answer = results["llm"]["replies"]

        session["chat_history"].append({"role": "user", "message": question})
        session["chat_history"].append({"role": "system", "message": answer})

        return render_template("ask_question.html", chat_history=session["chat_history"])

    return render_template("ask_question.html", chat_history=session["chat_history"])

if __name__ == "__main__":
    app.run(debug=True)
