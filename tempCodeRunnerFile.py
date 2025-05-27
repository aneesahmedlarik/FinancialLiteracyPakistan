@app.route("/ask-content-question", methods=["POST"])
def ask_content_question():
    data = request.get_json()
    content = data.get("content")
    question = data.get("question")
    if not content or not question:
        return jsonify({"error": "Missing content or question"}), 400

    # 1. Split the content into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents([content])

    # 2. Create embeddings and FAISS vector store
    embeddings = OpenAIEmbeddings()  # Uses your OPENAI_API_KEY from env
    db = FAISS.from_documents(docs, embeddings)

    # 3. Set up the retriever and QA chain
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0),  # Uses your OPENAI_API_KEY
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
    )

    # 4. Run the QA chain
    result = qa({"query": question})
    answer = result["result"]

    return jsonify({"answer": answer})