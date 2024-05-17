import streamlit as st
import fitz  # PyMuPDF
from langchain.llms import Ollama
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

#
def get_llm_response(input,content,prompt):  #input: how LLM model behave like, #image: To extract info, prompt: ask something
    # loading llama2 model
    model=Ollama(model='llama2')
    cont=str(content)
    response=model.invoke([input,cont,prompt]) #get response from model
    return response
# Function to extract text from PDF file
def extract_text_from_pdf(file):
    try:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error occurred while reading PDF file: {e}")
        return ""

# Main function
def main():
    # Set title and description
    st.title("PDF Chatbot")

    # Create a sidebar for file upload
    st.sidebar.title("Upload PDF File")
    uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=['pdf'])

    # Text input for prompt
    prompt = st.text_input("Ask a Question", "")

    # Submit button
    submitted = st.button("Submit")

    if submitted:
        if uploaded_file is not None:
            # Extract text from uploaded PDF file
            pdf_text = extract_text_from_pdf(uploaded_file)
            st.write(pdf_text)
            if pdf_text:
                try:
                    # Create embeddings
                    embeddings = HuggingFaceEmbeddings()

                    # Split text into chunks
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=20,
                        length_function=len,
                        is_separator_regex=False,
                    )
                    chunks = text_splitter.create_documents([pdf_text])

                    # Store chunks in ChromaDB
                    persist_directory = 'pdf_embeddings'
                    vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_directory)
                    vectordb.persist()  # Persist ChromaDB
                    st.write("Embeddings stored successfully in ChromaDB.")
                    st.write(f"Persist directory: {persist_directory}")

                    # Load persisted Chroma database
    
                    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
                    st.write(vectordb)
                    

                    # Load language model for QA
                    #llm = Ollama(model="llama2")

                    # Perform question answering
                    if prompt:
                        
                        #retriever=vectordb.as_retriever()
                        #docs=retriever.get_relevant_documents(prompt)
                        #st.write(docs[0])
                        # prompt_embedding = embeddings.embed_query([prompt])  # Wrap prompt in a list
                        # st.write(prompt_embedding)
                        docs = vectordb.similarity_search(prompt)
                        st.write(docs[0])
                        text=docs[0]
                        input_prompt="""You are an expert in understanding text contents. you will receive input pdf file and you will have to answer questions based on the input file."""
                        response=get_llm_response(input_prompt,text,prompt)
                        #response = llm.invoke([prompt,docs[0]])
                        st.subheader("Generated Answer:")
                        st.write(response)
                except Exception as e:
                    st.error(f"Error occurred during text processing: {e}")

if __name__ == "__main__":
    main()
