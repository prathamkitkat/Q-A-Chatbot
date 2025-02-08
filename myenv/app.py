from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain



load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
   If the question relates to IT, HR, Marketing, or Legal & Risk Management, provide a detailed answer as follows:

1. First, check the provided context and if the answer is found:
   - Answer with all available details from the context
   - Provide complete information as given in the context

2. If the answer is not in the context BUT the question is about IT, HR, Marketing, or Legal & Risk Management:
   - Provide a general best-practice answer based on standard industry knowledge
   - Clearly state: "While this specific information is not in the provided context, here is the standard practice:"
   - Follow with a detailed explanation

3. If the question is NOT related to IT, HR, Marketing, or Legal & Risk Management:
   - Respond only with: "This question is outside the scope of IT, HR, Marketing, and Legal & Risk Management. I can only answer questions related to these departments."

Context:\n {context}?\n
Question: \n{question}\n

Answer:
"""


    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = create_stuff_documents_chain(model, prompt)

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    # Updated to allow safe deserialization
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain.invoke(
        {"context":docs, "question": user_question}
        , return_only_outputs=True)

    print("Reply: ", response)

def main():
    # Get PDF file path from user
    pdf_path = [r"C:\Users\Lenovo\OneDrive\Desktop\hackathon\FAQ-10-Data-Protection.pdf",r"C:\Users\Lenovo\OneDrive\Desktop\hackathon\HR_Policies.pdf",r"C:\Users\Lenovo\OneDrive\Desktop\hackathon\Legal_Risk_Management_FAQ.pdf",r"C:\Users\Lenovo\OneDrive\Desktop\hackathon\Marketing_Department_FAQ.pdf",r"sample_policy_and_procedures_manual[1].pdf",r"C:\Users\Lenovo\OneDrive\Desktop\hackathon\Updated_IT_Policy_FAQ.pdf"]
    
  
   
    
    # Process the PDF
    pdf_docs = pdf_path
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)
    
    while True:
        user_question = input("\nAsk a question about the PDF (or type 'quit' to exit): ")
        if user_question.lower() == 'quit':
            break
        user_input(user_question)

if __name__ == "__main__":
    main()