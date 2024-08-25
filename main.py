import streamlit as st
from dotenv import load_dotenv
import PyPDF2
import os  
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub

#here just started the design 
css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
.stTextInput {
  background-color: white; 
  color: red; 
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">    
    <div class="message">{{MSG}}</div>
</div>
'''

def process_pdfs(pdf_directory):

    pdf_files = [os.path.join(pdf_directory, filename) for filename in os.listdir(pdf_directory) if filename.endswith(".pdf")]

    combined_text = ""
    for pdf_file in pdf_files:
        try:
            with open(pdf_file, "rb") as pdf_file_object:
                pdf_reader = PyPDF2.PdfReader(pdf_file_object)

                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    extracted_text = page.extract_text()
                    combined_text += extracted_text
        except FileNotFoundError:
            print(f"Error: PDF file not found: {pdf_file}")

    return combined_text



def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with Sandra")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
        st.session_state.chat_history = []

    file = r"C:\Users\Lenovo\Desktop\LV\data"  # الباث تبع الفولدر

    
    if not st.session_state.get("processed_data", False):
        with st.spinner("Preparing the chat with sandra it will take a few seconds..."):
            # Get PDF text
            raw_text = process_pdfs(file)

            # Get the text chunks
            chunks = get_text_chunks(raw_text)

            # Create vector store
            vectorstore = get_vectorstore(chunks)

            # Create conversation chain
            st.session_state.conversation = get_conversation_chain(vectorstore)

            # Mark data as processed
            st.session_state["processed_data"] = True

    st.header("Chat with Sandra")
    st.write('<style>{}</style>'.format('.stTextInput { background-color: red; color: white; }'), unsafe_allow_html=True)
    user_question = st.text_input("Ask a question about your mental health") 

    if user_question:
        handle_userinput(user_question)

        
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)


if __name__ == '__main__':
    main()
