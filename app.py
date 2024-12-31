import os
import time
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import chromadb
chromadb.api.client.SharedSystemClient.clear_system_cache()

load_dotenv()
api = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=api)

st.set_page_config('ChatBot')
st.subheader('PDF Chat',divider=True)

def get_docs(docs):
  file = genai.upload_file(docs, mime_type=docs.type)
  model = genai.GenerativeModel('gemini-1.5-flash')
  response = model.generate_content([
      file, """Extract the text from the given file"""])
  output = response.text
  return output

def text_chunks(output):
  splitter = RecursiveCharacterTextSplitter(
              chunk_size=10000,
              chunk_overlap=1000)
  chunk = splitter.split_text(output)
  return chunk

def get_vector(chunk):
  embedding = HuggingFaceEmbeddings(
              model_name="sentence-transformers/all-mpnet-base-v2")
  db = Chroma(
          collection_name='chat_pdf',
          embedding_function=embedding,
          persist_directory='chroma')
  db.add_texts(chunk)

def get_prompt():
  prompt_template = """
    Answer the question as detailed as possible from the provided context,
    make sure to provide all the details, if the answer is not in the 
    provided context just say, 'answer is not available in the context',
    don't provide the wrong answer
    Context:{context}
    Question:{question}
    Answer:
    """
  prompt = PromptTemplate(
                  template=prompt_template,
                  input_variables=['context','question'])
  return prompt

def get_response():
  embedding = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2")
  new_db = Chroma(
          collection_name='chat_pdf',
          embedding_function=embedding,
          persist_directory='chroma')
  retriver = new_db.as_retriever()
  prompt = get_prompt()
  chat_model = ChatGoogleGenerativeAI(
                model='gemini-1.5-flash',
                api_key=api)
  str_output = StrOutputParser()
  chain = {'context':retriver, 'question':RunnablePassthrough()} |\
              prompt | chat_model | str_output
  response = chain.invoke(user_question)
  return response

with st.sidebar:
    st.title('Menu')
    files = st.file_uploader('Upload File', accept_multiple_files=True)
    if st.button('Process'):
      with st.spinner('Processing...'):
        for file in files:
          raw_text = get_docs(file)
          chunks = text_chunks(raw_text)
          get_vector(chunks)
        st.success('Done')

user_question = st.chat_input('Ask a Question')
if user_question:
  st.markdown(f'User: {user_question}')
  response = get_response()
  with st.spinner():
    time.sleep(2)
  st.markdown(f'Response: {response}')