import os
import pickle
import streamlit as st
import tempfile
import pandas as pd
import asyncio
import json
import firebase_admin
import csv

# Import modules needed for building the chatbot application
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS

from firebase_admin import credentials
from firebase_admin import firestore
from datetime import datetime

# Set the Streamlit page configuration, including the layout and page title/icon
st.set_page_config(layout="wide", page_icon="💬", page_title="CVChat")

# Display the header for the application using HTML markdown
st.markdown(
    "<h1 style='text-align: center;'>Welcome to CVChat, an AI that knows Ilya better then anyone else! 💬</h1>",
    unsafe_allow_html=True)

# Allow the user to enter their OpenAI API key
user_api_key = st.secrets["openai_api"]

path = os.path.dirname(__file__)
if not firebase_admin._apps:
    key_dict = credentials.Certificate(json.loads(st.secrets["textkey"]))
    firebase_admin.initialize_app(key_dict)
db = firestore.client()
if 'auth' not in st.session_state:
            st.session_state['auth'] = False
if 'nick' not in st.session_state:
            st.session_state['nick'] = ''

async def main():
    
    # Check if the user has entered an OpenAI API key
    if user_api_key == "":
        
        # Display a message asking the user to enter their API key
        st.markdown(
            "<div style='text-align: center;'><h4>Enter your OpenAI API key to start chatting 😉</h4></div>",
            unsafe_allow_html=True)
        
    else:
        # Set the OpenAI API key as an environment variable
        os.environ["OPENAI_API_KEY"] = user_api_key
        query_params = st.experimental_get_query_params()

        if 'auth' in query_params and query_params['auth'] == 'True':
            st.session_state['auth'] = True
            
        else:
            holder1 = st.empty()
            holder2 = st.empty()
            username = holder1.text_input('Please create a nickname for yourself')
            login_button = holder2.button('Login')
            if login_button and username:
                st.session_state['auth'] = True
                st.session_state['nick'] = username
                doc_ref = db.collection("messages_collection").document(st.session_state['nick']).set({"id":1})
                holder1.empty()
                holder2.empty()
                
        if st.session_state['auth']:
            username = st.empty()
            login_button = st.empty()
            # Allow the user to upload a CSV file
            #uploaded_file = st.sidebar.file_uploader("upload", type="csv", label_visibility="hidden")
            # Retrieve all documents from "knowledge_base" collection
            docs = db.collection('knowledge_base').stream()

            # Define an empty list to hold the data
            data = []
            path = os.path.dirname(__file__)
            # Iterate through each document and append the question and answer fields to the data list
            for doc in docs:
                data.append(['"' + doc.to_dict()['question'] + '"', '"' + doc.to_dict()['answer'] + '"'])

            # Load the existing CSV file into a Pandas DataFrame and verify that each row has exactly two columns
            df = pd.read_csv(path+'/training_data.csv', header=0, names=['prompt', 'completion'])
            if len(df.columns) != 2:
                df = df[['prompt', 'completion']]
                df.to_csv(path+'/training_data.csv', index=False, quoting=2)

            # Append the new data to the DataFrame
            new_df = pd.DataFrame(data, columns=['prompt', 'completion'])
            df = pd.concat([df, new_df], ignore_index=True)

            # Verify that each row in the DataFrame has exactly two columns
            if len(df.columns) != 2:
                raise ValueError("The CSV file should have exactly two columns: 'prompt' and 'completion'.")

            # Write the updated DataFrame back to the CSV file, in append mode with double quotes around each field
            df.to_csv(path+'/training_data.csv', mode='w', header=True, index=False, quoting=1)

            # Delete all the processed documents from "knowledge_base" collection
            for doc in docs:
                doc.reference.delete()

            if os.path.exists(path+"/training_data.csv.pkl"):
                os.remove(path+"/training_data.csv.pkl")



            doc_ref = db.collection("messages_collection").document(st.session_state['nick'])
            path = os.path.dirname(__file__)
            uploaded_file = open(path+'/training_data.csv',"rb")
            #uploaded_file = uploaded_file.read("training_data.csv")
            #print(uploaded_file)
            # If the user has uploaded a file, display it in an expander
            if uploaded_file is not None:
                def show_user_file(uploaded_file):
                    file_container = st.expander("Your CSV file :")
                    shows = pd.read_csv(uploaded_file)
                    uploaded_file.seek(0)
                    file_container.write(shows)
                    
                show_user_file(uploaded_file)
                
            # If the user has not uploaded a file, display a message asking them to do so
            else :
                st.sidebar.info(
                "👆 Upload your CSV file to get started, "
                "sample for try : [fishfry-locations.csv](https://drive.google.com/file/d/18i7tN2CqrmoouaSqm3hDfAk17hmWx94e/view?usp=sharing)" 
                )
        
            if uploaded_file :
                print('About to start')
                try :
                    # Define an asynchronous function for storing document embeddings using Langchain and FAISS
                    print('About to start2')
                    async def storeDocEmbeds(file, filename):
                        print('Starting to create vectors...')
                        # Write the uploaded file to a temporary file
                        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tmp_file:
                            tmp_file.write(file)
                            tmp_file_path = tmp_file.name

                        # Load the data from the CSV file using Langchain
                        loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8")
                        dataset = loader.load()

                        # Create an embeddings object using Langchain
                        embeddings = OpenAIEmbeddings()
                        print('Starting to create vectors...Stage 2!')
                        # Store the embeddings vectors using FAISS
                        vectors = FAISS.from_documents(dataset, embeddings)
                        os.remove(tmp_file_path)

                        # Save the vectors to a pickle file
                        with open(filename + ".pkl", "wb") as f:
                            pickle.dump(vectors, f)
                        
                    # Define an asynchronous function for retrieving document embeddings
                    async def getDocEmbeds(file, filename):
                        
                        # Check if embeddings vectors have already been stored in a pickle file
                        if not os.path.isfile(filename + ".pkl"):
                            # If not, store the vectors using the storeDocEmbeds function
                            await storeDocEmbeds(file, filename)
                        
                        # Load the vectors from the pickle file
                        with open(filename + ".pkl", "rb") as f:
                            #global vectors
                            vectors = pickle.load(f)
                            
                        return vectors

                    # Define an asynchronous function for conducting conversational chat using Langchain
                    async def conversational_chat(query):
                        
                        # Use the Langchain ConversationalRetrievalChain to generate a response to the user's query
                        result = chain({"question": query, "chat_history": st.session_state['history']})
                        
                        # Add the user's query and the chatbot's response to the chat history
                        st.session_state['history'].append((query, result["answer"]))
                        
                        # Print the chat history for debugging purposes
                        print("Log: ")
                        print(st.session_state['history'])
                        
                        return result["answer"]

                    # Set up sidebar with various options
                    with st.sidebar.expander("🛠️ Settings", expanded=False):
                        
                        # Add a button to reset the chat history
                        if st.button("Reset Chat"):
                            st.session_state['reset_chat'] = True

                        # Allow the user to select a chatbot model to use
                        MODEL = st.selectbox(label='Model', options=['gpt-3.5-turbo'])

                    # If the chat history has not yet been initialized, do so now
                    if 'history' not in st.session_state:
                        st.session_state['history'] = []

                    # If the chatbot is not yet ready to chat, set the "ready" flag to False
                    if 'ready' not in st.session_state:
                        st.session_state['ready'] = False
                        
                    # If the "reset_chat" flag has not been set, set it to False
                    if 'reset_chat' not in st.session_state:
                        st.session_state['reset_chat'] = False
                    
                            # If a CSV file has been uploaded
                    if uploaded_file is not None:

                        # Display a spinner while processing the file
                        with st.spinner("Processing..."):

                            # Read the uploaded CSV file
                            uploaded_file.seek(0)
                            file = uploaded_file.read()
                            
                            # Generate embeddings vectors for the file
                            print('about to process your embeds')
                            vectors = await getDocEmbeds(file, uploaded_file.name)
                            print('about to process your embeds part #2')
                            # Use the Langchain ConversationalRetrievalChain to set up the chatbot
                            chain = ConversationalRetrievalChain.from_llm(llm = ChatOpenAI(temperature=0.0,model_name=MODEL),
                                                                        retriever=vectors.as_retriever())

                        # Set the "ready" flag to True now that the chatbot is ready to chat
                        st.session_state['ready'] = True

                    # If the chatbot is ready to chat
                    if st.session_state['ready']:

                        # If the chat history has not yet been initialized, initialize it now
                        if 'generated' not in st.session_state:
                            st.session_state['generated'] = ["Let me know what information you want to know, so that it'll be clear as day, that Ilya is the best candidate for Position at <company_name>"]

                        if 'past' not in st.session_state:
                            st.session_state['past'] = ["Hey ! 👋"]

                        # Create a container for displaying the chat history
                        response_container = st.container()
                        
                        # Create a container for the user's text input
                        container = st.container()

                        with container:
                            
                            # Create a form for the user to enter their query
                            with st.form(key='my_form', clear_on_submit=True):
                                
                                user_input = st.text_input("Query:", placeholder="You can ask questions about Ilya's education, skills, expirience that he had...", key='input')
                                submit_button = st.form_submit_button(label='Send')
                                
                                # If the "reset_chat" flag has been set, reset the chat history and generated messages
                                if st.session_state['reset_chat']:
                                    
                                    st.session_state['history'] = []
                                    st.session_state['past'] = ["Hey ! 👋"]
                                    st.session_state['generated'] = ["Let me know what information you want to know, so that it'll be clear as day, that Ilya is the best candidate for Position at <company_name>"]
                                    response_container.empty()
                                    st.session_state['reset_chat'] = False

                            # If the user has submitted a query
                            if submit_button and user_input:
                                
                                # Generate a response using the Langchain ConversationalRetrievalChain
                                output = await conversational_chat(user_input)
                                
                                # Add the user's input and the chatbot's output to the chat history
                                st.session_state['past'].append(user_input)
                                st.session_state['generated'].append(output)

                        # If there are generated messages to display
                        if st.session_state['generated']:
                            
                            # Display the chat history
                            with response_container:
                                
                                for i in range(len(st.session_state['generated'])):
                                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                                    message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
                                    
                            now = datetime.now()        
                            doc_ref.update({
                                        "message": firestore.ArrayUnion([{
                                                "question": st.session_state["past"][i],
                                                "answer": st.session_state["generated"][i],
                                                "timestamp": now.strftime("%d/%m/%Y %H:%M:%S")
                                            }])
                                        })          
                    #st.write(chain)

                except Exception as e:
                    st.error(f"Error: {str(e)}")

    # Create an expander for the "About" section
    about = st.sidebar.expander("About 🤖")
    
    # Write information about the chatbot in the "About" section 
    about.write("#### If you're satisfied with the answers - you can contact Ilya via [Linkedin](https://www.linkedin.com/in/linkilya/) or ⚡[Telegram](https://t.me/eli_eth)")
    logout = st.sidebar.button('Logout')
    if logout:
        st.session_state['auth']=False

#Run the main function using asyncio
if __name__ == "__main__":
    asyncio.run(main())

