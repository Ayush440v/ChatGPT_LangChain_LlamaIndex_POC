import os
import sys

from langchain.chains import ConversationalRetrievalChain
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_community.vectorstores import Chroma

import constants

os.environ["OPENAI_API_KEY"] = constants.APIKEY

# This is to be enabled only if we want to use the same data again and again to save the API calls
PERSIST = False

# Initialize the variable 'query' to None
query = None

# Check if any command-line arguments were passed
if len(sys.argv) > 1:
  
  # If yes, set 'query' to the first command-line argument  
  query = sys.argv[1]

# Check if the PERSIST flag is set and if a directory named "persist" exists
if PERSIST and os.path.exists("persist"):
  # If both conditions are met, print a message indicating that the existing index will be reused
  print("Reusing index...\n")
  # Create a Chroma vector store object, initializing it with the "persist" directory and an OpenAIEmbeddings function
  vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
  # Wrap the Chroma vector store object in a VectorStoreIndexWrapper to create the index
  index = VectorStoreIndexWrapper(vectorstore=vectorstore)
# If the PERSIST flag is not set or the "persist" directory does not exist
else:
  # Create a TextLoader object with "data/data.txt" as the data source
  loader = TextLoader("data/data.txt")
  # If the PERSIST flag is set
  if PERSIST:
    # Create a VectorstoreIndexCreator object with the "persist" directory as the argument
    # Call the from_loaders method with the TextLoader object to create the index
    index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
  # If the PERSIST flag is not set
  else:
    # Create a VectorstoreIndexCreator object without any arguments
    # Call the from_loaders method with the TextLoader object to create the index
    index = VectorstoreIndexCreator().from_loaders([loader])

# Create a ConversationalRetrievalChain object from a language model
# The language model used is ChatOpenAI with the model "gpt-3.5-turbo"
chain = ConversationalRetrievalChain.from_llm(
  llm=ChatOpenAI(model="gpt-3.5-turbo"),
  # The retriever used is the vectorstore from the previously created index
  # The search_kwargs parameter is set to {"k": 1}, meaning the retriever will return the top 1 most relevant result
  retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)

# Initialize an empty list to store the chat history
chat_history = []

# Start an infinite loop to continuously accept user input and generate responses
while True:
  # If the query is empty
  if not query:
    # Prompt the user for input
    query = input("Prompt: ")

  # If the user input is 'quit', 'q', or 'exit'
  if query in ['quit', 'q', 'exit']:
    # Exit the program
    sys.exit()

  # Pass the user query and the chat history to the ConversationalRetrievalChain object
  # The chain object will generate a response based on the query and chat history
  result = chain({"question": query, "chat_history": chat_history})

  # Print the generated response
  print(result['answer'])

  # Append the user query and the generated response to the chat history
  chat_history.append((query, result['answer']))

  # Reset the query to None for the next iteration
  query = None