import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer(ngram_range=(1, 4))
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response
        
counter = 0

def main():
    global counter
    st.title("Intents of Chatbot using NLP")

    # Create a sidebar menu with options
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Home Menu
    if choice == "Home":
        st.write("Welcome to the chatbot. Please type a message and press Enter to start the conversation.")

        # Check if the chat_log.csv file exists, and if not, create it with column names
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{counter}")

        if user_input:

            # Convert the user input to a string
            user_input_str = str(user_input)

            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_response_{counter}")

            # Get the current timestamp
            timestamp = datetime.datetime.now().strftime(f"%Y-%m-%d %H:%M:%S")

            # Save the user input and chatbot response to the chat_log.csv file
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()

    # Conversation History Menu
    elif choice == "Conversation History":
        # Display the conversation history in a collapsible expander
        st.header("Conversation History")
        # with st.beta_expander("Click to see Conversation History"):
        with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  # Skip the header row
            for row in csv_reader:
                st.text(f"User: {row[0]}")
                st.text(f"Chatbot: {row[1]}")
                st.text(f"Timestamp: {row[2]}")
                st.markdown("---")

    elif choice == "About":
        st.write(
        """
        Welcome to our Chatbot project! The goal is to build a smart and interactive chatbot 
        capable of understanding and responding to user input by leveraging Natural Language Processing (NLP) 
        and Logistic Regression. The chatbot interface is designed using **Streamlit**, 
        making it intuitive and visually appealing for users.
        """
    )

        st.subheader("🔍 Project Overview:")

        st.write(
        """
        This project is a blend of **machine learning** and **web development**. It is organized into two core components:
        
        1. **Training the Chatbot:** 
           - NLP techniques and the Logistic Regression algorithm are employed to train the chatbot. 
           - The training focuses on recognizing **intents** and extracting relevant **entities** from user input.
        
        2. **Building the Chatbot Interface:** 
           - A web-based chatbot interface is implemented using **Streamlit**, allowing seamless interaction with users. 
           - The interface bridges the trained model with users through an easy-to-use chat window.
        """
    )

        st.subheader("📊 Dataset Details:")

        st.write(
        """
        The dataset used in this project is a simple yet structured collection of labeled examples:
        
        - **Intents:** Reflect the purpose of user input, such as "greeting", "budget inquiry", or "general info".
        - **Entities:** Highlight key information in user inputs like "Hi", "How can I manage my budget?", "Tell me about yourself".
        - **Text:** Represents the actual user input to the chatbot.

        This dataset enables the chatbot to interpret a wide variety of user queries accurately.
        """
    )

        st.subheader("💬 Streamlit Chatbot Interface:")

        st.write(
        """
        The interface is crafted with **Streamlit**, offering a clean and interactive user experience:
        
        - A **text input box** allows users to type their messages.
        - A **chat window** displays user inputs alongside the chatbot's intelligent responses.
        - The chatbot leverages the trained model to ensure contextually accurate replies.
        
        This intuitive design ensures ease of use while making the interaction enjoyable and efficient.
        """
    )

        st.subheader("🌟 Conclusion:")

        st.write(
        """
        This project demonstrates the creation of a chatbot that combines **machine learning** and 
        **interactive web development** to deliver a functional and engaging tool. 

        Key Achievements:
        - Successfully trained a chatbot on labeled intents using **NLP** and **Logistic Regression**.
        - Designed a responsive and user-friendly interface with **Streamlit**.

        **Future Enhancements:**
        - Incorporate larger datasets for greater versatility.
        - Employ advanced NLP techniques like **transformers** or **deep learning models** (e.g., BERT, GPT).
        - Add features like multi-turn conversations, voice input, and multilingual support to make the chatbot even smarter.

        This project is a stepping stone to creating more sophisticated conversational AI systems. 🚀
        """
    )

if __name__ == '__main__':
    main()
