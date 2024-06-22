import tkinter as tk
from tkinter import scrolledtext
from transformers import AutoModelForCausalLM, AutoTokenizer
from gtts import gTTS
import os

# Initialize the LLM model and tokenizer
model_name = "EleutherAI/gpt-neo-1.3"  # You can use a different model
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create a Tkinter window
window = tk.Tk()
window.title("Chatbot")

# Create a scrolled text area for the chat
chat_area = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=40, height=10)
chat_area.grid(column=0, row=0, padx=10, pady=10, columnspan=2)

# Create a text input field for user messages
user_input = tk.Entry(window, width=30)
user_input.grid(column=0, row=1, padx=10, pady=10)

# Function to handle user input and display responses
def get_response():
    user_message = user_input.get()
    user_input.delete(0, tk.END)  # Clear the input field
    chat_area.insert(tk.END, f"You: {user_message}\n")

    # Get response from the chatbot model
    input_ids = tokenizer.encode(user_message, return_tensors="pt")
    response = model.generate(input_ids, max_length=50, num_return_sequences=1)
    bot_response = tokenizer.decode(response[0], skip_special_tokens=True)

    chat_area.insert(tk.END, f"Chatbot: {bot_response}\n")

    # Use gTTS to convert the bot's response to speech and play it
    tts = gTTS(text=bot_response, lang="en")
    tts.save("response.mp3")
    os.system("mpg321 response.mp3")

# Create a send button
send_button = tk.Button(window, text="Send", command=get_response)
send_button.grid(column=1, row=1, padx=10, pady=10)

# Start the Tkinter main loop
window.mainloop()
