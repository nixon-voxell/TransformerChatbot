Chatbot based on Transformer models
---

This is a chatbot trained on comment reply pairs using a Transformer model.
This Transfomer model is initially used to translate languages from one to other. If you want to learn more on what it is about, I highly recommend you to head to tensorflow's original website to learn more on this model:
https://www.tensorflow.org/tutorials/text/transformer

I'll try to explain all the functions on the website in my video playlist here: https://www.youtube.com/watch?v=G0j7q5oONbw&list=PLlnBGPe6GFdNBNwMxyXf2FFxVeXz_y3Ee

I have tweaked the code alot from the original code itself, mostly on the place where we sort our dataset for training. I'll try to change the model itself to improve it more for chatbot purpose.

Goal of this project:
The goal of this model is to train a chatbot that can remembers chat history and stores it in some form of states. The model will then be able to reply accurately based on current query and history states.