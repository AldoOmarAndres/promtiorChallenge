## Important: The link to the functional model is: https://appbot-g8hpe9dye6bpe7ba.eastus-01.azurewebsites.net/playground/.

This project was something completely new and challenging. I had never used these technologies before, so I had to rely on various sources such as documentation and videos to get an idea of what I needed to do, how to implement it, and then how to use it.

I was tasked with implementing a simple chatbot using LangChain, an LLM model, and the RAG architecture to answer specific questions.

Due to time constraints, I decided to go for a simple approach:
- The data used to feed the model was stored in a text file named `data.txt`.
- The models I used were **Ollama** and **Gemini**, as it is open-source and very flexible (which introduced certain challenges).
- To host the model, I decided to use **Azure Container Apps** along with **Docker**, also **Azure Web App**.

## Functionality:
The model consumes the data stored in local memory, and then, using **FastAPI** and **LangServe**, it can be accessed to answer the corresponding questions.

## Main Challenges:
Developing the app was not particularly difficult since the documentation was clear and easy to follow.

Deploying the model, however, was very complex. I wasn't able to fully deploy it, mainly due to my limited knowledge of **Docker**. 

After struggling with configuration files and failing to generate a single image containing both the app and the **Ollama model (Llama3.2:1b)**, I decided that a viable alternative was to deploy two separate images: one for the model and another for the app, and then have them communicate with each other.

Deploying the model was relatively straightforward (it is deployed) and accessible from anywhere. In fact, the app can be run locally while using the deployed model.

However, the app itself could not be deployed, as I was unable to connect the built image to Azure App Container. Even though I could access the deployed model's URL and test the chatbot before building the image, once deployed, it wouldn't work.

I also tried deploying the app as a **web app**, but the result was the same. Even attempting to download **Ollama** directly within the app, the error persisted, preventing the chatbot from accessing the model.

Finally, I am attaching an image where I run the chatbot locally (executing the `deploy.py` file). This connects to the **app container** where the **Llama3.2:1b** model is hosted and allows interaction with it.

## Alternative Approach:
To test a different approach, I also tried using **Gemini**. This alternative was much easier to implement. I only needed the **Google API_KEY**, and the model was the same one I had used for Llama.

This time, I was able to successfully deploy the app! 🎉

## Conclusion:
The challenge was great—clearly defined and with available resources. I feel a bit disappointed that I couldn't implement the chatbot with **Llama**, but I’m still happy that I managed to do it using **Gemini**.  

I truly appreciate this opportunity, not only as an evaluation but also as a chance to grow technically. I remain available for any further discussions.