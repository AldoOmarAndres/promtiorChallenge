## Important: The model is not deployed in the cloud.

This project was something completely new and challenging. I had never used these technologies before, so I had to rely on various sources such as documentation and videos to get an idea of what I needed to do, how to implement it, and then how to use it.

I was tasked with implementing a simple chatbot using LangChain, an LLM model, and the RAG architecture to answer specific questions.

Due to time constraints, I decided to go for a simple approach:
- The data used to feed the model was stored in a text file named `data.txt`.
- The model I used was **Ollama**, as it is open-source and very flexible (which introduced certain challenges).
- To host the model, I decided to use **Azure Container Apps** along with **Docker**.

## Functionality:
The model consumes the data stored in local memory, and then, using **FastAPI** and **LangServe**, it can be accessed to answer the corresponding questions.

## Main Challenges:
Developing the app was not particularly difficult since the documentation was clear and easy to follow.

Deploying the model, however, was very complex. I wasn't able to fully deploy it, mainly due to my limited knowledge of **Docker**. 

After struggling with configuration files and failing to generate a single image containing both the app and the **Ollama model (Llama3.2:1b)**, I decided that a viable alternative was to deploy two separate images: one for the model and another for the app, and then have them communicate with each other.

Deploying the model was relatively straightforward (it is deployed) and accessible from anywhere. In fact, the app can be run locally while using the deployed model.

However, the app itself could not be deployed, as I was unable to connect the built image to Azure App Container. Even though I could access the deployed model's URL and test the chatbot before building the image, once deployed, it wouldn't work.

I even created an instance of the app in Azure following the provided documentation, thinking the issue might be related to running Docker locally. However, when trying to access the service, I got no response—just a few minutes of loading followed by a disconnection due to lack of response.

## Conclusion:
The challenge was great—clearly defined and with available resources. It is frustrating not being able to complete it properly. Nonetheless, I appreciate the opportunity and remain available for any further discussions.
