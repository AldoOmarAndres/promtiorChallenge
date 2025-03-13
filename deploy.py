from fastapi import FastAPI
from pydantic import BaseModel
from langserve import add_routes
import uvicorn
from model import get_graph

# Importamos el chatbot definido en tu código
from langchain_core.runnables import Runnable

# Inicializar la aplicación FastAPI
app = FastAPI()

# Definir el modelo de entrada
class QuestionRequest(BaseModel):
    question: str

# Obtener el pipeline del chatbot
chat_chain: Runnable = get_graph()

@app.post("/chatbot")
def ask_chatbot(request: QuestionRequest):
    """Recibe una pregunta y devuelve la respuesta generada por el chatbot"""
    response = chat_chain.invoke({"question": request.question})
    return {"answer": response}

# Ejecutar la API localmente
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)