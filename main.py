import os
import streamlit as st
import requests

from os.path import join
from langchain_ollama.llms import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from consts import PDFS_DIRECTORY
from src.RAG.pdf_rag import (
    upload_pdf,
    load_pdf,
    split_text,
    index_documents,
    retrieve_documents,
    answer_question,
)

from src.utils.google_utils import write_an_event, name_the_next_10_events

# Run initializer.sh to set up the environment variable
os.environ["GOOGLE_CREDENTIAL_API"] = r"C:\Users\mario\Documents\client_secret.json"
print(os.getenv("GOOGLE_CREDENTIAL_API"))


def chatbot(
    vector_store: InMemoryVectorStore,
    model: OllamaLLM,
    template: str,
):
    uploaded_file = st.file_uploader("Sube el archivo PDF.", type=["pdf"])

    if uploaded_file:
        upload_pdf(uploaded_file)
        documents = load_pdf(join(PDFS_DIRECTORY, uploaded_file.name))
        documents = split_text(documents)
        index_documents(documents, vector_store)

        question = st.chat_input()
        if question:
            st.chat_message("user").write(question)
            related_documents = retrieve_documents(question, vector_store)
            answer = answer_question(question, related_documents, template, model)
            st.chat_message("bot").write(answer)


def chatbot_in_vscode(
    # vector_store: InMemoryVectorStore,
    model: OllamaLLM,
    template: str,
):
    documents = load_pdf(
        join(PDFS_DIRECTORY, r"2. Fiebre Tifoidea (Salmonella typhi y paratyphi).pdf")
    )
    documents = split_text(documents)
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local("data/faiss_index")
    vector_store = FAISS.load_local(
        "data/faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    question = input("Pregunta: ")
    if question:
        related_documents = retrieve_documents(question, vector_store)
        answer = answer_question(question, related_documents, template, model)
        print(answer)


def get_ticketmaster_events(api_key, place, max_results=10):
    """
    Search Ticketmaster events by place (city name)

    Args:
        api_key (str): Your Ticketmaster API key
        place (str): City name to search (e.g., "New York")
        max_results (int): Maximum number of results to return

    Returns:
        list: Event details or error message
    """
    endpoint = "https://app.ticketmaster.com/discovery/v2/events.json"

    params = {"apikey": api_key, "city": place, "size": max_results}

    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()  # Raise error for bad status codes

        data = response.json()

        # Handle empty results
        if "_embedded" not in data:
            return []
        events = []
        for event in data["_embedded"]["events"]:
            # Extract relevant information
            event_info = {
                "name": event.get("name", "N/A"),
                "url": event.get("url", "N/A"),
                "date": event["dates"]["start"].get("localDate", "N/A"),
                "time": event["dates"]["start"].get("localTime", "N/A"),
                "description": event.get("info", "No description available"),
                "venue": event.get("_embedded", {})
                .get("venues", [{}])[0]
                .get("name", "N/A"),
            }
            events.append(event_info)

        return events

    except requests.exceptions.RequestException as e:
        return f"API Error: {str(e)}"
    except ValueError:
        return "Error parsing JSON response"
    except KeyError:
        return "Unexpected response format"


# if __name__ == "__main__":
#     embeddings = OllamaEmbeddings(model="deepseek-r1:7b")
#     # vector_store = InMemoryVectorStore(embeddings)
#     model = OllamaLLM(model="deepseek-r1:7b")

#     template = """Eres un médico experto en tomar el examen de especialidad en México, el ENARM, y también eres un
#     asistente para resolver dudas médicas puntuales. Usa los siguientes fragmentos de contexto recuperados para
#     responder la pregunta. Si no sabes la respuesta, simplemente di que no sabes. Recuerda ser conciso y preciso.
#     Al responder, todo tu proceso de pensamiento interno (tu cadena de pensamiento) debe ser 100%
#     en español, comenzando con 'Bueno'. No utilices inglés ni mezcle idiomas en tu reflexión, a menos que sea necesario.
#     Ejemplo de una cadena de pensamiento correcta: "Bueno, el usuario pregunta sobre... [razonamiento en español, sin palabras
#     en inglés]". ¡Nunca utilices inglés en su cadena de pensamiento! ¡Se trata de tu pensamiento interno, no del resultado!
#     Pregunta: {question}
#     Contexto: {context}
#     Respuesta:
#     """

#     # chatbot(embeddings, vector_store, model, template)
#     # chatbot_in_vscode(vector_store, model, template)
#     chatbot_in_vscode(model, template)


if __name__ == "__main__":
    # Read json file with Ticketmaster API key
    with open("ticketmaster_key.json", "r") as file:
        ticketmaster_api_key = file.read().strip()
    ticketmaster_api_key_value = (
        ticketmaster_api_key.split(":")[1].strip().strip('"')[:-2]
    )
    print(ticketmaster_api_key_value, type(ticketmaster_api_key_value))
    # Get events from Ticketmaster API
    events = get_ticketmaster_events(
        ticketmaster_api_key_value, "Monterrey", max_results=3
    )
    print(events)

    # event_name = "Test Event"
    # event_description = "This is a test event created by the Google Calendar API."
    # location = "Monterrey, Nuevo León, México"
    # event_start = "2025-06-11T10:00:00-07:00"
    # event_end = "2025-06-11T11:00:00-07:00"
    # # write_an_event(event_name, event_description, location, event_start, event_end)
    # name_the_next_10_events()
