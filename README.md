# Peek: AI-Powered Event Recommendation & Scheduling Tool

## Overview

This is a prototype tool that demonstrates how an AI model can bridge two real-world APIs—Ticketmaster (for event discovery) and Google Calendar (for scheduling)—to provide a seamless, conversational experience for users. The AI interprets user intent, extracts relevant data, queries APIs, and presents actionable results in a human-friendly chat interface.

---

## Features

- **Conversational AI**: Uses a local LLM (via [Ollama](https://ollama.com/)) (model: qwen3:4b) to interpret user requests and maintain chat history.
- **Event Discovery**: Extracts city names from user queries and fetches relevant events from the Ticketmaster API.
- **Personalized Recommendations**: AI recommends events based on user preferences and event data.
- **Calendar Integration**: Schedules selected events directly into the user's Google Calendar.
- **Streamlit UI**: Interactive chat interface for a smooth user experience.

---

## Setup Instructions

### 1. Prerequisites

- Python 3.11
- [Streamlit](https://streamlit.io/)
- [Ollama](https://ollama.com/) (for local LLMs), download the model from the project using the command (keep in mind that this model is 2.6 GB):
  ```bash
  ollama pull qwen3:4b
  ```
- Ticketmaster API Key
- Google Calendar API credentials (`client_secret.json`)
- uv `universal virtualizer` for Python (optional, but recommended for managing dependencies)

### 2. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/peek.git
cd peek
uv sync
```

### 3. Configuration

- Place your Google API credentials at root folder as `client_secret.json`.
- Create a `ticketmaster_key.json` file in the project root:
  ```json
  {"TICKETMASTER_API_KEY": "your_ticketmaster_api_key"}
  ```

### 4. Running the App

```bash
streamlit run main.py
```

---

## Usage Example

1. **Ask for events**:  
   _"What events are happening in Monterrey this weekend?"_
2. **Get recommendations**:  
   The AI will extract the city, fetch events, and recommend the best options.
3. **Schedule an event**:  
   _"Add the Jazz Festival to my calendar."_
4. **Confirmation**:  
   The AI confirms the event has been scheduled in your Google Calendar.

---

## Key Files

- `main.py`: Main application logic, chat interface, and API integration.
- `src/RAG/pdf_rag.py`: AI prompt handling and response formatting.
- `src/utils/google_utils.py`: Google Calendar API utilities.
- `src/utils/ticketmaster_utils.py`: Ticketmaster API utilities.

---

## How It Works

1. **Conversation History**: Maintained in `st.session_state["history"]` for context-aware responses.
2. **City Extraction**: AI model outputs city names in `<city_name>` format after `<think> CORPUS </think>`. Regex extracts the city for API queries.
3. **Event Recommendation**: Ticketmaster API is queried, and results are passed back to the AI for personalized recommendations.
4. **Event Scheduling**: When the user selects an event, details are parsed and sent to Google Calendar via the API.

---

## Authentication & Error Handling

- **Google Calendar**: Uses OAuth2 credentials from `client_secret.json`.
- **Ticketmaster**: API key loaded from `ticketmaster_key.json`.

---

## Customization

- **Model**: Change the Ollama model in `main.py` (`OllamaLLM(model="qwen3:4b")`).
- **Prompt Templates**: Adjust prompt instructions in `main.py` for different behaviors.

---

## Next Steps

- Add automated tests for API utilities.
- Improve error handling to make it friendlier for users.
- Implement user authentication for personalized experiences.
- Support more event sources and calendar providers.
- Enhance UI for richer interactions.
- Deploy the app to a cloud service for wider accessibility.
- Consider using a more advanced LLM for better understanding and response generation.
- Explore additional features like reminders, event details, and user preferences.
- Remove the <think> tags from the AI output to make the conversation more natural.

---

## License

MIT License

---

## Demo

A sample walkthrough is expected to be shown during the interview.
