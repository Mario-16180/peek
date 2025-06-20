import os
import re
import ast
import streamlit as st
from langchain_ollama.llms import OllamaLLM
from src.RAG.pdf_rag import (
    answer_prompt_no_context,
    answer_with_events,
    answer_when_scheduled,
)

from src.utils.google_utils import write_an_event, name_the_next_10_events
from src.utils.ticketmaster_utils import get_ticketmaster_events

# Run initializer.sh to set up the environment variable
os.environ["GOOGLE_CREDENTIAL_API"] = "client_secret.json"
print(os.getenv("GOOGLE_CREDENTIAL_API"))


def chatbot(
    model: OllamaLLM,
    template: str,
    template_when_events_are_given: str,
    template_when_scheduling: str,
):
    pattern = r"<([^<>/]+)>"
    pattern_2 = r"\[[^\[\]]*\]"
    # Initialize history as a list of (role, message) tuples
    if "history" not in st.session_state:
        st.session_state["history"] = []
    prompt = st.chat_input()
    if prompt:
        # Add user message to history
        st.session_state["history"].append(("user", prompt))
        st.chat_message("user").write(prompt)

        # Build conversation history string
        history_str = ""
        for role, message in st.session_state["history"]:
            history_str += f"{role}: {message}\n"

        # Pass history to the model as context
        full_prompt = f"{history_str}user: {prompt}\n"

        answer = answer_prompt_no_context(full_prompt, template, model)

        # Add model response to history
        st.session_state["history"].append(("bot", answer))

        check = answer.split("</think>")[-1]
        matches = re.findall(pattern, check)
        matches_2 = re.findall(pattern_2, check)
        if matches:
            city_name = matches[0]
            # Read json file with Ticketmaster API key
            with open("ticketmaster_key.json", "r") as file:
                ticketmaster_api_key = file.read().strip()
            ticketmaster_api_key_value = (
                ticketmaster_api_key.split(":")[1].strip().strip('"')[:-2]
            )
            # Get events from Ticketmaster API
            events = get_ticketmaster_events(
                ticketmaster_api_key_value, city_name, max_results=5
            )
            answer_plus_events = answer_with_events(
                events, template_when_events_are_given, model
            )
            st.chat_message("bot").write(answer_plus_events)
        elif matches_2:
            event_details = matches_2[0]
            # Convert the string representation of the list to an actual list
            try:
                event_details_list = ast.literal_eval(event_details)
            except (ValueError, SyntaxError) as e:
                st.chat_message("bot").write(
                    "Error parsing event details. Please try again."
                )
            event_name = event_details_list[0]
            event_description = event_details_list[1]
            location = event_details_list[2]
            event_start = event_details_list[3]
            event_end = event_details_list[4]
            # Write the event to the user's Google Calendar
            write_an_event(
                event_name, event_description, location, event_start, event_end
            )
            answer_event_scheduled = answer_when_scheduled(
                event_details, template_when_scheduling, model
            )
            st.chat_message("bot").write(answer_event_scheduled)
        else:
            # If no city name is found, just return the answer
            st.chat_message("bot").write(answer)


def chatbot_in_vscode(
    model: OllamaLLM,
    template: str,
):
    prompt_user = input(" ")
    if prompt_user:
        answer = answer_prompt_no_context(prompt_user, template, model)
        print(answer)


if __name__ == "__main__":
    model = OllamaLLM(model="qwen3:4b")

    template = """You are an expert in retrieving specific information from a conversation with a user.
    Expect a user to ask you for events in a specific city. Retrieve that city from the conversation history
    in the following format: <city_name>. Do no use the characters that enclose the city name in any of your
    response except for the city name, that is do not use "<" or ">" in the rest of your responses except if
    you are going to provide the city name in the specified format.
    Then, I will extract that city name and use it to query the Ticketmaster API so I can give you context
    about the events in that city and you can recommend the best events to the user based on their preferences.
    If the user does not immediately ask for events, you should not provide any information about events or ask
    information, just wait for the user to ask for events in a specific city in a natural way. Don't let them
    know that you are waiting for them to ask for events, just wait for them to ask. Once they show implicit or explicit
    interest in specific event given the city, just provide the following info and nothing more according to the event details provided before so later we can schedule
    the event in the user's Google Calendar. Make sure to follow the format: [event_name, event_description, location, event_start, event_end].
    This is an example of how to format the information:
    ["Test Event", "This is a test event created by the Google Calendar API.", "Monterrey, Nuevo León, México", "2025-06-11T10:00:00-07:00", "2025-06-11T11:00:00-07:00"].
    Prompt: {prompt}
    Answer:
    """

    template_when_events_are_given = """These are the events that Ticketmaster API found for the city for the user.
    Make sure to use the information in the events to recommend the best events to the user based on their preferences.
    Events: {events}
    """

    template_when_scheduling = """You have now scheduled an event in the user's Google Calendar.
    """

    chatbot(model, template, template_when_events_are_given, template_when_scheduling)

    # name_the_next_10_events()
