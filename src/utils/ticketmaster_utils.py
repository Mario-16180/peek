import requests


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
                "genre": event["classifications"][0]["genre"].get("name", "N/A"),
                "subGenre": event["classifications"][0]["subGenre"].get("name", "N/A"),
            }
            events.append(event_info)

        return events

    except requests.exceptions.RequestException as e:
        return f"API Error: {str(e)}"
    except ValueError:
        return "Error parsing JSON response"
    except KeyError:
        return "Unexpected response format"
