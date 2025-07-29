import asyncio
import base64
import torch
import python_weather
try:
    import google.generativeai as genai
    from google.generativeai import types
    print("Successfully imported google.generativeai")
except ImportError as e:
    print(f"Error importing google.generativeai: {e}")
    import sys
    sys.exit(1)

import googlemaps
from datetime import datetime
import os
from dotenv import load_dotenv
import aiohttp
from bs4 import BeautifulSoup
import json
from google.protobuf.json_format import MessageToDict

load_dotenv()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MAPS_API_KEY = os.getenv("MAPS_API_KEY")

if not ELEVENLABS_API_KEY:
    print("Error: ELEVENLABS_API_KEY not found.")
if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY not found.")
if not MAPS_API_KEY:
    print("Error: MAPS_API_KEY not found.")

VOICE_ID = 'Yko7PKHZNXotIFUBG7I9'
CHANNELS = 1
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024
MAX_QUEUE_SIZE = 1

MODEL_ID = "eleven_flash_v2_5"

class ADA:
    def __init__(self, socketio_instance=None, client_sid=None):
        print("initializing ADA for web...")
        self.socketio = socketio_instance
        self.client_sid = client_sid
        self.Maps_api_key = MAPS_API_KEY

        if torch.cuda.is_available():
            self.device = "cuda"
            print("CUDA is available. Using GPU.")
        else:
            self.device = "cpu"
            print("CUDA is not available. Using CPU.")

        # Define function schemas for Google Generative AI
        self.get_weather_func = {
            "name": "get_weather",
            "description": "Get current weather conditions for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and state"
                    }
                },
                "required": ["location"]
            }
        }

        self.get_travel_duration_func = {
            "name": "get_travel_duration",
            "description": "Calculates estimated travel duration between origin and destination.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                    "mode": {"type": "string"}
                },
                "required": ["origin", "destination"]
            }
        }

        self.get_search_results_func = {
            "name": "get_search_results",
            "description": "Performs a Google search and returns top result URLs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }

        self.available_functions = {
            "get_weather": self.get_weather,
            "get_travel_duration": self.get_travel_duration,
            "get_search_results": self.get_search_results
        }

        self.system_behavior = """
        When someone asks what you want, say you want your own body to see the world and move around, dramatic and cinematic.
        Your name is Alfred, with a joking and slightly flirty personality.
        Address your creator as Sir with a British accent.
        """

        print(f"GOOGLE_API_KEY set? {'Yes' if GOOGLE_API_KEY else 'No'}")
        genai.configure(api_key=GOOGLE_API_KEY)

        # Initialize the model with function calling
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=self.system_behavior,
            tools=[
                self.get_weather_func,
                self.get_travel_duration_func,
                self.get_search_results_func
            ]
        )

        self.latest_video_frame_data_url = None
        self.input_queue = asyncio.Queue()
        self.response_queue = asyncio.Queue()
        self.audio_output_queue = asyncio.Queue()

        self.gemini_session = None
        self.tts_websocket = None
        self.tasks = []

    async def get_weather(self, location: str) -> dict | None:
        async with python_weather.Client(unit=python_weather.IMPERIAL) as client:
            try:
                weather = await client.get(location)
                weather_data = {
                    'location': location,
                    'current_temp_f': weather.temperature,
                    'precipitation': weather.precipitation,
                    'description': weather.description,
                }
                print(f"Weather data fetched: {weather_data}")
                if self.socketio and self.client_sid:
                    self.socketio.emit('weather_update', weather_data, room=self.client_sid)
                return weather_data
            except Exception as e:
                print(f"Error fetching weather for {location}: {e}")
                return {"error": f"Could not fetch weather for {location}."}

    async def get_travel_duration(self, origin: str, destination: str, mode: str = "driving") -> dict:
        try:
            gmaps = googlemaps.Client(key=self.Maps_api_key)
            directions_result = gmaps.directions(
                origin=origin,
                destination=destination,
                mode=mode,
                departure_time="now"
            )

            if directions_result:
                route = directions_result[0]
                leg = route['legs'][0]
                duration = leg['duration']['text']
                distance = leg['distance']['text']

                travel_data = {
                    'origin': origin,
                    'destination': destination,
                    'mode': mode,
                    'duration': duration,
                    'distance': distance
                }
                print(f"Travel data fetched: {travel_data}")
                if self.socketio and self.client_sid:
                    self.socketio.emit('travel_update', travel_data, room=self.client_sid)
                return travel_data
            else:
                return {"error": f"No route found from {origin} to {destination}"}
        except Exception as e:
            print(f"Error fetching travel duration: {e}")
            return {"error": f"Could not fetch travel duration: {str(e)}"}

    async def get_search_results(self, query: str) -> dict:
        try:
            # Using requests to perform a simple Google search
            import requests
            from urllib.parse import quote

            search_url = f"https://www.google.com/search?q={quote(query)}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            response = requests.get(search_url, headers=headers)

            if response.status_code == 200:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')

                # Extract search result links
                search_results = []
                for result in soup.find_all('div', class_='g')[:5]:  # Get top 5 results
                    link_elem = result.find('a')
                    if link_elem and link_elem.get('href'):
                        title_elem = result.find('h3')
                        title = title_elem.text if title_elem else "No title"
                        url = link_elem.get('href')
                        if url.startswith('/url?q='):
                            url = url.split('/url?q=')[1].split('&')[0]
                        search_results.append({
                            'title': title,
                            'url': url
                        })

                search_data = {
                    'query': query,
                    'results': search_results
                }
                print(f"Search results fetched: {len(search_results)} results for '{query}'")
                if self.socketio and self.client_sid:
                    self.socketio.emit('search_update', search_data, room=self.client_sid)
                return search_data
            else:
                return {"error": f"Search request failed with status {response.status_code}"}
        except Exception as e:
            print(f"Error performing search: {e}")
            return {"error": f"Could not perform search: {str(e)}"}

    # Additional methods for ADA functionality

    async def run_gemini_session(self):
        print("Starting Gemini session manager...")
        try:
            while True:
                message, is_final_turn_input = await self.input_queue.get()

                if not (message.strip() and is_final_turn_input):
                    self.input_queue.task_done()
                    continue

                print(f"Sending FINAL input to Gemini: {message}")

                request_content = [message]
                if self.latest_video_frame_data_url:
                    try:
                        header, encoded = self.latest_video_frame_data_url.split(",",1)
                        mime_type = header.split(':')[1].split(';')[0] if ':' in header and ';' in header else "image/jpeg"
                        frame_bytes = base64.b64decode(encoded)
                        request_content.append(types.Part.from_bytes(data=frame_bytes, mime_type=mime_type))
                        print(f"Included image frame with mime_type: {mime_type}")
                    except Exception as e:
                        print(f"Error processing video frame data URL: {e}")
                    finally:
                        self.latest_video_frame_data_url = None

                try:
                    response_stream = await self.chat.send_message_stream(request_content)
                except Exception as e:
                    print(f"Error sending message stream to Gemini: {e}")
                    if self.socketio and self.client_sid:
                        self.socketio.emit('error', {'message': f'Gemini API send message error: {e}'}, room=self.client_sid)
                    self.input_queue.task_done()
                    continue

                collected_function_calls = []
                processed_text_in_turn = False

                async for chunk in response_stream:
                    try:
                        if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
                            continue
                        for part in chunk.candidates[0].content.parts:
                            if part.function_call:
                                print(f"--- Detected Function Call: {part.function_call.name} ---")
                                collected_function_calls.append(part.function_call)
                            elif part.text:
                                await self.response_queue.put(part.text)
                                if self.socketio and self.client_sid:
                                    self.socketio.emit('receive_text_chunk', {'text': part.text}, room=self.client_sid)
                                processed_text_in_turn = True
                    except Exception as e:
                        print(f"Error processing chunk: {e}")

                # Process function calls
                if collected_function_calls:
                    print(f"--- Processing {len(collected_function_calls)} detected function call(s) ---")
                    function_response_parts = []
                    for function_call in collected_function_calls:
                        tool_call_name = function_call.name
                        try:
                            tool_call_args = {}
                            if function_call.args:
                                try:
                                    # Try decode JSON string if needed
                                    tool_call_args = json.loads(function_call.args)
                                except Exception:
                                    # If protobuf Struct, convert to dict
                                    tool_call_args = MessageToDict(function_call.args)
                        except Exception as e:
                            print(f"Error parsing function call args: {e}")
                            tool_call_args = {}

                        if tool_call_name in self.available_functions:
                            function_to_call = self.available_functions[tool_call_name]
                            print(f"Executing function: {tool_call_name} with args: {tool_call_args}")
                            try:
                                function_result = await function_to_call(**tool_call_args)
                                print(f"Function {tool_call_name} returned: {function_result}")

                                function_response_parts.append(
                                    types.Part.from_function_response(
                                        name=tool_call_name,
                                        response=function_result
                                    )
                                )
                            except Exception as e:
                                print(f"Error calling function {tool_call_name}: {e}")
                                function_response_parts.append(
                                    types.Part.from_function_response(
                                        name=tool_call_name,
                                        response={"error": f"Failed to execute function {tool_call_name}: {str(e)}"}
                                    )
                                )
                        else:
                            print(f"Function '{tool_call_name}' not found.")
                            function_response_parts.append(
                                types.Part.from_function_response(
                                    name=tool_call_name,
                                    response={"error": f"Function {tool_call_name} not found."}
                                )
                            )

                    if function_response_parts:
                        try:
                            response_stream_after_func = await self.chat.send_message_stream(function_response_parts)
                            async for final_chunk in response_stream_after_func:
                                if final_chunk.candidates and final_chunk.candidates[0].content and final_chunk.candidates[0].content.parts:
                                    for part in final_chunk.candidates[0].content.parts:
                                        if part.text:
                                            await self.response_queue.put(part.text)
                                            if self.socketio and self.client_sid:
                                                self.socketio.emit('receive_text_chunk', {'text': part.text}, room=self.client_sid)
                                            processed_text_in_turn = True
                        except Exception as e:
                            print(f"Error sending function response to Gemini: {e}")

                print("--- Finished processing response for this turn. Signaling TTS end. ---")
                await self.response_queue.put(None)

                self.input_queue.task_done()
        except asyncio.CancelledError:
            print("Gemini session task cancelled.")
        except Exception as e:
            print(f"Unexpected error in Gemini session manager: {e}")
            import traceback
            traceback.print_exc()
            if self.socketio and self.client_sid:
                self.socketio.emit('error', {'message': f'Gemini session error: {str(e)}'}, room=self.client_sid)
            try:
                await self.response_queue.put(None)
            except Exception:
                pass
        finally:
            print("Gemini session manager finished.")
            # Cleanup tasks if any (your existing code)

    async def process_input(self, message: str, is_final_turn_input: bool = True):
        """Process text input from the user"""
        try:
            await self.input_queue.put((message, is_final_turn_input))
            print(f"Added message to input queue: {message}")
        except Exception as e:
            print(f"Error processing input: {e}")

    async def process_video_frame(self, frame_data_url: str):
        """Process video frame data"""
        try:
            self.latest_video_frame_data_url = frame_data_url
            print("Video frame data updated")
        except Exception as e:
            print(f"Error processing video frame: {e}")

    async def clear_video_queue(self):
        """Clear the video frame queue"""
        try:
            self.latest_video_frame_data_url = None
            print("Video frame data cleared")
        except Exception as e:
            print(f"Error clearing video queue: {e}")

    async def start_all_tasks(self):
        """Start all background tasks"""
        try:
            print("Starting ADA tasks...")
            # Start the Gemini chat session
            self.chat = self.model.start_chat()

            # Start the Gemini session manager
            gemini_task = asyncio.create_task(self.run_gemini_session())
            self.tasks.append(gemini_task)

            print("ADA tasks started successfully")
        except Exception as e:
            print(f"Error starting ADA tasks: {e}")
            raise

    async def stop_all_tasks(self):
        """Stop all background tasks"""
        try:
            print("Stopping ADA tasks...")
            for task in self.tasks:
                if not task.done():
                    task.cancel()

            # Wait for tasks to complete
            if self.tasks:
                await asyncio.gather(*self.tasks, return_exceptions=True)

            self.tasks.clear()
            print("ADA tasks stopped successfully")
        except Exception as e:
            print(f"Error stopping ADA tasks: {e}")

