import asyncio
import os
from tempfile import NamedTemporaryFile
from typing import Optional, Union

import mimetypes
import httpx

from openai import AsyncOpenAI

from common.logger import log as logger
from config.settings import settings


client = AsyncOpenAI(api_key=settings.openai_api_key)


async def transcribe(source: Union[bytes, str], content_type: Optional[str] = None, language: str = "en-US") -> str:
  """Transcribe audio content to text.

  Args:
      source: Either binary audio data or a URL to an audio file
      content_type: The MIME type of the audio content (e.g., "audio/wav", "audio/mp3")
                    Optional if a URL is provided and the content type can be inferred
      language: Language code for transcription (default: "en-US")

  Returns:
      The transcribed text
  """
  temp_path = None
  try:
    # Initialize audio_content as bytes
    audio_content = None

    # Handle URL input
    if isinstance(source, str) and (source.startswith("http://") or source.startswith("https://")):
      async with httpx.AsyncClient() as http_client:
        response = await http_client.get(source)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses

        # Try to get content type from response headers if not provided
        if content_type is None:
          content_type = response.headers.get("content-type")

        # If still not found, try to guess from URL
        if content_type is None or not content_type.startswith("audio/"):
          guessed_type = mimetypes.guess_type(source)[0]
          if guessed_type and guessed_type.startswith("audio/"):
            content_type = guessed_type
          else:
            raise ValueError("Could not determine audio content type from URL. Please provide content_type.")

        audio_content = response.content
    # Handle bytes input
    elif isinstance(source, bytes):
      audio_content = source
    else:
      raise ValueError("Source must be either a URL or bytes data")

    if not content_type:
      raise ValueError("content_type is required when providing binary audio data")

    # Ensure audio_content is bytes
    if not isinstance(audio_content, bytes):
      raise ValueError("audio_content must be bytes at this point")

    # Extract the file extension from the content type
    file_extension = content_type.split("/")[1].lower()

    # Ensure we have a valid extension for OpenAI
    valid_extensions = ["flac", "m4a", "mp3", "mp4", "mpeg", "mpga", "oga", "ogg", "wav", "webm"]

    if file_extension not in valid_extensions:
      # Try to map the content type to a valid extension
      content_type_to_ext = {
        "audio/wave": "wav",
        "audio/x-wav": "wav",
        "audio/vnd.wave": "wav",
        "audio/mpeg3": "mp3",
        "audio/x-mpeg-3": "mp3",
        "audio/webm": "webm",
        "audio/ogg": "ogg",
      }

      file_extension = content_type_to_ext.get(content_type.lower(), "mp3")
      logger.debug(f"Mapped content type {content_type} to extension {file_extension}")

    # Create a temporary file
    with NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
      temp_file.write(audio_content)
      temp_file.flush()
      temp_path = temp_file.name

    logger.debug(f"Temporary file created at: {temp_path} with extension {file_extension}")

    # Verify the file exists and has content
    file_size = os.path.getsize(temp_path)
    logger.debug(f"File size: {file_size} bytes")

    if file_size == 0:
      raise ValueError("Audio file is empty")

    # Translation of the audio to english irrespective of the language

    # Open the temporary file and send it to OpenAI's Whisper API
    # with open(temp_path, "rb") as audio_file:
    #   # Make sure we're at the start of the file
    #   audio_file.seek(0)

    #   # For debugging, read a small chunk to verify the file isn't empty
    #   first_bytes = audio_file.read(16)
    #   audio_file.seek(0)  # Reset to beginning

    #   logger.info(f"First few bytes of file: {first_bytes.hex()[:32]}")

    #   # First detect the language if not English
    #   if language.lower() != "en-us" and language.lower() != "en":
    #     # Auto-detect the language
    #     detect_response = await client.audio.transcriptions.create(file=audio_file, model="whisper-1", response_format="verbose_json")
    #     audio_file.seek(0)  # Reset to beginning for next API call

    #     detected_language = detect_response.language
    #     logger.info(f"Detected language: {detected_language}")
    #     # If non-English language detected, use translation task
    #     if detected_language.lower() != "english":
    #       transcription = await client.audio.translations.create(
    #         file=audio_file,
    #         model="whisper-1",
    #         response_format="text",
    #       )
    #     else:
    #       # If English detected, use standard transcription
    #       transcription = await client.audio.transcriptions.create(
    #         file=audio_file,
    #         model="whisper-1",
    #         language="en",
    #         response_format="text"
    #       )
    #   else:
    #     # For English input, use standard transcription
    #     transcription = await client.audio.transcriptions.create(
    #       file=audio_file,
    #       model="whisper-1",
    #       language="en",
    #       response_format="text"
    #     )

    # Transliterate the audio to english

    # Open the temporary file and send it to OpenAI's Whisper API
    with open(temp_path, "rb") as audio_file:
      audio_file.seek(0)

      # For debugging, read a small chunk to verify the file isn't empty
      first_bytes = audio_file.read(16)
      audio_file.seek(0)  # Reset to beginning

      logger.debug(f"First few bytes of file: {first_bytes.hex()[:32]}")

      # Always use transcription with English as the output language
      transcription = await client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-1",
        language="en",  # Force English script output
        response_format="text",
      )
      return transcription

  except httpx.HTTPStatusError as e:
    raise ValueError(f"Error downloading audio from URL: {e}")
  except Exception as e:
    logger.error(f"Transcription error: {str(e)}")
    raise e
  finally:
    # Clean up by removing the temporary file
    if temp_path and os.path.exists(temp_path):
      os.unlink(temp_path)


# async def transcribe(source: Union[str, bytes], language: str = "en-US", content_type: Optional[str] = None) -> str:
#   """
#   Asynchronously transcribe audio to text from a local file path, URL, or bytes data.
#   Supports various audio formats by converting them to WAV using pydub.

#   Args:
#       source (Union[str, bytes]): Path to local audio file, URL to audio file, or raw audio bytes
#       language (str): Language code for transcription (default: "en-US")
#       content_type (Optional[str]): MIME type of the audio when source is bytes (e.g., "audio/mp3")

#   Returns:
#       str: Transcribed text or error message
#   """
#   recognizer = sr.Recognizer()
#   temp_files: List[str] = []

#   try:
#     # Handle bytes input
#     if isinstance(source, bytes):
#       if not content_type:
#         return "Error: content_type is required when providing binary audio data"

#       # Extract file extension from content type
#       if content_type.startswith("audio/"):
#         file_ext = f".{content_type.split('/')[1]}"
#       else:
#         return "Error: Invalid content type. Must start with 'audio/'"

#       # Save bytes data to a temporary file
#       temp_source = NamedTemporaryFile(delete=False, suffix=file_ext)
#       temp_source.write(source)
#       temp_source.close()
#       temp_files.append(temp_source.name)
#       audio_path: str = temp_source.name

#     # Handle URL input
#     elif isinstance(source, str) and source.startswith(("http://", "https://")):
#       # Download the file asynchronously
#       try:
#         async with httpx.AsyncClient(timeout=30.0) as client:
#           response = await client.get(source)
#           response.raise_for_status()

#         # Get content and extract filename correctly
#         content: bytes = response.content

#         # Parse URL to extract just the path without query parameters
#         parsed_url = urlparse(source)
#         path_without_query: str = unquote(parsed_url.path)
#         filename: str = os.path.basename(path_without_query)

#         # Get extension from filename
#         file_ext: str = os.path.splitext(filename)[1].lower()
#         if not file_ext:
#           # If no extension found, try to get it from content-type
#           content_type_header: str = response.headers.get("content-type", "")
#           if content_type_header.startswith("audio/"):
#             file_ext = f".{content_type_header.split('/')[1]}"
#           else:
#             file_ext = ".mp3"  # Default to mp3 if extension can't be determined

#         # Save to temporary file with proper extension
#         temp_source = NamedTemporaryFile(delete=False, suffix=file_ext)
#         temp_source.write(content)
#         temp_source.close()
#         temp_files.append(temp_source.name)
#         audio_path: str = temp_source.name
#       except httpx.HTTPError as e:
#         return f"Error downloading audio file: {str(e)}"
#     # Handle local file path
#     elif isinstance(source, str):
#       if not os.path.exists(source):
#         return f"Error: File {source} does not exist."
#       audio_path = source
#     else:
#       return "Error: source must be a file path, URL, or bytes"

#     # Convert audio to WAV format if it's not already
#     file_ext = os.path.splitext(audio_path)[1].lower()

#     if file_ext not in [".wav", ".aiff", ".flac"]:
#       # Use pydub to convert the file
#       try:
#         audio: AudioSegment
#         if file_ext == ".mp3":
#           audio = AudioSegment.from_mp3(audio_path)
#         elif file_ext == ".ogg":
#           audio = AudioSegment.from_ogg(audio_path)
#         elif file_ext == ".m4a":
#           audio = AudioSegment.from_file(audio_path, format="m4a")
#         else:
#           # Try generic loading
#           audio = AudioSegment.from_file(audio_path)

#         # Export as WAV
#         wav_temp = NamedTemporaryFile(delete=False, suffix=".wav")
#         wav_temp.close()
#         temp_files.append(wav_temp.name)
#         audio.export(wav_temp.name, format="wav")
#         audio_path = wav_temp.name
#       except Exception as e:
#         return f"Error converting audio format: {str(e)}"

#     # Process the audio file in a separate thread using run_in_executor
#     # since SpeechRecognition is blocking
#     loop = asyncio.get_event_loop()

#     def recognize_audio() -> str:
#       with sr.AudioFile(audio_path) as audio_source:
#         audio_data = recognizer.record(audio_source)
#         return recognizer.recognize_google(audio_data, language=language)

#     text: str = await loop.run_in_executor(None, recognize_audio)
#     return text

#   except sr.RequestError as e:
#     return f"API error: {str(e)}"
#   except sr.UnknownValueError:
#     return "Could not understand audio"
#   except Exception as e:
#     return f"Error: {str(e)}"
#   finally:
#     # Clean up temp files if they were created
#     for temp_file in temp_files:
#       if os.path.exists(temp_file):
#         os.unlink(temp_file)


if __name__ == "__main__":
  # Test URL transcription
  test = asyncio.run(
    transcribe(
      "https://s3.backendly.io/chats/3b23b62c-ea1b-464f-a1c3-6c9554017750/d35da558-8498-41b2-9e15-ac9b7756d030/harvard%20%281%29.wav?AWSAccessKeyId=OdA2IVj2d8LMclBd2LFO&Signature=pQ69d67%2FDsq2V5ROWgbfMngQTsM%3D&Expires=1747018272"
    )
  )
  print(test)

  # Uncomment to test bytes transcription
  """
  # Example bytes transcription
  import requests
  audio_url = "https://s3.backendly.io/chats/3b23b62c-ea1b-464f-a1c3-6c9554017750/d35da558-8498-41b2-9e15-ac9b7756d030/harvard%20%281%29.wav"
  response = requests.get(audio_url)
  audio_bytes = response.content

  test_bytes = asyncio.run(
    transcribe(
      source=audio_bytes,
      content_type="audio/wav"
    )
  )
  print(f"Bytes transcription: {test_bytes}")
  """
