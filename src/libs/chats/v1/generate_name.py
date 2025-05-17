import asyncio
import re

from agno.agent import Agent
from agno.models.deepseek import DeepSeek

from config.settings import settings

# Use a simplified prompt that works for any input
agent = Agent(
  model=DeepSeek(
    id="deepseek-chat",
    api_key=settings.deepseek_api_key,
  ),
  name="Chat Name Generator",
  instructions=(
    "You are a research assistant that generates concise 2 to 5 words titles from a given message."
    "If the message contains any greeting then do not generate the title just return 'New Chat'."
  ),
  expected_output="A concise 2 to 5 words title",
)


async def generate_chat_name(user_message: str) -> str:
  """
  Generate a chat session name based on the user's input.

  Args:
      user_message (str): The user's message or input to base the chat name on.

  Returns:
      str: The generated chat name, or a default name i.e. 'New Chat' if generation fails.
  """
  try:
    # Extract first 500 characters to focus on key content
    short_message = user_message[:500] + ("..." if len(user_message) > 500 else "")

    # Run the agent to generate a chat name
    response = await agent.arun(short_message)
    clean_title = response.content.strip()

    # Extract only the title if there's any extra text
    # Pattern to match: text that looks like a title (capitalized words)
    title_patterns = [
      r"^([A-Z][a-z0-9]+(?: [A-Za-z0-9]+){1,6})$",  # Simple title: 2-7 words, first word capitalized
      r"title:?\s*([A-Za-z0-9 ]{2,30})",  # "Title: Something"
      r"^[\"'](.+)[\"']$",  # Text in quotes
      r"^(.{2,30})$",  # Simple short text (2-30 chars)
    ]

    for pattern in title_patterns:
      match = re.search(pattern, clean_title)
      if match:
        extracted = match.group(1).strip()
        if extracted:
          return extracted

    # If we can't extract a clear title, but there's some text that's not too long, use it
    if clean_title and len(clean_title) < 40:
      return clean_title

    # Extract key nouns from the input as a last resort
    first_line = user_message.split("\n")[0].strip()
    # Look for capitalized phrases or important words
    important_phrase = re.search(r"([A-Z][a-z]+(?: [A-Z][a-z]+){1,3})", first_line)
    if important_phrase:
      return important_phrase.group(0)

    # If everything else fails, extract first 2-3 significant words
    words = [
      w
      for w in re.findall(r"\b[A-Za-z]{3,}\b", first_line)
      if w.lower() not in ("the", "and", "for", "with", "this", "that", "from", "what", "how", "when", "where")
    ]
    if words and len(words) >= 2:
      return " ".join(words[:3]).title()

    # Absolute last resort
    return "New Chat"

  except Exception as e:
    print(f"Error generating chat name: {e}")
    return "New Chat"


### For testing Purpose ###
async def main():
  # Example usage
  user_input = """Document formats refer to the various methods of encoding documents for storage or transmission. Here are some common document formats:

1. **PDF (Portable Document Format):**
   - Widely used for documents that require consistent formatting across different devices and platforms.
   - Maintains the original layout and cannot be easily altered without leaving a trace.

2. **DOC/DOCX:**
   - Microsoft Word document formats.
   - Widely used for text documents which can include formatted text, images, and other media.
   - DOCX is the newer, XML-based version that offers improved file compression and data recovery.

3. **TXT:**
   - Plain text format with no formatting.
   - Easily readable by any text editor and widely used for basic text files.

4. **ODT (OpenDocument Text):**
   - An open-source word processing format used primarily by OpenOffice and LibreOffice.
   - Similar in functionality to DOC/DOCX with support for text, images, and formatting.

5. **RTF (Rich Text Format):**
   - A cross-platform document format developed by Microsoft.
   - Supports text with basic formatting, making it more versatile than plain text.

6. **HTML (HyperText Markup Language):**
   - The standard markup language for creating web pages.
   - Allows for text, images, links, and multimedia content to be formatted for web browsers.

7. **XLS/XLSX:**
   - Microsoft Excel spreadsheet formats.
   - Used for storing data in tables with support for formulas, graphs, and macros.

8. **PPT/PPTX:**
   - Microsoft PowerPoint presentation formats.
   - Used for creating slideshows with text, images, videos, and animations.

If you can specify or describe a document, I can provide more detailed information about its format or contents.
  """  # noqa: E501
  chat_name = await generate_chat_name(user_input)
  print(chat_name)  # Directly print the chat name


if __name__ == "__main__":
  asyncio.run(main())
