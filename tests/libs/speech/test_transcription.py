import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.libs.speech.v1.transcription import transcribe

class TestTranscription:
    """Unit tests for speech transcription functionality."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings with API key."""
        with patch("src.libs.speech.v1.transcription.settings") as mock_settings:
            mock_settings.openai_api_key = "test-api-key"
            yield mock_settings

    @pytest.fixture
    def mock_mimetypes(self):
        """Mock mimetypes module to avoid bytes/str issues."""
        with patch("src.libs.speech.v1.transcription.mimetypes.guess_type") as mock_guess:
            mock_guess.return_value = ("audio/mp3", None)
            yield mock_guess

    @pytest.fixture
    def mock_temp_file(self):
        """Mock NamedTemporaryFile."""
        with patch("src.libs.speech.v1.transcription.NamedTemporaryFile") as mock_temp:
            mock_file = MagicMock()
            mock_file.name = "/tmp/test_audio.mp3"
            mock_temp.return_value.__enter__.return_value = mock_file
            yield mock_temp

    @pytest.fixture
    def mock_open(self):
        """Mock file open operation."""
        with patch("builtins.open", MagicMock()):
            yield

    @pytest.fixture
    def mock_os_operations(self):
        """Mock OS operations."""
        with patch("src.libs.speech.v1.transcription.os.path.getsize") as mock_getsize, \
             patch("src.libs.speech.v1.transcription.os.path.exists") as mock_exists, \
             patch("src.libs.speech.v1.transcription.os.unlink") as mock_unlink:
            mock_getsize.return_value = 1024  # File size
            mock_exists.return_value = True
            mock_unlink.return_value = None
            yield {
                "getsize": mock_getsize,
                "exists": mock_exists,
                "unlink": mock_unlink
            }

    @pytest.fixture
    def mock_httpx_client(self):
        """Mock HTTPX client for URL downloads."""
        with patch("src.libs.speech.v1.transcription.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance
            mock_instance.get = AsyncMock()
            mock_response = MagicMock()
            mock_response.content = b"audio content"
            mock_response.headers = {"content-type": "audio/mp3"}
            mock_instance.get.return_value = mock_response
            yield mock_instance

    @pytest.mark.asyncio
    async def test_transcribe_from_bytes(self, mock_settings, mock_temp_file, mock_open, mock_os_operations):
        """Test transcribing from bytes."""
        # Setup
        audio_bytes = b"test audio content"
        content_type = "audio/wav"

        # Mock the OpenAI API call
        with patch("src.libs.speech.v1.transcription.client.audio.transcriptions.create",
                  new_callable=AsyncMock) as mock_create:
            mock_create.return_value = "Transcribed text"

            # Execute
            result = await transcribe(source=audio_bytes, content_type=content_type)

            # Assert
            assert result == "Transcribed text"
            mock_create.assert_awaited_once()
            mock_os_operations["unlink"].assert_called_once()

    @pytest.mark.asyncio
    async def test_transcribe_from_url(self, mock_settings, mock_httpx_client, mock_temp_file, mock_open, mock_os_operations, mock_mimetypes):
        """Test transcribing from URL."""
        # Setup
        audio_url = "https://example.com/audio.mp3"

        # Mock the OpenAI API call
        with patch("src.libs.speech.v1.transcription.client.audio.transcriptions.create",
                  new_callable=AsyncMock) as mock_create:
            mock_create.return_value = "Transcribed text"

            # Execute
            result = await transcribe(source=audio_url)

            # Assert
            assert result == "Transcribed text"
            mock_httpx_client.get.assert_called_with(audio_url)
            mock_create.assert_awaited_once()
            mock_os_operations["unlink"].assert_called_once()

    @pytest.mark.asyncio
    async def test_transcribe_missing_content_type(self, mock_settings):
        """Test error when content_type is missing for bytes input."""
        # Setup
        audio_bytes = b"test audio content"

        # Execute and Assert
        with pytest.raises(ValueError) as excinfo:
            await transcribe(source=audio_bytes)

        assert "content_type is required" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_transcribe_invalid_source(self):
        """Test error when source is invalid."""
        # Execute and Assert
        with pytest.raises(ValueError) as excinfo:
            await transcribe(source=123)  # Not bytes or string

        assert "Source must be either a URL or bytes data" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_transcribe_url_content_type_inference(self, mock_settings, mock_httpx_client, mock_temp_file,
                                                         mock_open, mock_os_operations, mock_mimetypes):
        """Test content type inference from URL."""
        # Setup
        audio_url = "https://example.com/audio.mp3"
        mock_httpx_client.get.return_value.headers = {}  # No content-type header

        # Mock the OpenAI API call
        with patch("src.libs.speech.v1.transcription.client.audio.transcriptions.create",
                  new_callable=AsyncMock) as mock_create:
            mock_create.return_value = "Transcribed text"

            # Execute
            result = await transcribe(source=audio_url)

            # Assert
            assert result == "Transcribed text"
            # Should have guessed content type from URL extension
            assert mock_temp_file.call_args[1]['suffix'] == ".mp3"

    @pytest.mark.asyncio
    async def test_transcribe_url_download_error(self, mock_httpx_client):
        """Test error handling for URL download failures."""
        # Setup
        audio_url = "https://example.com/audio.mp3"
        mock_httpx_client.get.side_effect = Exception("Download error")

        # Execute and Assert
        with pytest.raises(Exception) as excinfo:
            await transcribe(source=audio_url)

        assert "Download error" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_transcribe_empty_file(self, mock_settings, mock_temp_file, mock_httpx_client, mock_os_operations, mock_mimetypes):
        """Test error handling for empty files."""
        # Setup
        audio_url = "https://example.com/audio.mp3"
        mock_os_operations["getsize"].return_value = 0  # Empty file

        # Execute and Assert
        with pytest.raises(ValueError) as excinfo:
            await transcribe(source=audio_url)

        assert "Audio file is empty" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_transcribe_api_error(self, mock_settings, mock_temp_file, mock_httpx_client, mock_os_operations, mock_mimetypes, mock_open):
        """Test error handling for OpenAI API errors."""
        # Setup
        audio_url = "https://example.com/audio.mp3"
        mock_file = mock_temp_file.return_value.__enter__.return_value
        mock_file.name = "/tmp/test_audio.mp3"

        # Mock the OpenAI API call with an error
        with patch("src.libs.speech.v1.transcription.client.audio.transcriptions.create",
                  new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = Exception("API Error")

            # Execute and Assert
            with pytest.raises(Exception) as excinfo:
                await transcribe(source=audio_url)

            assert "API Error" in str(excinfo.value)
            mock_os_operations["unlink"].assert_called_once()  # Ensure temp file cleanup

    @pytest.mark.asyncio
    async def test_transcribe_with_language_parameter(self, mock_settings, mock_temp_file, mock_httpx_client,
                                                      mock_os_operations, mock_mimetypes, mock_open):
        """Test transcribing with language parameter (note: currently always uses 'en')."""
        # Setup
        audio_url = "https://example.com/audio.mp3"
        language = "es-ES"
        mock_file = mock_temp_file.return_value.__enter__.return_value
        mock_file.name = "/tmp/test_audio.mp3"

        # Mock the OpenAI API call
        with patch("src.libs.speech.v1.transcription.client.audio.transcriptions.create",
                  new_callable=AsyncMock) as mock_create:
            mock_create.return_value = "Transcribed text"

            # Execute
            result = await transcribe(source=audio_url, language=language)

            # Assert
            assert result == "Transcribed text"
            # Function always uses "en" regardless of input language parameter
            mock_create.assert_awaited_once()
            assert mock_create.call_args[1].get('language') == "en"

    @pytest.mark.asyncio
    async def test_transcribe_content_type_mapping(self, mock_settings, mock_temp_file, mock_open, mock_os_operations):
        """Test content type mapping for unusual extensions."""
        # Setup
        audio_bytes = b"test audio content"
        content_type = "audio/x-wav"  # Non-standard content type

        # Mock the OpenAI API call
        with patch("src.libs.speech.v1.transcription.client.audio.transcriptions.create",
                  new_callable=AsyncMock) as mock_create:
            mock_create.return_value = "Transcribed text"

            # Execute
            result = await transcribe(source=audio_bytes, content_type=content_type)

            # Assert
            assert result == "Transcribed text"
            # Should have mapped to .wav extension
            assert mock_temp_file.call_args[1]['suffix'] == ".wav"

    @pytest.mark.asyncio
    async def test_transcribe_cleanup_on_error(self, mock_settings, mock_temp_file, mock_httpx_client, mock_os_operations, mock_mimetypes):
        """Test temp file cleanup on error."""
        # Setup
        audio_url = "https://example.com/audio.mp3"
        mock_os_operations["getsize"].side_effect = Exception("File error")

        # Execute and Assert
        with pytest.raises(Exception):
            await transcribe(source=audio_url)

        # Verify temp file cleanup was attempted
        mock_os_operations["unlink"].assert_called_once()