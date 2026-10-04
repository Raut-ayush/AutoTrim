import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types


# Load .env from project root
load_dotenv()


class GeminiProvider:
    """
    Gemini API provider for AutoTrim.

    Uses Google's official google-genai SDK.
    """

    def __init__(
        self,
        model="gemini-3.8-flash",
        max_retries=4,
    ):
        self.model = model
        self.max_retries = max_retries

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable is not set."
            )

        self.client = genai.Client(api_key=api_key)

    def generate(
        self,
        prompt,
        system=None,
        temperature=0.2,
    ):
        """
        Generate a response using Gemini.

        Automatically retries transient API errors such as:
        - 503 UNAVAILABLE
        - 429 RESOURCE_EXHAUSTED
        """

        start = time.perf_counter()

        config = types.GenerateContentConfig(
            temperature=temperature,
        )

        if system:
            config.system_instruction = system

        last_error = None

        for attempt in range(self.max_retries + 1):

            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )

                elapsed = time.perf_counter() - start

                return {
                    "model": self.model,
                    "response": response.text.strip(),
                    "total_time": elapsed,
                    "prompt_tokens": None,
                    "output_tokens": None,
                }

            except Exception as e:

                last_error = e

                error_text = str(e)

                is_retryable = (
                    "503" in error_text
                    or "UNAVAILABLE" in error_text
                    or "429" in error_text
                    or "RESOURCE_EXHAUSTED" in error_text
                    or "500" in error_text
                    or "502" in error_text
                    or "504" in error_text
                )

                if not is_retryable:
                    raise

                if attempt >= self.max_retries:
                    raise RuntimeError(
                        f"Gemini request failed after "
                        f"{self.max_retries + 1} attempts.\n"
                        f"Last error: {last_error}"
                    ) from last_error

                # Exponential backoff:
                # 2s → 4s → 8s → 16s
                wait_time = 2 ** (attempt + 1)

                print(
                    f"Gemini temporary error "
                    f"(attempt {attempt + 1}/{self.max_retries + 1}). "
                    f"Retrying in {wait_time}s..."
                )

                time.sleep(wait_time)

        raise RuntimeError(
            f"Gemini generation failed: {last_error}"
        )

    def check_connection(self):
        """
        Check whether Gemini API is accessible.
        """

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents="Reply with exactly: OK",
            )

            return response.text.strip() == "OK"

        except Exception:
            return False