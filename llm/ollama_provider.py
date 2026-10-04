import json
import time
import urllib.request
import urllib.error


class OllamaProvider:
    """
    Minimal Ollama client for AutoTrim.

    Uses Ollama's local HTTP API directly so we don't need
    an additional Python Ollama dependency.
    """

    def __init__(
        self,
        model="qwen2.5-coder:7b",
        host="http://localhost:11434",
    ):
        self.model = model
        self.host = host.rstrip("/")

    def _request(self, endpoint, payload):
        url = f"{self.host}{endpoint}"

        data = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(request, timeout=600) as response:
            return json.loads(response.read().decode("utf-8"))

    def generate(
        self,
        prompt,
        system=None,
        temperature=0.2,
    ):
        """
        Generate a response using Ollama.

        keep_alive=0 tells Ollama to unload the model
        immediately after generation.
        """

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": 0,
            "options": {
                "temperature": temperature,
            },
        }

        if system:
            payload["system"] = system

        start = time.perf_counter()

        response = self._request("/api/generate", payload)

        elapsed = time.perf_counter() - start

        return {
            "model": self.model,
            "response": response.get("response", "").strip(),
            "total_time": elapsed,
            "prompt_tokens": response.get("prompt_eval_count"),
            "output_tokens": response.get("eval_count"),
        }

    def check_connection(self):
        """Check whether Ollama is running."""

        try:
            with urllib.request.urlopen(
                f"{self.host}/api/tags",
                timeout=5,
            ) as response:
                return response.status == 200

        except (urllib.error.URLError, TimeoutError):
            return False