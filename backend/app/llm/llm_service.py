import json
import os
from pathlib import Path
from socket import timeout as SocketTimeout
from urllib.parse import urlsplit
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv


BACKEND_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(BACKEND_ROOT / ".env")

DEFAULT_BASE_URL = "http://127.0.0.1:11434"


class LLMUnavailableError(RuntimeError):
    pass


class OllamaRuntime:
    def __init__(self, base_url: str, timeout: float = 3.0):
        self.base_url = base_url.rstrip("/")
        parsed_url = urlsplit(self.base_url)
        if parsed_url.scheme not in {"http", "https"} or parsed_url.hostname not in {
            "localhost",
            "127.0.0.1",
            "::1",
        }:
            raise ValueError("Ollama base URL must point to a loopback address")
        if parsed_url.username or parsed_url.password:
            raise ValueError("Credentials must not be embedded in the Ollama base URL")
        if timeout <= 0:
            raise ValueError("Ollama health-check timeout must be greater than zero")
        self.timeout = timeout

    def check_model(self, model: str) -> dict:
        if not model:
            return {"available": False, "detail": "LLM_MODEL is empty."}
        request = Request(f"{self.base_url}/api/tags", method="GET")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read())
        except HTTPError as error:
            return {
                "available": False,
                "detail": f"Ollama returned HTTP {error.code} while listing local models.",
            }
        except (TimeoutError, SocketTimeout):
            return {
                "available": False,
                "detail": f"Ollama did not respond within {self.timeout:g} seconds while listing models.",
            }
        except (URLError, ConnectionError, OSError) as error:
            reason = getattr(error, "reason", error)
            return {
                "available": False,
                "detail": f"Cannot connect to Ollama at {self.base_url}: {reason}. Start the local Ollama service.",
            }
        except (ValueError, TypeError):
            return {"available": False, "detail": "Ollama returned an invalid model-list response."}

        models = result.get("models", []) if isinstance(result, dict) else []
        if not isinstance(models, list):
            return {"available": False, "detail": "Ollama returned an invalid model-list response."}
        names = [item.get("name") for item in models if isinstance(item, dict)]
        if model not in names:
            return {
                "available": False,
                "detail": f"Model {model!r} is not installed locally. Run `ollama pull {model}`.",
                "installed_models": names,
            }
        return {"available": True, "detail": "Ollama is reachable and the configured model is installed."}

    def is_available(self, model: str) -> bool:
        return self.check_model(model)["available"]

    def generate(
        self,
        model: str,
        prompt: str,
        context: str,
        system_prompt: str,
        timeout: float,
    ) -> str:
        user_content = f"Question:\n{prompt}\n\nRetrieved context:\n{context}"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "stream": False,
            "options": {"temperature": 0},
        }
        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read())
        except HTTPError as error:
            try:
                body = json.loads(error.read().decode("utf-8", errors="replace"))
            except (ValueError, OSError):
                body = {}
            detail = body.get("error") if isinstance(body, dict) else None
            if error.code == 404:
                message = detail or f"Model {model!r} is unavailable to Ollama. Run `ollama pull {model}`."
            else:
                message = detail or f"Ollama returned HTTP {error.code} during generation."
            raise LLMUnavailableError(str(message)) from error
        except (TimeoutError, SocketTimeout) as error:
            raise LLMUnavailableError(
                f"Ollama generation timed out after {timeout:g} seconds."
            ) from error
        except URLError as error:
            reason = getattr(error, "reason", error)
            raise LLMUnavailableError(
                f"Cannot connect to Ollama at {self.base_url}: {reason}. Start the local Ollama service."
            ) from error
        except (ConnectionError, OSError) as error:
            raise LLMUnavailableError(
                f"Ollama request failed at {self.base_url}: {error}."
            ) from error
        except (ValueError, TypeError) as error:
            raise LLMUnavailableError("Ollama returned an invalid generation response.") from error

        message = result.get("message") if isinstance(result, dict) else None
        answer = message.get("content") if isinstance(message, dict) else None
        if not isinstance(answer, str) or not answer.strip():
            raise LLMUnavailableError("Local LLM returned an empty response.")
        return answer.strip()


class LLMService:
    """Configurable local-generation service; business logic uses this interface."""

    def __init__(
        self,
        provider: str | None = None,
        model_name: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        runtime=None,
    ):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "ollama")).lower()
        self.model_name = (
            model_name if model_name is not None else os.getenv("LLM_MODEL", "")
        ).strip()
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.timeout = (
            timeout
            if timeout is not None
            else float(os.getenv("LLM_TIMEOUT_SECONDS", "120"))
        )
        if self.timeout <= 0:
            raise ValueError("LLM_TIMEOUT_SECONDS must be greater than zero")
        self.availability_timeout = float(os.getenv("LLM_AVAILABILITY_TIMEOUT_SECONDS", "3"))
        if self.availability_timeout <= 0:
            raise ValueError("LLM_AVAILABILITY_TIMEOUT_SECONDS must be greater than zero")
        self.runtime = runtime
        if self.runtime is None and self.provider == "ollama":
            self.runtime = OllamaRuntime(self.base_url, timeout=self.availability_timeout)

    def is_available(self) -> bool:
        if not self.model_name or self.runtime is None:
            return False
        return self.runtime.is_available(self.model_name)

    def health_check(self) -> dict:
        if self.provider != "ollama":
            return {
                "status": "unavailable",
                "provider": self.provider,
                "model": self.model_name or None,
                "available": False,
                "detail": f"Unsupported local LLM provider: {self.provider}.",
            }
        if not self.model_name or self.runtime is None:
            return {
                "status": "unavailable",
                "provider": self.provider,
                "model": self.model_name or None,
                "available": False,
                "detail": "LLM_MODEL is empty; local answer generation is disabled.",
            }
        if hasattr(self.runtime, "check_model"):
            check = self.runtime.check_model(self.model_name)
        else:
            available = self.runtime.is_available(self.model_name)
            check = {"available": available, "detail": "Ollama model check completed."}
        return {
            "status": "ok" if check["available"] else "unavailable",
            "provider": self.provider,
            "model": self.model_name,
            "available": bool(check["available"]),
            "base_url": self.base_url,
            "detail": check.get("detail"),
            **({"installed_models": check["installed_models"]} if "installed_models" in check else {}),
        }

    def generate(
        self,
        prompt: str,
        context: str = "",
        system_prompt: str = "",
    ) -> str:
        if not self.model_name or self.runtime is None:
            raise LLMUnavailableError("Local LLM is not configured.")
        if self.provider != "ollama":
            raise LLMUnavailableError(
                f"Unsupported local LLM provider: {self.provider}"
            )
        return self.runtime.generate(
            model=self.model_name,
            prompt=prompt,
            context=context,
            system_prompt=system_prompt,
            timeout=self.timeout,
        )
