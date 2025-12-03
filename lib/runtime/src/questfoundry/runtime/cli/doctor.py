"""
QuestFoundry Doctor - Diagnostic command for configuration and provider health.

Provides:
- Configuration summary with masked secrets
- Provider status checks (unconfigured / unavailable / ready)
- Spec source validation
- Connectivity tests
"""

import os
import platform
import sys
from enum import Enum
from typing import Any
from urllib import request
from urllib.error import URLError

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from questfoundry.runtime.config import QuestFoundrySettings, get_settings


class ProviderStatus(Enum):
    """Provider availability states."""

    READY = "ready"  # Configured and responding
    UNAVAILABLE = "unavailable"  # Configured but connection failed
    UNCONFIGURED = "unconfigured"  # No API key or URL set


def mask_secret(value: str | None, visible_chars: int = 4) -> str:
    """Mask a secret value, showing only first few characters."""
    if not value:
        return "(not set)"
    if len(value) <= visible_chars:
        return "*" * len(value)
    return value[:visible_chars] + "*" * (len(value) - visible_chars)


def get_env_var_status(var_name: str) -> tuple[bool, str]:
    """Check if environment variable is set and return masked value."""
    value = os.getenv(var_name)
    if value:
        return True, mask_secret(value)
    return False, "(not set)"


class ProviderChecker:
    """Checks provider configuration and connectivity."""

    # Map provider to (env_var_name, check_function)
    PROVIDERS = {
        "anthropic": ("ANTHROPIC_API_KEY", "_check_anthropic"),
        "openai": ("OPENAI_API_KEY", "_check_openai"),
        "google": ("GOOGLE_API_KEY", "_check_google"),
        "ollama": (None, "_check_ollama"),  # No env var required
        "litellm": ("LITELLM_API_BASE", "_check_litellm"),
    }

    def __init__(self, settings: QuestFoundrySettings):
        self.settings = settings
        self.results: dict[str, dict[str, Any]] = {}

    def check_all(self) -> dict[str, dict[str, Any]]:
        """Check all providers and return results."""
        for provider, (env_var, check_method) in self.PROVIDERS.items():
            self.results[provider] = self._check_provider(provider, env_var, check_method)
        return self.results

    def _check_provider(
        self, provider: str, env_var: str | None, check_method: str
    ) -> dict[str, Any]:
        """Check a single provider."""
        result: dict[str, Any] = {
            "status": ProviderStatus.UNCONFIGURED,
            "message": "",
            "details": {},
        }

        # Check if configured
        if env_var:
            value = os.getenv(env_var)
            if not value:
                result["message"] = f"{env_var} not set"
                return result
            result["details"]["credential"] = mask_secret(value)

        # Provider is configured, check connectivity
        try:
            check_fn = getattr(self, check_method)
            check_result = check_fn()
            result["status"] = ProviderStatus.READY
            result["message"] = check_result.get("message", "OK")
            result["details"].update(check_result.get("details", {}))
        except Exception as e:
            result["status"] = ProviderStatus.UNAVAILABLE
            result["message"] = str(e)

        return result

    def _check_anthropic(self) -> dict[str, Any]:
        """Check Anthropic API connectivity."""
        try:
            from anthropic import Anthropic

            client = Anthropic()
            # Minimal API call - just check auth works
            # Using messages.create with max_tokens=1 is cheapest
            client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}],
            )
            return {"message": "API responding", "details": {"model": "claude-3-haiku"}}
        except ImportError as e:
            raise RuntimeError("anthropic package not installed") from e
        except Exception as e:
            error_msg = str(e)
            if "authentication" in error_msg.lower() or "api key" in error_msg.lower():
                raise RuntimeError("Invalid API key") from None
            raise RuntimeError(f"API error: {error_msg[:50]}") from None

    def _check_openai(self) -> dict[str, Any]:
        """Check OpenAI API connectivity."""
        try:
            from openai import OpenAI

            client = OpenAI()
            # List models is a cheap way to verify auth
            list(client.models.list())[:5]  # Just verify we can list
            return {"message": "API responding", "details": {"models_available": True}}
        except ImportError as e:
            raise RuntimeError("openai package not installed") from e
        except Exception as e:
            error_msg = str(e)
            if "authentication" in error_msg.lower() or "api key" in error_msg.lower():
                raise RuntimeError("Invalid API key") from None
            raise RuntimeError(f"API error: {error_msg[:50]}") from None

    def _check_google(self) -> dict[str, Any]:
        """Check Google AI API connectivity."""
        try:
            import google.generativeai as genai

            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            # List models to verify auth
            models = list(genai.list_models())
            return {
                "message": "API responding",
                "details": {"models_available": len(models) > 0},
            }
        except ImportError as e:
            raise RuntimeError("google-generativeai package not installed") from e
        except Exception as e:
            error_msg = str(e)
            if "api key" in error_msg.lower():
                raise RuntimeError("Invalid API key") from None
            raise RuntimeError(f"API error: {error_msg[:50]}") from None

    def _check_ollama(self) -> dict[str, Any]:
        """Check Ollama server connectivity."""
        base_url = self.settings.llm.ollama_host

        try:
            # Check if server is running
            with request.urlopen(f"{base_url}/api/tags", timeout=5) as resp:
                import json

                data = json.loads(resp.read().decode())
                models = [m.get("name", "?") for m in data.get("models", [])]
                if not models:
                    return {
                        "message": "Server running, no models",
                        "details": {"host": base_url, "models": []},
                    }
                return {
                    "message": f"{len(models)} model(s) available",
                    "details": {"host": base_url, "models": models[:5]},
                }
        except URLError:
            raise RuntimeError(f"Cannot connect to {base_url}") from None
        except Exception as e:
            raise RuntimeError(f"Error: {e}") from None

    def _check_litellm(self) -> dict[str, Any]:
        """Check LiteLLM proxy connectivity."""
        api_base = self.settings.llm.litellm_api_base
        if not api_base:
            raise RuntimeError("LITELLM_API_BASE not configured")

        try:
            # Check if proxy is running
            health_url = f"{api_base.rstrip('/')}/health"
            with request.urlopen(health_url, timeout=5):
                return {
                    "message": "Proxy responding",
                    "details": {"api_base": api_base},
                }
        except URLError:
            # Try models endpoint as fallback
            try:
                models_url = f"{api_base.rstrip('/')}/models"
                with request.urlopen(models_url, timeout=5):
                    return {
                        "message": "Proxy responding",
                        "details": {"api_base": api_base},
                    }
            except Exception:
                raise RuntimeError(f"Cannot connect to {api_base}") from None
        except Exception as e:
            raise RuntimeError(f"Error: {e}") from None


def check_spec_status(settings: QuestFoundrySettings) -> dict[str, Any]:
    """Check spec source and availability."""
    from questfoundry.runtime.core.spec_fetcher import get_spec_source_preference

    result: dict[str, Any] = {
        "source": get_spec_source_preference(),
        "status": "unknown",
        "path": None,
    }

    # Check if spec is available based on source
    source = result["source"]

    if source == "bundled" or source == "auto":
        # Check bundled spec
        try:
            from importlib.resources import files

            spec_pkg = files("questfoundry.runtime") / "spec_bundle"
            if spec_pkg.is_dir():
                result["status"] = "available"
                result["path"] = str(spec_pkg)
            else:
                result["status"] = "not found"
        except Exception:
            result["status"] = "not found"

    if source == "download" or (source == "auto" and result["status"] != "available"):
        # Check downloaded spec
        cache_dir = settings.paths.cache_dir
        spec_dir = cache_dir / "spec"
        if spec_dir.exists():
            result["status"] = "available"
            result["path"] = str(spec_dir)
            # Check for version metadata
            metadata_file = spec_dir / ".questfoundry-spec.json"
            if metadata_file.exists():
                import json

                try:
                    metadata = json.loads(metadata_file.read_text())
                    result["version"] = metadata.get("tag", "unknown")
                except Exception:
                    pass

    if source == "local":
        local_path = os.getenv("QF_SPEC_PATH")
        if local_path and os.path.isdir(local_path):
            result["status"] = "available"
            result["path"] = local_path
        else:
            result["status"] = "not found"
            result["path"] = local_path

    return result


def format_config_dump(settings: QuestFoundrySettings) -> str:
    """Format all configuration values with masked secrets."""
    lines = []

    # Runtime settings
    lines.append("[bold]Runtime:[/bold]")
    rt = settings.runtime
    lines.append(f"  max_iterations: {rt.max_iterations}")
    lines.append(f"  max_ping_pong: {rt.max_ping_pong}")
    lines.append(f"  max_role_executions: {rt.max_role_executions}")
    lines.append(f"  execution_reset_threshold: {rt.execution_reset_threshold}")
    lines.append(f"  max_consecutive_role_executions: {rt.max_consecutive_role_executions}")
    lines.append(f"  max_parallel_roles: {rt.max_parallel_roles}")
    lines.append("")

    # LLM settings
    lines.append("[bold]LLM:[/bold]")
    llm = settings.llm
    lines.append(f"  default_provider: {llm.default_provider or 'auto'}")
    lines.append(f"  default_temperature: {llm.default_temperature}")
    lines.append(f"  default_max_tokens: {llm.default_max_tokens}")
    lines.append(f"  ollama_host: {llm.ollama_host}")
    lines.append(f"  ollama_num_ctx: {llm.ollama_num_ctx}")
    if llm.litellm_api_base:
        lines.append(f"  litellm_api_base: {llm.litellm_api_base}")
    if llm.litellm_api_key:
        lines.append(f"  litellm_api_key: {mask_secret(llm.litellm_api_key)}")
    lines.append("")

    # Memory settings
    lines.append("[bold]Memory:[/bold]")
    mem = settings.memory
    lines.append(f"  prompt_error_threshold: {mem.prompt_error_threshold}")
    lines.append(f"  prompt_warning_threshold: {mem.prompt_warning_threshold}")
    lines.append(f"  memory_cap: {mem.memory_cap}")
    lines.append(f"  summarize_messages_threshold: {mem.summarize_messages_threshold}")
    lines.append(f"  summarize_chars_threshold: {mem.summarize_chars_threshold}")
    lines.append(f"  prior_conversation_max_chars: {mem.prior_conversation_max_chars}")
    lines.append("")

    # Paths settings
    lines.append("[bold]Paths:[/bold]")
    paths = settings.paths
    lines.append(f"  project_dir: {paths.project_dir}")
    lines.append(f"  project_id: {paths.project_id}")
    lines.append(f"  cache_dir: {paths.cache_dir}")
    lines.append("")

    # Logging settings
    lines.append("[bold]Logging:[/bold]")
    log = settings.logging
    lines.append(f"  level: {log.level}")
    lines.append(f"  format: {log.format}")
    lines.append("")

    # Network settings
    lines.append("[bold]Network:[/bold]")
    net = settings.network
    lines.append(f"  spec_fetch_timeout: {net.spec_fetch_timeout}")
    lines.append(f"  spec_download_timeout: {net.spec_download_timeout}")

    return "\n".join(lines)


def run_doctor(
    console: Console,
    show_config: bool = False,
    skip_network: bool = False,
    output_json: bool = False,
) -> int:
    """
    Run the doctor diagnostic checks.

    Returns:
        Exit code (0 = all good, 1 = issues found)
    """
    settings = get_settings()
    has_issues = False

    if output_json:
        return _run_doctor_json(settings, skip_network)

    # Header
    console.print()
    console.print(
        Panel(
            "[bold cyan]QuestFoundry Doctor[/bold cyan]\n"
            "Checking configuration and provider connectivity...",
            border_style="cyan",
        )
    )
    console.print()

    # Environment info
    console.print("[bold]Environment:[/bold]")
    console.print(f"  Python: {sys.version.split()[0]}")
    console.print(f"  Platform: {platform.system()} {platform.release()}")
    console.print(f"  Working Dir: {os.getcwd()}")
    console.print()

    # Config sources
    console.print("[bold]Config Sources:[/bold]")
    env_file = ".env"
    if os.path.exists(env_file):
        console.print(f"  [green].[/green] {env_file} (found)")
    else:
        console.print(f"  [dim].[/dim] {env_file} (not found)")

    # Check for QF_ prefixed env vars
    qf_vars = [k for k in os.environ if k.startswith("QF_")]
    if qf_vars:
        console.print(f"  [green].[/green] Environment variables: {len(qf_vars)} QF_* vars set")
    else:
        console.print("  [dim].[/dim] Environment variables: no QF_* vars set")
    console.print()

    # Provider status table
    console.print("[bold]Provider Status:[/bold]")
    console.print()

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("Provider", style="cyan")
    table.add_column("Status")
    table.add_column("Details")

    if skip_network:
        # Just show configuration status without network checks
        for provider, (env_var, _) in ProviderChecker.PROVIDERS.items():
            if env_var:
                is_set, masked = get_env_var_status(env_var)
                if is_set:
                    status = "[yellow]? Configured[/yellow]"
                    details = f"{env_var}={masked}"
                else:
                    status = "[dim]- Unconfigured[/dim]"
                    details = f"{env_var} not set"
            else:
                # Ollama - always "configured"
                status = "[yellow]? Configured[/yellow]"
                details = f"host={settings.llm.ollama_host}"
            table.add_row(provider, status, details)
    else:
        # Full connectivity check
        checker = ProviderChecker(settings)
        results = checker.check_all()

        ready_count = 0
        for provider, result in results.items():
            status_enum = result["status"]
            message = result["message"]
            details = result.get("details", {})

            if status_enum == ProviderStatus.READY:
                status = "[green]+ Ready[/green]"
                ready_count += 1
                # Format details
                detail_parts = []
                if "models" in details:
                    models = details["models"]
                    if models:
                        model_str = ", ".join(models[:3])
                        if len(models) > 3:
                            model_str += f" (+{len(models)-3})"
                        detail_parts.append(f"models: {model_str}")
                if "host" in details:
                    detail_parts.append(f"host: {details['host']}")
                detail_str = "; ".join(detail_parts) if detail_parts else message
            elif status_enum == ProviderStatus.UNAVAILABLE:
                status = "[red]x Unavailable[/red]"
                detail_str = message
                has_issues = True
            else:  # UNCONFIGURED
                status = "[dim]- Unconfigured[/dim]"
                detail_str = message

            table.add_row(provider, status, detail_str)

        console.print(table)
        console.print()

        if ready_count == 0:
            console.print(
                "[red bold]No providers available![/red bold] "
                "Set at least one API key to use QuestFoundry."
            )
            has_issues = True

    console.print(table)
    console.print()

    # Spec status
    console.print("[bold]Spec Status:[/bold]")
    spec_status = check_spec_status(settings)
    source = spec_status["source"]
    status = spec_status["status"]

    if status == "available":
        console.print(f"  [green]+[/green] Source: {source}")
        if "version" in spec_status:
            console.print(f"  [green]+[/green] Version: {spec_status['version']}")
        console.print(f"  [dim]Path: {spec_status['path']}[/dim]")
    else:
        console.print(f"  [red]x[/red] Source: {source} ({status})")
        if spec_status.get("path"):
            console.print(f"  [dim]Expected: {spec_status['path']}[/dim]")
        has_issues = True
    console.print()

    # Config dump
    if show_config:
        console.print("[bold]Current Configuration:[/bold]")
        console.print()
        config_text = format_config_dump(settings)
        console.print(config_text)
        console.print()

    # Summary
    if has_issues:
        console.print(
            Panel(
                "[yellow]Some issues detected.[/yellow] "
                "Review the output above for details.",
                border_style="yellow",
            )
        )
        return 1
    else:
        console.print(
            Panel(
                "[green]All checks passed![/green] QuestFoundry is ready to use.",
                border_style="green",
            )
        )
        return 0


def _run_doctor_json(settings: QuestFoundrySettings, skip_network: bool) -> int:
    """Output doctor results as JSON."""
    import json

    result: dict[str, Any] = {
        "environment": {
            "python": sys.version.split()[0],
            "platform": f"{platform.system()} {platform.release()}",
            "cwd": os.getcwd(),
        },
        "providers": {},
        "spec": check_spec_status(settings),
        "config": {
            "runtime": {
                "max_iterations": settings.runtime.max_iterations,
                "max_ping_pong": settings.runtime.max_ping_pong,
            },
            "llm": {
                "default_provider": settings.llm.default_provider,
                "ollama_host": settings.llm.ollama_host,
            },
            "paths": {
                "project_dir": str(settings.paths.project_dir),
                "project_id": settings.paths.project_id,
            },
        },
    }

    if not skip_network:
        checker = ProviderChecker(settings)
        for provider, check_result in checker.check_all().items():
            result["providers"][provider] = {
                "status": check_result["status"].value,
                "message": check_result["message"],
            }
    else:
        for provider, (env_var, _) in ProviderChecker.PROVIDERS.items():
            if env_var:
                is_set, _ = get_env_var_status(env_var)
                status = "configured" if is_set else "unconfigured"
            else:
                status = "configured"
            result["providers"][provider] = {"status": status}

    print(json.dumps(result, indent=2))

    # Return 1 if no ready providers
    ready = [p for p, r in result["providers"].items() if r["status"] == "ready"]
    return 0 if ready or skip_network else 1
