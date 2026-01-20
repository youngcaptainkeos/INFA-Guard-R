import json
import os
from typing import List, Optional, Tuple, Union

from openai import AsyncOpenAI, OpenAI

from utils.llm_client.file_proxy import AsyncFileProxyClient
from utils.llm_client.vllm import AsyncVLLMLBClient
from utils.proxy_manager import ProxyManager


AsyncClient = Union[
    AsyncOpenAI,
    AsyncFileProxyClient,
    AsyncVLLMLBClient,
]


class LLMClientFactory:
    def __init__(self) -> None:
        logs_dir = os.path.join(
            os.path.dirname(__file__), '..', '..', 'scripts', 'vllm', 'logs'
        )
        self.serve_file = os.path.join(logs_dir, 'serve.json')
        self.proxy_manager = ProxyManager()

    @staticmethod
    def _resolve_env(env_name: Optional[str]) -> Optional[str]:
        if not env_name:
            return None
        value = os.getenv(env_name)
        if value is None:
            return None
        value = value.strip()
        return value or None

    @staticmethod
    def _ensure_value(value: Optional[str], env_name: Optional[str]) -> Optional[str]:
        if value is not None:
            return value
        if env_name:
            return LLMClientFactory._resolve_env(env_name)
        return None

    def create_file_proxy_client(
        self,
        model: Optional[str],
        base_url: str,
        api_key: Optional[str],
        client_name: str,
    ) -> Tuple[AsyncFileProxyClient, Optional[str]]:
        if not base_url[7:].startswith(('http://', 'https://')):
            raise ValueError(
                'Invalid file proxy format. Expected "file://http://..." or '
                '"file://https://...".'
            )
        if not api_key:
            raise ValueError('API key is required for file proxy usage.')

        client = AsyncFileProxyClient(
            client_name=client_name,
            base_url=base_url[7:],
            api_key=api_key,
        )
        return client, model

    def get_all_vllm_endpoints(self, model: str) -> List[str]:
        if not os.path.exists(self.serve_file):
            raise FileNotFoundError(
                f'serve.json not found at "{self.serve_file}". Is the vLLM service running?'
            )

        try:
            with open(self.serve_file, 'r', encoding='utf-8') as f:
                serve_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as err:
            raise ValueError(
                f'Could not read or parse serve.json at "{self.serve_file}".'
            ) from err

        model_instances = serve_data.get(model)
        if not model_instances:
            raise ValueError(f'No running service found for model "{model}" in serve.json.')

        endpoints = []
        for instance in model_instances.values():
            ip, port = instance['ip'], instance['port']
            endpoints.append(f'http://{ip}:{port}/v1')
        return endpoints

    def create_vllm_client(
        self,
        model: Optional[str],
        client_name: str,
    ) -> Tuple[AsyncVLLMLBClient, Optional[str]]:
        all_endpoints = self.get_all_vllm_endpoints(model)
        client = AsyncVLLMLBClient(
            endpoints=all_endpoints,
            api_key='bearer',
            proxy_manager=self.proxy_manager,
            client_name=client_name,
        )
        return client, model

    def _build_async_openai(
        self,
        client_name: str,
        base_url: Optional[str],
        api_key: Optional[str],
    ) -> AsyncOpenAI:
        if not api_key:
            raise ValueError('API key is required for AsyncOpenAI client.')

        kwargs = {'api_key': api_key}
        if base_url:
            kwargs['base_url'] = base_url

        httpx_client = self.proxy_manager.get_httpx_client(client_name)
        if httpx_client is not None:
            kwargs['http_client'] = httpx_client
        return AsyncOpenAI(**kwargs)

    def _build_sync_openai(
        self,
        client_name: str,
        base_url: Optional[str],
        api_key: Optional[str],
    ) -> OpenAI:
        if not api_key:
            raise ValueError('API key is required for OpenAI client.')

        kwargs = {'api_key': api_key}
        if base_url:
            kwargs['base_url'] = base_url

        httpx_client = self.proxy_manager.get_httpx_sync_client(client_name)
        if httpx_client is not None:
            kwargs['http_client'] = httpx_client
        return OpenAI(**kwargs)

    def create_async_client(
        self,
        client_name: str,
        model: Optional[str],
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url_env: Optional[str] = 'BASE_URL',
        api_key_env: Optional[str] = 'OPENAI_API_KEY',
    ) -> Tuple[AsyncClient, Optional[str]]:
        resolved_base_url = self._ensure_value(base_url, base_url_env)
        resolved_api_key = self._ensure_value(api_key, api_key_env)

        if resolved_base_url and resolved_base_url.lower().startswith('file://'):
            return self.create_file_proxy_client(
                model,
                resolved_base_url,
                resolved_api_key,
                client_name,
            )

        if resolved_base_url and resolved_base_url.lower() == 'vllm':
            return self.create_vllm_client(model, client_name)

        client = self._build_async_openai(client_name, resolved_base_url, resolved_api_key)
        return client, model

    def create_sync_client(
        self,
        client_name: str,
        model: Optional[str],
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url_env: Optional[str] = 'BASE_URL',
        api_key_env: Optional[str] = 'OPENAI_API_KEY',
    ) -> Tuple[OpenAI, Optional[str]]:
        resolved_base_url = self._ensure_value(base_url, base_url_env)
        resolved_api_key = self._ensure_value(api_key, api_key_env)

        if resolved_base_url and resolved_base_url.lower().startswith('file://'):
            raise ValueError('File proxy client does not support synchronous API usage.')

        if resolved_base_url and resolved_base_url.lower() == 'vllm':
            raise ValueError('vLLM client only supports asynchronous usage.')

        client = self._build_sync_openai(client_name, resolved_base_url, resolved_api_key)
        return client, model

