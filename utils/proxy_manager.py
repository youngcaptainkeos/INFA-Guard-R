import os
from typing import Dict, Optional

import httpx


class ProxyManager:
    """Manage per-client HTTP proxies for OpenAI-style clients."""

    def __init__(self) -> None:
        self._proxies: Dict[str, str] = {}
        for key, value in os.environ.items():
            if not value:
                continue
            if key.endswith('_PROXY'):
                client_key = key[:-6].lower()
                self._proxies[client_key] = value

    def get_proxy(self, key: str) -> str:
        if key is None:
            return ''
        return self._proxies.get(key.lower(), '')

    def create_httpx_client(self, proxy_url: str) -> httpx.AsyncClient:
        proxies = {'http://': proxy_url, 'https://': proxy_url}
        return httpx.AsyncClient(proxies=proxies, http2=True)

    def create_httpx_sync_client(self, proxy_url: str) -> httpx.Client:
        proxies = {'http://': proxy_url, 'https://': proxy_url}
        return httpx.Client(proxies=proxies, http2=True)

    def get_httpx_client(self, key: str) -> Optional[httpx.AsyncClient]:
        proxy_url = self.get_proxy(key)
        if not proxy_url:
            return None
        return self.create_httpx_client(proxy_url)

    def get_httpx_sync_client(self, key: str) -> Optional[httpx.Client]:
        proxy_url = self.get_proxy(key)
        if not proxy_url:
            return None
        return self.create_httpx_sync_client(proxy_url)

