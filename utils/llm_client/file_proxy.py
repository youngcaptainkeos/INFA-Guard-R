import asyncio
import json
import os
import time
import uuid

from openai.types.chat import ChatCompletion


BASE_PROXY_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'scripts', 'vllm', 'proxies')
REQUESTS_PATH = os.path.join(BASE_PROXY_DIR, 'requests')
RESPONSES_PATH = os.path.join(BASE_PROXY_DIR, 'responses')
ERRORS_PATH = os.path.join(BASE_PROXY_DIR, 'errors')


__ALL__ = ['AsyncFileProxyClient']


class FileProxyChatCompletions:

    def __init__(
        self,
        client_name: str,
        base_url: str,
        api_key: str,
        timeout: int = 300,
    ) -> None:
        self.client_name = client_name
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout

    async def create(self, **kwargs) -> ChatCompletion:
        request_id = str(uuid.uuid4())
        request_payload = {
            'proxy_target': {
                'client_name': self.client_name,
                'base_url': self.base_url,
                'api_key': self.api_key,
            },
            'payload': kwargs,
        }

        request_file = os.path.join(REQUESTS_PATH, f'{request_id}.json')
        tmp_request_file = f'{request_file}.tmp'
        with open(tmp_request_file, 'w', encoding='utf-8') as f:
            json.dump(request_payload, f, ensure_ascii=False)
        os.rename(tmp_request_file, request_file)

        response_file = os.path.join(RESPONSES_PATH, f'{request_id}.json')
        err_file = os.path.join(ERRORS_PATH, f'{request_id}.json')

        start_time = time.time()
        while time.time() - start_time < self.timeout:
            if os.path.exists(response_file):
                with open(response_file, 'r', encoding='utf-8') as f:
                    response_data = json.load(f)
                os.remove(response_file)
                return ChatCompletion.model_validate(response_data)

            if os.path.exists(err_file):
                with open(err_file, 'r', encoding='utf-8') as f:
                    error_data = json.load(f)
                os.remove(err_file)
                raise RuntimeError(
                    f'LLM File Proxy Error: {error_data.get("error", "Unknown error")}'
                )

            await asyncio.sleep(0.5)

        if os.path.exists(request_file):
            os.remove(request_file)
        raise TimeoutError(f'LLM File Proxy timed out after {self.timeout}s.')


class FileProxyChat:

    def __init__(
        self,
        client_name: str,
        base_url: str,
        api_key: str,
        timeout: int = 300,
    ) -> None:
        self.completions = FileProxyChatCompletions(client_name, base_url, api_key, timeout)


class AsyncFileProxyClient:

    def __init__(
        self,
        client_name: str,
        base_url: str,
        api_key: str,
        timeout: int = 3000,
    ) -> None:
        os.makedirs(REQUESTS_PATH, exist_ok=True)
        os.makedirs(RESPONSES_PATH, exist_ok=True)
        os.makedirs(ERRORS_PATH, exist_ok=True)

        self.chat = FileProxyChat(client_name, base_url, api_key, timeout)

