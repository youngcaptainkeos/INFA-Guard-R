import asyncio
from asyncio import AbstractEventLoop
import json
import os
import sys

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler

sys.path.append('.')
from utils.proxy_manager import ProxyManager

BASE_PROXY_DIR = os.path.join(os.path.dirname(__file__), '..', 'proxies')
REQUESTS_PATH = os.path.join(BASE_PROXY_DIR, 'requests')
PROCESSING_PATH = os.path.join(BASE_PROXY_DIR, 'processing')
RESPONSES_PATH = os.path.join(BASE_PROXY_DIR, 'responses')
ERRORS_PATH = os.path.join(BASE_PROXY_DIR, 'errors')


class RequestHandler(FileSystemEventHandler):

    def __init__(self, loop: AbstractEventLoop):
        super().__init__()
        self.loop = loop
        self.proxy_manager = ProxyManager()

    async def process_request(self, file_path: str, request_id: str):
        client = None
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                payload = json.load(f)

            target_info = payload.get('proxy_target')
            if not target_info or 'api_key' not in target_info or 'base_url' not in target_info:
                raise ValueError('Request JSON is missing "proxy_target" information.')

            request_payload = payload['payload']
            target_client_name = target_info['client_name']
            target_base_url = target_info['base_url']
            target_api_key = target_info['api_key']

            httpx_client = self.proxy_manager.get_httpx_client(target_client_name)
            client = AsyncOpenAI(
                base_url=target_base_url,
                api_key=target_api_key,
                http_client=httpx_client,
            )
            print(f'[{request_id}] Routing request to "{target_base_url}"')

            response: ChatCompletion = await client.chat.completions.create(**request_payload)
            response_file = os.path.join(RESPONSES_PATH, f'{request_id}.json')
            with open(response_file, 'w', encoding='utf-8') as f:
                f.write(response.model_dump_json())
            print(f'[{request_id}] Successfully processed and responded.')
        
        except Exception as e:
            err_file = os.path.join(ERRORS_PATH, f'{request_id}.json')
            err_payload = {
                'error': f'Error during proxy processing: {type(e).__name__} - {e}'
            }
            with open(err_file, 'w', encoding='utf-8') as f:
                json.dump(err_payload, f)
            print(f'[{request_id}] Error processing request: {e}')

        finally:
            if client:
                await client.close()
            if os.path.exists(file_path):
                os.remove(file_path)

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.json'):
            request_id = os.path.basename(event.src_path).split('.')[0]
            print(f'[{request_id}] New request detected.')

            processing_path = os.path.join(PROCESSING_PATH, f'{request_id}.json')
            try:
                os.rename(event.src_path, processing_path)
                asyncio.run_coroutine_threadsafe(
                    self.process_request(processing_path, request_id),
                    self.loop
                )
            except FileNotFoundError:
                pass
            except Exception as e:
                print(f'Error {event.src_path}: {e}')


async def main():
    print('Running File Proxy Server')

    for path in [REQUESTS_PATH, PROCESSING_PATH, RESPONSES_PATH, ERRORS_PATH]:
        os.makedirs(path, exist_ok=True)

    loop = asyncio.get_running_loop()
    event_handler = RequestHandler(loop)

    observer = PollingObserver(timeout=1)
    observer.schedule(event_handler, REQUESTS_PATH, recursive=False)
    observer.start()

    print(
        f'File proxy server started. Watching for requests in: '
        f'{os.path.abspath(REQUESTS_PATH)}'
    )

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print('\nShutting down server...')
    finally:
        observer.stop()
        observer.join()
        print('Server stopped.')


if __name__ == '__main__':
    asyncio.run(main())
