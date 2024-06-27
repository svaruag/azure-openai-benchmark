import logging
from openai import OpenAI
from .request_stats import RequestStats
import aiohttp
import backoff
import time

REQUEST_ID_HEADER = "apim-request-id"
UTILIZATION_HEADER = "azure-openai-deployment-utilization"
RETRY_AFTER_MS_HEADER = "retry-after-ms"
MAX_RETRY_SECONDS = 60.0


def _terminal_http_code(e) -> bool:
    # we only retry on 429
    return e.response.status != 429


class VLLMRequester:
    """
    A requester for the vLLM server that makes a streaming call and collects
    corresponding statistics.
    :param url: Full URL of the vLLM server (e.g., "http://localhost:8000/v1").
    :param model: The model to use for inference.
    :param backoff: Whether to retry throttled or unsuccessful requests.
    """
    def __init__(self, url: str, model: str, backoff=False):
        self.url = url
        self.model = model
        self.backoff = backoff
        self.client = OpenAI(api_key="EMPTY", base_url=url)

    async def call(self, session: aiohttp.ClientSession, body: dict) -> RequestStats:
        stats = RequestStats()
        body["stream"] = True
        body["model"] = self.model
        
        try:
            await self._call(session, body, stats)
        except Exception as e:
            import traceback
            traceback.print_exc()
            stats.last_exception = e

        return stats

    @backoff.on_exception(backoff.expo,
                          aiohttp.ClientError,
                          jitter=backoff.full_jitter,
                          max_time=MAX_RETRY_SECONDS,
                          giveup=_terminal_http_code)
    async def _call(self, session: aiohttp.ClientSession, body: dict, stats: RequestStats):
        headers = {
            "Content-Type": "application/json",
        }
        stats.request_start_time = time.time()
        while stats.calls == 0 or time.time() - stats.request_start_time < MAX_RETRY_SECONDS:
            stats.calls += 1
            response = await session.post(self.url, headers=headers, json=body)
            stats.response_status_code = response.status
            if response.status != 429:
                break
            if self.backoff and RETRY_AFTER_MS_HEADER in response.headers:
                try:
                    retry_after_str = response.headers[RETRY_AFTER_MS_HEADER]
                    retry_after_ms = float(retry_after_str)
                    logging.debug(f"retry-after sleeping for {retry_after_ms}ms")
                    await asyncio.sleep(retry_after_ms / 1000.0)
                except ValueError as e:
                    logging.warning(f"unable to parse retry-after header value: {UTILIZATION_HEADER}={retry_after_str}: {e}")
                    break
            else:
                break

        if response.status != 200 and response.status != 429:
            logging.warning(f"call failed: {REQUEST_ID_HEADER}={response.headers[REQUEST_ID_HEADER]} {response.status}: {response.reason}")
        if self.backoff:
            response.raise_for_status()
        if response.status == 200:
            await self._handle_response(response, stats)

    async def _handle_response(self, response: aiohttp.ClientResponse, stats: RequestStats):
        async with response:
            stats.response_time = time.time()
            async for line in response.content:
                if not line.startswith(b'data:'):
                    continue
                if stats.first_token_time is None:
                    stats.first_token_time = time.time()
                if stats.generated_tokens is None:
                    stats.generated_tokens = 0
                stats.generated_tokens += 1
            stats.response_end_time = time.time()


if __name__ == "__main__":
    import asyncio
    
    async def test_vllm_requester():
        url = "http://0.0.0.0:8000/v1/chat/completions"
        model = "/dev/shm/phi/azureml/old/Phi-3-mini-4k-instruct/mlflow_model_folder/data/model"
        requester = VLLMRequester(url, model)
        
        # Sample request body
        body = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"}
            ],
            "max_tokens": 100,
        }
        
        async with aiohttp.ClientSession() as session:
            stats = await requester.call(session, body)
        
        print(f"Request started at: {stats.request_start_time}")
        print(f"Response status code: {stats.response_status_code}")
        # print(f"Time to first token: {stats.first_token_time - stats.request_start_time:.4f} seconds")
        # print(f"Total response time: {stats.response_end_time - stats.request_start_time:.4f} seconds")
        print(f"Generated tokens: {stats.generated_tokens}")
        print(f"Number of API calls made: {stats.calls}")
        if stats.last_exception:
            print(f"Last exception: {stats.last_exception}")

    asyncio.run(test_vllm_requester())
