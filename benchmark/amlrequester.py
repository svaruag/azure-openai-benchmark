import asyncio
import logging
import json
import time
from typing import Optional

import aiohttp
import backoff

from .request_stats import RequestStats


REQUEST_ID_HEADER = "apim-request-id"
UTILIZATION_HEADER = "azureml-deployment-utilization"
RETRY_AFTER_MS_HEADER = "retry-after-ms"
MAX_RETRY_SECONDS = 60.0

DEPLOYMENT_NAME_HEADER = "azureml-model-deployment"


def _get_gpt4o_header():
    import json

    additional_headers = ''
    return json.loads(additional_headers)


def _terminal_http_code(e) -> bool:
    return e.response.status != 429

class AzureMLRequester:
    def __init__(self, api_key: str, deployment_name: str, url: str, backoff=False):
        self.api_key = api_key
        self.url = url
        self.backoff = backoff
        self.deployment_name = deployment_name

    async def call(self, session: aiohttp.ClientSession, body: dict) -> RequestStats:
        stats = RequestStats()
        body["stream"] = True
        body["ignore_eos"] = True
        try:
            await self._call(session, body, stats)
        except Exception as e:
            stats.last_exception = e

        return stats

    @backoff.on_exception(backoff.expo,
                          aiohttp.ClientError,
                          jitter=backoff.full_jitter,
                          max_time=MAX_RETRY_SECONDS,
                          giveup=_terminal_http_code)
    async def _call(self, session: aiohttp.ClientSession, body: dict, stats: RequestStats):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            DEPLOYMENT_NAME_HEADER: self.deployment_name,
            "extra-parameters": "pass-through",
        }
        stats.request_start_time = time.time()
        while stats.calls == 0 or time.time() - stats.request_start_time < MAX_RETRY_SECONDS:
            stats.calls += 1
            response = await session.post(self.url, headers=headers, json=body)
            stats.response_status_code = response.status
            self._read_utilization(response, stats)
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
            content = []
            async for line in response.content:
                if not line.startswith(b'data:'):
                    continue
                if stats.first_token_time is None:
                    stats.first_token_time = time.time()
                if stats.generated_tokens is None:
                    stats.generated_tokens = 0
                stats.generated_tokens += 1

                # Extract and collect content
                data = None
                line = line.decode('utf-8').strip()
                if line.startswith('data: '):
                    line = line[6:]  # Remove 'data: ' prefix
                    if line.strip() and line.strip() != '[DONE]':
                        try:
                            data = json.loads(line)
                            if 'choices' in data and len(data['choices']) > 0:
                                delta = data['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    content.append(delta['content'])
                        except json.JSONDecodeError:
                            logging.warning(f"Failed to parse JSON: {line}")

            stats.generated_text = ''.join(content)
            stats.response_end_time = time.time()

    def _read_utilization(self, response: aiohttp.ClientResponse, stats: RequestStats):
        if UTILIZATION_HEADER in response.headers:
            util_str = response.headers[UTILIZATION_HEADER]
            if len(util_str) == 0:
                logging.warning(f"got empty utilization header {UTILIZATION_HEADER}")
            elif util_str[-1] != '%':
                logging.warning(f"invalid utilization header value: {UTILIZATION_HEADER}={util_str}")
            else:
                try:
                    stats.deployment_utilization = float(util_str[:-1])
                except ValueError as e:
                    logging.warning(f"unable to parse utilization header value: {UTILIZATION_HEADER}={util_str}: {e}")


async def main():
    api_key = ""
    deployment_name = "phi-3-mini-4k-baseline-opt2"
    url = "https://phi3a100bench.westus3.inference.ml.azure.com/chat/completions"
    body = {
        "messages": [
            {
                "role": "user",
                "content": "please share the flight info from Miami to Seattle."
            }
        ],
        "temperature": 0.8,
        "stream": True,
        "max_tokens": 10,
    }

    async with aiohttp.ClientSession() as session:
        requester = AzureMLRequester(api_key, deployment_name, url, backoff=True)
        stats = await requester.call(session, body)
        # print(f"Request stats: {stats}")
        print(f"Request stats: {repr(stats)}")
        print(f"Generated text: {stats.generated_text}")


if __name__ == "__main__":
    asyncio.run(main())