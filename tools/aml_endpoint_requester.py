import asyncio
import json
import time
from typing import Optional, Dict, Tuple
import random

import math
from wonderwords import RandomWord

import aiohttp

from transformers import AutoTokenizer

from rich.console import Console
from rich.table import Table
from rich import box


class RequestStats:
    def __init__(self):
        self.request_start_time: Optional[float] = None
        self.response_status_code: int = 0
        self.response_time: Optional[float] = None
        self.first_token_time: Optional[float] = None
        self.response_end_time: Optional[float] = None
        self.generated_tokens: Optional[int] = None
        self.generated_text: Optional[str] = None

    def __repr__(self) -> str:
        return f"RequestStats(\n" \
               f"    request_start_time: {self.request_start_time}\n" \
               f"    response_status_code: {self.response_status_code}\n" \
               f"    response_time: {self.response_time}\n" \
               f"    first_token_time: {self.first_token_time}\n" \
               f"    response_end_time: {self.response_end_time}\n" \
               f"    generated_tokens: {self.generated_tokens}\n" \
               f"    generated_text: {self.generated_text}\n" \
               f")"

class AzureMLRequester:
    def __init__(self, api_key: str, deployment_name: str, url: str):
        self.api_key = api_key
        self.url = url
        self.deployment_name = deployment_name

    async def call(self, session: aiohttp.ClientSession, body: dict) -> RequestStats:
        stats = RequestStats()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "azureml-model-deployment": self.deployment_name,
            "extra-parameters": "pass-through",
        }

        stats.request_start_time = time.time()
        async with session.post(self.url, headers=headers, json=body) as response:
            stats.response_status_code = response.status
            if response.status != 200:
                raise Exception(f"Response status code: {response.status}")
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
                            print(f"Failed to parse JSON: {line}")

        stats.generated_text = ''.join(content)
        stats.response_end_time = time.time()
        return stats


# Global variables for caching
CACHED_PROMPT = ""
CACHED_PROMPT_TOKENS = 0
TOKENIZER = None

def initialize_tokenizer(model_name: str):
    global TOKENIZER
    TOKENIZER = AutoTokenizer.from_pretrained(model_name)

def generate_messages(prompt_token_len: int, gen_token_len: int) -> list:
    global CACHED_PROMPT, CACHED_PROMPT_TOKENS, TOKENIZER

    if TOKENIZER is None:
        raise ValueError("Tokenizer not initialized. Call initialize_tokenizer() first.")

    messages = [{"role": "user", "content": f"Write a long essay about life in at least {gen_token_len} tokens. "}]
    
    if CACHED_PROMPT_TOKENS < prompt_token_len:
        words = []
        current_tokens = CACHED_PROMPT_TOKENS
        while current_tokens < prompt_token_len:
            word = random.choice(list(TOKENIZER.vocab))
            word_tokens = len(TOKENIZER.encode(word)) - 1  # Subtract 1 to account for potential special tokens
            if current_tokens + word_tokens <= prompt_token_len + 5:  # Allow up to 5 extra tokens
                words.append(word)
                current_tokens += word_tokens
            if current_tokens >= prompt_token_len:
                break
        
        CACHED_PROMPT = " ".join(words)
        CACHED_PROMPT_TOKENS = current_tokens

    messages.insert(0, {"role": "user", "content": CACHED_PROMPT})
    return messages


async def process_deployment(api_key: str, deployment_name: str, url: str, shape: Tuple[int, int]) -> Dict[str, float]:
    prompt_tokens, gen_tokens = shape
    body = {
        "messages": generate_messages(prompt_tokens, gen_tokens),
        "temperature": 0.8,
        "stream": True,
        "max_tokens": gen_tokens,
        "ignore_eos": True,
    }

    async with aiohttp.ClientSession() as session:
        requester = AzureMLRequester(api_key, deployment_name, url)
        stats = await requester.call(session, body)
        
        return {
            "deployment": deployment_name,
            "shape": shape,
            "latency": stats.response_end_time - stats.request_start_time,
            "ttft": stats.first_token_time - stats.request_start_time if stats.first_token_time else None,
            "generated_tokens": stats.generated_tokens,
            "generated_text": stats.generated_text
        }

async def main():
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    api_key = ""
    url = "https://phi3a100bench.westus3.inference.ml.azure.com/chat/completions"

    deployments = [
        "phi-3-mini-4k-baseline-opt2",
        "phi31-mini4k-rc-opt2"
    ]

    shapes = [
        (500, 100), 
        (500, 200),
        (500, 400),
        (500, 500),
        (500, 800),
        (500, 1200),
        (500, 2000)
    ]

    initialize_tokenizer(model_name)

    tasks = []
    for deployment in deployments:
        for shape in shapes:
            tasks.append(process_deployment(api_key, deployment, url, shape))


    results = await asyncio.gather(*tasks)

    # # Group results by shape
    results_by_shape = {}
    for result in results:
        shape = result['shape']
        if shape not in results_by_shape:
            results_by_shape[shape] = []
        results_by_shape[shape].append(result)

    console = Console()

    for shape, shape_results in results_by_shape.items():
        table = Table(title=f"Shape Comparison: {shape}", box=box.ROUNDED)
        
        table.add_column("Deployment", style="cyan", no_wrap=True)
        table.add_column("Latency (s)", style="magenta")
        table.add_column("TTFT (s)", style="green")
        table.add_column("Generated Tokens", style="yellow")

        for result in shape_results:
            table.add_row(
                result['deployment'],
                f"{result['latency']:.3f}",
                f"{result['ttft']:.3f}" if result['ttft'] else "N/A",
                str(result['generated_tokens']),
            )

        console.print(table)
        console.print()  # Add a blank line between tables

if __name__ == "__main__":
    asyncio.run(main())
