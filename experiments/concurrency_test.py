import asyncio
import httpx
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

API_URL = "http://localhost:8000/api/v1/predict"
IMAGE_PATH = "demo/sample_image.jpg"
NUM_REQUESTS = 20

async def send_request(client, image_data):
    files = {'file': ('test.jpg', image_data, 'image/jpeg')}
    start = time.perf_counter()
    await client.post(API_URL, files=files)
    return time.perf_counter() - start

async def run_async_test(image_data):
    async with httpx.AsyncClient() as client:
        tasks = [send_request(client, image_data) for _ in range(NUM_REQUESTS)]
        start_time = time.perf_counter()
        await asyncio.gather(*tasks)
        return time.perf_counter() - start_time

def run_sequential_test(image_data):
    total_time = 0
    with httpx.Client() as client:
        for _ in range(NUM_REQUESTS):
            files = {'file': ('test.jpg', image_data, 'image/jpeg')}
            start = time.perf_counter()
            client.post(API_URL, files=files)
            total_time += (time.perf_counter() - start)
    return total_time

def run_threaded_test(image_data):
    def sync_request():
        with httpx.Client() as client:
            files = {'file': ('test.jpg', image_data, 'image/jpeg')}
            client.post(API_URL, files=files)
    start_time = time.perf_counter()
    with ThreadPoolExecutor(max_workers=10) as executor:
        list(executor.map(lambda _: sync_request(), range(NUM_REQUESTS)))
    return time.perf_counter() - start_time

async def main():
    try:
        with open(IMAGE_PATH, "rb") as f: image_data = f.read()
    except FileNotFoundError:
        print("❌ Sample image not found in demo/ folder."); return
    
    seq_time = run_sequential_test(image_data)
    thread_time = run_threaded_test(image_data)
    async_time = await run_async_test(image_data)

    df = pd.DataFrame({"Method": ["Sequential", "ThreadPool", "AsyncIO"], "Total Time (s)": [seq_time, thread_time, async_time]})
    print("--- PERFORMANCE RESULTS ---", df.to_string(index=False))

if __name__ == "__main__":
    asyncio.run(main())