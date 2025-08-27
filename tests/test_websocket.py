#!/usr/bin/env python3

import asyncio
import json
import threading

import requests
import websockets


async def websocket_listener():
    """Listen to WebSocket updates"""
    try:
        uri = "ws://localhost:8000/ws/test_client"
        async with websockets.connect(uri) as websocket:
            print("🔗 WebSocket connected!")

            async for message in websocket:
                try:
                    data = json.loads(message)
                    if data.get("type") == "job_update":
                        job_id = data.get("job_id", "")[:8]
                        status = data.get("status")
                        progress = data.get("progress")
                        print(f"📡 Job {job_id}... | {status} | {progress}")

                        if status == "completed":
                            result = data.get("result", {})
                            print(
                                f"✅ Result: {result.get('size', 0)} people, density: {result.get('density', 0):.4f}"
                            )
                except json.JSONDecodeError:
                    print(f"📨 Raw: {message}")

    except Exception as e:
        print(f"❌ WebSocket error: {e}")


def start_analysis():
    """Start analysis via REST API"""
    try:
        print("🚀 Starting analysis...")

        with open(
            "/home/ryan/PycharmProjects/match_engine/data/test_batch.csv", "rb"
        ) as f:
            files = {"file": f}
            data = {"min_density": 0.3}

            response = requests.post(
                "http://localhost:8000/analyze", files=files, data=data
            )

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Analysis started! Job ID: {result['job_id']}")
        else:
            print(f"❌ Failed: {response.status_code}")

    except Exception as e:
        print(f"❌ Error: {e}")


async def main():
    # Start WebSocket listener
    websocket_task = asyncio.create_task(websocket_listener())

    # Wait a moment then start analysis
    await asyncio.sleep(1)

    # Start analysis in background thread
    analysis_thread = threading.Thread(target=start_analysis)
    analysis_thread.start()

    # Wait for WebSocket messages
    try:
        await websocket_task
    except KeyboardInterrupt:
        print("\n👋 Disconnecting...")


if __name__ == "__main__":
    asyncio.run(main())
