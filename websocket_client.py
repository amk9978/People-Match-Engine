#!/usr/bin/env python3

import asyncio
import json
import sys

import requests
import websockets


class WebSocketClient:
    def __init__(self, client_id: str = "test_client"):
        self.client_id = client_id
        self.websocket_url = f"ws://localhost:8000/ws/{client_id}"
        self.api_url = "http://localhost:8000"

    async def listen_for_updates(self):
        """Connect to WebSocket and listen for job updates"""
        try:
            async with websockets.connect(self.websocket_url) as websocket:
                print(f"🔗 Connected to WebSocket as {self.client_id}")
                print("📡 Listening for real-time updates...")
                print("=" * 50)

                async for message in websocket:
                    try:
                        data = json.loads(message)
                        await self.handle_message(data)
                    except json.JSONDecodeError:
                        print(f"📨 Raw message: {message}")

        except websockets.exceptions.ConnectionClosed:
            print("🔌 WebSocket connection closed")
        except Exception as e:
            print(f"❌ WebSocket error: {e}")

    async def handle_message(self, data: dict):
        """Handle incoming WebSocket messages"""
        msg_type = data.get("type", "unknown")

        if msg_type == "job_update":
            job_id = data.get("job_id", "unknown")
            status = data.get("status", "unknown")
            progress = data.get("progress", "")

            print(f"🔄 Job {job_id[:8]}... | Status: {status}")
            if progress:
                print(f"   📋 {progress}")

            if status == "completed":
                result = data.get("result", {})
                if result:
                    print(
                        f"   ✅ Found {result.get('size', 0)} people with density {result.get('density', 0):.4f}"
                    )
                print("   🎉 Analysis completed!")

            elif status == "failed":
                error = data.get("error", "Unknown error")
                print(f"   ❌ Error: {error}")

            print("-" * 30)

        elif msg_type == "echo":
            print(f"📡 Echo: {data.get('message', '')}")
        else:
            print(f"📨 Unknown message type: {msg_type}")
            print(f"   Data: {data}")

    def start_analysis(self, csv_file_path: str, min_density: float = 0.3):
        """Start analysis job via REST API"""
        try:
            print(f"🚀 Starting analysis of {csv_file_path}")

            with open(csv_file_path, "rb") as f:
                files = {"file": f}
                data = {"min_density": min_density}

                response = requests.post(
                    f"{self.api_url}/analyze", files=files, data=data
                )

            if response.status_code == 200:
                result = response.json()
                job_id = result["job_id"]
                print(f"✅ Analysis started! Job ID: {job_id}")
                return job_id
            else:
                print(f"❌ Failed to start analysis: {response.status_code}")
                print(response.text)
                return None

        except Exception as e:
            print(f"❌ Error starting analysis: {e}")
            return None


async def main():
    client = WebSocketClient("demo_client")

    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        print(f"📁 Using CSV file: {csv_file}")

        # Start WebSocket listener in background
        websocket_task = asyncio.create_task(client.listen_for_updates())

        # Wait a moment for WebSocket to connect
        await asyncio.sleep(1)

        # Start analysis
        job_id = client.start_analysis(csv_file)

        if job_id:
            # Keep listening for updates
            await websocket_task
        else:
            websocket_task.cancel()
    else:
        print("Usage: python websocket_client.py <csv_file_path>")
        print("Example: python websocket_client.py data/test_batch.csv")


if __name__ == "__main__":
    asyncio.run(main())
