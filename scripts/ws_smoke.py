"""Connect to the edge-node WebSocket, read a few snapshots, send a control,
and verify the response path. Exits with code 0 on success.
"""

from __future__ import annotations

import asyncio
import json
import sys

import websockets


async def main() -> int:
    url = "ws://127.0.0.1:8081/"
    try:
        async with websockets.connect(url, open_timeout=5) as ws:
            # Expect "hello"
            hello = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            assert hello.get("type") == "hello", hello
            print("OK  hello:", hello.get("server_time"))

            # Read up to 5 messages to find a snapshot
            for _ in range(5):
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                if msg.get("type") == "snapshot":
                    scene = msg["scene"]
                    print(f"OK  snapshot: scenario={scene['scenario']} "
                          f"frame={scene['frame_count']} victims={len(scene['victims'])}")
                    break
            else:
                print("NO snapshot in first 5 messages"); return 1

            # Send a scenario change and observe subsequent snapshot
            await ws.send(json.dumps({"type": "set_scenario", "scenario": "fire_structure"}))
            print("SEND set_scenario -> fire_structure")

            # Read up to 10 messages, expecting updated scenario
            for _ in range(10):
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                if msg.get("type") == "snapshot" and msg["scene"]["scenario"] == "fire_structure":
                    print("OK  scenario change acknowledged in broadcast")
                    break
            else:
                print("scenario change never observed in broadcast"); return 1

            return 0
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
