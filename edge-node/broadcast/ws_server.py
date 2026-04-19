"""Dashboard broadcast server.

Serves the receiver dashboard over HTTP on one port and pushes scene state +
accepts control events (confirm tag, change scenario, generate MIST) over
WebSocket on another. Pure asyncio + stdlib where possible.

Protocol (JSON, line-delimited over WS):
  server → client:
    {"type": "snapshot", "scene": {...}}            # full scene, sent on every tick
    {"type": "transcript", "text": "...", "ts": ..}
    {"type": "mist", "victim_id": "...", "mist": {...}}
    {"type": "audit", "event": {...}}
    {"type": "hello", "server_time": ..}

  client → server:
    {"type": "confirm_tag", "victim_id": "...", "tag": "RED", "actor": "medic"}
    {"type": "set_scenario", "scenario": "fire_structure"}
    {"type": "generate_mist", "victim_id": "..."}
    {"type": "note", "victim_id": "...", "text": "TQ on 14:02"}
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import socket
from collections import deque
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from typing import Any, Awaitable, Callable, Deque, Dict, Optional, Set, Tuple

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
except ImportError:  # pragma: no cover
    websockets = None
    WebSocketServerProtocol = Any  # type: ignore

logger = logging.getLogger(__name__)


class BroadcastServer:
    def __init__(self,
                 http_host: str = "0.0.0.0",
                 http_port: int = 8080,
                 ws_host: str = "0.0.0.0",
                 ws_port: int = 8081,
                 dashboard_dir: str = "dashboard",
                 on_control: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None) -> None:
        self.http_host = http_host
        self.http_port = http_port
        self.ws_host = ws_host
        self.ws_port = ws_port
        self.dashboard_dir = os.path.abspath(dashboard_dir)
        self.on_control = on_control

        self._clients: Set[Any] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ws_server_task: Optional[asyncio.Task] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._http_thread: Optional[threading.Thread] = None
        self._http_server: Optional[ThreadingHTTPServer] = None
        self._stop_event: Optional[asyncio.Event] = None

        # Replay buffer of recent non-snapshot events so a reconnecting
        # client can catch up without waiting for the next heartbeat.
        # Snapshots aren't buffered — they're cumulative by design, so
        # the next tick is enough to restore state. What we can lose
        # across a reconnect are point-in-time events (scan_ready,
        # timer_milestone, audit, …); those we persist here.
        #
        # ``_event_seq`` is a monotonic counter the server stamps on
        # every event; the client echoes the last one it has so we can
        # replay everything newer. 200 slots ~= 30–60 seconds of busy
        # scene traffic; older gaps are handled by the next snapshot.
        self._event_seq: int = 0
        self._event_buffer: Deque[Tuple[int, str]] = deque(maxlen=200)
        self._EVENT_BUFFER_BYPASS = {"snapshot", "hello"}

    # ------------------------------------------------------------------
    def start(self) -> None:
        self._start_http()
        self._start_ws()

    def stop(self) -> None:
        if self._http_server is not None:
            try:
                self._http_server.shutdown()
            except Exception:
                pass
        if self._loop is not None and self._stop_event is not None:
            # Signal the WS serve-loop to exit cleanly. The async context
            # manager takes care of closing the server and cancelling tasks.
            self._loop.call_soon_threadsafe(self._stop_event.set)
            if self._ws_thread is not None:
                self._ws_thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    def broadcast(self, message: Dict[str, Any]) -> None:
        """Thread-safe fan-out. Can be called from the main pipeline thread.

        Events other than ``snapshot`` / ``hello`` get a monotonic ``seq``
        stamped onto the payload and a copy stored in the replay buffer
        so reconnecting clients can catch up from the last seq they saw.
        """
        if not self._loop or self._loop.is_closed():
            return
        mtype = str(message.get("type") or "")
        if mtype and mtype not in self._EVENT_BUFFER_BYPASS:
            self._event_seq += 1
            message = dict(message)
            message["seq"] = self._event_seq
        payload = json.dumps(message, default=_json_default)
        if mtype and mtype not in self._EVENT_BUFFER_BYPASS:
            self._event_buffer.append((self._event_seq, payload))
        fut = asyncio.run_coroutine_threadsafe(self._broadcast_async(payload), self._loop)
        try:
            fut.result(timeout=0.5)
        except Exception:
            pass  # don't let a slow client block the pipeline

    async def _broadcast_async(self, payload: str) -> None:
        dead: Set[Any] = set()
        for client in list(self._clients):
            try:
                await client.send(payload)
            except Exception:
                dead.add(client)
        self._clients -= dead

    # ------------------------------------------------------------------
    def _start_http(self) -> None:
        dashboard_dir = self.dashboard_dir

        class Handler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=dashboard_dir, **kwargs)

            def log_message(self, fmt, *args):
                return  # silence default logging

            def do_GET(self):
                # /api/scans/<id>/{frame,crop,face}.jpg → in-memory JPEG store
                path = self.path.split("?", 1)[0]
                if path.startswith("/api/scans/"):
                    parts = path.strip("/").split("/")
                    if len(parts) == 4 and parts[0] == "api" and parts[1] == "scans":
                        scan_id, fname = parts[2], parts[3]
                        if _serve_scan_jpeg(self, scan_id, fname):
                            return
                        self.send_error(404, "scan artifact not found")
                        return
                return super().do_GET()

        class ReusingServer(ThreadingHTTPServer):
            allow_reuse_address = True

        self._http_server = ReusingServer((self.http_host, self.http_port), Handler)
        # On Windows set SO_EXCLUSIVEADDRUSE off so re-binds during development
        # don't hit TIME_WAIT. (allow_reuse_address alone isn't enough on Win.)
        try:
            self._http_server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        except Exception:
            pass
        self._http_thread = threading.Thread(target=self._http_server.serve_forever,
                                             name="dashboard-http", daemon=True)
        self._http_thread.start()
        print(f"[broadcast] Dashboard http://{self.http_host}:{self.http_port}/")

    def _start_ws(self) -> None:
        if websockets is None:
            print("[broadcast] websockets not installed; running without dashboard push.")
            return

        async def _serve() -> None:
            self._stop_event = asyncio.Event()
            # websockets >= 13 expects to be entered from an already-running
            # loop; ``serve()`` is itself an async context manager in v13+.
            async with websockets.serve(self._ws_handler, self.ws_host, self.ws_port):
                print(f"[broadcast] WebSocket ws://{self.ws_host}:{self.ws_port}/")
                # Park here until stop() signals us. Exiting this coroutine
                # cleanly lets the async-with close the server and drain tasks,
                # so Ctrl+C shutdowns don't spew "Task was destroyed" warnings.
                await self._stop_event.wait()

        def run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            try:
                loop.run_until_complete(_serve())
            except (asyncio.CancelledError, KeyboardInterrupt):
                pass
            except Exception as exc:
                print(f"[broadcast] WS server error: {exc}")
            finally:
                try:
                    # Cancel any remaining tasks before closing the loop.
                    pending = asyncio.all_tasks(loop=loop)
                    for task in pending:
                        task.cancel()
                    if pending:
                        loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True)
                        )
                except Exception:
                    pass
                try:
                    loop.close()
                except Exception:
                    pass

        self._ws_thread = threading.Thread(target=run, name="dashboard-ws", daemon=True)
        self._ws_thread.start()

    async def _ws_handler(self, ws, *_args) -> None:
        self._clients.add(ws)
        try:
            await ws.send(json.dumps({
                "type": "hello",
                "server_time": time.time(),
                "seq": self._event_seq,
            }))
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                # Client is reconnecting and wants any events after the
                # last seq it successfully processed. Replay them in
                # order; anything older than the ring-buffer window is
                # simply dropped — the next snapshot recovers the state.
                if msg.get("type") == "resume":
                    try:
                        last_seq = int(msg.get("last_seq") or 0)
                    except (TypeError, ValueError):
                        last_seq = 0
                    replayed = 0
                    for seq, payload in list(self._event_buffer):
                        if seq > last_seq:
                            try:
                                await ws.send(payload)
                                replayed += 1
                            except Exception:
                                break
                    try:
                        await ws.send(json.dumps({
                            "type": "resume_ack",
                            "replayed": replayed,
                            "server_seq": self._event_seq,
                            "from_seq": last_seq,
                        }))
                    except Exception:
                        pass
                    continue
                if self.on_control:
                    try:
                        await self.on_control(msg)
                    except Exception as exc:
                        logger.warning("control handler failed: %s", exc)
        except Exception:
            pass
        finally:
            self._clients.discard(ws)


def _json_default(o: Any) -> Any:
    if hasattr(o, "to_dict"):
        return o.to_dict()
    if hasattr(o, "__dict__"):
        return o.__dict__
    return str(o)


def _serve_scan_jpeg(handler: SimpleHTTPRequestHandler, scan_id: str, fname: str) -> bool:
    """Serve a scan JPEG out of pipeline.scan_store. Returns True on hit."""
    try:
        from pipeline.scan_store import store as scan_store
    except Exception:
        return False
    if fname == "frame.jpg":
        payload = scan_store.get_frame(scan_id)
    elif fname == "crop.jpg":
        payload = scan_store.get_crop(scan_id)
    elif fname == "face.jpg":
        payload = scan_store.get_face(scan_id)
    else:
        return False
    if not payload:
        return False
    handler.send_response(200)
    handler.send_header("Content-Type", "image/jpeg")
    handler.send_header("Content-Length", str(len(payload)))
    # Scan JPEGs are immutable per ``scan_id`` — allow caching so the
    # dashboard can re-render tiles every snapshot tick without
    # re-downloading face crops (was causing visible avatar flicker).
    handler.send_header("Cache-Control", "public, max-age=604800, immutable")
    handler.end_headers()
    try:
        handler.wfile.write(payload)
    except Exception:
        pass
    return True
