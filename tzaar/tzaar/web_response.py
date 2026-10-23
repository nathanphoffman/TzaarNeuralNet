"""Sending a response from the small web servers (the game server and the training dashboard)."""
import json

JSON_TYPE = "application/json"
HTML_TYPE = "text/html"

def send(handler, content_type, body, status=200, no_store=False):
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)

    if no_store:
        handler.send_header("Cache-Control", "no-store")

    handler.end_headers()
    handler.wfile.write(body)

def send_json(handler, data, status=200, no_store=False):
    send(handler, JSON_TYPE, json.dumps(data).encode(), status, no_store)
