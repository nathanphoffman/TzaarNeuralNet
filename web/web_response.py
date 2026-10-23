"""Sending a response from the small web servers (the game server and the training dashboard)."""
import json

HTTP_OK = 200
HTTP_BAD_REQUEST = 400
HTTP_NOT_FOUND = 404
JSON_TYPE = "application/json"
HTML_TYPE = "text/html"

def send(handler, content_type, body, status=HTTP_OK, no_store=False):
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)

    if no_store:
        handler.send_header("Cache-Control", "no-store")

    handler.end_headers()
    handler.wfile.write(body)

def send_json(handler, data, status=HTTP_OK, no_store=False):
    send(handler, JSON_TYPE, json.dumps(data).encode(), status, no_store)
