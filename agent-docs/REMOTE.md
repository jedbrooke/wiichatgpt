# WiiChatGPT Remote Client

## Overview

A Wii homebrew chat application that connects to a remote OpenAI-compatible API server and streams responses to the TV. The Wii acts as a thin client: all inference runs on the server, the console handles UI, networking, and conversation state. Uses the same UI described in [UI.md](UI.md) (both console and graphical modes).

Compatible with any server implementing the OpenAI chat completions endpoint: OpenAI itself, ollama, llama.cpp server, vLLM, text-generation-webui, LM Studio, LocalAI, etc.

## Architecture

```
Wii (thin client)                         Remote Server
+---------------------------+             +----------------------+
| UI layer (console or SDL) |             | OpenAI-compatible    |
|   token callback renders  |             | POST /v1/chat/...    |
|   to screen as tokens     |             | stream: true (SSE)   |
|   arrive                  |             |                      |
| SSE stream parser         |--HTTP(S)--->| Any model, any HW    |
|   line-by-line read       |<--SSE-------|                      |
|   JSON delta extraction   |             +----------------------+
| Conversation manager      |
|   message history array   |
|   token-budget truncation |
| Network (libogc TCP)      |
| TLS (mbedtls/wolfSSL)     |
| Config (SD card)          |
+---------------------------+
```

## OpenAI Chat Completions API Surface

The client implements one endpoint: `POST /v1/chat/completions` with `stream: true`.

### Request

```json
{
  "model": "gpt-4o-mini",
  "messages": [
    {"role": "system", "content": "You are a helpful Wii assistant."},
    {"role": "user", "content": "What are the best Wii games?"},
    {"role": "assistant", "content": "Mario Kart Wii is a classic..."},
    {"role": "user", "content": "Tell me more about it"}
  ],
  "stream": true,
  "temperature": 0.7,
  "max_tokens": 256
}
```

Headers: `Content-Type: application/json`, `Authorization: Bearer <api_key>` (omitted for keyless LAN servers).

### Response (SSE stream)

```
data: {"id":"chatcmpl-abc","choices":[{"delta":{"role":"assistant"},"index":0}]}

data: {"id":"chatcmpl-abc","choices":[{"delta":{"content":"Mario"},"index":0}]}

data: {"id":"chatcmpl-abc","choices":[{"delta":{"content":" Kart"},"index":0}]}

data: [DONE]
```

Each `data:` line is either a JSON object with a `choices[0].delta.content` field (a token) or the literal `[DONE]` sentinel. Blank lines separate events. The first event may contain a `role` delta instead of `content` (ignored by the parser).

### Error responses

| HTTP Status | Meaning | Client behavior |
|---|---|---|
| 200 | Success (SSE stream) | Parse and render |
| 400 | Bad request (malformed body) | Display error, do not retry |
| 401 | Invalid API key | Display error, prompt for config check |
| 429 | Rate limited | Display error, retry with backoff |
| 500+ | Server error | Display error, retry with backoff |

Non-200 responses return a JSON body with an `error.message` field. The client displays this message in the chat stream as a system message.

## SSE Streaming Client

The core adaptation layer between the network and the UI's token callback.

### Parse loop

```
read HTTP response headers
verify status 200 and content-type: text/event-stream
loop:
    read line from socket (up to \n)
    if line is empty or "\n":
        continue (event boundary)
    if line starts with "data: ":
        payload = line after "data: " prefix
        if payload == "[DONE]":
            signal generation complete
            break
        parse payload as JSON
        extract choices[0].delta.content
        if content exists:
            invoke token_callback(content, userdata)
    pump UI events (SDL_PollEvent / check for cancel)
    if cancel requested:
        close connection
        break
```

### Chunked transfer encoding

Most servers send SSE responses with `Transfer-Encoding: chunked`. libcurl handles chunk decoding transparently when using `CURLOPT_WRITEFUNCTION`. For a raw-socket implementation, the client must strip chunk headers (`<hex-length>\r\n ... \r\n`).

### Partial reads

SSE lines can span multiple TCP segments. The client maintains a line buffer and accumulates bytes until a `\n` is found. Maximum line length: 4 KB (sufficient for any single token delta).

### Connection lifecycle

1. Open connection (TCP + optional TLS handshake)
2. Send HTTP POST with JSON body
3. Read and parse SSE stream until `[DONE]` or error
4. Close connection

Connections are not reused across requests (no HTTP keep-alive). This simplifies the implementation and avoids stale-connection issues on the Wii's network stack.

## Conversation Manager

Maintains the message history as an array of `{role, content}` objects.

### Turn cycle

1. User submits text
2. Append `{"role": "user", "content": "<input>"}` to history
3. Send full history array in the API request
4. Accumulate streamed tokens into a content string
5. On completion, append `{"role": "assistant", "content": "<full response>"}` to history
6. Ready for next turn

### System prompt

A configurable system message is prepended to the messages array on every request. Default: `"You are a helpful Wii assistant. Keep responses concise."` Configurable via `remote.conf`.

### Token-budget truncation

The remote model has a finite context window. The client tracks approximate token count (1 token ~ 4 characters as a rough heuristic) and drops the oldest user/assistant pairs when history exceeds a configurable limit (`max_history_tokens` in config, default 4096). The system prompt is never dropped.

## Network Stack (Wii)

### TCP/IP

`libogc` provides `network.h` with BSD-style sockets (`net_init`, `net_socket`, `net_connect`, `net_send`, `net_recv`). WiFi is configured through the Wii system settings menu before launching the app.

### HTTP

Two options, in order of preference:

1. **libcurl (devkitPPC port):** Available via the devkitPro pacman repository (`ppc-libcurl`). Handles HTTP framing, chunked encoding, redirects, timeouts. The SSE parser hooks into `CURLOPT_WRITEFUNCTION` to receive data as it arrives. This is the recommended path.

2. **Raw sockets:** Build the HTTP request manually (`net_send` the headers and body), parse the response line-by-line. More code, but zero additional dependencies. Viable for LAN-only setups where TLS is not needed.

### TLS

Required for public API endpoints (api.openai.com, etc.). Optional for LAN servers.

| Library | Notes |
|---|---|
| mbedtls | devkitPro port available (`ppc-mbedtls`). Small footprint, well-tested on embedded. |
| wolfSSL | Also ported. Slightly larger but more TLS feature coverage. |

Certificate handling: bundle a CA certificate file (`cacert.pem`) on the SD card. libcurl reads it via `CURLOPT_CAINFO`. For trusted LAN servers, TLS verification can be disabled (`CURLOPT_SSL_VERIFYPEER = 0`).

### DNS

`libogc` `net_gethostbyname` provides basic DNS resolution. For LAN setups, use IP addresses directly to avoid DNS dependency.

## Configuration

`sd:/apps/wiichatgpt/remote.conf` on the SD card. Plain text key-value format.

```
endpoint=http://192.168.1.100:8080/v1/chat/completions
api_key=
model=llama3
system_prompt=You are a helpful Wii assistant. Keep responses concise.
max_history_tokens=4096
temperature=0.7
max_tokens=256
timeout_seconds=30
```

- `endpoint`: full URL to the chat completions endpoint
- `api_key`: bearer token (empty string for keyless servers)
- `model`: model identifier string sent in the request body
- `system_prompt`: prepended to every request
- `max_history_tokens`: approximate token budget for conversation history
- `temperature`: sampling temperature
- `max_tokens`: maximum tokens per response
- `timeout_seconds`: connection and response timeout

Defaults are tuned for a LAN server (no TLS, no API key, short timeout).

## Memory Budget

Without model weights, KV cache, or PLE tables, nearly all of MEM1 and MEM2 are available.

### MEM2 (64 MB)

| Component | Size |
|---|---|
| UI (framebuffer, font, text cache, OSK) | ~6 MB |
| libcurl + TLS library | ~0.5 MB |
| HTTP send/receive buffers | ~0.1 MB |
| Conversation history (text) | ~0.1 MB (generous for hundreds of turns) |
| CA certificate bundle | ~0.2 MB |
| **Total** | **~7 MB** |
| **Free** | **~57 MB** |

### MEM1 (24 MB)

| Component | Size |
|---|---|
| Code + stack | ~1 MB |
| Network stack (libogc) | ~0.1 MB |
| Line/JSON parse buffers | ~0.01 MB |
| **Total** | **~1.1 MB** |
| **Free** | **~23 MB** |

The massive headroom means conversation history length is effectively unlimited by client memory. The remote model's context window is the binding constraint.

## Error Handling and UX

| Scenario | UI behavior |
|---|---|
| WiFi not configured | "No network. Configure WiFi in Wii settings." |
| Connection refused | "Cannot reach server at <endpoint>." |
| DNS failure | "Cannot resolve hostname." |
| TLS handshake failure | "TLS error. Check certificate config." |
| HTTP 401 | "Invalid API key. Check remote.conf." |
| HTTP 429 | "Rate limited. Retrying in <N>s..." |
| HTTP 5xx | "Server error. Retrying in <N>s..." |
| Stream interrupted mid-response | Display partial response, append "(connection lost)" |
| Timeout (no data for `timeout_seconds`) | "Request timed out." |
| User cancels (Escape/Home) | Close connection, keep partial response in history |

Retry logic: exponential backoff starting at 2s, max 30s, max 3 retries. After 3 failures, stop retrying and display the error.

All error messages appear inline in the chat stream as system messages, not as modal dialogs.

## Token Callback Integration

The UI's `llm_token_callback` interface (defined in [UI.md](UI.md)) is reused directly. The SSE parser invokes the callback for each `delta.content` string received:

```c
// In the SSE parse loop:
if (delta_content != NULL) {
    token_callback(delta_content, userdata);
}
```

Console mode: the callback calls `printf`. Graphical mode: the callback appends to the in-progress message surface and redraws.

The UI does not know or care whether tokens come from a local model or a remote API. The callback interface is the abstraction boundary.

## Source Files

```
remote/
  remote_main.cpp    # init, config loading, main loop
  api_client.cpp     # HTTP request construction, SSE parsing
  api_client.h       # public interface
  conversation.cpp   # message history management, truncation
  conversation.h
```

UI source files from [UI.md](UI.md) are shared. LLM code is not linked.

## Desktop Build

Same source, different link targets:

| Component | Wii | Desktop |
|---|---|---|
| HTTP | ppc-libcurl | system libcurl |
| TLS | ppc-mbedtls | system OpenSSL |
| Sockets | libogc network.h | POSIX sockets (via libcurl) |
| UI | sdl-wii / libogc console | SDL 1.2 / terminal |

The desktop build is used for development and testing against a local ollama or llama.cpp server before deploying to Wii hardware.