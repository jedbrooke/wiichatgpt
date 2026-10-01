# WiiChatGPT OpenAI-Compatible Server

## Overview

An OpenAI-compatible HTTP server that wraps the local inference stack, allowing a second Wii running the remote client (see [REMOTE.md](REMOTE.md)) to connect over LAN. One Wii runs the model and serves responses; the other acts as a thin chat client.

LAN-only, no TLS, single concurrent connection. The server implements the minimal API surface that [REMOTE.md](REMOTE.md)'s client consumes: `POST /v1/chat/completions` with SSE streaming.

Compatible with any OpenAI-format client, not just the Wii remote client. `curl`, Python `openai` library, and other tools work against this server for testing.

## Architecture

```
Wii A (server)                                    Wii B (client)
+--------------------------------------------+    +---------------------------+
| Inference stack (DESIGN.md, MODEL.md)      |    | UI layer (UI.md)          |
|   dense core weights in MEM2               |    |   token callback renders  |
|   PLE table on SD card                     |    |   to screen               |
|   CPU paired-singles + GPU TEV             |    | SSE stream parser         |
|                                            |    | Conversation manager      |
| HTTP server                                |    | HTTP client (libcurl)     |
|   net_socket listener on port 8080         |    |   POST /v1/chat/...       |
|   request parser (headers + JSON body)     |    |   stream: true            |
|   SSE writer (token callback -> socket)    |    |                           |
|   single connection, blocking              |    | Config (remote.conf)      |
+-----------|--------------------------------+    +------------|---------------+
            |           LAN (WiFi)                            |
            +------------------------------------------------|+
```

Both Wiis connect to the same WiFi network via Wii system settings. The client's `remote.conf` points `endpoint` at the server's IP and port.

## API Surface

### `POST /v1/chat/completions`

The only endpoint used for inference. Accepts the same request format the client sends (documented in [REMOTE.md](REMOTE.md)).

#### Request body (JSON)

| Field | Type | Required | Handling |
|---|---|---|---|
| `messages` | array of `{role, content}` | Yes | Concatenated into a prompt string |
| `stream` | boolean | No | Must be `true` or omitted. Non-streaming mode is not implemented (returns 400). |
| `model` | string | No | Ignored (the server runs one model). Echoed back in response. |
| `temperature` | number | No | Passed to sampler. Default: 0.7 |
| `max_tokens` | number | No | Passed to generation. Default: 256 |
| `top_p` | number | No | Passed to sampler if present. Default: 0.9 |

All other OpenAI fields (`tools`, `response_format`, `logprobs`, `n`, etc.) are silently ignored.

#### Response (SSE stream)

Content-Type: `text/event-stream`. Transfer-Encoding: none (raw write, no chunked encoding). Connection: close.

```
data: {"id":"wii-0001","object":"chat.completion.chunk","model":"wiichatgpt","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"wii-0001","object":"chat.completion.chunk","model":"wiichatgpt","choices":[{"index":0,"delta":{"content":"Mario"},"finish_reason":null}]}

data: {"id":"wii-0001","object":"chat.completion.chunk","model":"wiichatgpt","choices":[{"index":0,"delta":{"content":" Kart"},"finish_reason":null}]}

data: {"id":"wii-0001","object":"chat.completion.chunk","model":"wiichatgpt","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

The `id` field is a monotonically incrementing counter prefixed with `wii-`. The `model` field echoes the configured model name.

### `GET /v1/models`

Returns a static model list for discoverability.

```json
{
  "object": "list",
  "data": [
    {
      "id": "wiichatgpt",
      "object": "model",
      "owned_by": "local"
    }
  ]
}
```

### All other paths

HTTP 404 with JSON error body.

## HTTP Server Implementation

Single-threaded, blocking, one connection at a time. The Wii can only run one inference at a time anyway, so concurrent connections have no benefit.

### Listen loop

```
bind to 0.0.0.0:<port>
listen (backlog = 1)
loop:
    client_fd = accept()
    read HTTP request line (e.g. "POST /v1/chat/completions HTTP/1.1\r\n")
    parse method and path
    read headers until blank line (\r\n\r\n)
    extract Content-Length
    read body (up to Content-Length bytes, max 64 KB)
    route request:
        POST /v1/chat/completions -> handle_completions(client_fd, body)
        GET  /v1/models           -> handle_models(client_fd)
        *                         -> send_error(client_fd, 404, "Unknown endpoint")
    close(client_fd)
```

### Request parser

Uses [picohttpparser](https://github.com/h2o/picohttpparser) (two files: `picohttpparser.c` + `picohttpparser.h`, MIT license). Zero-allocation, zero-copy HTTP/1.x parser used in production by the h2o HTTP server. Returns pointers directly into the receive buffer for method, path, and headers.

```c
char buf[4096];
const char *method, *path;
size_t method_len, path_len, num_headers = 32;
int minor_version;
struct phr_header headers[32];

size_t bytes = sock_recv(client_fd, buf, sizeof(buf), 0);
int parsed = phr_parse_request(buf, bytes,
    &method, &method_len, &path, &path_len,
    &minor_version, headers, &num_headers, 0);
```

After parsing headers, the `Content-Length` value is found by iterating the `headers` array and the body is read from the socket in a second `recv` call (up to `max_body_size` bytes, default 64 KB). Requests exceeding the limit get a 413 response.

No support for chunked request encoding, `Expect: 100-continue`, or HTTP pipelining.

### Connection handling

No keep-alive. Every request-response cycle ends with `close()`. The `Connection: close` header is sent in every response. This matches the client's behavior (documented in [REMOTE.md](REMOTE.md): connections are not reused).

### Busy rejection

If a request arrives while inference is running (not possible in the single-threaded model, but guards against future changes), the server returns HTTP 503:

```json
{"error": {"message": "Server busy, inference in progress", "type": "server_error", "code": 503}}
```

## JSON Parsing

Uses [cJSON](https://github.com/DaveGamble/cJSON) (single-file C library, ~1500 lines, MIT license). Already battle-tested on embedded targets including ESP32.

### Request parsing

```c
cJSON* root = cJSON_Parse(body);
cJSON* messages = cJSON_GetObjectItem(root, "messages");
cJSON* stream = cJSON_GetObjectItem(root, "stream");
// ... extract temperature, max_tokens, top_p
```

The `messages` array is iterated to build a prompt string. Each message's `role` and `content` are extracted. The prompt format follows the model's chat template (e.g., `<role>: <content>\n` pairs, with the system prompt first).

### Response construction

SSE JSON chunks are built with `cJSON_CreateObject` / `cJSON_PrintUnformatted`. The printed string is written to the socket prefixed with `data: ` and suffixed with `\n\n`.

After printing, the cJSON object is freed. One allocation + free per token. At ~18 tok/s this is ~18 malloc/free pairs per second, negligible overhead.

## Token Callback to SSE Bridge

The `llm_token_callback` interface from [UI.md](UI.md) connects inference to the socket.

```c
typedef struct {
    int client_fd;
    int token_count;
    char request_id[32];
    const char* model_name;
} SseContext;

void sse_token_callback(const char* piece, void* userdata) {
    SseContext* ctx = (SseContext*)userdata;
    ctx->token_count++;

    // Build SSE chunk: data: {"id":"...","choices":[{"delta":{"content":"<piece>"}}]}\n\n
    cJSON* root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "id", ctx->request_id);
    cJSON_AddStringToObject(root, "object", "chat.completion.chunk");
    cJSON_AddStringToObject(root, "model", ctx->model_name);
    // ... build choices array with delta.content = piece
    char* json = cJSON_PrintUnformatted(root);

    char line[4096];
    snprintf(line, sizeof(line), "data: %s\n\n", json);
    net_send(ctx->client_fd, line, strlen(line), 0);

    cJSON_Delete(root);
    free(json);
}
```

On generation complete (EOS or max tokens), the server sends the final chunk with `finish_reason: "stop"`, then `data: [DONE]\n\n`, then closes the connection.

### Error during generation

If `net_send` fails (client disconnected), the callback sets a flag that causes `llm_generate_cb` to stop early. Partial inference is abandoned, resources freed, and the server returns to the accept loop.

## Conversation Handling

The server is **stateless per-request**. The client sends the full message history in every request (as specified in [REMOTE.md](REMOTE.md)). The server does not maintain conversation state between requests.

### Prompt construction from messages array

```
for each message in messages:
    if role == "system":
        prepend to prompt as system context
    else:
        append "<role>: <content>\n"
append "assistant:" (to prime the model for generation)
```

The exact format depends on the model's chat template. The server applies a simple concatenation format. No special tokens beyond what the tokenizer handles.

### Token budget

The server does not truncate the messages array. If the concatenated prompt exceeds the model's 512-token context window, the tokenizer truncates from the left (dropping oldest context). The client is responsible for managing history length (via `max_history_tokens` in `remote.conf`).

## Concurrency Model

No threads, no async I/O. The server processes one request at a time:

```
accept -> parse request -> run inference (streaming to socket) -> close -> accept
```

During inference (~54 ms per token, ~256 tokens max = ~14 seconds), the server socket is not accepting new connections. The listen backlog of 1 means one pending connection can queue; additional connections are refused by the OS.

This is acceptable for the target use case (one client Wii). For multi-client scenarios, the 503 rejection and client retry logic (documented in [REMOTE.md](REMOTE.md)) handle the contention.

## Networking

### Wii

`libogc` provides `net_init`, `net_socket`, `net_bind`, `net_listen`, `net_accept`, `net_recv`, `net_send`, `net_close`. These are BSD socket analogs.

WiFi is configured through the Wii system settings menu. The server binds to `0.0.0.0` (all interfaces). The client connects using the server Wii's IP address (visible in Wii system settings or discoverable via the server's console output at startup).

### Desktop

Standard POSIX sockets (`socket`, `bind`, `listen`, `accept`, `recv`, `send`, `close`). Same source code with a thin portability shim:

```c
#ifdef WII
  #include <network.h>
  #define sock_send  net_send
  #define sock_recv  net_recv
  // ...
#else
  #include <sys/socket.h>
  #define sock_send  send
  #define sock_recv  recv
  // ...
#endif
```

### No TLS

LAN-only deployment. No certificate management, no handshake overhead. The client's `remote.conf` uses `http://` endpoints. If TLS is ever needed (exposing the server beyond LAN), a reverse proxy (nginx, caddy) on a separate machine is the intended path rather than adding TLS to the Wii binary.

## Memory Budget

The server runs alongside the full inference stack. From [MODEL.md](MODEL.md), MEM2 has ~6.7 MB free after model weights and UI. The server has no UI (headless), so the ~8 MB UI allocation is freed, giving ~14.7 MB available.

### MEM2 additions

| Component | Size |
|---|---|
| picohttpparser code | ~3 KB |
| cJSON library code | ~50 KB |
| HTTP request buffer (max body) | ~64 KB |
| SSE write buffer | ~4 KB |
| cJSON working memory (per-token alloc/free) | ~2 KB peak |
| **Total server overhead** | **~123 KB** |

### MEM1 additions

| Component | Size |
|---|---|
| Socket file descriptors + libogc net internals | ~10 KB |
| Prompt string buffer | ~8 KB |
| **Total** | **~18 KB** |

The server adds negligible memory overhead. The inference stack's memory layout from [MODEL.md](MODEL.md) and [DESIGN.md](DESIGN.md) is unchanged.

## Configuration

`sd:/apps/wiichatgpt/server.conf` on the SD card. Plain text key-value format.

```
port=8080
model_name=wiichatgpt
max_body_size=65536
default_temperature=0.7
default_max_tokens=256
```

- `port`: TCP listen port
- `model_name`: string returned in SSE response `model` field and `/v1/models`
- `max_body_size`: maximum HTTP request body in bytes (rejects larger with 413)
- `default_temperature`: used when client omits `temperature`
- `default_max_tokens`: used when client omits `max_tokens`

## Error Responses

All error responses use the OpenAI error format for client compatibility:

```json
{"error": {"message": "...", "type": "...", "code": <http_status>}}
```

| HTTP Status | Condition | Message |
|---|---|---|
| 400 | Malformed JSON body | "Invalid JSON in request body" |
| 400 | Missing `messages` array | "messages field is required" |
| 400 | `stream` set to `false` | "Non-streaming mode is not supported" |
| 404 | Unknown endpoint | "Unknown endpoint" |
| 405 | Wrong HTTP method | "Method not allowed" |
| 413 | Body exceeds `max_body_size` | "Request body too large" |
| 503 | Inference already in progress | "Server busy, inference in progress" |

## Server Startup

```
load server.conf from SD card
load model weights into MEM2 (same as standalone inference)
load PLE table index
load tokenizer
net_init()
bind + listen on configured port
print "WiiChatGPT server listening on <ip>:<port>"
enter accept loop
```

The startup message prints the Wii's IP address (obtained via `net_gethostip()` on Wii, or by iterating interfaces on desktop). This is what the client operator types into `remote.conf`.

## Desktop Build

Same source, different link targets:

| Component | Wii | Desktop |
|---|---|---|
| Sockets | libogc `network.h` | POSIX `sys/socket.h` |
| JSON | cJSON (same source) | cJSON (same source) |
| Inference | devkitPPC build | native build |
| Config path | `sd:/apps/wiichatgpt/server.conf` | `./server.conf` |

The desktop build is the primary development target. Test with `curl`:

```
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hello"}],"stream":true}'
```

Or with the Python `openai` library:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")
for chunk in client.chat.completions.create(
    model="wiichatgpt",
    messages=[{"role": "user", "content": "hello"}],
    stream=True
):
    print(chunk.choices[0].delta.content or "", end="")
```

## Source Files

```
server/
  server_main.cpp      # config loading, model init, accept loop
  http_server.cpp      # socket setup, routing, response helpers
  http_server.h        # public interface
  sse_writer.cpp       # token callback, SSE formatting, socket writes
  sse_writer.h
vendor/
  picohttpparser.c     # HTTP request parser (h2o/picohttpparser, MIT)
  picohttpparser.h
  cJSON.c              # JSON parser/builder (DaveGamble/cJSON, MIT)
  cJSON.h
```

LLM code linked in from the inference build. UI code is not linked (server is headless).
