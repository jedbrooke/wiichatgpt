# WiiChatGPT UI Specification

## Overview

Single-session chat interface. No persistent threads, no conversation history across sessions. Two implementation tiers: a console mode that works immediately with the existing inference CLI, and a graphical mode with a proper chat layout and wiimote-driven on-screen keyboard.

## Console UI

A text terminal on the TV screen. `printf` output, `fgets` input. Looks like a headless Linux TTY.

### Wii implementation

libogc provides `console_init`, which sets up a framebuffer-backed scrolling text console with VT100 escape code support. `libwiikeyboard` hooks a USB HID keyboard into stdin, so `fgets` and `getchar` work natively. Together, these give a working interactive terminal with ~30 lines of init boilerplate.

```
+----------------------------------+
|  > hello                         |
|  Hey! What Wii game are you      |
|  playing today?                  |
|  > mario kart                    |
|  Mario Kart Wii is one of the    |
|  best-selling Wii games ever,    |
|  with over 37 million copies     |
|  sold worldwide._                |
|  >                               |
+----------------------------------+
```

The existing `llm_generate` already prints tokens to stdout via `safe_printf`. The chat loop is:

1. Print prompt marker (`> `)
2. `fgets` from stdin (blocks on USB keyboard via libwiikeyboard's spin-wait)
3. Call `llm_generate` with the user's input
4. Tokens stream to the TV via `printf`
5. Repeat

### Desktop implementation

The existing CLI main (`./llm_inference model.bin -i "prompt"`) already does single-shot inference. For interactive mode, add a REPL loop that repeatedly reads from stdin and calls `llm_generate`. No new dependencies.

### Limitations

- Requires USB keyboard (no wiimote input)
- No scrollback beyond what the framebuffer holds (~30 lines at standard font size)
- No word wrap (text wraps at framebuffer width, mid-word breaks accepted)
- No visual distinction between user and assistant messages beyond the `> ` prefix
- Single-turn only for v1 (no conversation history passed to the model)
- Busy-spin on stdin read (burns CPU while waiting for input, but inference is idle anyway)


## Full Graphical UI

### Display

| Parameter | Value |
|---|---|
| Framebuffer | 640 x 480 (standard Wii homebrew) |
| Safe area | 576 x 432 (5% margin: 32px horizontal, 24px vertical) |
| Orientation | Landscape (fixed) |

All interactive elements stay within the safe area. An overscan offset config option allows adjustment for CRTs with non-standard overscan.

### Layout

```
+----------------------------------+
|          (safe area)             |
|  +----------------------------+  |
|  |     Message stream         |  |
|  |     (scrollable)           |  |
|  |                            |  |
|  |  You: hello                |  |
|  |  Wii: Hey! What Wii game  |  |
|  |       are you playing?     |  |
|  |  You: mario kart           |  |
|  |  Wii: Mario Kart Wii is_  |  |
|  |       (streaming...)       |  |
|  |                            |  |
|  +----------------------------+  |
|  |  [text input bar]    [Send]|  |
|  +----------------------------+  |
|  |  [on-screen keyboard]      |  |
|  +----------------------------+  |
+----------------------------------+
```

Three vertical zones within the safe area:

| Zone | Height (approx) | Content |
|---|---|---|
| Message stream | ~55% (~240px) | Scrollable chat history |
| Input bar | ~10% (~40px) | Text cursor + send button |
| On-screen keyboard | ~35% (~150px) | QWERTY grid (hides when USB keyboard detected) |

When a USB keyboard is detected via libwiikeyboard, the on-screen keyboard hides and the message stream expands into the freed space.

### Message stream

- User messages left-aligned with "You:" prefix
- Assistant messages left-aligned with "Wii:" prefix
- Tokens stream in real-time (character-by-character append as generated)
- Auto-scrolls to bottom during generation
- Manual scroll via D-pad up/down or USB keyboard arrow keys
- Word wrap at safe area width
- Multi-turn: full conversation history displayed, and prior turns included in the prompt up to the 512-token context window

### Text input bar

- Single-line text field with blinking cursor
- Enter (USB keyboard) or Send button (pointer tap) submits
- Input disabled during generation (visual indicator: grayed out or "thinking..." text)

### On-screen keyboard

Use `dborth/libgui`'s built-in `GuiKeyboard` class. This is the standard OSK in the Wii homebrew scene (used by Snes9x GX, VBA GX, FCE Ultra GX). It handles:

- QWERTY layout rendering
- Wiimote IR pointer tracking and key tap detection
- Shift/caps toggle
- Key press visual feedback

No custom OSK implementation needed. Development and testing of the graphical UI happens in Dolphin emulator.

### Graphics stack

| Component | Library |
|---|---|
| Graphics/window | SDL 1.2 via sdl-wii |
| Font rendering | SDL_ttf (Wii FreeType port) |
| On-screen keyboard | libgui (`dborth/libgui`) GuiKeyboard |
| USB keyboard input | SDL_KEYDOWN via libwiikeyboard |
| Pointer input | Wiimote IR mapped to SDL mouse by sdl-wii |

SDL_ttf render mode: `TTF_RenderText_Solid` or `TTF_RenderText_Shaded`. Avoid `TTF_RenderText_Blended` on Wii (per-pixel alpha blend is expensive at 729 MHz).

### Font

Liberation Sans (regular weight, ~130 KB). OFL licensed, metric-compatible with Arial, clean proportional sans-serif. Readable at 14-16px on a 480p display. ~12px for keyboard labels.

### Text surface caching

SDL_ttf re-rasterizes text on every call. Cache rendered surfaces:

- Each finalized chat message rendered once, then blitted from cache
- The in-progress assistant message re-rendered only when a new token arrives
- Keyboard key labels rendered once at init
- Cache invalidated on font size change

### Input handling

Standard SDL 1.2 `SDL_PollEvent` loop:

| Event | Action |
|---|---|
| `SDL_QUIT` | Exit |
| `SDL_KEYDOWN` (printable) | Append to input buffer |
| `SDL_KEYDOWN` (Enter) | Submit message |
| `SDL_KEYDOWN` (Backspace) | Delete last character |
| `SDL_KEYDOWN` (Escape) | Cancel generation |
| `SDL_KEYDOWN` (Up/Down) | Scroll message history |
| `SDL_MOUSEMOTION` | Update pointer position (OSK hover) |
| `SDL_MOUSEBUTTONDOWN` | Tap OSK key or Send button |

Wiimote mapping (via sdl-wii):

| Wiimote | SDL event | UI action |
|---|---|---|
| IR pointer | `SDL_MOUSEMOTION` | Cursor over OSK |
| A button | `SDL_MOUSEBUTTONDOWN` (left) | Tap key / Send |
| D-pad up/down | `SDL_KEYDOWN` | Scroll messages |
| Home | `SDL_QUIT` | Exit |

## LLM Integration

### Token callback API

Add a callback-based generation function alongside the existing `llm_generate`:

```c
typedef void (*llm_token_callback)(const char* piece, void* userdata);

void llm_generate_cb(LlmModel* model, LlmTokenizer* tokenizer,
                     LlmSampler* sampler, const char* prompt, int steps,
                     llm_token_callback cb, void* userdata);
```

Console mode uses the existing `llm_generate` (prints to stdout). Graphical mode uses `llm_generate_cb` to route tokens into the UI.

### Generation flow (graphical mode)

1. User submits text via input bar or OSK
2. UI disables input, shows generation indicator
3. Prompt is constructed from conversation history (all prior turns concatenated, truncated to fit within 512-token context window)
4. `llm_generate_cb` called with a callback that:
   - Appends the token to the current assistant message string
   - Re-renders the in-progress message surface
   - Blits the updated frame
   - Pumps `SDL_PollEvent` (keeps UI responsive, allows escape to cancel)
5. On completion (EOS or max tokens), UI re-enables input

### Cooperative multitasking

No threads on Wii. The token callback is the yield point: after each token (~54 ms), the callback processes SDL events and redraws. At ~18 tok/s, the UI updates at ~18 fps during generation.

## Memory Budget (Wii, graphical mode)

From MODEL.md, ~8 MB allocated for UI in MEM2.

| Component | Estimate |
|---|---|
| SDL framebuffer (640x480x4, double-buffered) | ~2.4 MB |
| Font data (TTF file) | ~0.1 MB |
| Text surface cache | ~1-2 MB |
| libgui / OSK surfaces | ~0.5 MB |
| Input buffer, message strings | ~0.1 MB |
| SDL_ttf / FreeType working memory | ~1-2 MB |
| **Total** | **~4-6 MB** |
| **Headroom** | **~2-4 MB** |

Console mode uses only the framebuffer (~600 KB for 640x480 YUV) and libwiikeyboard's 256-byte ring buffer.

If we want to support CJK/unicode in the future, we can use Noto Sans CJK/GNU Unifont

## Build Integration

### Console UI

No new dependencies beyond libwiikeyboard (already part of libogc). Add a REPL loop to the existing `llm_inference` target (behind a `-c` flag or as default when no `-i` prompt given).

On Wii: link `-lwiikeyboard`, add video/keyboard init boilerplate.

### Full Graphical UI

Built and tested in Dolphin emulator with devkitPPC toolchain.

Dependencies: SDL 1.2 (sdl-wii), SDL_ttf, libgui, FreeTypeGX, libwiikeyboard.

Source files:

```
ui/
  chat_main.cpp    # SDL init, event loop, main()
  chat_ui.cpp      # rendering, layout, text cache
  chat_ui.h        # public interface
```

LLM code compiled with `-DLLM_NO_MAIN` (existing guard) to avoid duplicate `main()`.
