# Architecture Documentation — Phase 1.1 (regenerated) + Phase 1.2

## 1. High-Level Architecture

```
                         ┌─────────────────────┐
                         │   Speech Recognition │  (interface exists; Whisper
                         └──────────┬───────────┘   adapter is optional extra)
                                    ▼
                         ┌─────────────────────┐
                         │   Intent Detection    │  (Phase 1.4)
                         └──────────┬───────────┘
                                    ▼
        ┌───────────────────────────────────────────────┐
        │              NavigationEngine (1.2)             │
        │  routing (Dijkstra/A*) + localization (snapping) │
        │  + NavigationStateMachine + SafetyEngine          │
        └───────┬───────────────────────────────┬────────┘
                │                                │
     SAFETY lane │ (deterministic,     CONVERSATIONAL lane │ (LLM-mediated,
     <100ms budget, no LLM)            latency-tolerant, Phase 1.4)
                │                                │
                ▼                                ▼
     ┌─────────────────────┐          ┌─────────────────────┐
     │  VoiceGuidance        │          │  MLLM Interface       │
     │  (SpeechPriority.     │          │  → natural language   │
     │   SAFETY)             │          │  → TTS (CONVERSATIONAL)│
     └─────────────────────┘          └─────────────────────┘

Phase 1.2 concretely implements the SAFETY lane end-to-end:
  SafetyEngine.evaluate() → NavEvent + alert text → VoiceGuidance.announce_safety_alert()
  → TextToSpeechInterface.speak(text, priority=SpeechPriority.SAFETY)

No LLM sits anywhere in that path. Turn-by-turn instructions (from
instructions.py) go through the CONVERSATIONAL lane, but are themselves
template-generated, not LLM-generated — Phase 1.4 is where the MLLM enters,
strictly for open-ended Q&A, not for anything safety- or route-relevant.
```

## 2. Folder Structure

See the README's project structure section for the full tree. Key points:

- `navigation/` contains zero framework imports (no FastAPI, no
  SQLAlchemy) — it's testable in isolation, which is why 200+ of the 243
  tests need no database or HTTP client at all.
- `interfaces/` depends only on the standard library, `numpy`, and other
  `interfaces/` modules — never on `infrastructure/` or `navigation/`
  concrete implementations, keeping the dependency arrow pointing from
  concrete → abstract (Dependency Inversion).
- `api/` is a thin adapter layer: routers translate HTTP requests into
  calls on `NavigationEngine` and translate results back into Pydantic
  response models. No routing/safety logic lives in `api/`.

## 3. Module Dependency Diagram

```
config.settings ◄────────────────────────────┐
     ▲                                        │
core.logging_config                    core.exceptions
     ▲                                        ▲
     │                                        │
infrastructure.* ──implements──► interfaces.* ◄──depends-on── navigation.engine.*
     │                                                              │
infrastructure.database.session ──uses──► infrastructure.database.models
                                                                     │
navigation.graph.* ◄──uses── navigation.routing.* ◄──uses── navigation.engine.*
     ▲                              ▲                              │
     │                              │                              ▼
navigation.graph.geometry    navigation.localization.*      navigation.engine.{instructions,
                                                              voice_guidance, safety_engine}
                                                                     │
                                                                     ▼
                                                              api.routers.* ──uses──► api.schemas.*
```

## 4. Sequence: Start Navigation + Position Update (Phase 1.2)

```
Client          API Router         NavigationEngine       SafetyEngine    VoiceGuidance
  │  POST /navigation/sessions ──►
  │                                start_navigation() ──► find_route() (A*/Dijkstra)
  │                                                    ──► generate_instructions()
  │                                                    ──► announce_instruction() ──► TTS
  │  ◄── 201 {session_id, route, first_instruction}
  │
  │  POST .../position ──►
  │                                update_position() ──► localize_fix() (snap + confidence)
  │                                                    ──► evaluate() ──► SafetyDecision
  │                                                    ◄──────────────────
  │                                if alert: announce_safety_alert() ──► TTS (SAFETY priority)
  │                                if event valid: state_machine.transition()
  │                                if next instruction due: announce_instruction() ──► TTS
  │  ◄── 200 {state, current_node_id, next_instruction, safety_alert}
```

## 5. Testing Strategy

- **Unit tests (`tests/unit/`, ~200 tests)**: pure logic, no I/O.
  - Graph: construction, validation, serialization round-trips, malformed
    JSON handling.
  - Routing: Dijkstra and A* cross-validated against each other for
    *every node pair* in the pilot graph (`test_astar.py`) — this is what
    caught a real admissibility bug during development (see Section 8).
  - Accessibility: STRICT mode must never return a route using an
    inaccessible edge; PREFERRED must favor accessible routes but allow
    fallback; NONE ignores accessibility entirely.
  - Localization: snapping distance/confidence math tested independently
    of any GPS hardware via `MockGPSProvider`.
  - State machine: every legal transition and a representative sample of
    illegal ones (asserting `InvalidStateTransitionError`).
  - Exception hierarchy: a parametrized test walks every exception class
    in `core.exceptions` and asserts it is Recoverable XOR Fatal, never
    both — this is a regression test for the exact MRO bug pattern the
    hierarchy is designed to prevent.
- **Integration tests (`tests/integration/`, ~40 tests)**:
  - `NavigationEngine` exercised end-to-end with a `NullTTS` fake — no
    audio hardware, but the *real* routing, snapping, safety, and state
    machine code all run together.
  - Full FastAPI request/response cycles via `TestClient`, including error
    paths (unknown node → 404, illegal pause → 409, malformed route → 400).
- **Fixtures (`tests/conftest.py`)**: a hand-built 5-node graph with a
  known accessible/inaccessible route pair (fast, exact algorithm checks)
  and the actual IIT-H pilot fixture graph (realistic-scale checks).
- Coverage: 88% overall. The three files at 0% (`opencv_camera.py`,
  `tts_engine.py`, `whisper_stt.py`) require real camera/audio/ML hardware
  to exercise meaningfully — mocking `cv2.VideoCapture` or `pyttsx3.init()`
  internals would test the mock, not the adapter. These are covered by
  interface-contract tests instead (`test_hardware_interfaces.py`).

## 6. Deployment Notes

- `pip install -e .` + `uvicorn campus_nav.api.main:create_app --factory`
  is sufficient to run the Phase 1.2 API locally.
- SQLite remains fine for the pilot; `DATABASE_URL` is the only change
  needed for PostgreSQL.
- **Known scaling limitation, by design for now**: `app.state.navigation_engines`
  is an in-memory Python dict, keyed by navigation_session_id. It does not
  survive a process restart and does not work across multiple server
  instances behind a load balancer. Acceptable for a single-process pilot;
  MUST be replaced with a shared store (Redis, or database-backed session
  reconstruction) before any multi-instance production deployment. This is
  isolated behind `api/dependencies.py` so the fix touches one file.
- The campus graph is loaded once at startup from a JSON file
  (`data/sample_campus_graph.json` by default) and treated as read-only,
  versioned data — not something the API mutates.

## 7. Completed Work

**Phase 1.1 (regenerated):**
- `pyproject.toml`: src layout, editable installs, pytest/coverage/ruff/black/isort/mypy all configured
- `openai-whisper` moved to optional `[speech]` extra; all default deps verified Python 3.12-compatible
- Improved exception hierarchy (single-branch-per-leaf constraint, tested)
- Structured logging: text or JSON output, rotating file handler, session context
- Hardware interfaces (Camera, GPS, STT, TTS) + adapters (OpenCV, Mock GPS, Whisper, pyttsx3, NullTTS)
- Database infrastructure: engine/session management, Phase 1.1 + 1.2 models

**Phase 1.2 (new):**
- Campus graph domain model: `Node`, `Edge`, `CampusGraph` with accessibility/landmark metadata, validation, JSON serialization
- Dijkstra and A* routing (A* heuristic-admissibility verified against the fixture graph)
- Accessibility-aware routing (`NONE`/`PREFERRED`/`STRICT` modes)
- Multiple route generation (Yen's k-shortest-paths algorithm)
- GPS-to-graph snapping (nearest node + nearest edge projection) with confidence estimation
- `NavigationStateMachine` — explicit, exhaustively-tested transition table
- Deterministic `SafetyEngine` — route-adherence and confidence monitoring, zero LLM involvement
- Turn-by-turn `instructions.py` and `VoiceGuidance` (correct `SpeechPriority` routing)
- `NavigationEngine` orchestrator tying all of the above together
- Navigation database models (`NavigationSessionRecord`, `RouteRecord`, `NavigationEventRecord`)
- FastAPI endpoints: graph inspection, full navigation session lifecycle, route alternatives
- 243 tests (unit + integration), 88% coverage, clean Ruff/Black/isort/mypy

## 8. Bugs Found and Fixed During This Phase

Documented here deliberately — catching these was the point of the testing
investment, not an embarrassment to hide:

1. **NavigationEngine ignored its own method arguments.** The first draft
   of `update_position(latitude, longitude, gps_accuracy_meters)` accepted
   a raw fix but then queried an independently-injected `GPSInterface`
   instead of using the arguments — meaning a caller supplying a phone's
   GPS reading via the API would have been silently ignored in favor of
   whatever a separate mock/hardware GPS source reported. Fixed by
   extracting a shared `localize_fix()` function used directly by
   `update_position()`, with the `GPSInterface`-polling path kept only as
   an alternative for a hypothetical embedded/hardware-polling daemon.
2. **A* admissibility violation from inconsistent fixture data.** The
   original pilot fixture graph had node lat/lon coordinates that didn't
   correspond to their edges' stated `distance_meters` (e.g. two nodes
   270m apart in a straight line connected by an edge labeled 60m). This
   silently broke A*'s heuristic admissibility guarantee, causing A* to
   return a 220m route when Dijkstra correctly found 155m. Caught by a
   test that cross-validates A* against Dijkstra across every node pair
   in the pilot graph. Fixed by regenerating the fixture from a
   geometrically consistent local coordinate system, then verifying
   `haversine(a, b) <= edge.distance_meters` holds for every edge before
   shipping the fixture.

## 9. Risks

- **Indoor localization gap** — GPS snapping degrades gracefully (returns
  `None` → `UNCERTAIN` state) but there is still no real indoor
  positioning source (BLE beacons, WiFi RTT). Phase 1.3's landmark
  recognition is the intended eventual fallback.
- **In-memory navigation engine registry** — see Section 6; a known,
  explicitly-scoped scaling limitation, not an oversight.
- **TTS platform portability** — `pyttsx3` behavior varies by OS/backend;
  the API layer sidesteps this today via `NullTTS`, but a real deployment
  target needs verification.
- **YOLOv8 licensing (AGPL-3.0)** — unchanged from Phase 1.1 review; still
  fine for pilot/academic use, revisit if commercialized.
- **Privacy** — unchanged from Phase 1.1 review; still applies once Phase
  1.3 adds continuous camera capture.
- **Liability framing** — unchanged from Phase 1.1 review; still must be
  consistently communicated once conversational/LLM output is added in
  Phase 1.4.

## 10. Future Improvements

- Replace the in-memory navigation engine registry with Redis or
  database-backed reconstruction before multi-instance deployment.
- Composite localization provider (GPS + indoor fallback) once a real
  indoor positioning source exists.
- Swap SQLite → PostgreSQL for concurrent multi-user sessions.
- Add a cloud TTS adapter for the conversational lane only, keeping
  `pyttsx3`/offline as the safety-lane engine.
- `SafetyEngine` currently evaluates route-adherence and confidence only;
  Phase 1.3 will extend its mandate to vision-based obstacle detection,
  feeding into the same deterministic, LLM-free SAFETY lane.
