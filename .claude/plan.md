# Implementation Requirements: Lightweight AI Voice Assistant with System Analytics

## 1. Objective

Build a lightweight, portable AI voice assistant that can understand spoken English questions, generate answers using Groq, and respond using my own voice.

The application must also include a monitoring and analytics dashboard that provides visibility into CPU, memory, disk, network, model performance, voice generation latency, API usage and errors.

This is intended as a working technical demo, not a large production platform.

**Primary goals:**
- Clone and use my own voice for AI-generated responses.
- Use lightweight open-source audio models wherever practical.
- Use Groq for speech recognition and LLM inference where appropriate.
- Monitor resource consumption and model performance.
- Run on GitHub Codespaces and local machines.
- Keep deployment and setup simple.
- Avoid unnecessary infrastructure, dependencies and operating costs.

## 2. First task: Explore and evaluate before implementation

Before writing or modifying code, inspect the entire existing repository.

Do not assume the repository is empty or create a new implementation without checking what already exists.

Perform the following assessment:

1. Review the existing code, dependencies, configuration and documentation.
2. Identify the current application architecture and reusable components.
3. Check whether any voice processing, TTS, ASR, Groq or monitoring implementation already exists.
4. Identify unnecessary dependencies and potential compatibility issues.
5. Research current lightweight open-source voice models and their official GitHub repositories.
6. Check model licenses, supported languages, CPU requirements, memory requirements and voice-cloning capabilities.
7. Evaluate whether the selected models can run within GitHub Codespaces resource limits.
8. Identify potential performance bottlenecks and deployment limitations.

Prepare an implementation plan with:
- Existing functionality that can be reused.
- Recommended models and alternatives.
- Required dependencies.
- Expected CPU and RAM requirements.
- API requirements and external services.
- Potential technical limitations.
- Implementation phases and testing strategy.

After completing the assessment, proceed with implementation if the approach is clear and does not require destructive changes. If a significant architectural decision or external service dependency is unresolved, explain it before proceeding.

Do not spend time building a large framework or introducing unnecessary abstractions.

## 3. Voice cloning and text-to-speech

The primary requirement is to generate speech using my own recorded English voice.

### Primary model: PocketTTS

GitHub: https://github.com/kyutai-labs/pocket-tts

Evaluate PocketTTS as the default TTS engine.

Requirements:
- Support reference-audio-based voice cloning.
- Accept an uploaded WAV recording.
- Generate speech that retains the reference speaker's characteristics.
- Support CPU inference without requiring a dedicated GPU.
- Support streaming audio generation if practical.
- Allow the voice profile to be reused without reprocessing the original recording on every request.
- Support loading and saving a reusable voice state.
- Measure voice initialization and speech-generation latency.
- Provide a way to regenerate audio without restarting the application.

Use the official PocketTTS Python API and documented voice-state export functionality.

Do not describe reference-audio conditioning as model fine-tuning. Actual weight fine-tuning is not required for the first version.

### Voice recording and profile management

Implement a simple voice setup experience.

Requirements:
- Upload or record my own voice.
- Preview the recording.
- Allow the user to replace the recording.
- Generate and save a reusable voice profile.
- Display whether a voice profile is configured.
- Provide a test sentence to evaluate the generated voice.
- Display errors when voice initialization fails.

Use only recordings belonging to the user or recordings for which explicit permission has been provided.

Do not upload or expose voice recordings unnecessarily. Do not provide public access to voice profiles.

### Alternative TTS models to evaluate

Compare the following models before finalizing the implementation.

**PocketTTS**
https://github.com/kyutai-labs/pocket-tts

- Primary candidate.
- CPU-oriented.
- Approximately 100M parameters for the main model.
- Reference-audio voice cloning.
- Reusable voice states.

**Kokoro-82M**
https://github.com/hexgrad/kokoro

- Lightweight TTS model.
- Approximately 82M parameters.
- Good alternative for preset voice generation.
- Evaluate CPU performance and English pronunciation.
- Important: preset voice generation is not equivalent to cloning my personal voice.

**Sopro**
https://github.com/search?q=Sopro+TTS+voice+cloning&type=repositories

- Investigate its current implementation and availability.
- Check whether reference-audio voice cloning is supported.
- Verify actual model size, license and CPU compatibility before considering it.

Do not install every model. Select one primary TTS engine and optionally one fallback after comparing their performance.

## 4. Speech recognition (ASR)

The application must accept spoken English questions.

Evaluate these options:

**Option A: Groq Whisper API**

Official documentation:
https://console.groq.com/docs/speech-to-text

Evaluate:
- whisper-large-v3-turbo
- whisper-large-v3

Use the currently supported model identifiers confirmed in the official documentation.

Advantages:
- No local ASR model installation.
- Reduced local CPU and memory consumption.
- Simple API integration.

Disadvantages:
- Requires internet connectivity.
- Uses external API processing.
- API availability and usage limits apply.

**Option B: Faster-Whisper**

GitHub:
https://github.com/SYSTRAN/faster-whisper

Evaluate:
- Tiny or Base models first.
- CPU inference.
- INT8 quantization.
- Memory consumption and transcription latency.

**Option C: Whisper.cpp**

GitHub:
https://github.com/ggml-org/whisper.cpp

Evaluate its suitability for lightweight local transcription and portability.

### ASR implementation requirements

- Accept browser microphone recordings.
- Convert audio into an appropriate supported format.
- Use English as the initial language.
- Return the transcription to the application.
- Display the recognized text before or alongside answer generation.
- Handle empty recordings and transcription failures.
- Measure transcription latency.
- Support switching between cloud and local ASR if the implementation remains simple.

Default to Groq ASR for the initial Codespaces demo unless benchmarking shows a compelling reason to use local inference.

## 5. Voice activity detection

Evaluate Silero VAD:

https://github.com/snakers4/silero-vad

Purpose:
- Detect when a user starts and stops speaking.
- Avoid unnecessary processing of silence.
- Improve microphone recording workflows.
- Support voice activity metrics.

Keep VAD optional in the first version. Do not introduce it if browser microphone recording already provides a reliable and simpler experience.

## 6. Groq integration and AI responses

Use Groq for answer generation.

Official documentation:
https://console.groq.com/docs

Requirements:
- Accept recognized speech as input.
- Send the question to a configured Groq chat model.
- Generate concise, natural English answers.
- Return the answer to the TTS engine.
- Support conversation history.
- Provide a configurable system prompt.
- Handle API timeouts, rate limits and errors.
- Measure API response latency.
- Record token usage when the API exposes it.

Keep model identifiers configurable.

Do not hardcode API keys or expose them to the browser.

Use environment variables or the platform's secret-management facility.

The application should support replacing the Groq model without changing the overall application architecture.

## 7. Complete voice interaction

The main user workflow should be straightforward:

1. Open the application.
2. Configure or select the saved voice profile.
3. Click the microphone button.
4. Ask a question in English.
5. Convert the recording into text.
6. Send the text to Groq.
7. Receive an AI-generated answer.
8. Generate speech using the selected voice profile.
9. Play the generated audio in the browser.
10. Display the complete conversation history.

Show the status of each processing stage.

Suggested statuses:
- Ready
- Recording
- Transcribing
- Generating response
- Generating speech
- Completed
- Failed

Avoid unnecessary waiting indicators or full-page reloads.

## 8. System monitoring and analytics dashboard

This is a major requirement.

Build an integrated dashboard that provides visibility into the resources consumed by the application.

Use lightweight Python monitoring libraries.

### Primary monitoring library: Psutil

Official documentation:
https://psutil.readthedocs.io/

GitHub:
https://github.com/giampaolo/psutil

Use Psutil to collect the following metrics.

### CPU monitoring

Display:
- Overall CPU utilization.
- Application process CPU utilization.
- CPU core count.
- Per-core utilization where available.
- CPU utilization history.
- Peak CPU utilization.
- Average CPU utilization.

### Memory monitoring

Display:
- Total system-visible memory.
- Available memory.
- Used memory.
- Memory utilization percentage.
- Application process RSS memory.
- Application process VMS memory where supported.
- Peak application memory consumption.
- Memory utilization history.

Distinguish between system/container memory and the application's own process memory.

### Disk monitoring

Display:
- Total disk capacity.
- Used disk space.
- Available disk space.
- Disk utilization percentage.
- Application data directory size.
- Model cache size if it can be measured reliably.

### Network monitoring

Display:
- Bytes sent.
- Bytes received.
- Network throughput.
- Network activity history.

Where possible, distinguish application-specific network activity from system-wide network statistics. Clearly label metrics when this distinction is not available.

### Process monitoring

Display:
- Application process ID.
- Process CPU consumption.
- Process memory consumption.
- Application uptime.
- Thread count.
- Process status.

### GPU monitoring

GPU support must be optional.

If a GPU is available, investigate a lightweight way to report GPU utilization and memory usage.

If no GPU is available, show GPU as unavailable rather than reporting misleading values.

Do not add heavyweight GPU monitoring dependencies solely for this optional feature.

## 9. Voice and AI performance analytics

In addition to system monitoring, measure application-specific performance.

Track the following metrics for every completed or failed request.

| Metric | Description |
|---|---|
| Request ID | Unique request identifier |
| Transcription latency | Time taken for ASR |
| LLM latency | Time taken by Groq |
| TTS latency | Time taken to generate speech |
| Total latency | End-to-end processing time |
| Audio duration | Length of the recorded input |
| Generated audio duration | Length of generated output |
| Request status | Success or failure |
| Error category | Failure reason where applicable |
| Model name | ASR, LLM and TTS model identifiers |
| Timestamp | Request creation time |

Calculate:
- Total voice requests.
- Successful requests.
- Failed requests.
- Average response time.
- Median response time.
- P95 response time where sufficient samples exist.
- Average transcription latency.
- Average LLM latency.
- Average TTS latency.
- Total audio generated.
- Total API usage where available.

Do not invent metrics or display placeholder values as actual measurements.

## 10. Analytics UI requirements

Keep the UI simple, modern and responsive.

Use Streamlit with Plotly or another lightweight charting library if the existing application does not already use a suitable frontend.

The dashboard should include:

**System overview**
- CPU usage.
- Memory usage.
- Disk usage.
- Network activity.
- Application uptime.

**Voice performance**
- Transcription latency.
- Groq response latency.
- TTS generation latency.
- End-to-end response time.

**Usage analytics**
- Total questions.
- Successful requests.
- Failed requests.
- Recent activity.
- Model usage.

**Historical charts**
- CPU utilization over time.
- Memory utilization over time.
- TTS latency over time.
- API latency over time.
- Request volume over time.

Include:
- Current values.
- Historical trends.
- Clear units.
- Time ranges such as 15 minutes, 1 hour and 24 hours where historical data exists.
- Empty states when no data is available.

Do not overload the dashboard with too many charts.

## 11. Metrics storage

Use SQLite for lightweight historical analytics.

Requirements:
- Store request-level metrics.
- Store periodic resource measurements.
- Store application errors.
- Include timestamps.
- Support filtering by time range.
- Avoid duplicate records.
- Use appropriate indexes.
- Provide configurable retention or cleanup for older resource samples.

Do not store raw voice recordings or complete sensitive conversation content in analytics tables by default.

The application should continue functioning if historical analytics storage is temporarily unavailable.

Live monitoring must not depend on the database.

## 12. Lightweight and portable deployment

The application must run in:

1. GitHub Codespaces.
2. Local macOS.
3. Linux.
4. Windows where dependencies support it.
5. Docker-based environments.

Prioritize Codespaces and local macOS/Linux for the first implementation.

Requirements:
- Simple installation.
- Minimal system dependencies.
- No mandatory GPU.
- No Kubernetes.
- No mandatory Prometheus.
- No mandatory Grafana.
- No unnecessary background services.
- No complex orchestration.
- Docker support where useful.
- Environment-based configuration.
- Persistent storage for the saved voice profile and SQLite analytics.

Provide a simple startup command.

Do not introduce infrastructure that is unnecessary for a single-user demo.

### Codespaces-specific considerations

Validate the actual runtime environment.

Check:
- Available CPU cores.
- Available memory.
- Disk capacity.
- Python version.
- Audio library compatibility.
- Microphone access through the browser.
- Forwarded port behavior.
- Model download and cache persistence.

Important: system metrics collected inside Codespaces may represent the container or virtual machine rather than the user's physical laptop.

Label this accurately in the dashboard.

Do not assume GPU access in Codespaces.

## 13. Performance requirements

The application should remain lightweight.

Priorities:
- Low memory consumption.
- Limited CPU usage during idle periods.
- Avoid repeatedly loading models.
- Cache loaded models.
- Reuse voice profiles.
- Avoid unnecessary audio conversions.
- Avoid blocking the UI during long operations.
- Use streaming TTS where practical.
- Prevent multiple simultaneous TTS operations if the selected model is not thread-safe.
- Use bounded resource-monitoring intervals.
- Avoid excessive database writes.

Measure actual performance instead of making unsupported performance claims.

For the first demo, prioritize stable sequential voice requests over complex concurrent processing.

## 14. Security and privacy

Treat voice recordings and generated voice profiles as sensitive personal data.

Requirements:
- Only process the user's own voice or explicitly authorized recordings.
- Never expose uploaded recordings through public URLs.
- Do not commit voice recordings, voice profiles or API keys to Git.
- Add sensitive audio and profile files to Git ignore rules.
- Keep API credentials server-side.
- Use environment variables or Codespaces secrets.
- Restrict access to the application.
- Avoid logging raw audio or sensitive transcripts.
- Provide a way to delete the saved voice profile and recordings.
- Do not implement voice impersonation of other people.
- Make it clear that generated speech is AI-generated.

## 15. Error handling and reliability

Handle:
- Missing Groq API key.
- Invalid API key.
- Groq rate limits.
- Network failures.
- Unsupported audio formats.
- Empty microphone recordings.
- Failed transcription.
- Failed TTS generation.
- Invalid voice recordings.
- Missing voice profiles.
- Insufficient disk space.
- Model download failures.
- SQLite errors.

Show useful error messages in the UI.

Avoid silently swallowing exceptions.

Keep the application usable when optional components such as GPU monitoring or VAD are unavailable.

## 16. Testing requirements

Implement practical tests for:

**Voice**
- Reference audio upload.
- Voice profile creation.
- Voice profile reload.
- TTS generation.
- Audio playback.
- Invalid audio handling.

**AI integration**
- Groq authentication.
- Speech transcription.
- Answer generation.
- API timeout handling.
- Rate-limit handling.

**Monitoring**
- CPU collection.
- Memory collection.
- Disk collection.
- Network collection.
- Process metrics.
- Historical metric persistence.
- Dashboard data accuracy.

**Performance**
- Model initialization time.
- Voice profile initialization time.
- TTS latency.
- Memory usage before and after loading the model.
- CPU usage during speech generation.
- Idle resource consumption.

**Deployment**
- Clean environment installation.
- Codespaces startup.
- Local startup.
- Docker startup if implemented.
- Persistent voice profile and database after restart.

Tests must not require paid API access for every unit test. Mock external API calls where appropriate.

## 17. Model benchmarking and selection

Before finalizing model dependencies, compare the available options.

Use this evaluation matrix:

| Area | Evaluation |
|---|---|
| Model size | Download and disk footprint |
| Memory | Peak RAM consumption |
| CPU | Average and peak CPU |
| Latency | Time to first audio and total generation time |
| Quality | Voice similarity and pronunciation |
| Compatibility | Codespaces and local execution |
| License | Model and dependency licensing |
| Maintenance | Active official repository and documentation |

Recommended initial candidates:

- TTS: PocketTTS.
- TTS fallback: Kokoro-82M for preset voices.
- ASR cloud: Groq Whisper Large V3 Turbo, subject to current availability.
- ASR local: Faster-Whisper Base or Tiny.
- VAD: Silero VAD, optional.
- System monitoring: Psutil.
- Charts: Plotly.
- Metrics database: SQLite.

Do not install all candidates simultaneously. Benchmark them independently and select the smallest option that meets the quality requirements.

Check the official repository and model card for the latest model sizes, supported Python versions, license terms and inference requirements.

## 18. Reference implementations and documentation

Use these as starting points for research.

**Voice and TTS**

PocketTTS:
https://github.com/kyutai-labs/pocket-tts

Kokoro:
https://github.com/hexgrad/kokoro

**Speech recognition**

Faster-Whisper:
https://github.com/SYSTRAN/faster-whisper

Whisper.cpp:
https://github.com/ggml-org/whisper.cpp

**Voice activity detection**

Silero VAD:
https://github.com/snakers4/silero-vad

**Monitoring**

Psutil:
https://github.com/giampaolo/psutil

**UI and analytics**

Streamlit:
https://github.com/streamlit/streamlit

Plotly:
https://github.com/plotly/plotly.py

**AI inference**

Groq documentation:
https://console.groq.com/docs

**Browser microphone**

Streamlit audio input:
https://docs.streamlit.io/develop/api-reference/widgets/st.audio_input

Review these projects for implementation patterns, but do not copy unnecessary frameworks or dependencies.

## 19. Final deliverables

The completed implementation should provide:

1. A working English voice assistant.
2. Voice profile creation using my own recording.
3. Voice profile persistence and reload.
4. Groq-powered speech recognition and answer generation.
5. AI responses spoken using the selected voice.
6. A system resource dashboard.
7. Voice and API performance analytics.
8. Historical metrics stored locally.
9. Portable deployment for Codespaces and local environments.
10. Environment configuration and setup documentation.
11. A clear list of dependencies and model licenses.
12. Automated tests for core functionality.

## 20. Implementation approach

Work in the following sequence.

**Phase 1: Repository assessment**
- Inspect existing code.
- Research models.
- Identify reusable components.
- Validate dependencies and runtime compatibility.
- Document findings and the implementation plan.

**Phase 2: Voice proof of concept**
- Integrate PocketTTS.
- Add voice recording upload.
- Generate a reusable voice profile.
- Generate and play a test sentence.
- Measure resource usage.

**Phase 3: AI integration**
- Integrate Groq speech recognition.
- Integrate Groq answer generation.
- Connect answers to PocketTTS.
- Implement conversation history.

**Phase 4: Analytics**
- Add Psutil monitoring.
- Track application-specific resource consumption.
- Record voice request metrics.
- Add SQLite persistence.
- Implement the analytics dashboard.

**Phase 5: Portability and validation**
- Validate Codespaces.
- Validate local execution.
- Add Docker support if useful.
- Test performance and error handling.
- Update documentation.

## 21. Important engineering instructions

- Prioritize a working end-to-end demo over architectural complexity.
- Do not introduce heavyweight models without demonstrating a clear benefit.
- Do not assume every open-source model supports voice cloning.
- Verify model licenses and API capabilities against official sources.
- Do not claim actual model training when using reference-audio conditioning.
- Do not introduce unnecessary services or infrastructure.
- Reuse existing code wherever possible.
- Avoid breaking existing functionality.
- Keep model selection configurable.
- Keep secrets outside the repository.
- Measure actual resource usage and report real numbers.
- Do not use fabricated dashboard data.
- Do not create a large abstraction layer for a small demo.
- Document any trade-offs and known limitations.

**Final objective:** Deliver a simple, lightweight and portable AI voice assistant that can speak using my own voice, answer questions through Groq, and provide clear analytics about CPU, memory, network, model performance and overall application usage.

Start by inspecting the existing repository. Research, validate and document the approach, then implement the solution in incremental phases.