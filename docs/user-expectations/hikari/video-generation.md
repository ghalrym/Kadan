# Video generation

Video generation uses MiniMax H3 FL2VA INT8 + Turbo. Select a hardware tier and download its model before queuing work. The page reads that tier's duration and resolution defaults: Go uses five seconds at 480p, Plus eight seconds at 480p, and Pro five seconds at 768p. The initial aspect is 16:9. A defaults-read error must remain visible rather than silently submitting invented settings.

Users can request whole durations from one to fifteen seconds, 480p or 768p resolution, and 16:9, 9:16 or 1:1 aspect. Frame rate is fixed at 24 fps. The prompt must be nonblank and at most 8,000 characters. Negative prompts are unsupported. The API seed defaults to 42; the current browser form does not expose a seed control. Explicit clip settings still need actual memory admission and can fail rather than being silently reduced.

Queue video returns an accepted job, not a completed video. The queue reports Queued, Rendering, Done, Failed or Cancelled. Pending jobs are polled approximately every three seconds. Only Done with an available output supplies playable MP4 content. A placeholder, progress percentage or HTTP 202 is not finished media. Refresh queue reloads the server's current job list.

Cancel request while submitting stops waiting in that browser. It does not cancel a job the server has already accepted. Refresh the queue before submitting again after an uncertain submission, or duplicate jobs can result. Leaving the page also stops its browser reads and submission wait; accepted jobs remain owned by the server. Use Cancel on the queued or rendering job to request actual job cancellation, then check its resulting state. Cleanup can outlast the button press.

The current job index exists only in the API process. Reloading the browser can recover that process's jobs, but restarting the API loses their listing and status. Completed MP4 files are written to the configured video directory; retained files alone do not reconstruct the lost index or make their old content routes available. Download completed media before an API restart if it must remain accessible. This queue is not a durable media library or backup.
