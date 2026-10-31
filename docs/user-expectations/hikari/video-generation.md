# Video generation

Hikari generates video clips with audio. The Video page starts with a five-second, 480p clip. Users can choose clips from one to fifteen seconds, 480p or 768p resolution, and wide, portrait or square aspect. Settings that cannot run must fail rather than be silently reduced.

**Queue video** accepts a job; it does not return a finished video. The queue shows waiting, rendering, completed, failed or cancelled work. Only a completed job with available output supplies playable media. **Refresh queue** checks the current list.

**Cancel request** while submitting stops the browser's wait. A job already accepted can keep running. After an uncertain submission, refresh the queue before submitting again to avoid duplicates. Leaving the page also leaves accepted work running.

Use **Cancel** on a queued or rendering job to request cancellation, then check its state. Cleanup can outlast the button press.

Reloading the browser can recover the current server's job list. Restarting Hikari's API loses that list and its download access, even when completed files remain on disk. Download required media while available; the queue is not a durable media library.

See [Video worker architecture](../../architecture/hikari/worker-video.md) for defaults, job ownership and file retention.
