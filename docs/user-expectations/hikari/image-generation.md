# Image generation

Generate from text starts with an empty prompt, square 1:1 aspect, four images and a blank seed. Choose one, two or four images and a 1:1, 4:3, 3:4 or 16:9 aspect. The nonblank prompt accepts up to 8,000 characters. A blank seed requests random seeds; a supplied seed is recorded with the result. A seed alone is not a guarantee of identical output across different models or settings.

Generation uses the image checkpoint selected through Hikari Settings. The model must be downloaded, and the request still needs memory admission. Multiple requested images are an output count, not a guarantee they all execute simultaneously. The API's base-model step default is 50; the Turbo checkpoint uses its saved eight-step schedule instead.

The progress panel distinguishes waiting, model loading and generation. Percentages and remaining time are estimates from observed progress, not a completion deadline. Missing or stale progress must not be presented as completed work. A successful response supplies the requested PNG images; an error or incomplete image payload must not create substitute artwork.

Results are held in the current page's gallery. Hikari currently returns an empty image-history list and does not persist generated image bytes. Leaving or reloading the page, including switching between Generate and Edit, loses these displayed results and form state. Save the PNGs while available if they must be kept; Kadan's separate image-delivery flow saves its artwork through Tsuchi.

Cancel request aborts the browser request and ignores its late result. Leaving the page also aborts it. Native cancellation and resource cleanup can take longer than the browser action. An error is not proof that no computation occurred, and submitting again is another generation request.

See [Image editing](image-editing.md) for source uploads and reference mode, and [Inference operations](inference.md) for resource waiting and model-loading limits.
