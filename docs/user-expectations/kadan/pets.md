# Pets

Kadan lets users create or import an animated pet for each agent. Every agent starts with Kadan's shipped default animated pet, which supports all animations and cursor-following look directions. Open **Edit pet** on the [agent card](navigation.md#agent-card) that opens when you click the pet in chat, or under the pet on an empty chat, to browse candidates and preview their animations and look directions. Creating or importing a candidate does not change the current pet. **Use pet** selects it for the agent identity, home and chat; **Use default pet** returns to Kadan's default pet.

If Kadan cannot load the selected pet, it reports the failure and offers **Retry** to load it again. The default pet remains visible until the selected pet is available.

Under **New pet**, describe the character and art style, then choose **Create pet**. Hikari generates the artwork in 12 steps—one canonical character still followed by 11 animation and look-direction strips guided by built-in pose templates (including a potted sprout motif for the working animation)—and Tsuchi saves the progress and results. Creation can wait for models or available hardware. Closing the dialog does not cancel it. Only one creation or repair can be active for an agent at a time.

Existing pets and creations made with earlier per-frame methods keep their original method and saved artwork; resuming or repairing a candidate keeps its method and saved artwork. Preview every animation and look direction before selecting a candidate.

**Cancel creation** can remain pending while inference finishes cleanup. Failed, cancelled or interrupted creation can be resumed explicitly, reusing completed steps. A failed job must leave the current pet unchanged. After an uncertain submission, reopen the candidate list and check saved progress before creating another job.

**Import** accepts a supported spritesheet, a folder containing its pet description and sheet, or a ZIP of that folder. Imports require no inference and become candidates for explicit selection. Older sheets without look directions do not follow the cursor. Invalid packages produce an error.

For generated candidates, **Repair** creates a revised candidate while keeping the other saved animations (or frames, for earlier pets) and the previous candidate. Describe the problem, regenerate, then preview before selecting the revision. Imported pets do not support repair. Successful generation does not guarantee consistent appearance or motion; users judge the preview.

Trashing artwork used by the selected pet clears its selection. Restoring that artwork does not automatically select the pet again.

On the subscription-image branch, pet creation and repair explicitly retain Hikari's local generation and reference editing, even when conversation image generation uses ChatGPT. Pets still require the local image model and available memory. This preserves their saved seeds and reference workflow; subscription pet generation is not exposed. This branch behavior awaits owner validation and is not asserted as verified on master.
