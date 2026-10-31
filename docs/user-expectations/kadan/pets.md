# Pets

Kadan lets users create or import an animated pet for each agent. Open **Edit pet** to browse candidates and preview their animations and look directions. Creating or importing a candidate does not change the current pet. **Use pet** selects it for the agent identity, home and chat; **Use hatchling** returns to the default.

If Kadan cannot load the selected pet, it reports the failure and offers **Retry** to load it again. The default hatchling remains visible until the selected pet is available.

Under **New pet**, describe the character and art style, then choose **Create pet**. Hikari generates the artwork and Tsuchi saves the progress and results. Creation can wait for models or available hardware. Closing the dialog does not cancel it. Only one creation or repair can be active for an agent at a time.

**Generation reference** defaults to **Single image**. Choose **Character sheet** to generate a reference showing the same character from several views and poses. Hikari then draws individual animation frames using that sheet. This adds a generation step and can take longer. The saved sheet can be opened from the candidate; it is a design reference, not the finished animation. Reference-sheet uploads are not supported by this option. **Import** still accepts finished pet packages.

Character sheets are an experimental alternative, with no demonstrated quality advantage. They can still produce inconsistent characters, unwanted repeated figures or incorrect poses. Preview every animation and look direction before selecting a candidate. Existing pets and creations keep their original method; resuming or repairing a candidate keeps its reference choice and saved artwork.

**Cancel creation** can remain pending while inference finishes cleanup. Failed, cancelled or interrupted creation can be resumed explicitly, reusing completed steps. A failed job must leave the current pet unchanged. After an uncertain submission, reopen the candidate list and check saved progress before creating another job.

**Import** accepts a supported spritesheet, a folder containing its pet description and sheet, or a ZIP of that folder. Imports require no inference and become candidates for explicit selection. Older sheets without look directions do not follow the cursor. Invalid packages produce an error.

For generated candidates, **Repair a frame** creates a revised candidate while keeping the other saved frames and the previous candidate. Describe the problem, regenerate, then preview before selecting the revision. Imported pets do not support frame repair. Successful generation does not guarantee consistent appearance or motion; users judge the preview.

Trashing artwork used by the selected pet clears its selection. Restoring that artwork does not automatically select the pet again.
