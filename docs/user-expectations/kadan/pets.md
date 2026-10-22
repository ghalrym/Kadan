# Pets

Kadan lets users create or import an animated pet for each agent. Open **Edit pet** to browse candidates and preview their animations and look directions. Creating or importing a candidate does not change the current pet. **Use pet** selects it for the roster, home and chat; **Use hatchling** returns to the default.

Under **New pet**, describe the character and art style, then choose **Create pet**. Hikari generates the artwork and Tsuchi saves the progress and results. Creation can wait for models or available hardware. Closing the dialog does not cancel it. Only one creation or repair can be active for an agent at a time.

**Cancel creation** can remain pending while inference finishes cleanup. Failed, cancelled or interrupted creation can be resumed explicitly, reusing completed steps. A failed job must leave the current pet unchanged. After an uncertain submission, reopen the candidate list and check saved progress before creating another job.

**Import** accepts a supported spritesheet, a folder containing its pet description and sheet, or a ZIP of that folder. Imports require no inference and become candidates for explicit selection. Older sheets without look directions do not follow the cursor. Invalid packages produce an error.

For generated candidates, **Repair a frame** creates a revised candidate while keeping the other saved frames and the previous candidate. Describe the problem, regenerate, then preview before selecting the revision. Imported pets do not support frame repair. Successful generation does not guarantee consistent appearance or motion; users judge the preview.

Trashing artwork used by the selected pet clears its selection. Restoring that artwork does not automatically select the pet again.
