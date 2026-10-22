# Pet Doctor

Kadan's **Doctor** button checks a completed generated pet for changes in body size between frames. It saves a separate candidate, preserving the original and the pet currently in use. Preview the result and choose **Use pet** to select it.

Doctor compares recognizable visual details against the original character reference. When those details agree on a size difference, it resizes the entire frame's character proportionally. Raising an arm or crouching alone should not trigger resizing. Each frame keeps its existing foot height, including when airborne. Doctor does not correct floating feet, anatomy or character identity.

The report lists corrected frames, frames already within tolerance, and frames skipped because their scale could not be measured reliably or resizing would clip the character. Sparse details, changed viewpoints and substantial pose changes can prevent correction. Skipped frames remain unchanged; they are not certified as consistent. Imported sheets lack the original reference and do not support Doctor.

Doctor requires no model inference. It runs in the background and continues after the dialog closes. Cancel stops further work while retaining completed checks. Failed, cancelled or interrupted jobs can be resumed. A failure leaves the original candidate and selection unchanged.
