# Reviewed-host CPU software test policy

Andrew requested85 C warning and90 C abort for future software test windows.
`thermal_policy.py` and `thermal_guard.py` implement that policy for the verified
5955WX host only. This is software supervision, not a firmware setting or a change
to hardware throttling/protection. No GPU window has run under the new policy.

AMD lists the5955WX maximum operating junction temperature as95 C:
[AMD product specification](https://www.amd.com/en/support/downloads/drivers.html/processors/ryzen-threadripper-pro/ryzen-threadripper-pro-5000wx-series/amd-ryzen-threadripper-pro-5955wx.html).
The warning/abort values are nominally10/5 C below that specification. They are not
certified junction margins: sampled control/CCD readings need not equal the hottest
junction, and brief peaks between samples remain possible.

## Installed mapping and evidence

Read-only host inspection found AuthenticAMD, family25 (19h), model8 (08h), stepping2,
model name `AMD Ryzen Threadripper PRO 5955WX 16-Cores`. The loaded kernel is
6.17.9-76061709-generic; k10temp srcversion is A594DFB9AAE185D37A6B0D8. The package
version ends in df6b2b6, resolved to Pop!_OS source commit
df6b2b6295d1ac5414690b4a74f35020f0327f7d.

[Packaged k10temp source](https://github.com/pop-os/linux/blob/df6b2b6295d1ac5414690b4a74f35020f0327f7d/drivers/hwmon/k10temp.c)
selects the Zen3 Threadripper Chagall CCD register mapping for family19h/model08h.
Its label table maps temp1 to Tctl and temp5/temp7 to Tccd3/Tccd5. Its model-specific
Tctl offset table covers family17h entries, not this family19h model; there is no
separately exported Tdie here. No offset is invented or subtracted by this policy.
The source warns that CCD conversion is experimentally derived rather than confirmed
from chip datasheets. This establishes driver channel identity, not calibration to
the hottest physical junction or mapping CCD labels to Linux logical CPU numbers.
The installed binary was identified through package/module metadata, not rebuilt
and bit-for-bit compared with source.

| Installed PCI0000:00:18.3, vendor1022/device1653 | Role | Warning | Abort |
| --- | --- | ---: | ---: |
| temp1_input / Tctl | Control reading | >=85 C | >=90 C |
| temp5_input / Tccd3 | CCD reading | >=85 C | >=90 C |
| temp7_input / Tccd5 | CCD reading | >=85 C | >=90 C |

All three are mandatory. Any additional correctly numbered Tccd1..8 channel on the
same verified device is also evaluated, not discarded. Unexpected labels (including
an unexpected Tdie), driver/PCI/CPU/kernel/module changes, duplicate identities,
missing mandatory channels, read errors and invalid numbers reject the sample and
record the unresolved mapping. hwmon numbering may change; discovery checks labels,
channel filenames and physical PCI identity. Every available numeric reading is
retained even when its label or metadata is unreadable.

## Boundaries and unchanged safeguards

85 exactly warns;89.999 still warns and is accepted;90 exactly aborts. No averaging
or Tctl preference hides a hot CCD. Warnings are logged and retained in per-sensor
JSONL assessments. Missing/stale data fails closed. The independent watchdog's0.5 s
sampling,1 s freshness deadline and blocked-I/O abort thread are unchanged. Cooler
admission still requires every sensor strictly below60 C. Limit overrides above90 C
are rejected. CUDA-wait review criteria explicitly bind the policy ID and85/90 limits.

The historical80.75 C Tccd3 /69.125 C Tctl sample would pass this CPU temperature
policy if supplied with a verified current mapping; that is a policy comparison,
not retrospective evidence that the failed image request succeeded.

This change affects the host test guard. The live Python rank transport has its
own80 C guard and remains unchanged to preserve the no-GPU/no-restart boundary.
Consequently an API image test can still be stopped by that independent80 C guard.
A separate reviewed API integration change is required before claiming the complete
API window uses85/90. Frozen earlier launchers with literal80 C also remain historical
artifacts; do not use them as the new policy entry point. Existing approved bundles
are not rewritten, and new source/config bindings require independent review before
any GPU execution.

Deterministic tests cover every sensor at84.999/85/89.999/90/90.001, cool admission,
hot CCD with cool Tctl, unknown/new/duplicate/missing sensors, unreadable metadata,
changed CPU/kernel/module identity, warning persistence and abort persistence.
Existing watchdog tests cover stale data and blocked sensor/evidence I/O independently.
