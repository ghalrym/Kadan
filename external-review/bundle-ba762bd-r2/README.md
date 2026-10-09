# External execution binding review only

Not a product merge candidate and not execution approval. This is an unchanged
copy of the current external bundle; binding.json is complete, including all
263 product runtime and 71 external runtime SHA256 entries.

Product source_commit: ba762bdf984dbbd712700a28f870ac53f1dcbccf
External source snapshot: 4d30c5ee4bf3d85a96d907e34b576e0657d7616c
Binding SHA256: e13e449d3d243359d5910ab912fbb49110739adea14bab91a2056bfa5b1312a1
Wrapper SHA256: 960d83b51e6cfbb2cd6b39690427ffe4dd7742af52e8a43c8dd0b8872909dc89
Logging SHA256: 37676b93aa4e9c6949b3d29e24cfdf8290613948fb2ba4e5f0694f09e2ed2190

queue_image_a.py is byte-identical to
4d30c5ee4bf3d85a96d907e34b576e0657d7616c:docs/real-weight-denoiser/reviewed_image_a.py.
No remaining wrapper delta exists. logging.json is likewise byte-identical to
that snapshot's docs/real-weight-denoiser/reviewed_logging.json.
Every external_runtime entry was verified against that immutable snapshot;
every runtime entry was verified against source_commit. No source code changes,
API activation, model execution or approval record were introduced.
