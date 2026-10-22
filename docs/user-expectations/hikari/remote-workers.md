# Workers on other computers

Hikari can use inference workers on the API computer or other reachable computers. Each worker serves an inference type. The API itself needs no GPU; the selected worker and its computer must support the requested workload. Connecting a worker does not load a model or demonstrate that a request will fit.

Workers receive the selected settings, and required checkpoints are downloaded on the computers that serve them. Downloads and first loads can delay inference. Local copies remain on their worker computers; adding a remote worker does not merge storage or combine all machines' memory into one larger model allocation.

Hikari selects workers for requests and manages waiting when resources are busy. Adding workers can provide more places to execute work, but does not guarantee a particular speedup or immediate start. Workers on the same computer share that computer's available resources.

If a worker disconnects, waiting requests can move to another connected worker of the same type. Running requests fail rather than silently restart their inference. A replacement worker can need to load models again. After an uncertain result, check saved jobs or output before submitting another request.

Operators configure connectivity and the shared worker API key. There is no browser workflow for provisioning remote machines. See the [remote worker setup](../../architecture/hikari/inference.md#workers-on-another-computer) and [security expectations](../security.md).
