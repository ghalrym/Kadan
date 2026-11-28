# AI models

Kadan Server's **AI models** section manages the one fixed model behind each ability. Users never choose between competing models.

## Downloading

- Each ability is listed by what it does (Chat & thinking, Images, Dictation, Reading aloud) with a one-line description, the model's own name and its download size underneath.
- **Download everything** downloads every missing model; an ability that is missing or stopped has its own **Download** or **Try again** button. Before anything starts, a confirmation shows the total size and any license notices, which must be accepted.
- Downloads continue in the background while the page is closed or Kadan is in use. Progress is an estimate until complete. Completion means the model is on disk, not that it is loaded or that a request will fit the computer's memory.
- An ability whose part of the server is not running shows **Not running** and suggests restarting Kadan Server and checking its logs.
- If a request to start downloads fails or times out, the page says some downloads may have started and offers **Check again** before retrying.

## Using a subscription instead

**Use a subscription instead** lets users sign in to Claude or ChatGPT and choose what serves each ability:

- **Chat & thinking**: This computer, Claude or ChatGPT.
- **Images**: This computer or ChatGPT. Claude cannot create images, and editing pictures and pets always uses this computer.

A choice stays unavailable until its account is signed in. **Sign in** opens the provider's page. For Claude, approve access and paste the code Claude shows back into Kadan Server. For ChatGPT, enter the code Kadan Server shows on the ChatGPT page; Kadan Server notices when sign-in finishes. **Sign out** returns that account's abilities to this computer. See [Subscription language providers](../hikari/subscriptions.md) for limits and failure behavior.

## Advanced

Collapsed by default, **Advanced** changes the chat model:

- **Chat model size**: **Smaller** fits a 24 GB graphics card and answers quickly; **Higher quality** needs more memory and disk space.
- **Content filter**: **Standard** may decline some requests; **Unfiltered** answers more freely, and the user is responsible for how it is used.

A different choice may need its own download before Kadan can use it. Earlier downloads stay on disk. See [Model settings and downloads](../hikari/model-settings.md) for storage and verification details.
