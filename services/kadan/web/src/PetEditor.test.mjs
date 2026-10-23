import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { runInNewContext } from "node:vm";
import ts from "typescript";
import * as pet from "./pet.ts";
import * as petImport from "./petImport.ts";
import * as petCreation from "./petCreation.ts";

const source = ts.transpileModule(
  readFileSync(new URL("./PetEditor.tsx", import.meta.url), "utf8"),
  { compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX } },
).outputText;
const settle = () => new Promise(resolve => setImmediate(resolve));
const event = { preventDefault() {} };
function deferred() {
  let resolve, reject;
  const promise = new Promise((done, fail) => { resolve = done; reject = fail; });
  return { promise, resolve, reject };
}
const file = (name, text = "", path = "") => ({
  name, size: 100, lastModified: 1, webkitRelativePath: path,
  text: () => typeof text === "string" ? Promise.resolve(text) : text.promise,
});
const manifest = text => file("pet.json", text);
const sheet = () => file("spritesheet.webp");
const imported = {
  id: "candidate", version: 1, origin: "imported", status: "ready", prompt: "Fox",
  steps: [], manifest: { sprite_version: 2 }, reference_mode: "single_image",
};

// Call the component with in-memory hooks, files, image probes and API boundaries.
// No mounted UI, browser, service or image decoding is involved.
async function editor(t) {
  const slots = [], effects = [], probes = [], urls = [], revoked = [], requests = [], selections = [];
  let cursor = 0, tree, uuid = 0, importResponse = imported, mounted = true, lateWrites = 0;
  const react = {
    useState(initial) {
      const index = cursor++;
      if (!(index in slots)) slots[index] = typeof initial === "function" ? initial() : initial;
      return [slots[index], value => {
        if (!mounted) lateWrites++;
        slots[index] = typeof value === "function" ? value(slots[index]) : value;
      }];
    },
    useRef(initial) {
      const index = cursor++;
      return slots[index] ??= { current: initial };
    },
    useMemo(factory, dependencies) {
      const index = cursor++;
      if (!slots[index] || dependencies.some((value, i) => value !== slots[index].dependencies[i])) {
        slots[index] = { dependencies, value: factory() };
      }
      return slots[index].value;
    },
    useCallback(callback, dependencies) { return react.useMemo(() => callback, dependencies); },
    useEffect(effect, dependencies) {
      const index = cursor++;
      const previous = slots[index];
      if (!previous || dependencies.some((value, i) => value !== previous.dependencies[i])) {
        const slot = slots[index] = { dependencies };
        effects.push(() => { previous?.cleanup?.(); slot.cleanup = effect(); });
      }
    },
  };
  const jsx = (type, props) => ({ type, props });
  const modules = {
    react, "react/jsx-runtime": { jsx, jsxs: jsx }, "lucide-react": {},
    "./pet": pet, "./petImport": petImport, "./petCreation": petCreation,
    "./Pet": { agentPet: () => ({ url: "hatchling", version: 2 }) },
    "./PetStage": { default: "PetStage" },
    "./PetContext": {
      petSheet: job => job?.manifest ? { url: "saved-sheet", version: 2 } : null,
      usePetImage: () => ({ ready: true, failed: false }),
    },
    "./uuid": { randomUuid: () => `import-${++uuid}` },
    "./api": {
      agentPath: key => `/agents/${key}`,
      ApiError: class extends Error {},
      async api(path, body) {
        requests.push({ path, body });
        if (path.endsWith("/imports")) return importResponse;
        assert.equal(path, "/agents/guide/pets");
        assert.equal(body, undefined);
        return [];
      },
    },
  };
  const exports = {};
  runInNewContext(source, {
    exports, require: name => { assert.ok(name in modules, name); return modules[name]; },
    sessionStorage: { getItem: () => null, setItem() {}, removeItem() {} },
    window: { setTimeout: () => 1 }, clearTimeout() {},
    URL: {
      createObjectURL(file) { const url = `blob:${urls.length}`; urls.push({ file, url }); return url; },
      revokeObjectURL(url) { revoked.push(url); },
    },
    Image: class { constructor() { probes.push(this); } },
    FormData: class {
      fields = new Map();
      set(key, value) { this.fields.set(key, value); }
      append(key, value) { this.fields.set(key, [...(this.fields.get(key) ?? []), value]); }
    },
  });
  const props = {
    agentKey: "guide", agentName: "Guide", selected: null,
    onSelect: async id => { selections.push(id); }, onRefreshSelection: async () => {}, onClose() {},
  };
  function render() {
    cursor = 0;
    tree = exports.default(props);
    effects.splice(0).forEach(effect => effect());
  }
  function elements(node = tree) {
    if (!node || typeof node !== "object") return [];
    return [node, ...[node.props?.children].flat(Infinity).flatMap(child => elements(child ?? null))];
  }
  const find = predicate => elements().find(predicate);
  const button = label => find(node => node.type === "button" && node.props.children === label);
  const pickerButton = label => find(node => node.type === "button" && node.props.children?.at?.(-1) === label);
  const fileInputs = () => elements().filter(node => node.type === "input" && node.props.type === "file");
  const clearButton = () => find(node => node.props?.["aria-label"] === "Choose different files");
  const nameInput = () => find(node => node.props?.id === "pet-import-name");
  function unmount() {
    if (!mounted) return;
    mounted = false;
    slots.forEach(slot => slot?.cleanup?.());
  }
  t.after(unmount);
  render();
  await settle();
  render();
  find(node => node.props?.role === "tab" && node.props.children.at(-1) === " Import").props.onClick();
  render();
  return {
    probes, urls, revoked, requests, selections, unmount, render,
    get lateWrites() { return lateWrites; },
    async flush() { await settle(); render(); },
    pick(files, folder = false) {
      const input = fileInputs()[folder ? 1 : 0];
      assert.ok(!input.props.disabled, "only enabled file inputs can be changed");
      input.props.onChange({ target: { files } });
      render();
    },
    drop(read) {
      let reads = 0;
      find(node => node.props?.onDrop).props.onDrop({
        ...event,
        dataTransfer: { items: [{ webkitGetAsEntry: () => ({
          isFile: true, fullPath: "/pet.json",
          file: (resolve, reject) => { reads++; read.promise.then(resolve, reject); },
        }) }] },
      });
      render();
      return reads;
    },
    clear() {
      assert.ok(!clearButton().props.disabled, "only an enabled clear button can be clicked");
      clearButton().props.onClick();
      render();
    },
    rename(value) {
      assert.ok(!nameInput().props.disabled, "only an enabled name input can be changed");
      nameInput().props.onChange({ target: { value } });
      render();
    },
    importControlsDisabled: () => ({
      clear: clearButton()?.props.disabled === true,
      name: nameInput()?.props.disabled === true,
      files: fileInputs()[0].props.disabled === true,
      folder: fileInputs()[1].props.disabled === true,
      chooseFiles: pickerButton(" Choose files").props.disabled === true,
      chooseFolder: pickerButton(" Choose folder").props.disabled === true,
    }),
    newPet() { find(node => node.props?.className?.includes("pet-card new")).props.onClick(); render(); },
    ready: () => !(button("Import pet") ?? button("Importing…")).props.disabled,
    preview: () => find(node => node.type === "PetStage")?.props.sheet.url,
    name: () => nameInput()?.props.value,
    alerts: () => elements().filter(node => node.props?.role === "alert").map(node => node.props.children),
    onSubmit: () => find(node => node.type === "form").props.onSubmit,
    async submit() { await find(node => node.type === "form").props.onSubmit(event); render(); },
    imports: () => requests.filter(request => request.path.endsWith("/imports")),
    respondWith(value) { importResponse = value; },
    usePet() { button("Use pet").props.onClick(); },
  };
}
function loaded(probe, width = 1536, height = 2288) {
  probe.naturalWidth = width;
  probe.naturalHeight = height;
  probe.onload();
}

for (const outcome of ["success", "invalid", "rejected"]) {
  for (const next of ["clear", "replacement", "current error"]) {
    test(`stale manifest ${outcome} cannot change a ${next} selection`, async t => {
      const view = await editor(t);
      const read = deferred();
      view.pick([manifest(read), sheet()]);
      await view.flush();
      if (next === "clear") view.pick([]);
      else if (next === "replacement") {
        view.pick([file("new.zip")]);
        await view.flush();
      } else {
        view.pick([manifest("{")]);
        await view.flush();
      }
      const before = { name: view.name(), alerts: view.alerts(), ready: view.ready() };
      if (outcome === "rejected") read.reject(new Error("Old manifest read failed"));
      else read.resolve(outcome === "success" ? '{"displayName":"Old pet"}' : "{");
      await view.flush();
      assert.deepEqual({ name: view.name(), alerts: view.alerts(), ready: view.ready() }, before);
      assert.equal(view.probes.length, 0, "stale manifests never allocate an image preview");
    });
  }
}

test("a pending replacement cannot submit a previously ready ZIP", async t => {
  const view = await editor(t);
  view.pick([file("old.zip")]);
  await view.flush();
  assert.equal(view.ready(), true);
  const read = deferred();
  view.pick([manifest(read), sheet()]);
  assert.equal(view.ready(), false);
  await view.submit();
  assert.equal(view.imports().length, 0);
  read.resolve("{}");
  await view.flush();
  assert.equal(view.ready(), false, "image probing is still pending");
});

for (const replacement of ["zip", "webp"]) {
  test(`a retained submit handler cannot import an old package after a replacement ${replacement} becomes ready before rendering`, async t => {
    const view = await editor(t);
    view.pick([file("old.zip")]);
    await view.flush();
    assert.equal(view.ready(), true);
    const oldSubmit = view.onSubmit();
    const next = file(`new.${replacement}`);
    view.pick([next]);
    await settle();
    if (replacement === "webp") loaded(view.probes[0]);

    await oldSubmit(event);
    assert.equal(view.imports().length, 0, "the retained handler must not upload the old package");

    view.render();
    assert.equal(view.ready(), true);
    await view.submit();
    assert.equal(view.imports().length, 1);
    assert.equal(view.imports()[0].body.fields.get("name"), "new");
    assert.deepEqual(view.imports()[0].body.fields.get("files"), [next]);
  });
}

for (const outcome of ["success", "failure"]) {
  test(`stale dropped-file collection ${outcome} cannot replace a new package`, async t => {
    const view = await editor(t);
    view.pick([file("old.zip")]);
    await view.flush();
    const read = deferred();
    view.drop(read);
    assert.equal(view.ready(), false, "dropping clears readiness before folder traversal finishes");
    await view.submit();
    assert.equal(view.imports().length, 0);
    view.pick([file("new.zip")]);
    await view.flush();
    if (outcome === "success") read.resolve(manifest("{}")); else read.reject(new Error("Old folder unavailable"));
    await view.flush();
    assert.equal(view.name(), "new");
    assert.equal(view.ready(), true);
    assert.deepEqual(view.alerts(), []);
  });
}

test("current dropped-file collection failure remains visible", async t => {
  const view = await editor(t);
  const read = deferred();
  view.drop(read);
  read.reject(new Error("Folder files are unavailable"));
  await view.flush();
  assert.match(view.alerts()[0], /Folder files are unavailable/);
  assert.equal(view.ready(), false);
});

for (const outcome of ["load", "error"]) {
  for (const next of ["clear", "replacement", "current error"]) {
    test(`stale image ${outcome} cannot change a ${next} selection`, async t => {
      const view = await editor(t);
      view.pick([sheet()]);
      await view.flush();
      const old = view.probes[0];
      if (next === "clear") view.clear();
      else {
        view.pick(next === "replacement" ? [file("new.webp")] : [manifest("{")]);
        await view.flush();
        if (next === "replacement") { loaded(view.probes[1]); await view.flush(); }
      }
      assert.ok(view.revoked.includes(old.src), "obsolete URLs are released before their probe settles");
      const before = { preview: view.preview(), name: view.name(), alerts: view.alerts(), ready: view.ready() };
      if (outcome === "load") loaded(old); else old.onerror();
      await view.flush();
      assert.deepEqual({ preview: view.preview(), name: view.name(), alerts: view.alerts(), ready: view.ready() }, before);
    });
  }
}

test("current valid image becomes ready only after probing and imports without selecting it", async t => {
  const view = await editor(t);
  const image = sheet();
  view.pick([manifest('{"displayName":"Copper Fox"}'), image]);
  await view.flush();
  assert.equal(view.name(), "Copper Fox");
  assert.equal(view.ready(), false);
  await view.submit();
  assert.equal(view.imports().length, 0);
  loaded(view.probes[0]);
  await view.flush();
  assert.equal(view.ready(), true);
  assert.equal(view.preview(), view.probes[0].src);
  await view.submit();
  assert.equal(view.imports().length, 1);
  assert.equal(view.imports()[0].body.fields.get("name"), "Copper Fox");
  assert.equal(view.imports()[0].body.fields.get("files")[1], image);
  assert.deepEqual(view.selections, []);
  assert.ok(view.revoked.includes(view.probes[0].src));
  view.usePet();
  await view.flush();
  assert.deepEqual(view.selections, [imported.id]);
});

test("current image failures and unsupported dimensions stay visible and cannot import", async t => {
  const view = await editor(t);
  view.pick([sheet()]);
  await view.flush();
  view.probes[0].onerror();
  await view.flush();
  assert.match(view.alerts()[0], /could not be read as an image/);
  assert.ok(view.revoked.includes(view.probes[0].src));
  assert.equal(view.ready(), false);
  await view.submit();
  assert.equal(view.imports().length, 0);
  view.pick([sheet()]);
  await view.flush();
  loaded(view.probes[1], 100, 100);
  await view.flush();
  assert.match(view.alerts()[0].join(""), /100x100/);
  assert.equal(view.ready(), false);
  await view.submit();
  assert.equal(view.imports().length, 0);
});

test("a current failed manifest read reports its error", async t => {
  const view = await editor(t);
  const read = deferred();
  view.pick([manifest(read), sheet()]);
  read.reject(new Error("Manifest file is unavailable"));
  await view.flush();
  assert.match(view.alerts()[0], /Manifest file is unavailable/);
  assert.equal(view.ready(), false);
});

for (const outcome of ["load", "error", "ready"]) {
  test(`unmount releases a ${outcome} image and ignores later callbacks`, async t => {
    const view = await editor(t);
    view.pick([sheet()]);
    await view.flush();
    const probe = view.probes[0];
    if (outcome === "ready") { loaded(probe); await view.flush(); }
    view.unmount();
    assert.ok(view.revoked.includes(probe.src));
    if (outcome === "error") probe.onerror(); else loaded(probe);
    assert.equal(view.lateWrites, 0);
  });
}

for (const outcome of ["success", "failure"]) {
  test(`unmount ignores pending manifest ${outcome}`, async t => {
    const view = await editor(t);
    const read = deferred();
    view.pick([manifest(read), sheet()]);
    await view.flush();
    view.unmount();
    if (outcome === "success") read.resolve("{}"); else read.reject(new Error("Read failed"));
    await settle();
    assert.equal(view.urls.length, 0);
    assert.equal(view.lateWrites, 0);
  });
}

test("uncertain import retries retain their id and keep the upload error visible", async t => {
  const view = await editor(t);
  view.pick([file("fox.zip")]);
  await view.flush();
  view.respondWith(Promise.reject(new Error("Upload response lost")));
  await view.submit();
  assert.ok(view.alerts().includes("Upload response lost"));
  view.respondWith(imported);
  await view.submit();
  assert.equal(view.imports().length, 2);
  assert.equal(view.imports()[0].body.fields.get("id"), view.imports()[1].body.fields.get("id"));
  assert.deepEqual(view.selections, []);
});

for (const outcome of ["success", "failure"]) {
  test(`package and name controls stay locked until import ${outcome}`, async t => {
    const view = await editor(t);
    const packageA = file("fox.zip");
    view.pick([packageA]);
    await view.flush();
    view.rename("Copper Fox");
    const unlocked = { clear: false, name: false, files: false, folder: false, chooseFiles: false, chooseFolder: false };
    assert.deepEqual(view.importControlsDisabled(), unlocked);

    const response = deferred();
    view.respondWith(response.promise);
    const submission = view.submit();
    view.render();
    assert.deepEqual(view.importControlsDisabled(), { clear: true, name: true, files: true, folder: true, chooseFiles: true, chooseFolder: true });
    assert.equal(view.ready(), false);
    // The drop handler needs its own guard; it is not a disabled native control.
    const read = deferred();
    assert.equal(view.drop(read), 0, "a busy drop must not start collecting files");
    assert.equal(view.name(), "Copper Fox");
    assert.equal(view.imports().length, 1);
    assert.equal(view.imports()[0].body.fields.get("name"), "Copper Fox");
    assert.deepEqual(view.imports()[0].body.fields.get("files"), [packageA]);

    if (outcome === "success") response.resolve(imported);
    else response.reject(new Error("Upload response lost"));
    await submission;
    assert.deepEqual(view.selections, [], "import completion never selects the pet");
    if (outcome === "success") {
      assert.equal(view.name(), undefined, "success opens the imported candidate");
      view.newPet();
      assert.equal(view.name(), undefined, "success clears the submitted package internally");
      assert.equal(view.ready(), false);
    } else {
      assert.equal(view.name(), "Copper Fox");
      assert.ok(view.alerts().includes("Upload response lost"));
      assert.equal(view.ready(), true);
      assert.deepEqual(view.importControlsDisabled(), unlocked);
      view.clear();
    }

    const packageB = file("otter.zip", "", "otter/otter.zip");
    view.pick([packageB], true);
    await view.flush();
    view.rename("River Otter");
    assert.equal(view.name(), "River Otter");
    assert.equal(view.ready(), true);
    assert.deepEqual(view.importControlsDisabled(), unlocked);
    assert.deepEqual(view.alerts(), []);
  });
}
