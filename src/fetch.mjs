/**
 * fetch.mjs — cached network access shared by every family.
 *
 * Two problems this solves:
 *
 * 1. Hugging Face redirects to a *signed* CDN URL that changes per request, so
 *    the browser HTTP cache never hits for model weights. We store responses
 *    under the stable `huggingface.co/.../resolve/<rev>/...` URL in the Cache
 *    API instead, so a page refresh is free — offline works too.
 * 2. Config files (calibration, manifests, tokenizers) used to be fetched with
 *    plain `fetch()` — or worse, `fetch(url, {cache: "no-store"})` — so every
 *    visit re-requested them. They now go through `fetchCachedJson` too, which
 *    is what keeps the demo's own request counter at zero on a reload.
 *
 * `fetchCachedBytes` / `fetchOnnxExternalData` / `requestPersistentStorage`
 * were born in laya.mjs and are re-exported there for compatibility.
 */

export const MODEL_CACHE = "kf-models-v1";

export class HttpError extends Error {
  constructor(url, status) {
    super(`fetch ${url} failed (HTTP ${status ?? "no response"})`);
    this.name = "HttpError";
    this.url = url;
    this.status = status;
  }
}

export class JsonParseError extends Error {
  constructor(url, cause) {
    super(`fetch ${url}: response is not valid JSON: ${cause?.message ?? cause}`);
    this.name = "JsonParseError";
    this.url = url;
    this.cause = cause;
  }
}

async function modelCacheStore(cacheName = MODEL_CACHE) {
  try {
    if (typeof caches === "undefined") return null;
    // Real CacheStorage needs open(); test doubles implement match/put directly.
    if (typeof caches.open === "function") return await caches.open(cacheName);
    return caches;
  } catch {
    return null;
  }
}

/** Ask for persistent storage once so big weights survive eviction pressure. */
export async function requestPersistentStorage() {
  try {
    return (await globalThis.navigator?.storage?.persist?.()) ?? false;
  } catch {
    return false;
  }
}

/**
 * Fetch bytes with a Cache-API layer (see the module header).
 *
 * @returns {Promise<Uint8Array>}
 */
export async function fetchCachedBytes(url, {
  fetchImpl = fetch,
  onProgress = null,
  cacheName = MODEL_CACHE,
  force = false,
  persist = false,
} = {}) {
  const store = await modelCacheStore(cacheName);
  if (persist) await requestPersistentStorage();
  if (store && !force) {
    try {
      const hit = await store.match(url);
      if (hit) {
        const data = new Uint8Array(await hit.arrayBuffer());
        onProgress?.({ phase: "cache", file: String(url).split("/").pop(), loaded: data.length, total: data.length });
        return data;
      }
    } catch {
      /* cache read failed — fall through to the network */
    }
  }
  const res = await fetchImpl(url);
  if (!res?.ok) throw new HttpError(url, res?.status);
  const data = await readBytesWithProgress(res, url, onProgress);
  if (store) {
    try {
      await store.put(url, new Response(data, {
        headers: { "content-type": "application/octet-stream", "content-length": String(data.length) },
      }));
    } catch {
      /* quota exceeded / private mode — caching is best-effort */
    }
  }
  return data;
}

/**
 * Fetch and parse JSON through the byte cache.
 *
 * @param {object} [opts]
 *  nullOn404 — return null instead of throwing for a 404 (callers probing
 *  optional files walk the filename list without catching). A *definite* 404
 *  is remembered in the Cache API, so a reload does not re-probe it; `force`
 *  skips that memo like it skips the byte cache.
 * @returns {Promise<any|null>}
 */
export async function fetchCachedJson(url, { nullOn404 = false, force = false, cacheName = MODEL_CACHE, ...opts } = {}) {
  const absentKey = `${url}${JSON_404}`;
  if (nullOn404 && !force && await readCachedBytes(absentKey, cacheName)) return null;
  let bytes;
  try {
    bytes = await fetchCachedBytes(url, { cacheName, force, ...opts });
  } catch (e) {
    if (nullOn404 && e instanceof HttpError && e.status === 404) {
      await writeCachedBytes(absentKey, new TextEncoder().encode("1"), cacheName);
      return null;
    }
    // Fakes (and some runtimes) hand back a parsed body; a SyntaxError from
    // res.json() surfaces here and is a JSON problem, not a transport one.
    if (e instanceof SyntaxError) throw new JsonParseError(url, e);
    throw e;
  }
  try {
    return JSON.parse(new TextDecoder().decode(bytes));
  } catch (e) {
    throw new JsonParseError(url, e);
  }
}

/** Read a cached response body without touching the network (null on miss). */
async function readCachedBytes(url, cacheName = MODEL_CACHE) {
  const store = await modelCacheStore(cacheName);
  if (!store) return null;
  try {
    const hit = await store.match(url);
    return hit ? new Uint8Array(await hit.arrayBuffer()) : null;
  } catch {
    return null;
  }
}

/** Best-effort cache write (quota exceeded / private mode are ignored). */
async function writeCachedBytes(url, bytes, cacheName = MODEL_CACHE) {
  const store = await modelCacheStore(cacheName);
  if (!store) return false;
  try {
    await store.put(url, new Response(bytes, {
      headers: { "content-type": "application/octet-stream", "content-length": String(bytes.length) },
    }));
    return true;
  } catch {
    return false;
  }
}

async function readBytesWithProgress(res, url, onProgress) {
  const total = Number(res.headers?.get?.("content-length")) || 0;
  const file = String(url).split("/").pop();
  if (res.body?.getReader) {
    const reader = res.body.getReader();
    let received = 0;
    let data = total ? new Uint8Array(total) : null;
    const chunks = total ? null : [];
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (data) {
        if (received + value.length > data.length) {
          // content-length can describe the *encoded* size (gzip/br), so the
          // decoded stream may be larger — grow instead of overflowing.
          const grown = new Uint8Array(Math.max(data.length * 2, received + value.length));
          grown.set(data.subarray(0, received));
          data = grown;
        }
        data.set(value, received);
      } else {
        chunks.push(value);
      }
      received += value.length;
      onProgress?.({ phase: "download", file, loaded: received, total: total || null });
    }
    if (!data) {
      data = new Uint8Array(received);
      let off = 0;
      for (const c of chunks) { data.set(c, off); off += c.length; }
    } else {
      // the buffer may have grown past the decoded length (content-length is
      // the *encoded* size) — always trim to what was actually received
      data = data.subarray(0, received);
    }
    return data;
  }
  let data;
  if (typeof res.arrayBuffer === "function") {
    data = new Uint8Array(await res.arrayBuffer());
  } else if (typeof res.text === "function") {
    data = new TextEncoder().encode(await res.text());
  } else if (typeof res.json === "function") {
    data = new TextEncoder().encode(JSON.stringify(await res.json()));
  } else {
    throw new Error(`fetch ${url}: response has no readable body`);
  }
  onProgress?.({ phase: "download", file, loaded: data.length, total: data.length });
  return data;
}

// Negative-cache key suffix for "this graph has no external-data sibling".
const NO_EXTERNAL_DATA = "#jev-no-external-data";

// Negative-cache key suffix for "this JSON file is a definite 404".
const JSON_404 = "#jev-json-404";

/**
 * ONNX external-data loader.
 *
 * The pinned Laya export stores weights in `<model>.onnx.data` siblings
 * (encoder ~468 MB, head ~53 MB). onnxruntime-web cannot resolve those from a
 * URL by itself (`Module.MountedFiles is not available`); it needs the bytes
 * passed via `session_options.externalData`. Fetch the sibling (streaming,
 * with progress) and hand it over.
 *
 * Probes `<model>.onnx.data` then `<model>.onnx_data`; a definite 404 on both
 * is remembered in the Cache API so a reload does not re-probe two URLs per
 * graph into the host's request counter.
 *
 * @returns {Promise<{path:string, data:Uint8Array}|null>} null when the model
 *   has no external data.
 */
export async function fetchOnnxExternalData(url, { fetchImpl = fetch, onProgress = null, force = false } = {}) {
  const store = await modelCacheStore();
  const marker = `${url}${NO_EXTERNAL_DATA}`;
  if (store && !force) {
    try {
      if (await store.match(marker)) return null;
    } catch { /* marker read is best-effort */ }
  }
  let absent = false;
  let uncertain = false;
  for (const suffix of [".data", "_data"]) {
    const dataUrl = `${url}${suffix}`;
    const path = `${String(url).split("/").pop()}${suffix}`;
    // Probe cheaply first: a 404 means "try the next suffix". Any other
    // unhappy HEAD (5xx, or a proxy that blocks HEAD) only skips the *probe* —
    // fall through to the GET, which is the operation we actually need; a
    // blocked HEAD must never be the reason existing weights go unfetched.
    const head = await fetchImpl(dataUrl, { method: "HEAD" }).catch(() => null);
    if (head && !head.ok && head.status === 404) {
      absent = true;
      continue;
    }
    try {
      const data = await fetchCachedBytes(dataUrl, { fetchImpl, onProgress, force, persist: true });
      return { path, data };
    } catch (e) {
      if (e instanceof HttpError && e.status === 404) absent = true;
      else uncertain = true;
    }
  }
  if (store && absent && !uncertain) {
    try {
      await store.put(marker, new Response("1", { headers: { "content-type": "text/plain" } }));
    } catch { /* best-effort */ }
  }
  return null;
}
