// Screenshot the live demos over the Chrome DevTools Protocol.
//
// Why not `chrome --screenshot`: it fires at the load event, which is before
// this demo's async boot (hardware detect -> manifest fetch -> render) has
// painted the panel, and --virtual-time-budget starves the network so the page
// comes out blank. Driving CDP directly lets us wait for a real condition and
// capture when the pixels are actually there.
//
// Zero dependencies: Node 22+ has a global WebSocket and fetch.
//
//   node tools/shot.mjs <jobs.json>
//
// jobs.json: [{ "url": "...", "out": "...", "waitFor": "css", "waitMs": 1500 }]

import { spawn } from "node:child_process";
import { mkdir, writeFile, rm } from "node:fs/promises";
import path from "node:path";
import os from "node:os";

const CHROME = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const PORT = 9333;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function cdpTargets() {
  const res = await fetch(`http://127.0.0.1:${PORT}/json/list`);
  return res.json();
}

class Session {
  constructor(ws) {
    this.ws = ws;
    this.id = 0;
    this.pending = new Map();
    this.events = [];
    this.console = [];
    this.errors = [];
    ws.addEventListener("message", (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.id && this.pending.has(msg.id)) {
        const { resolve, reject } = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        msg.error ? reject(new Error(JSON.stringify(msg.error))) : resolve(msg.result);
      } else if (msg.method) {
        if (msg.method === "Runtime.consoleAPICalled") {
          this.console.push({
            level: msg.params.type,
            text: (msg.params.args ?? []).map((a) => a.value ?? a.description ?? a.type).join(" "),
          });
        } else if (msg.method === "Runtime.exceptionThrown") {
          const d = msg.params.exceptionDetails;
          this.errors.push(d.exception?.description ?? d.text ?? "exception");
        } else if (msg.method === "Log.entryAdded") {
          const e = msg.params.entry;
          this.console.push({ level: e.level, text: `[${e.source}] ${e.text}` });
        }
        this.events.push(msg);
      }
    });
  }
  send(method, params = {}) {
    const id = ++this.id;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.ws.send(JSON.stringify({ id, method, params }));
      setTimeout(() => {
        if (this.pending.has(id)) {
          this.pending.delete(id);
          reject(new Error(`CDP timeout: ${method}`));
        }
      }, 60_000);
    });
  }
  async evaluate(expression) {
    const r = await this.send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
    if (r.exceptionDetails) throw new Error(r.exceptionDetails.text ?? "evaluate threw");
    return r.result?.value;
  }
}

// Poll a JS expression until truthy, so we never capture a half-painted frame.
async function waitFor(s, expression, timeoutMs = 30_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      if (await s.evaluate(expression)) return true;
    } catch { /* page may still be swapping documents */ }
    await sleep(250);
  }
  return false;
}

async function shoot(jobs, extraArgs = []) {
  const profile = path.join(os.tmpdir(), "opencode", "cdp-profile");
  await rm(profile, { recursive: true, force: true });
  const chrome = spawn(CHROME, [
    "--headless=new",
    `--remote-debugging-port=${PORT}`,
    `--user-data-dir=${profile}`,
    "--no-first-run",
    "--no-default-browser-check",
    "--hide-scrollbars",
    "--force-device-scale-factor=1",
    "--window-size=1440,900",
    // Lets headless expose a software WebGPU/WebGL adapter instead of nothing,
    // so the demo exercises its real GPU path rather than only the CPU one.
    "--enable-unsafe-swiftshader",
    ...extraArgs,
    "about:blank",
  ], { stdio: "ignore", detached: false });

  // wait for the debugging endpoint
  for (let i = 0; i < 60; i++) {
    try { await fetch(`http://127.0.0.1:${PORT}/json/version`); break; } catch { await sleep(500); }
  }

  const out = [];
  try {
    for (const job of jobs) {
      const targets = await cdpTargets();
      const page = targets.find((t) => t.type === "page");
      if (!page) throw new Error("no page target");

      const ws = new WebSocket(page.webSocketDebuggerUrl);
      await new Promise((res, rej) => {
        ws.addEventListener("open", res, { once: true });
        ws.addEventListener("error", rej, { once: true });
      });
      const s = new Session(ws);

      await s.send("Page.enable");
      await s.send("Runtime.enable");
      await s.send("Log.enable");
      await s.send("Emulation.setDeviceMetricsOverride", {
        width: job.width ?? 1440, height: job.height ?? 900, deviceScaleFactor: 1, mobile: false,
      });
      await s.send("Page.navigate", { url: job.url });

      // The real readiness signal: the demo's own DOM says it is ready.
      const ok = job.waitFor
        ? await waitFor(s, job.waitFor, job.timeoutMs ?? 45_000)
        : (await sleep(job.waitMs ?? 2500), true);
      if (job.waitFor && !ok) out.push({ out: job.out, warn: "waitFor never became true" });

      if (job.script) await s.evaluate(job.script);
      // Second phase: after driving the page (e.g. clicking "Load model" and
      // waiting out a 348 MB weight download), wait for *that* to settle.
      if (job.thenWaitFor) {
        const ok2 = await waitFor(s, job.thenWaitFor, job.thenTimeoutMs ?? 900_000);
        if (!ok2) out.push({ out: job.out, warn: "thenWaitFor never became true" });
      }
      if (job.after) await sleep(job.after);

      const shot = await s.send("Page.captureScreenshot", {
        format: "png",
        captureBeyondViewport: !!job.fullPage,
        ...(job.fullPage
          ? { clip: await s.evaluate(`(()=>{const r=document.documentElement.getBoundingClientRect();return {x:0,y:0,width:Math.ceil(r.width),height:Math.ceil(r.height),scale:1};})()`) }
          : {}),
      });
      await mkdir(path.dirname(job.out), { recursive: true });
      await writeFile(job.out, Buffer.from(shot.data, "base64"));

      // Report what the page itself measured, so the caption can be honest.
      let facts = null;
      if (job.report) facts = await s.evaluate(job.report).catch(() => null);
      out.push({ out: job.out, url: job.url, facts, errors: s.errors, console: job.quiet ? undefined : s.console.slice(-12) });

      ws.close();
    }
  } finally {
    try { await fetch(`http://127.0.0.1:${PORT}/json/close/x`).catch(() => {}); } catch { /* ignore */ }
    chrome.kill();
  }
  return out;
}

const jobs = JSON.parse(process.argv[2] ? await (await import("node:fs/promises")).readFile(process.argv[2], "utf8") : "[]");
const res = await shoot(jobs, process.argv[3] ? process.argv[3].split(" ").filter(Boolean) : []);
for (const r of res) console.log(JSON.stringify(r));
