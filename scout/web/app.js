"use strict";
const $ = (id) => document.getElementById(id);
const MAX_PHOTOS = 10;
const MAX_BYTES = 25_000_000;
const PHASES = [["survey", "Survey"], ["crop", "Crop"], ["weeds", "Weeds"], ["check", "Check the picture"],
                ["zoom", "Zoom"], ["name", "Name the weeds"], ["measure", "Measure"]];
const TOOL_STEPS = new Set(["survey", "segment", "mark", "zoom", "classify", "measure", "finish"]);
const WALK_ID = /^[0-9a-f]{16}$/;

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

function clock(seconds) {
  const s = Math.max(Math.round(seconds), 0);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

// Upload view
const form = $("walk-form");
const filesInput = $("files");
const dropZone = $("drop-zone");
const scoutButton = $("scout-button");
let chosen = [];

function showError(message) {
  $("form-error").textContent = message || "";
  $("form-error").hidden = !message;
}

function setFiles(files) {
  chosen = Array.from(files).filter((f) => f.type.startsWith("image/") || /\.(heic|heif)$/i.test(f.name));
  const thumbs = $("thumbs");
  thumbs.replaceChildren();
  let bytes = 0;
  for (const file of chosen) {
    bytes += file.size;
    if (thumbs.children.length < MAX_PHOTOS && file.type.startsWith("image/") && !/hei[cf]/i.test(file.type)) {
      const img = el("img");
      img.alt = "";
      img.src = URL.createObjectURL(file);
      img.onload = () => URL.revokeObjectURL(img.src);
      thumbs.append(img);
    }
  }
  if (chosen.length) thumbs.append(el("span", null, `${chosen.length} photo${chosen.length === 1 ? "" : "s"} · ${(bytes / 1e6).toFixed(1)} MB`));
  $("drop-text").textContent = chosen.length ? "Change photos" : "Drop photos here or tap to choose";
  showError("");
}

filesInput.addEventListener("change", () => setFiles(filesInput.files));
for (const type of ["dragenter", "dragover"]) {
  dropZone.addEventListener(type, (event) => { event.preventDefault(); dropZone.classList.add("dragging"); });
}
for (const type of ["dragleave", "drop"]) {
  dropZone.addEventListener(type, (event) => { event.preventDefault(); dropZone.classList.remove("dragging"); });
}
dropZone.addEventListener("drop", (event) => setFiles(event.dataTransfer.files));

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const crop = form.elements.crop.value;
  if (!crop) return showError("Choose the crop first.");
  if (!chosen.length) return showError("Add 1 to 10 photos from one walk.");
  if (chosen.length > MAX_PHOTOS) return showError(`That's ${chosen.length} photos; the limit is ${MAX_PHOTOS}.`);
  const big = chosen.find((f) => f.size > MAX_BYTES);
  if (big) return showError(`${big.name} is over 25 MB.`);
  const body = new FormData();
  body.append("crop", crop);
  for (const file of chosen) body.append("files", file, file.name);
  scoutButton.disabled = true;
  scoutButton.textContent = "Uploading…";
  try {
    const response = await fetch("/walks", { method: "POST", body });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      showError(typeof data.detail === "string" ? data.detail : "Upload failed. Try again.");
      return;
    }
    openLive(data.walk_id, crop, true);
  } catch {
    showError("Couldn't reach the server. Try again.");
  } finally {
    scoutButton.disabled = false;
    scoutButton.textContent = "Scout this field";
  }
});

fetch("/samples").then((r) => r.json()).then((samples) => {
  if (!samples.length) return;
  const list = $("sample-list");
  for (const sample of samples) {
    const card = el("div", "sample");
    const img = el("img");
    img.alt = "";
    if (sample.thumbnail) img.src = sample.thumbnail;
    const body = el("div");
    const meta = [sample.stops ? `${sample.stops} stops` : "", sample.decision || ""].filter(Boolean).join(" · ");
    const links = el("div", "links");
    const watch = el("a", null, "Watch the agent");
    watch.href = `/?walk=${sample.walk_id}`;
    const report = el("a", null, "Report");
    report.href = `/walks/${sample.walk_id}/report`;
    links.append(watch, report);
    body.append(el("div", "title", sample.name), el("div", "meta", meta), links);
    card.append(img, body);
    list.append(card);
  }
  $("samples").hidden = false;
}).catch(() => {});

// Live view
const live = { stops: new Map(), focus: null, pinned: false, total: 0, done: 0, replay: false, budget: null,
               reportUrl: "", timer: null, startedAt: 0, source: null };

function openLive(walkId, crop, push) {
  if (push) history.pushState(null, "", `/?walk=${walkId}`);
  $("upload-view").hidden = true;
  $("live-view").hidden = false;
  $("live-title").textContent = crop ? `Scouting your ${crop} field` : "Scouting the field";
  live.reportUrl = `/walks/${walkId}/report`;
  $("report-link").href = live.reportUrl;
  $("skip-link").href = live.reportUrl;
  live.startedAt = Date.now();
  live.timer = setInterval(() => { if (!live.replay) $("elapsed").textContent = clock((Date.now() - live.startedAt) / 1000); }, 1000);
  live.source = new EventSource(`/walks/${walkId}/events`);
  live.source.onmessage = (message) => handle(JSON.parse(message.data));
  live.source.onerror = () => {
    if (live.source.readyState === EventSource.CLOSED) {
      setStatus("This walk isn't running any more.", true);
    } else {
      setStatus("Reconnecting…");
    }
  };
}

function setStatus(text, offerUpload) {
  const status = $("live-status");
  status.replaceChildren(text ? el("span", null, `${text} `) : "");
  if (offerUpload) {
    const link = el("a", null, "Start a new walk");
    link.href = "/";
    status.append(link);
  }
}

function stopFor(index, name) {
  let stop = live.stops.get(index);
  if (!stop) {
    stop = { index, name: name || `Photo ${index + 1}`, status: "waiting", picture: null, caption: "", thumb: null,
             thought: "", detail: [], problem: "", final: null, startT: null, lastT: 0, calls: 0, estimate: null,
             plants: 0, areas: 0,
             phases: Object.fromEntries(PHASES.map(([key]) => [key, { state: "pending", summary: "" }])) };
    const button = el("button");
    button.type = "button";
    button.append(el("span", "mark"));
    button.addEventListener("click", () => pin(index));
    stop.button = button;
    live.stops.set(index, stop);
    const ordered = [...live.stops.values()].sort((a, b) => a.index - b.index);
    $("strip").replaceChildren(...ordered.map((s) => s.button));
    if (live.focus === null) live.focus = index;
  } else if (name) {
    stop.name = name;
  }
  return stop;
}

function pin(index) {
  live.pinned = true;
  live.focus = index;
  $("follow-live").hidden = false;
  render();
}

$("follow-live").addEventListener("click", () => {
  live.pinned = false;
  $("follow-live").hidden = true;
  render();
});

function touch(stop) {
  if (!live.pinned) live.focus = stop.index;
  render();
}

function phaseOf(stop, event) {
  const summary = event.summary || "";
  const weedsDone = stop.phases.weeds.state === "done";
  switch (event.tool) {
    case "survey": return "survey";
    case "mark":
      if (summary.startsWith("Mark crop")) return weedsDone ? "check" : "crop";
      return weedsDone ? "check" : "weeds";
    case "segment": return summary.includes('"crop row"') ? "crop" : (weedsDone ? "check" : "weeds");
    case "zoom": return "zoom";
    case "classify": return "name";
    case "measure": return "measure";
    default: return null;
  }
}

function applyStep(stop, event) {
  const summary = event.summary || "";
  if (TOOL_STEPS.has(event.tool)) stop.calls += 1;
  if (event.t != null) stop.lastT = event.t;
  const estimate = summary.match(/about (\d+) weeds/);
  if (event.tool === "survey" && estimate) stop.estimate = Number(estimate[1]);
  const weeds = summary.match(/^Mark weed: (\d+) of/);
  if (weeds) stop.plants += Number(weeds[1]);
  const areas = summary.match(/^Mark patch: (\d+) of/);
  if (areas) stop.areas += Number(areas[1]);
  if (event.ok === false) stop.problem = summary;
  const key = phaseOf(stop, event);
  if (!key) return;
  const at = PHASES.findIndex(([k]) => k === key);
  for (const [k] of PHASES) if (stop.phases[k].state === "current") stop.phases[k].state = "pending";
  stop.phases[key] = { state: event.ok === false ? "failed" : "done", summary };
  const next = PHASES.slice(at + 1).find(([k]) => stop.phases[k].state === "pending");
  if (next) stop.phases[next[0]].state = "current";
}

function finishPhases(stop) {
  for (const [k] of PHASES) {
    if (stop.phases[k].state === "pending" || stop.phases[k].state === "current") stop.phases[k].state = "skipped";
  }
}

function countsLine(stop) {
  const parts = [];
  if (stop.estimate != null) parts.push(`about ${stop.estimate} weeds seen`);
  if (stop.plants || stop.areas) {
    let marked = `marked ${stop.plants} plant${stop.plants === 1 ? "" : "s"}`;
    if (stop.areas) marked += ` and ${stop.areas} weed area${stop.areas === 1 ? "" : "s"}`;
    parts.push(marked);
  }
  if (live.budget && stop.status === "working" && stop.startT != null) {
    const calls = Math.max(live.budget.calls - stop.calls, 0);
    parts.push(`${calls} calls and ${clock(live.budget.seconds - (stop.lastT - stop.startT))} left`);
  }
  if (stop.final && stop.final.status !== "failed") parts.push(`cover ${stop.final.cover}% · ${stop.final.weeds} weeds`);
  return parts.join(" · ");
}

function showPicture(url, caption) {
  const img = $("stage-img");
  img.alt = caption;
  if (!url || img.dataset.src === url) return;
  img.dataset.src = url;
  img.classList.add("fading");
  const next = new Image();
  next.onload = () => {
    if (img.dataset.src !== url) return;
    img.src = url;
    img.classList.remove("fading");
  };
  next.src = url;
}

function render() {
  for (const stop of live.stops.values()) {
    const b = stop.button;
    b.dataset.status = stop.status;
    b.setAttribute("aria-label", `${stop.name}, ${stop.status}`);
    b.setAttribute("aria-current", String(stop.index === live.focus));
    if (stop.thumb) b.style.backgroundImage = `url("${stop.thumb}")`;
    b.querySelector(".mark").textContent = stop.status === "done" ? "✓" : stop.status === "failed" ? "!" : "";
  }
  const stop = live.stops.get(live.focus);
  if (!stop) return;
  showPicture(stop.picture, `${stop.name}: ${stop.caption}`);
  const caption = $("stage-caption");
  caption.textContent = `${stop.name} · ${stop.problem || stop.caption || "waiting to start"}`;
  caption.classList.toggle("problem", Boolean(stop.problem) && stop.status === "failed");
  $("thought-text").textContent = stop.thought ? `“${stop.thought}”` : "";
  $("thought-detail").replaceChildren(...stop.detail.slice(0, 3).map((line) => el("li", null, line)));
  $("checklist").replaceChildren(...PHASES.map(([key, label]) => {
    const phase = stop.phases[key];
    const item = el("li", phase.state);
    item.append(el("span", "step-name", label));
    if (phase.summary) item.append(el("span", "step-summary", phase.summary));
    return item;
  }));
  $("stage-counts").textContent = countsLine(stop);
}

function progress() {
  $("progress-bar").style.width = live.total ? `${(live.done / live.total) * 100}%` : "0";
  $("live-progress-text").textContent = `${live.done} of ${live.total} photos done`;
}

function ended() {
  clearInterval(live.timer);
  if (live.source) live.source.close();
  $("live-end").hidden = false;
  $("live-progress-text").textContent = "Report ready";
  setTimeout(() => { if (!live.pinned) location.href = live.reportUrl; }, 3000);
}

function handle(event) {
  setStatus("");
  if (event.replay && !live.replay) {
    live.replay = true;
    $("replay-badge").hidden = false;
    $("skip-link").hidden = false;
  }
  if (live.replay && event.t != null) $("elapsed").textContent = clock(event.t);
  switch (event.type) {
    case "walk_start":
      live.total = event.stops.length;
      live.budget = event.budget || null;
      $("live-title").textContent = `Scouting your ${event.crop} field`;
      for (const s of event.stops) stopFor(s.stop, s.name);
      progress();
      render();
      break;
    case "stop_start": {
      const stop = stopFor(event.stop);
      stop.status = "working";
      stop.startT = event.t;
      stop.lastT = event.t;
      stop.thumb = event.picture;
      stop.picture = event.picture;
      stop.caption = "Photo";
      stop.phases.survey.state = "current";
      touch(stop);
      break;
    }
    case "step": {
      const stop = stopFor(event.stop);
      applyStep(stop, event);
      if (event.picture) {
        stop.picture = event.picture;
        stop.caption = event.tool === "zoom" ? "Close-up" : "What is marked now: green crop, magenta weeds, blue weed areas";
      }
      touch(stop);
      break;
    }
    case "thought": {
      const stop = stopFor(event.stop);
      stop.thought = event.text;
      stop.detail = event.detail || [];
      touch(stop);
      break;
    }
    case "stop_done": {
      const stop = stopFor(event.stop);
      stop.status = event.status === "failed" ? "failed" : "done";
      stop.final = event;
      if (event.picture) {
        stop.picture = event.picture;
        stop.caption = "Final overlay";
      }
      finishPhases(stop);
      live.done += 1;
      progress();
      touch(stop);
      break;
    }
    case "report_building":
      $("live-progress-text").textContent = "All photos done. Building the report…";
      break;
    case "report_ready":
      ended();
      break;
    case "error":
      if (live.source) live.source.close();
      clearInterval(live.timer);
      setStatus(event.message, true);
      break;
  }
}

window.addEventListener("popstate", () => location.reload());
const startWalk = new URLSearchParams(location.search).get("walk");
if (startWalk && WALK_ID.test(startWalk)) openLive(startWalk, "", false);
