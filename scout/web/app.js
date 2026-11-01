const form = document.getElementById("walk-form");
const errorBox = document.getElementById("form-error");
const stopsBox = document.getElementById("stops");
const statusBox = document.getElementById("status");
const button = form.querySelector("button");

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function stopList(index) {
  let list = document.getElementById(`stop-${index}`);
  if (!list) {
    const section = document.createElement("section");
    const heading = document.createElement("h3");
    heading.textContent = `Stop ${index + 1}`;
    list = document.createElement("ol");
    list.id = `stop-${index}`;
    section.append(heading, list);
    stopsBox.appendChild(section);
  }
  return list;
}

function addLine(index, text, className) {
  const item = document.createElement("li");
  item.textContent = text;
  if (className) item.className = className;
  stopList(index).appendChild(item);
}

function follow(walkId) {
  document.getElementById("progress").hidden = false;
  stopsBox.replaceChildren();
  statusBox.textContent = "Working…";
  const events = new EventSource(`/walks/${walkId}/events`);
  events.onopen = () => {
    statusBox.textContent = "Working…";
  };
  events.onmessage = (message) => {
    const event = JSON.parse(message.data);
    if (event.type === "step") {
      addLine(event.stop, event.summary, event.ok ? "" : "failed");
    } else if (event.type === "stop_done") {
      addLine(event.stop, event.status === "ok" ? "Stop done" : `Stop ${event.status.replaceAll("_", " ")}`, "done");
    } else if (event.type === "report_ready") {
      events.close();
      button.disabled = false;
      window.location.href = `/walks/${walkId}/report`;
    } else if (event.type === "error") {
      events.close();
      button.disabled = false;
      statusBox.textContent = event.message;
    }
  };
  // The browser reconnects by itself and sends Last-Event-ID, so an error is not the end. It gives up (CLOSED)
  // when the server no longer knows the walk, for example after a restart.
  events.onerror = () => {
    if (events.readyState === EventSource.CLOSED) {
      button.disabled = false;
      statusBox.textContent = "Lost track of this walk. Reload and upload the photos again.";
    } else {
      statusBox.textContent = "Connection dropped. Reconnecting…";
    }
  };
}

form.addEventListener("submit", async (submitEvent) => {
  submitEvent.preventDefault();
  errorBox.hidden = true;
  const count = form.elements.files.files.length;
  if (count < 1 || count > 10) {
    showError("Pick 1 to 10 photos.");
    return;
  }
  button.disabled = true;
  button.textContent = "Uploading…";
  try {
    const response = await fetch("/walks", { method: "POST", body: new FormData(form) });
    const body = await response.json();
    if (!response.ok) {
      showError(typeof body.detail === "string" ? body.detail : "Upload failed. Try again.");
      button.disabled = false;
      return;
    }
    follow(body.walk_id);
  } catch {
    showError("Couldn't reach the server. Try again.");
    button.disabled = false;
  } finally {
    button.textContent = "Scout this field";
  }
});

fetch("/samples")
  .then((response) => response.json())
  .then((samples) => {
    if (!samples.length) return;
    const list = document.getElementById("sample-list");
    for (const sample of samples) {
      const item = document.createElement("li");
      const link = document.createElement("a");
      link.href = `/walks/${sample.walk_id}/report`;
      link.textContent = sample.name;
      item.appendChild(link);
      list.appendChild(item);
    }
    document.getElementById("samples").hidden = false;
  })
  .catch(() => {});
