const TILE_FILES = {
  road: "/assets/road.jpg",
  grass: "/assets/grass.jpg",
  mud: "/assets/mud.jpg",
  ford: "/assets/ford.jpg",
  obstacle: "/assets/obstacle.jpg",
  close: "/assets/close.jpg",
  maintenance: "/assets/maintenance.jpg",
};

const TRANSPARENT_SPRITES = {
  agent: "/assets/a.jpg",
  rival: "/assets/r.jpg",
  stop1: "/assets/1.jpg",
  stop2: "/assets/2.jpg",
  stop3: "/assets/3.jpg",
};

const IMAGES = {};
let stateData = null;
let isAutoPlaying = false;
let isStepping = false;
let countdownTimer = null;
let remainingSec = 10.0;
let lastOverrides = null;

function removeWhiteBackground(img) {
  const off = document.createElement("canvas");
  off.width = img.width;
  off.height = img.height;
  const octx = off.getContext("2d");
  octx.drawImage(img, 0, 0);
  const imgData = octx.getImageData(0, 0, off.width, off.height);
  const d = imgData.data;

  const bgR = d[0], bgG = d[1], bgB = d[2];
  for (let i = 0; i < d.length; i += 4) {
    const r = d[i], g = d[i + 1], b = d[i + 2];
    const distCorner = Math.abs(r - bgR) + Math.abs(g - bgG) + Math.abs(b - bgB);
    const isNearWhite = r > 232 && g > 232 && b > 232;
    if (distCorner < 45 || isNearWhite) {
      d[i + 3] = 0;
    }
  }
  octx.putImageData(imgData, 0, 0);
  return off;
}

function preloadImages() {
  const promises = [];
  for (const [key, src] of Object.entries(TILE_FILES)) {
    promises.push(new Promise((resolve) => {
      const img = new Image();
      img.onload = () => { IMAGES[key] = img; resolve(); };
      img.onerror = () => resolve();
      img.src = src;
    }));
  }
  for (const [key, src] of Object.entries(TRANSPARENT_SPRITES)) {
    promises.push(new Promise((resolve) => {
      const img = new Image();
      img.onload = () => { IMAGES[key] = removeWhiteBackground(img); resolve(); };
      img.onerror = () => resolve();
      img.src = src;
    }));
  }
  return Promise.all(promises);
}

function getTimerLimit() {
  return Math.max(3, parseFloat(document.getElementById("pTimerLimit").value) || 10);
}

function startStepCountdown() {
  if (countdownTimer) clearInterval(countdownTimer);
  remainingSec = getTimerLimit();
  updateClockDisplay();

  countdownTimer = setInterval(async () => {
    remainingSec = Math.max(0, remainingSec - 0.1);
    updateClockDisplay();

    if (remainingSec <= 0) {
      remainingSec = getTimerLimit();
      if (isAutoPlaying) {
        await stepSimulation();
      }
    }
  }, 100);
}

function updateClockDisplay() {
  const clockImg = document.getElementById("clockIcon");
  const timerCard = document.getElementById("timerCard");
  const timerText = document.getElementById("timerText");

  timerText.textContent = `${remainingSec.toFixed(1)}s`;
  if (remainingSec <= 2.0) {
    if (!clockImg.src.endsWith("/assets/clock_red.jpg")) {
      clockImg.src = "/assets/clock_red.jpg";
    }
    timerCard.classList.add("urgent");
  } else {
    if (!clockImg.src.endsWith("/assets/clock.jpg")) {
      clockImg.src = "/assets/clock.jpg";
    }
    timerCard.classList.remove("urgent");
  }
}

function parseStops(text) {
  const pairs = text.split(";").map((s) => s.trim()).filter((s) => s.length > 0);
  return pairs.map((p) => {
    const parts = p.split(",").map((x) => parseInt(x.trim(), 10));
    if (parts.length !== 2 || parts.some((x) => Number.isNaN(x))) {
      throw new Error(`Bad stop "${p}". Use: row,col; row,col; row,col`);
    }
    return parts;
  });
}

function gatherOverrides() {
  let stops = lastOverrides ? lastOverrides.STOPS : [];
  try {
    stops = parseStops(document.getElementById("pStops").value);
  } catch (err) {
    alert(err.message);
  }
  return {
    SEED: parseInt(document.getElementById("pSeed").value, 10),
    TURN_TIME_LIMIT_SEC: parseFloat(document.getElementById("pTimerLimit").value),
    SLIP_PROBABILITY: parseFloat(document.getElementById("pSlip").value),
    WEIGHTED_ASTAR_EPSILON: parseFloat(document.getElementById("pEps").value),
    RUSH_HOUR_MULTIPLIER: parseFloat(document.getElementById("pRushMult").value),
    RUSH_HOUR_CYCLE: parseInt(document.getElementById("pRushPeriod").value, 10),
    RUSH_HOUR_END: parseInt(document.getElementById("pRushDur").value, 10),
    ADVERSARY_SEARCH_DEPTH: parseInt(document.getElementById("pDepth").value, 10),
    LOCAL_GAME_WINDOW_RADIUS: parseInt(document.getElementById("pWindow").value, 10),
    STOPS: stops,
    TWIST_RIVAL: document.getElementById("pTwistRival").checked,
    TWIST_RUSH: document.getElementById("pTwistRush").checked,
    TWIST_CLOSURE: document.getElementById("pTwistClosure").checked,
  };
}

function formatDirArrow(allowedDirs) {
  if (!allowedDirs || allowedDirs.length === 0) return "X";
  const s = new Set(allowedDirs);
  if (s.has("LEFT") && s.has("RIGHT")) return "↔";
  if (s.has("UP") && s.has("DOWN")) return "↕";
  if (s.has("RIGHT")) return "→";
  if (s.has("LEFT")) return "←";
  if (s.has("DOWN")) return "↓";
  if (s.has("UP")) return "↑";
  return "!";
}

function renderGrid() {
  if (!stateData) return;
  const canvas = document.getElementById("gridCanvas");
  const ctx = canvas.getContext("2d");
  const rows = stateData.rows;
  const cols = stateData.cols;
  const cellW = canvas.width / cols;
  const cellH = canvas.height / rows;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // 1. Draw Terrain Tiles (-1=obstacle, 0=road, 1=grass, 2=mud, 3=ford)
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const val = stateData.grid[r][c];
      let tileImg = IMAGES.road;
      if (val < 0) tileImg = IMAGES.obstacle;
      else if (val === 0) tileImg = IMAGES.road;
      else if (val === 1) tileImg = IMAGES.grass;
      else if (val === 2) tileImg = IMAGES.mud;
      else if (val === 3) tileImg = IMAGES.ford;

      if (tileImg) {
        ctx.drawImage(tileImg, c * cellW, r * cellH, cellW, cellH);
      } else {
        ctx.fillStyle = val < 0 ? "#1e293b" : "#334155";
        ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
      }

      ctx.strokeStyle = "rgba(15, 23, 42, 0.35)";
      ctx.lineWidth = 0.7;
      ctx.strokeRect(c * cellW, r * cellH, cellW, cellH);
    }
  }

  // 2. Draw Dispatcher Avoid Zones
  for (const [ar, ac] of stateData.avoid_cells) {
    ctx.fillStyle = "rgba(249, 115, 22, 0.45)";
    ctx.fillRect(ac * cellW, ar * cellH, cellW, cellH);
  }

  // 3. Draw Active & Upcoming Road Closures / Maintenance WITH live countdown timer & directional arrow badge
  for (const cl of stateData.closures) {
    const [cr, cc] = cl.cell;
    const x = cc * cellW;
    const y = cr * cellH;

    if (cl.status === "ACTIVE" && cl.mode === "closed" && IMAGES.close) {
      ctx.drawImage(IMAGES.close, x, y, cellW, cellH);
    } else if (IMAGES.maintenance) {
      ctx.drawImage(IMAGES.maintenance, x, y, cellW, cellH);
    }

    // Draw high-contrast timer/direction pill directly on the block
    ctx.save();
    ctx.fillStyle = cl.status === "ACTIVE" ? "rgba(220, 38, 38, 0.88)" : "rgba(234, 88, 12, 0.92)";
    ctx.fillRect(x + 1, y + 1, cellW - 2, 11);

    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 9px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    let label = "";
    if (cl.status === "UPCOMING") {
      // Shows countdown steps until closure/maintenance activates (e.g., "-3t" or "-2t→")
      const suffix = cl.mode === "directional" ? formatDirArrow(cl.allowed_dirs) : "X";
      label = `${cl.steps_until}t${suffix}`;
    } else if (cl.mode === "closed") {
      label = `X(${cl.steps_left}t)`;
    } else {
      label = `${formatDirArrow(cl.allowed_dirs)}(${cl.steps_left}t)`;
    }
    ctx.fillText(label, x + cellW / 2, y + 6.5);
    ctx.restore();
  }

  // 4. Draw Agent Historical Trail
  if (stateData.agent_trail && stateData.agent_trail.length > 1) {
    ctx.beginPath();
    stateData.agent_trail.forEach(([r, c], idx) => {
      const x = (c + 0.5) * cellW;
      const y = (r + 0.5) * cellH;
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = "rgba(34, 197, 94, 0.55)";
    ctx.lineWidth = 3;
    ctx.stroke();
  }

  // 5. Draw Active Planned Path
  const planPath = stateData.plan.path || [];
  if (planPath.length > 1) {
    ctx.beginPath();
    planPath.forEach(([r, c], idx) => {
      const x = (c + 0.5) * cellW;
      const y = (r + 0.5) * cellH;
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 4;
    ctx.setLineDash([6, 4]);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // 6. Draw Delivery Stops 1, 2, 3
  const stopSprites = [IMAGES.stop1, IMAGES.stop2, IMAGES.stop3];
  stateData.stops.forEach(([sr, sc], idx) => {
    const isDelivered = stateData.delivered.some(([dr, dc]) => dr === sr && dc === sc);
    const isStolen = stateData.stolen.some(([tr, tc]) => tr === sr && tc === sc);

    ctx.save();
    if (isDelivered) {
      ctx.fillStyle = "rgba(34, 197, 94, 0.45)";
      ctx.fillRect(sc * cellW, sr * cellH, cellW, cellH);
    } else if (isStolen) {
      ctx.fillStyle = "rgba(239, 68, 68, 0.45)";
      ctx.fillRect(sc * cellW, sr * cellH, cellW, cellH);
    }
    const sp = stopSprites[idx];
    if (sp) {
      ctx.drawImage(sp, sc * cellW + 1, sr * cellH + 1, cellW - 2, cellH - 2);
    }
    ctx.restore();
  });


  // 7. Draw Rival (R) with a translucent penalty halo showing the tactical engagement field
  const [rr, rc] = stateData.rival_pos;
  const radiusPenalty = 3;
  ctx.save();
  ctx.fillStyle = "rgba(239, 68, 68, 0.15)";
  for (let dr = -radiusPenalty; dr <= radiusPenalty; dr++) {
    for (let dc = -radiusPenalty; dc <= radiusPenalty; dc++) {
      const d = Math.abs(dr) + Math.abs(dc);
      if (d === 0 || d > radiusPenalty) continue;
      const r2 = rr + dr, c2 = rc + dc;
      if (r2 < 0 || r2 >= rows || c2 < 0 || c2 >= cols) continue;
      ctx.fillRect(c2 * cellW, r2 * cellH, cellW, cellH);
    }
  }
  ctx.restore();
  if (IMAGES.rival) {
    ctx.drawImage(IMAGES.rival, rc * cellW, rr * cellH, cellW, cellH);
  }

  // 8. Draw Delivery Agent (A)
  const [ar, ac] = stateData.agent_pos;
  if (IMAGES.agent) {
    ctx.drawImage(IMAGES.agent, ac * cellW, ar * cellH, cellW, cellH);
  }
}

function updateUI(data) {
  stateData = data;
  document.getElementById("stampText").textContent = data.stamp;
  document.getElementById("eventBanner").textContent = data.last_event_log;
  document.getElementById("algoSelect").value = data.algorithm;

  if (data.overrides) {
    lastOverrides = data.overrides;
    document.getElementById("pSeed").value = data.overrides.SEED;
    document.getElementById("pTimerLimit").value = data.overrides.TURN_TIME_LIMIT_SEC;
    document.getElementById("pSlip").value = data.overrides.SLIP_PROBABILITY;
    document.getElementById("vSlip").textContent = data.overrides.SLIP_PROBABILITY;
    document.getElementById("pEps").value = data.overrides.WEIGHTED_ASTAR_EPSILON;
    document.getElementById("vEps").textContent = data.overrides.WEIGHTED_ASTAR_EPSILON;
    document.getElementById("pRushMult").value = data.overrides.RUSH_HOUR_MULTIPLIER;
    document.getElementById("vRushMult").textContent = data.overrides.RUSH_HOUR_MULTIPLIER;
    document.getElementById("pRushPeriod").value = data.overrides.RUSH_HOUR_CYCLE;
    document.getElementById("pRushDur").value = data.overrides.RUSH_HOUR_END;
    document.getElementById("pDepth").value = data.overrides.ADVERSARY_SEARCH_DEPTH;
    document.getElementById("pWindow").value = data.overrides.LOCAL_GAME_WINDOW_RADIUS;
    if (document.activeElement.id !== "pStops") {
      document.getElementById("pStops").value = data.overrides.STOPS.map((s) => `${s[0]},${s[1]}`).join("; ");
    }
    document.getElementById("pTwistRival").checked = data.overrides.TWIST_RIVAL;
    document.getElementById("pTwistRush").checked = data.overrides.TWIST_RUSH;
    document.getElementById("pTwistClosure").checked = data.overrides.TWIST_CLOSURE;
  }

  document.getElementById("mStep").textContent = data.t;
  document.getElementById("mRush").textContent = data.rush_hour
    ? `RUSH HOUR (${data.overrides.RUSH_HOUR_MULTIPLIER}x road)`
    : "Normal (1x)";
  document.getElementById("mRush").className = data.rush_hour ? "m-val bad" : "m-val good";
  document.getElementById("mCost").textContent = data.total_cost;
  document.getElementById("mSlips").textContent = data.slip_count;
  document.getElementById("mDelivered").textContent = `${data.delivered.length} / ${data.stops.length}`;
  document.getElementById("mStolen").textContent = `${data.stolen.length} / ${data.stops.length}`;
  document.getElementById("mPlanCost").textContent = data.plan.cost;
  document.getElementById("mNodes").textContent = data.plan.nodes_expanded;

  renderGrid();
}

async function apiPost(endpoint, payload = {}) {
  const res = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  let data = null;
  try {
    data = await res.json();
  } catch (err) {
    data = null;
  }
  if (!res.ok || !data || data.error) {
    alert((data && data.error) || `Request failed (HTTP ${res.status})`);
    return null;
  }
  updateUI(data);
  return data;
}

async function stepSimulation() {
  if (isStepping) return;
  isStepping = true;
  try {
    remainingSec = getTimerLimit();
    updateClockDisplay();
    await apiPost("/api/step", { overrides: gatherOverrides() });
  } finally {
    isStepping = false;
  }
}

async function runAutoPlayLoop() {
  while (isAutoPlaying) {
    await stepSimulation();
    if (!isAutoPlaying) break;
    await new Promise((r) => setTimeout(r, 350));
  }
}

document.getElementById("btnStep").addEventListener("click", stepSimulation);

document.getElementById("btnPlay").addEventListener("click", (e) => {
  if (isAutoPlaying) {
    isAutoPlaying = false;
    e.target.textContent = "Auto-Play";
  } else {
    isAutoPlaying = true;
    e.target.textContent = "Pause";
    // The countdown timer is the only automatic step driver; restart it for the next step.
    remainingSec = getTimerLimit();
    updateClockDisplay();
  }
});

document.getElementById("btnReset").addEventListener("click", async () => {
  isAutoPlaying = false;
  document.getElementById("btnPlay").textContent = "Auto-Play";
  remainingSec = getTimerLimit();
  await apiPost("/api/reset", { overrides: gatherOverrides() });
});

document.getElementById("algoSelect").addEventListener("change", async (e) => {
  await apiPost("/api/plan", { algorithm: e.target.value, overrides: gatherOverrides() });
});

document.getElementById("btnApplyParams").addEventListener("click", async () => {
  remainingSec = getTimerLimit();
  await apiPost("/api/plan", {
    algorithm: document.getElementById("algoSelect").value,
    overrides: gatherOverrides(),
  });
});

document.getElementById("btnResetWithParams").addEventListener("click", async () => {
  remainingSec = getTimerLimit();
  await apiPost("/api/reset", { overrides: gatherOverrides() });
});

document.getElementById("btnRestoreDefaults").addEventListener("click", async () => {
  isAutoPlaying = false;
  document.getElementById("btnPlay").textContent = "Auto-Play";
  await apiPost("/api/restore_defaults", {});
  remainingSec = getTimerLimit();
  updateClockDisplay();
});

[
  ["pSlip", "vSlip"],
  ["pEps", "vEps"],
  ["pRushMult", "vRushMult"],
].forEach(([inputId, labelId]) => {
  document.getElementById(inputId).addEventListener("input", (e) => {
    document.getElementById(labelId).textContent = e.target.value;
  });
});

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    document.getElementById("dispatchInput").value = chip.dataset.prompt;
    document.getElementById("btnDispatch").click();
  });
});

document.getElementById("btnDispatch").addEventListener("click", async () => {
  const input = document.getElementById("dispatchInput");
  const instruction = input.value.trim();
  if (!instruction) return;
  document.getElementById("taoOutput").textContent = "Running ReAct Sense -> Reason -> Act loop...";
  const data = await apiPost("/api/dispatch", { instruction });
  if (data && data.dispatch_result) {
    document.getElementById("taoOutput").textContent = JSON.stringify(data.dispatch_result, null, 2);
  }
});

(async function init() {
  await preloadImages();
  const res = await fetch("/api/state");
  const initialData = await res.json();
  updateUI(initialData);
  startStepCountdown();
})();