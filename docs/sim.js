/**
 * Real-time Client-side Neural Connectome & Continuous-Time ODE Simulator.
 * Simulates 68 Desikan-Killiany cortical parcels and Kuramoto/Wilson-Cowan dynamics at 60 FPS.
 * Research and Simulation Architecture by Saket Dixit.
 */

(function () {
  const NUM_NODES = 68;
  const canvas = document.getElementById("connectomeCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  // DOM Elements
  const kappaSlider = document.getElementById("kappaSlider");
  const kappaVal = document.getElementById("kappaVal");
  const stimSlider = document.getElementById("stimSlider");
  const stimVal = document.getElementById("stimVal");
  const leakSlider = document.getElementById("leakSlider");
  const leakVal = document.getElementById("leakVal");
  const targetSubSelect = document.getElementById("targetSubSelect");
  const tmsBtn = document.getElementById("tmsBtn");
  const resetBtn = document.getElementById("resetBtn");
  const pauseBtn = document.getElementById("pauseBtn");

  const rMetricEl = document.getElementById("rMetric");
  const regimeMetricEl = document.getElementById("regimeMetric");
  const latencyMetricEl = document.getElementById("latencyMetric");

  // Subnetworks definition (Cortical Functional Palette)
  const SUBNETWORKS = [
    { name: "Default Mode", color: "#3B82F6", start: 0, end: 17 },
    { name: "Somatomotor", color: "#F59E0B", start: 17, end: 34 },
    { name: "Visual", color: "#A855F7", start: 34, end: 51 },
    { name: "Frontoparietal", color: "#10B981", start: 51, end: 68 },
  ];

  // Parameters
  let kappa = parseFloat(kappaSlider ? kappaSlider.value : 0.85);
  let stimAmp = parseFloat(stimSlider ? stimSlider.value : 0.0);
  let leakAlpha = parseFloat(leakSlider ? leakSlider.value : 0.15);
  let isPaused = false;
  let tmsPulseActive = false;
  let tmsTimer = 0;
  let simTime = 0.0;
  const dt = 0.03;

  // Generate 68 parcel positions across left and right cortical hemispheres
  const nodes = [];
  const width = canvas.width;
  const height = canvas.height;
  const cx = width / 2;
  const cy = height / 2;

  for (let i = 0; i < NUM_NODES; i++) {
    const isLeft = i < NUM_NODES / 2;
    const subIdx = Math.min(Math.floor(i / 17), 3);
    const sub = SUBNETWORKS[subIdx];

    const hemisphereOffset = isLeft ? -135 : 135;
    const localIdx = isLeft ? i : i - NUM_NODES / 2;
    const angle = (localIdx / (NUM_NODES / 2)) * Math.PI * 2;
    const rx = 90 + (localIdx % 3) * 16;
    const ry = 145 + (localIdx % 4) * 12;

    const x = cx + hemisphereOffset + Math.cos(angle) * rx + (Math.random() - 0.5) * 6;
    const y = cy + Math.sin(angle) * ry + (Math.random() - 0.5) * 6;

    nodes.push({
      id: i,
      x: x,
      y: y,
      subnetwork: subIdx,
      color: sub.color,
      phase: Math.random() * Math.PI * 2,
      omega: 1.0 + (Math.random() - 0.5) * 0.4,
      activity: 0.2 + Math.random() * 0.2,
      dxdt: 0.0,
    });
  }

  // Precompute modular structural connectivity matrix W
  const adj = [];
  for (let i = 0; i < NUM_NODES; i++) {
    adj[i] = [];
    for (let j = 0; j < NUM_NODES; j++) {
      if (i === j) {
        adj[i][j] = 0;
        continue;
      }
      const dx = nodes[i].x - nodes[j].x;
      const dy = nodes[i].y - nodes[j].y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const sameSub = nodes[i].subnetwork === nodes[j].subnetwork;
      const modularFactor = sameSub ? 1.85 : 0.35;
      const weight = Math.exp(-0.014 * dist) * modularFactor;
      adj[i][j] = weight > 0.22 ? weight : 0;
    }
  }

  // R(t) rolling history for mini-sparkline chart
  const rHistory = [];
  const maxHistory = 120;

  // Event Listeners
  if (kappaSlider) {
    kappaSlider.addEventListener("input", (e) => {
      kappa = parseFloat(e.target.value);
      if (kappaVal) kappaVal.textContent = kappa.toFixed(2);
    });
  }
  if (stimSlider) {
    stimSlider.addEventListener("input", (e) => {
      stimAmp = parseFloat(e.target.value);
      if (stimVal) stimVal.textContent = stimAmp.toFixed(2);
    });
  }
  if (leakSlider) {
    leakSlider.addEventListener("input", (e) => {
      leakAlpha = parseFloat(e.target.value);
      if (leakVal) leakVal.textContent = leakAlpha.toFixed(2);
    });
  }
  if (tmsBtn) {
    tmsBtn.addEventListener("click", () => {
      tmsPulseActive = true;
      tmsTimer = 2.5; // 2.5 seconds pulse
    });
  }
  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      for (const node of nodes) {
        node.phase = Math.random() * Math.PI * 2;
        node.activity = 0.2 + Math.random() * 0.2;
      }
      rHistory.length = 0;
    });
  }
  if (pauseBtn) {
    pauseBtn.addEventListener("click", () => {
      isPaused = !isPaused;
      pauseBtn.textContent = isPaused ? "Resume" : "Pause";
    });
  }

  // Simulation Step: Coupled Kuramoto-Wilson-Cowan ODE
  function stepSimulation() {
    const tStart = performance.now();
    simTime += dt;

    if (tmsPulseActive) {
      tmsTimer -= dt;
      if (tmsTimer <= 0) tmsPulseActive = false;
    }

    const targetSub = targetSubSelect ? parseInt(targetSubSelect.value, 10) : 3;

    // Numerical integration step for each node
    for (let i = 0; i < NUM_NODES; i++) {
      const node = nodes[i];

      // Structural coupling input: sum_j W_ij * sin(theta_j - theta_i)
      let phaseCoupling = 0.0;
      let activityInput = 0.0;

      for (let j = 0; j < NUM_NODES; j++) {
        const w_ij = adj[i][j];
        if (w_ij > 0) {
          phaseCoupling += w_ij * Math.sin(nodes[j].phase - node.phase);
          activityInput += w_ij * nodes[j].activity;
        }
      }

      // External TMS stimulation
      let stim = 0.0;
      if (tmsPulseActive || stimAmp > 0) {
        const isTarget = targetSub === -1 || node.subnetwork === targetSub;
        if (isTarget) {
          const effAmp = tmsPulseActive ? Math.max(stimAmp, 1.4) : stimAmp;
          stim = effAmp * Math.sin(2.0 * Math.PI * 10.0 * simTime);
        }
      }

      // Phase derivative d(theta)/dt
      const dtheta_dt = node.omega + (kappa / NUM_NODES) * phaseCoupling + stim * 0.5;
      node.phase = (node.phase + dtheta_dt * dt) % (Math.PI * 2);

      // Excitatory activity derivative dE/dt = -alpha * E + Sigmoid(input)
      const totalInput = 2.5 * node.activity + kappa * activityInput + stim + 0.8;
      const sigmoid = 1.0 / (1.0 + Math.exp(-1.3 * (totalInput - 3.0)));
      node.activity += ((-leakAlpha * node.activity + sigmoid) / 1.0) * dt;
      node.activity = Math.max(0.05, Math.min(node.activity, 1.25));
    }

    // Compute Kuramoto Order Parameter R(t) = | (1/N) sum exp(i * theta_j) |
    let sumCos = 0.0;
    let sumSin = 0.0;
    for (let i = 0; i < NUM_NODES; i++) {
      sumCos += Math.cos(nodes[i].phase);
      sumSin += Math.sin(nodes[i].phase);
    }
    const rOrder = Math.sqrt(sumCos * sumCos + sumSin * sumSin) / NUM_NODES;
    rHistory.push(rOrder);
    if (rHistory.length > maxHistory) rHistory.shift();

    const tEnd = performance.now();
    const computeMs = tEnd - tStart;

    // Update UI Badges
    if (rMetricEl) rMetricEl.textContent = rOrder.toFixed(3);
    if (latencyMetricEl) latencyMetricEl.textContent = `${computeMs.toFixed(2)} ms`;

    if (regimeMetricEl) {
      if (rOrder < 0.32) {
        regimeMetricEl.textContent = "Asynchronous (I)";
        regimeMetricEl.style.color = "#94A3B8";
      } else if (rOrder < 0.72) {
        regimeMetricEl.textContent = "Metastable (II)";
        regimeMetricEl.style.color = "#F59E0B";
      } else {
        regimeMetricEl.textContent = "Phase-Locked (III)";
        regimeMetricEl.style.color = "#10B981";
      }
    }
  }

  // Draw Anatomical Hemispheric Outline
  function drawChamberGrid() {
    // Subtle anatomical midline
    ctx.strokeStyle = "rgba(255, 255, 255, 0.06)";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(cx, 30);
    ctx.lineTo(cx, height - 30);
    ctx.stroke();
    ctx.setLineDash([]);

    // Anatomical hemispheric boundary outlines
    ctx.strokeStyle = "rgba(148, 163, 184, 0.14)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.ellipse(cx - 135, cy, 118, 175, 0, 0, Math.PI * 2);
    ctx.stroke();

    ctx.beginPath();
    ctx.ellipse(cx + 135, cy, 118, 175, 0, 0, Math.PI * 2);
    ctx.stroke();

    // Clean anatomical labels
    ctx.fillStyle = "rgba(148, 163, 184, 0.45)";
    ctx.font = "11px Inter, sans-serif";
    ctx.fillText("Left Hemisphere", cx - 180, 45);
    ctx.fillText("Right Hemisphere", cx + 90, 45);
  }

  // Render Loop
  function draw() {
    ctx.clearRect(0, 0, width, height);
    drawChamberGrid();

    // Draw structural connectivity edges
    for (let i = 0; i < NUM_NODES; i++) {
      for (let j = i + 1; j < NUM_NODES; j++) {
        const w = adj[i][j];
        if (w > 0.22) {
          const phaseDiff = Math.abs(Math.sin(nodes[i].phase - nodes[j].phase));
          const alpha = Math.min(0.55, (1.0 - phaseDiff * 0.5) * (w * 0.45));

          ctx.beginPath();
          ctx.moveTo(nodes[i].x, nodes[i].y);
          ctx.lineTo(nodes[j].x, nodes[j].y);
          ctx.strokeStyle = `rgba(147, 197, 253, ${alpha})`;
          ctx.lineWidth = w * 1.6;
          ctx.stroke();
        }
      }
    }

    // Draw nodes
    for (const node of nodes) {
      const radius = 5.5 + node.activity * 6.5;

      // Glow halo based on activity
      const grad = ctx.createRadialGradient(node.x, node.y, radius * 0.2, node.x, node.y, radius * 2.4);
      grad.addColorStop(0, node.color);
      grad.addColorStop(1, "rgba(0, 0, 0, 0)");

      ctx.beginPath();
      ctx.arc(node.x, node.y, radius * 2.4, 0, Math.PI * 2);
      ctx.fillStyle = grad;
      ctx.fill();

      // Core parcel node
      ctx.beginPath();
      ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = node.color;
      ctx.fill();
      ctx.strokeStyle = "#FFFFFF";
      ctx.lineWidth = 1.3;
      ctx.stroke();

      // Phase angle vector indicator
      const px = node.x + Math.cos(node.phase) * (radius + 5);
      const py = node.y + Math.sin(node.phase) * (radius + 5);
      ctx.beginPath();
      ctx.moveTo(node.x, node.y);
      ctx.lineTo(px, py);
      ctx.strokeStyle = "#FFFFFF";
      ctx.lineWidth = 1.6;
      ctx.stroke();
    }

    // Draw mini sparkline graph of R(t) in bottom-right corner
    const sparkW = 180;
    const sparkH = 46;
    const sparkX = width - sparkW - 16;
    const sparkY = height - sparkH - 16;

    ctx.fillStyle = "rgba(15, 23, 42, 0.9)";
    ctx.fillRect(sparkX, sparkY, sparkW, sparkH);
    ctx.strokeStyle = "rgba(148, 163, 184, 0.25)";
    ctx.strokeRect(sparkX, sparkY, sparkW, sparkH);

    ctx.fillStyle = "#94A3B8";
    ctx.font = "9px Inter, sans-serif";
    ctx.fillText("KURAMOTO ORDER R(t)", sparkX + 8, sparkY + 12);

    if (rHistory.length > 1) {
      ctx.beginPath();
      for (let i = 0; i < rHistory.length; i++) {
        const x = sparkX + (i / maxHistory) * sparkW;
        const y = sparkY + sparkH - rHistory[i] * (sparkH - 18) - 4;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = "#10B981";
      ctx.lineWidth = 1.8;
      ctx.stroke();
    }
  }

  function animationLoop() {
    if (!isPaused) {
      stepSimulation();
    }
    draw();
    requestAnimationFrame(animationLoop);
  }

  // Start simulation
  animationLoop();
})();
