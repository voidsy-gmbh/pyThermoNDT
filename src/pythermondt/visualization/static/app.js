const API_PREFIX = "/api/v1";
const MAX_VECTOR_POINTS = 20000;
const MATRIX_ROW_COUNT = 20;
const MATRIX_COL_COUNT = 20;
const HEATMAP_TRACE_TYPE = "heatmap";

const statusElement = document.getElementById("status");
const treeRootElement = document.getElementById("tree-root");
const nodeJsonElement = document.getElementById("node-json");

const plotPanelElement = document.getElementById("plot-panel");
const plotInfoElement = document.getElementById("plot-info");
const plotMessageElement = document.getElementById("plot-message");
const plotContainerElement = document.getElementById("plot-container");
const matrixContainerElement = document.getElementById("matrix-container");

const viewModeElement = document.getElementById("view-mode");
const modeOriginElement = document.getElementById("mode-origin");

const frameControlsElement = document.getElementById("frame-controls");
const frameAxisElement = document.getElementById("frame-axis");
const frameSliderElement = document.getElementById("frame-slider");
const frameLabelElement = document.getElementById("frame-label");

let selectedNodePath = null;
let selectedNodeType = null;
let selectedTreeItem = null;
let selectedPlotMeta = null;
let selectionVersion = 0;

let frameRequestController = null;
let frameRequestVersion = 0;

let currentMode = null;
let modeOrigin = "auto";
let renderedMode = null;
let renderedPath = null;
let frameRenderScheduled = false;

function setStatus(message, isError = false) {
  statusElement.textContent = message;
  statusElement.className = isError ? "status-error" : "status-ok";
}

function setPlotMessage(message, isError = false) {
  plotMessageElement.textContent = message;
  plotMessageElement.className = isError ? "plot-message plot-message-error" : "plot-message";
}

function setModeOriginLabel() {
  modeOriginElement.textContent = modeOrigin === "manual" ? "Manual" : "Auto";
}

async function fetchJson(url) {
  const response = await fetch(url);
  let payload = null;

  try {
    payload = await response.json();
  } catch {
    payload = null;
  }

  if (!response.ok) {
    if (payload && payload.error) {
      throw new Error(payload.error);
    }
    throw new Error(`HTTP ${response.status}`);
  }

  return payload;
}

function parseShapeHeader(shapeHeader) {
  if (!shapeHeader) {
    return [];
  }
  return shapeHeader
    .split(",")
    .filter((part) => part.length > 0)
    .map((part) => Number.parseInt(part, 10));
}

function normalizePlotMeta(meta) {
  if (!meta || typeof meta !== "object") {
    return {
      ndim: 0,
      shape: [],
      available_modes: ["line"],
      default_mode: "line",
      frame_axis_options: [],
      default_frame_axis: null,
      numel: 0,
    };
  }

  const shape = Array.isArray(meta.shape) ? meta.shape : [];
  const ndim = Number.isInteger(meta.ndim) ? meta.ndim : shape.length;

  let availableModes = Array.isArray(meta.available_modes) ? meta.available_modes : [];

  if (availableModes.length === 0) {
    if (ndim === 1) {
      availableModes = ["matrix", "line"];
    } else if (ndim === 2 || ndim === 3) {
      availableModes = ["matrix", "line", "heatmap"];
    } else {
      availableModes = ["line"];
    }
  }

  let defaultMode = typeof meta.default_mode === "string" ? meta.default_mode : null;
  if (!defaultMode || !availableModes.includes(defaultMode)) {
    if (meta.render_mode === "frames" || meta.render_mode === "heatmap") {
      defaultMode = availableModes.includes("heatmap") ? "heatmap" : availableModes[0];
    } else if (meta.render_mode === "line") {
      defaultMode = availableModes.includes("line") ? "line" : availableModes[0];
    } else {
      defaultMode = availableModes[0];
    }
  }

  let frameAxisOptions = Array.isArray(meta.frame_axis_options) ? meta.frame_axis_options : [];
  if (ndim === 3 && frameAxisOptions.length === 0 && shape.length === 3) {
    frameAxisOptions = [0, 1, 2];
  }

  let defaultFrameAxis = Number.isInteger(meta.default_frame_axis) ? meta.default_frame_axis : null;
  if (ndim === 3 && defaultFrameAxis === null) {
    defaultFrameAxis = 2;
  }

  return {
    ...meta,
    ndim,
    shape,
    available_modes: availableModes,
    default_mode: defaultMode,
    frame_axis_options: frameAxisOptions,
    default_frame_axis: defaultFrameAxis,
  };
}

async function fetchBinary(url, signal) {
  const response = await fetch(url, { signal });

  if (!response.ok) {
    let message = `HTTP ${response.status}`;
    try {
      const payload = await response.json();
      if (payload.error) {
        message = payload.error;
      }
    } catch {
      // Keep generic fallback.
    }
    throw new Error(message);
  }

  const dtype = response.headers.get("X-PTNDT-Dtype");
  if (dtype !== "float32") {
    throw new Error(`Unsupported binary dtype '${dtype}'.`);
  }

  const shape = parseShapeHeader(response.headers.get("X-PTNDT-Shape"));
  const buffer = await response.arrayBuffer();
  return {
    values: new Float32Array(buffer),
    shape,
  };
}

function nodeIcon(nodeType) {
  if (nodeType === "root") {
    return "R";
  }
  if (nodeType === "group") {
    return "G";
  }
  return "D";
}

function clearSelection() {
  if (selectedTreeItem) {
    selectedTreeItem.classList.remove("selected");
  }
}

function clearPlot() {
  if (window.Plotly && plotContainerElement.dataset.hasPlot === "true") {
    window.Plotly.purge(plotContainerElement);
  }
  plotContainerElement.dataset.hasPlot = "false";
  plotContainerElement.dataset.plotType = "";
  plotContainerElement.innerHTML = "";
  plotContainerElement.hidden = true;

  matrixContainerElement.innerHTML = "";
  matrixContainerElement.hidden = true;

  frameControlsElement.hidden = true;
  setPlotMessage("");
  plotInfoElement.textContent = "";

  renderedMode = null;
  renderedPath = null;
}

function hidePlotPanel() {
  selectedPlotMeta = null;
  plotPanelElement.hidden = true;
  clearPlot();
}

function reshapeToMatrix(values, rows, cols) {
  if (rows * cols !== values.length) {
    throw new Error(`Unexpected frame size: expected ${rows * cols} values, received ${values.length}.`);
  }

  const matrix = new Array(rows);
  for (let row = 0; row < rows; row += 1) {
    const start = row * cols;
    matrix[row] = Array.from(values.subarray(start, start + cols));
  }
  return matrix;
}

function resetFrameRequests() {
  frameRequestVersion += 1;
  if (frameRequestController) {
    frameRequestController.abort();
    frameRequestController = null;
  }
}

function getSelectedFrameAxis() {
  return Number.parseInt(frameAxisElement.value, 10);
}

function getSelectedFrameIndex() {
  return Number.parseInt(frameSliderElement.value, 10);
}

function updateFrameLabel(index, frameCount) {
  frameLabelElement.textContent = `${index} / ${Math.max(frameCount - 1, 0)}`;
}

function buildFrameAxisOptions(meta, selectedAxis) {
  frameAxisElement.innerHTML = "";
  for (const axis of meta.frame_axis_options) {
    const option = document.createElement("option");
    option.value = String(axis);
    option.textContent = `Axis ${axis} (size ${meta.shape[axis]})`;
    frameAxisElement.appendChild(option);
  }

  frameAxisElement.value = String(selectedAxis);
}

function syncFrameControls(meta, frameAxis, frameIndex = 0) {
  const frameCount = meta.shape[frameAxis];
  frameSliderElement.min = "0";
  frameSliderElement.max = String(Math.max(frameCount - 1, 0));
  frameSliderElement.step = "1";
  frameSliderElement.value = String(Math.min(frameIndex, Math.max(frameCount - 1, 0)));
  updateFrameLabel(getSelectedFrameIndex(), frameCount);
}

function shouldShowFrameControls(mode, meta) {
  return meta.ndim === 3 && (mode === "heatmap" || mode === "matrix");
}

function configureFrameControls(mode, meta) {
  if (!shouldShowFrameControls(mode, meta)) {
    frameControlsElement.hidden = true;
    return;
  }

  const previousAxis = Number.parseInt(frameAxisElement.value || "-999", 10);
  const previousIndex = Number.parseInt(frameSliderElement.value || "0", 10);
  const axisIsValid = meta.frame_axis_options.includes(previousAxis);
  const targetAxis = axisIsValid ? previousAxis : Number.parseInt(String(meta.default_frame_axis), 10);

  frameControlsElement.hidden = false;
  buildFrameAxisOptions(meta, targetAxis);
  syncFrameControls(meta, targetAxis, previousIndex);
}

function buildModeOptions(meta) {
  viewModeElement.innerHTML = "";
  for (const mode of meta.available_modes) {
    const option = document.createElement("option");
    option.value = mode;
    option.textContent = mode[0].toUpperCase() + mode.slice(1);
    viewModeElement.appendChild(option);
  }

  viewModeElement.value = meta.default_mode;
}

function renderMatrixTable(payload) {
  matrixContainerElement.innerHTML = "";
  matrixContainerElement.hidden = false;
  plotContainerElement.hidden = true;

  const rows = payload.shape[0];
  const cols = payload.shape[1];
  const showingRows = payload.row_end - payload.row_start;
  const showingCols = payload.col_end - payload.col_start;

  const caption = document.createElement("p");
  caption.className = "matrix-caption";
  caption.textContent = `Showing rows ${payload.row_start}..${payload.row_end - 1} of ${rows}, columns ${payload.col_start}..${payload.col_end - 1} of ${cols}`;
  matrixContainerElement.appendChild(caption);

  const table = document.createElement("table");
  table.className = "matrix-table";

  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");
  const corner = document.createElement("th");
  corner.textContent = "r\\c";
  headerRow.appendChild(corner);

  for (let col = 0; col < showingCols; col += 1) {
    const th = document.createElement("th");
    th.textContent = String(payload.col_start + col);
    headerRow.appendChild(th);
  }
  thead.appendChild(headerRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  payload.values.forEach((rowValues, rowIdx) => {
    const tr = document.createElement("tr");
    const rowHeader = document.createElement("th");
    rowHeader.textContent = String(payload.row_start + rowIdx);
    tr.appendChild(rowHeader);

    rowValues.forEach((value) => {
      const td = document.createElement("td");
      td.textContent = Number(value).toFixed(4);
      tr.appendChild(td);
    });

    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  matrixContainerElement.appendChild(table);
}

async function renderLineMode(path, token) {
  const totalValues = selectedPlotMeta.numel;
  const stride = Math.max(1, Math.ceil(totalValues / MAX_VECTOR_POINTS));
  const count = Math.min(MAX_VECTOR_POINTS, totalValues);

  const vectorUrl =
    `${API_PREFIX}/plot/vector.bin?path=${encodeURIComponent(path)}` +
    `&start=0&count=${count}&stride=${stride}`;

  setPlotMessage("Loading vector data...");
  const payload = await fetchBinary(vectorUrl, null);
  if (token !== selectionVersion) {
    return;
  }

  const yValues = Array.from(payload.values);
  const xValues = new Array(yValues.length);
  for (let idx = 0; idx < yValues.length; idx += 1) {
    xValues[idx] = idx * stride;
  }

  const trace = {
    type: "scattergl",
    mode: "lines",
    x: xValues,
    y: yValues,
    line: { color: "#2f6fab", width: 2 },
    name: path,
  };

  const layout = {
    margin: { l: 55, r: 20, t: 20, b: 45 },
    xaxis: { title: "Index" },
    yaxis: { title: "Value" },
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff",
  };

  const config = {
    displaylogo: false,
    responsive: true,
    modeBarButtonsToRemove: ["lasso2d", "select2d"],
  };

  plotContainerElement.hidden = false;
  matrixContainerElement.hidden = true;
  await window.Plotly.react(plotContainerElement, [trace], layout, config);
  plotContainerElement.dataset.hasPlot = "true";

  if (stride > 1) {
    setPlotMessage(`Displaying downsampled data (stride ${stride}).`);
  } else {
    setPlotMessage("");
  }

  plotInfoElement.textContent = `${path} | line mode | ${totalValues} values`;
}

async function loadFrameBinary(path, frameAxis, frameIndex) {
  resetFrameRequests();
  const requestVersion = frameRequestVersion;
  const controller = new AbortController();
  frameRequestController = controller;

  const frameUrl =
    `${API_PREFIX}/plot/frame.bin?path=${encodeURIComponent(path)}` +
    `&frame_axis=${frameAxis}&frame_index=${frameIndex}`;

  const payload = await fetchBinary(frameUrl, controller.signal);
  if (requestVersion !== frameRequestVersion) {
    return null;
  }

  if (payload.shape.length !== 2) {
    throw new Error(`Expected a 2D frame payload, got shape '${payload.shape}'.`);
  }

  return payload;
}

async function renderHeatmapMode(path, token) {
  const incrementalUpdate =
    plotContainerElement.dataset.hasPlot === "true" && plotContainerElement.dataset.plotType === "heatmap";

  let frameAxis = -1;
  let frameIndex = 0;

  if (selectedPlotMeta.ndim === 3) {
    frameAxis = getSelectedFrameAxis();
    frameIndex = getSelectedFrameIndex();
  }

  if (!incrementalUpdate) {
    setPlotMessage("Loading heatmap data...");
  }
  const payload = await loadFrameBinary(path, frameAxis, frameIndex);
  if (payload === null || token !== selectionVersion) {
    return;
  }

  const rows = payload.shape[0];
  const cols = payload.shape[1];
  const zValues = reshapeToMatrix(payload.values, rows, cols);

  const titleText = selectedPlotMeta.ndim === 3 ? `Frame ${frameIndex} (axis ${frameAxis})` : "Matrix";

  plotContainerElement.hidden = false;
  matrixContainerElement.hidden = true;

  if (!incrementalUpdate) {
    const trace = {
      type: HEATMAP_TRACE_TYPE,
      z: zValues,
      colorscale: "Viridis",
      zsmooth: false,
      colorbar: { title: "Value" },
      hovertemplate: "x=%{x}<br>y=%{y}<br>z=%{z}<extra></extra>",
    };

    const layout = {
      margin: { l: 60, r: 20, t: 30, b: 45 },
      title: { text: titleText, font: { size: 13 } },
      xaxis: { title: "Column" },
      yaxis: { title: "Row", autorange: "reversed" },
      paper_bgcolor: "#ffffff",
      plot_bgcolor: "#ffffff",
    };

    const config = {
      displaylogo: false,
      responsive: true,
      modeBarButtonsToRemove: ["lasso2d", "select2d"],
    };

    await window.Plotly.react(plotContainerElement, [trace], layout, config);
  } else {
    await window.Plotly.restyle(plotContainerElement, { z: [zValues] }, [0]);
    await window.Plotly.relayout(plotContainerElement, { "title.text": titleText });
  }

  plotContainerElement.dataset.hasPlot = "true";
  plotContainerElement.dataset.plotType = "heatmap";
  setPlotMessage("");
  plotInfoElement.textContent = `${path} | heatmap ${rows} x ${cols} | dtype float32`;
}

async function renderMatrixMode(path, token) {
  let frameAxis = -1;
  let frameIndex = 0;
  if (selectedPlotMeta.ndim === 3) {
    frameAxis = getSelectedFrameAxis();
    frameIndex = getSelectedFrameIndex();
  }

  const matrixUrl =
    `${API_PREFIX}/plot/matrix?path=${encodeURIComponent(path)}` +
    `&frame_axis=${frameAxis}&frame_index=${frameIndex}` +
    `&row_start=0&row_count=${MATRIX_ROW_COUNT}&col_start=0&col_count=${MATRIX_COL_COUNT}`;

  setPlotMessage("Loading matrix table...");
  const payload = await fetchJson(matrixUrl);
  if (token !== selectionVersion) {
    return;
  }

  renderMatrixTable(payload);
  setPlotMessage("");
  plotInfoElement.textContent = `${path} | matrix mode | shape ${payload.shape[0]} x ${payload.shape[1]}`;
}

async function renderCurrentMode(token) {
  if (!selectedNodePath || !selectedPlotMeta || !currentMode) {
    return;
  }

  const rendererChanged = renderedMode !== currentMode || renderedPath !== selectedNodePath;
  const showFrameControls = shouldShowFrameControls(currentMode, selectedPlotMeta);
  const controlsNeedRefresh =
    rendererChanged ||
    (showFrameControls && frameControlsElement.hidden) ||
    (!showFrameControls && !frameControlsElement.hidden);

  if (rendererChanged) {
    clearPlot();
    renderedMode = currentMode;
    renderedPath = selectedNodePath;
  }

  if (controlsNeedRefresh) {
    configureFrameControls(currentMode, selectedPlotMeta);
  }

  if (currentMode === "line") {
    await renderLineMode(selectedNodePath, token);
    return;
  }

  if (currentMode === "heatmap") {
    await renderHeatmapMode(selectedNodePath, token);
    return;
  }

  if (currentMode === "matrix") {
    await renderMatrixMode(selectedNodePath, token);
    return;
  }

  setPlotMessage(`Unknown mode '${currentMode}'.`, true);
}

async function applyMode(mode, origin = "manual") {
  if (!selectedPlotMeta || !selectedPlotMeta.available_modes.includes(mode)) {
    return;
  }

  currentMode = mode;
  modeOrigin = origin;
  setModeOriginLabel();

  viewModeElement.value = mode;
  const token = selectionVersion;

  try {
    await renderCurrentMode(token);
  } catch (error) {
    if (error.name === "AbortError") {
      return;
    }
    setPlotMessage(String(error), true);
    setStatus(String(error), true);
  }
}

async function preparePlotModes(path, meta) {
  selectedPlotMeta = meta;
  buildModeOptions(meta);
  setModeOriginLabel();

  currentMode = meta.default_mode;
  modeOrigin = "auto";
  setModeOriginLabel();
  await applyMode(currentMode, "auto");
  plotPanelElement.hidden = false;
}

async function selectNode(path, nodeType, treeItem) {
  clearSelection();
  selectedTreeItem = treeItem;
  selectedTreeItem.classList.add("selected");

  selectionVersion += 1;
  const localVersion = selectionVersion;

  selectedNodePath = path;
  selectedNodeType = nodeType;
  resetFrameRequests();

  const detailsUrl = `${API_PREFIX}/node?path=${encodeURIComponent(path)}`;
  const payload = await fetchJson(detailsUrl);
  if (localVersion !== selectionVersion) {
    return;
  }

  nodeJsonElement.textContent = JSON.stringify(payload, null, 2);

  if (selectedNodeType !== "dataset") {
    hidePlotPanel();
    return;
  }

  const plotMetaUrl = `${API_PREFIX}/plot/meta?path=${encodeURIComponent(path)}`;
  const plotMetaRaw = await fetchJson(plotMetaUrl);
  const plotMeta = normalizePlotMeta(plotMetaRaw);
  if (localVersion !== selectionVersion) {
    return;
  }

  await preparePlotModes(path, plotMeta);
}

async function loadChildren(path, listElement) {
  const payload = await fetchJson(`${API_PREFIX}/tree?path=${encodeURIComponent(path)}`);
  listElement.innerHTML = "";

  for (const child of payload.children) {
    const item = createTreeNodeElement(child);
    listElement.appendChild(item);
  }
}

function createTreeNodeElement(node) {
  const listItem = document.createElement("li");
  listItem.className = "tree-item";

  const row = document.createElement("div");
  row.className = "tree-row";

  const toggleButton = document.createElement("button");
  toggleButton.type = "button";
  toggleButton.className = "tree-toggle";
  toggleButton.textContent = node.has_children ? ">" : "";
  toggleButton.disabled = !node.has_children;
  row.appendChild(toggleButton);

  const labelButton = document.createElement("button");
  labelButton.type = "button";
  labelButton.className = "tree-label";
  labelButton.textContent = `[${nodeIcon(node.node_type)}] ${node.name}`;
  row.appendChild(labelButton);

  listItem.appendChild(row);

  const childrenList = document.createElement("ul");
  childrenList.className = "tree-children";
  childrenList.hidden = true;
  listItem.appendChild(childrenList);

  labelButton.addEventListener("click", async () => {
    try {
      await selectNode(node.path, node.node_type, row);
      setStatus(`Selected ${node.path}`);
    } catch (error) {
      setStatus(String(error), true);
    }
  });

  if (node.has_children) {
    toggleButton.addEventListener("click", async () => {
      try {
        if (!childrenList.dataset.loaded) {
          await loadChildren(node.path, childrenList);
          childrenList.dataset.loaded = "true";
        }

        const isExpanded = !childrenList.hidden;
        childrenList.hidden = isExpanded;
        toggleButton.textContent = isExpanded ? ">" : "v";
      } catch (error) {
        setStatus(String(error), true);
      }
    });
  }

  return listItem;
}

function scheduleFrameRender() {
  if (frameRenderScheduled) {
    return;
  }

  frameRenderScheduled = true;
  window.requestAnimationFrame(() => {
    frameRenderScheduled = false;
    void applyMode(currentMode, modeOrigin);
  });
}

async function initializeTree() {
  const treeList = document.createElement("ul");
  treeList.className = "tree-list";
  treeRootElement.appendChild(treeList);

  const rootNode = {
    path: "/",
    name: "/",
    node_type: "root",
    has_children: true,
  };

  const rootItem = createTreeNodeElement(rootNode);
  treeList.appendChild(rootItem);

  const rootChildren = rootItem.querySelector(".tree-children");
  const rootToggle = rootItem.querySelector(".tree-toggle");
  const rootRow = rootItem.querySelector(".tree-row");

  await loadChildren("/", rootChildren);
  rootChildren.dataset.loaded = "true";
  rootChildren.hidden = false;
  rootToggle.textContent = "v";

  await selectNode("/", "root", rootRow);
}

viewModeElement.addEventListener("change", async () => {
  if (!selectedPlotMeta) {
    return;
  }
  await applyMode(viewModeElement.value, "manual");
});

frameAxisElement.addEventListener("change", async () => {
  if (!selectedPlotMeta || !currentMode) {
    return;
  }

  if (!shouldShowFrameControls(currentMode, selectedPlotMeta)) {
    return;
  }

  const frameAxis = getSelectedFrameAxis();
  syncFrameControls(selectedPlotMeta, frameAxis);
  scheduleFrameRender();
});

frameSliderElement.addEventListener("input", async () => {
  if (!selectedPlotMeta || !currentMode) {
    return;
  }

  if (!shouldShowFrameControls(currentMode, selectedPlotMeta)) {
    return;
  }

  const frameAxis = getSelectedFrameAxis();
  updateFrameLabel(getSelectedFrameIndex(), selectedPlotMeta.shape[frameAxis]);
  scheduleFrameRender();
});

async function initialize() {
  if (!window.Plotly) {
    setStatus("Plotly bundle failed to load.", true);
    return;
  }

  try {
    await fetchJson(`${API_PREFIX}/health`);
    await initializeTree();
    setStatus("Viewer ready");
  } catch (error) {
    setStatus(String(error), true);
  }
}

initialize();
