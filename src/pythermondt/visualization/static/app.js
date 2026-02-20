const API_PREFIX = "/api/v1";
const DEFAULT_LIMIT = 256;

const statusElement = document.getElementById("status");
const treeRootElement = document.getElementById("tree-root");
const nodeJsonElement = document.getElementById("node-json");
const previewControlsElement = document.getElementById("preview-controls");
const previewJsonElement = document.getElementById("preview-json");
const previewWindowElement = document.getElementById("preview-window");
const previewPreviousButton = document.getElementById("preview-prev");
const previewNextButton = document.getElementById("preview-next");

let selectedNodePath = null;
let selectedNodeType = null;
let selectedTreeItem = null;
let previewOffset = 0;
let previewTotal = 0;

function setStatus(message, isError = false) {
  statusElement.textContent = message;
  statusElement.className = isError ? "status-error" : "status-ok";
}

async function fetchJson(url) {
  const response = await fetch(url);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `HTTP ${response.status}`);
  }
  return payload;
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

async function selectNode(path, nodeType, treeItem) {
  clearSelection();
  selectedTreeItem = treeItem;
  selectedTreeItem.classList.add("selected");

  selectedNodePath = path;
  selectedNodeType = nodeType;
  previewOffset = 0;
  previewTotal = 0;

  const detailsUrl = `${API_PREFIX}/node?path=${encodeURIComponent(path)}`;
  const payload = await fetchJson(detailsUrl);
  nodeJsonElement.textContent = JSON.stringify(payload, null, 2);

  if (selectedNodeType === "dataset") {
    previewControlsElement.hidden = false;
    await loadPreview();
  } else {
    previewControlsElement.hidden = true;
    previewJsonElement.textContent = "";
    previewWindowElement.textContent = "";
  }
}

async function loadPreview() {
  if (!selectedNodePath || selectedNodeType !== "dataset") {
    return;
  }

  const previewUrl =
    `${API_PREFIX}/preview?path=${encodeURIComponent(selectedNodePath)}` +
    `&offset=${previewOffset}&limit=${DEFAULT_LIMIT}`;
  const payload = await fetchJson(previewUrl);
  previewTotal = payload.total;

  previewJsonElement.textContent = JSON.stringify(payload, null, 2);
  previewWindowElement.textContent =
    `Offset: ${payload.offset}, Returned: ${payload.returned}, Total: ${payload.total}`;

  previewPreviousButton.disabled = previewOffset === 0;
  previewNextButton.disabled = previewOffset + DEFAULT_LIMIT >= previewTotal;
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

previewPreviousButton.addEventListener("click", async () => {
  previewOffset = Math.max(0, previewOffset - DEFAULT_LIMIT);
  try {
    await loadPreview();
  } catch (error) {
    setStatus(String(error), true);
  }
});

previewNextButton.addEventListener("click", async () => {
  previewOffset = Math.min(previewTotal, previewOffset + DEFAULT_LIMIT);
  try {
    await loadPreview();
  } catch (error) {
    setStatus(String(error), true);
  }
});

async function initialize() {
  try {
    await fetchJson(`${API_PREFIX}/health`);
    await initializeTree();
    setStatus("Viewer ready");
  } catch (error) {
    setStatus(String(error), true);
  }
}

initialize();
