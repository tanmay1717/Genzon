/**
 * Genzon — Popup Script
 * Handles the extension popup UI.
 */

document.addEventListener("DOMContentLoaded", async () => {
    const dot = document.getElementById("dot");
    const statusText = document.getElementById("status-text");
    const btnReanalyze = document.getElementById("btn-reanalyze");
    const btnSettings = document.getElementById("btn-settings");
    const btnSave = document.getElementById("btn-save");
    const settingsPanel = document.getElementById("settings-panel");
    const apiUrlInput = document.getElementById("api-url");
  
    // Load saved API URL
    const stored = await chrome.storage.local.get("apiUrl");
    if (stored.apiUrl) {
      apiUrlInput.value = stored.apiUrl;
    }
  
    // Check API status
    const apiUrl = apiUrlInput.value;
    try {
      const response = await fetch(`${apiUrl}/health`);
      const data = await response.json();
  
      if (data.status === "healthy" && data.model_loaded) {
        dot.className = "dot online";
        statusText.textContent = `Connected — ${data.device}`;
      } else {
        dot.className = "dot offline";
        statusText.textContent = "API running but models not loaded";
      }
    } catch {
      dot.className = "dot offline";
      statusText.textContent = "API offline — start the server";
    }
  
    // Re-analyze button
    btnReanalyze.addEventListener("click", async () => {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (tab && tab.url && tab.url.includes("amazon")) {
        chrome.scripting.executeScript({
          target: { tabId: tab.id },
          func: () => {
            // Remove existing badges and re-run
            document.querySelectorAll(".genzon-badge, .genzon-aggregate").forEach((el) => el.remove());
            window.__genzonLoaded = false;
            // Re-inject content script
            const script = document.createElement("script");
            script.src = chrome.runtime.getURL("scripts/content.js");
            document.head.appendChild(script);
          },
        });
        window.close();
      } else {
        statusText.textContent = "Open an Amazon page first";
      }
    });
  
    // Settings toggle
    btnSettings.addEventListener("click", () => {
      const visible = settingsPanel.style.display !== "none";
      settingsPanel.style.display = visible ? "none" : "block";
      btnSettings.textContent = visible ? "Settings" : "Hide Settings";
    });
  
    // Save settings
    btnSave.addEventListener("click", async () => {
      const url = apiUrlInput.value.replace(/\/+$/, ""); // remove trailing slash
      await chrome.storage.local.set({ apiUrl: url });
      statusText.textContent = "Settings saved! Reload the page.";
    });
  });