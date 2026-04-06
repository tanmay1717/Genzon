/**
 * Genzon — Background Service Worker
 * Routes API calls from content script to bypass HTTPS→HTTP restriction.
 */

const API_URL = "http://localhost:8000";

// Install
chrome.runtime.onInstalled.addListener((details) => {
  console.log("[Genzon] Extension installed:", details.reason);
  chrome.storage.local.set({ apiUrl: API_URL });
});

// Handle messages from content script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "HEALTH_CHECK") {
    fetch(`${API_URL}/health`)
      .then((res) => res.json())
      .then((data) => sendResponse({ ok: true, data }))
      .catch((err) => sendResponse({ ok: false, error: err.message }));
    return true; // keep channel open for async response
  }

  if (message.type === "PREDICT") {
    fetch(`${API_URL}/api/v1/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reviews: message.reviews }),
    })
      .then((res) => res.json())
      .then((data) => sendResponse({ ok: true, data }))
      .catch((err) => sendResponse({ ok: false, error: err.message }));
    return true;
  }

  if (message.type === "GET_API_URL") {
    sendResponse({ apiUrl: API_URL });
  }
});