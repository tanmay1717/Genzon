/**
 * Genzon — API Client
 * Routes all API calls through the background service worker
 * to bypass HTTPS→HTTP restriction on Amazon pages.
 */

const GenzonAPI = {
  /**
   * Send reviews to the backend for scoring.
   */
  async predict(reviews) {
    return new Promise((resolve, reject) => {
      chrome.runtime.sendMessage(
        { type: "PREDICT", reviews },
        (response) => {
          if (chrome.runtime.lastError) {
            reject(new Error(chrome.runtime.lastError.message));
            return;
          }
          if (response && response.ok) {
            resolve(response.data);
          } else {
            reject(new Error(response?.error || "API request failed"));
          }
        }
      );
    });
  },

  /**
   * Check if the API is reachable.
   */
  async healthCheck() {
    return new Promise((resolve) => {
      chrome.runtime.sendMessage({ type: "HEALTH_CHECK" }, (response) => {
        if (chrome.runtime.lastError) {
          resolve(false);
          return;
        }
        resolve(
          response?.ok &&
          response?.data?.status === "healthy" &&
          response?.data?.model_loaded === true
        );
      });
    });
  },
};