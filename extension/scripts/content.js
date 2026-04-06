/**
 * Genzon — Content Script
 * Scrapes Amazon review data, sends to API, injects scores inline.
 */

(function () {
    "use strict";
  
    // Prevent running twice
    if (window.__genzonLoaded) return;
    window.__genzonLoaded = true;
  
    console.log("[Genzon] Content script loaded");
  
    // ── Config ──
    const MAX_REVIEWS = 50;
    const RETRY_DELAY = 2000;
  
    // ── Scrape reviews from Amazon DOM ──
  
    function scrapeReviews() {
      const reviews = [];
  
      // Amazon review selectors (they change occasionally)
      const reviewElements = document.querySelectorAll(
        '[data-hook="review"], .review, .a-section.review'
      );
  
      reviewElements.forEach((el, index) => {
        if (index >= MAX_REVIEWS) return;
  
        // Review text
        const textEl = el.querySelector(
          '[data-hook="review-body"] span, .review-text-content span, .review-text span'
        );
        const reviewText = textEl ? textEl.textContent.trim() : "";
  
        if (!reviewText || reviewText.length < 5) return;
  
        // Star rating
        const starEl = el.querySelector(
          '[data-hook="review-star-rating"] .a-icon-alt, .review-rating .a-icon-alt'
        );
        let starRating = 3;
        if (starEl) {
          const match = starEl.textContent.match(/(\d\.?\d?)/);
          if (match) starRating = Math.round(parseFloat(match[1]));
        }
  
        // Verified purchase
        const verifiedEl = el.querySelector(
          '[data-hook="avp-badge"], .a-size-mini .a-color-state'
        );
        const verifiedPurchase = verifiedEl
          ? verifiedEl.textContent.toLowerCase().includes("verified")
          : false;
  
        // Helpful votes
        const helpfulEl = el.querySelector(
          '[data-hook="helpful-vote-statement"], .cr-vote-text'
        );
        let helpfulVotes = 0;
        if (helpfulEl) {
          const match = helpfulEl.textContent.match(/(\d+)/);
          if (match) helpfulVotes = parseInt(match[1]);
        }
  
        // Has media (images/videos)
        const hasMedia =
          el.querySelector(
            '[data-hook="review-image-tile"], .review-image-tile-section, video'
          ) !== null;
  
        // Review date
        const dateEl = el.querySelector(
          '[data-hook="review-date"], .review-date'
        );
        const reviewDate = dateEl ? dateEl.textContent.trim() : null;
  
        reviews.push({
          element: el,
          data: {
            review_text: reviewText,
            star_rating: starRating,
            verified_purchase: verifiedPurchase,
            helpful_votes: helpfulVotes,
            has_media: hasMedia,
            review_date: reviewDate,
          },
        });
      });
  
      console.log(`[Genzon] Scraped ${reviews.length} reviews`);
      return reviews;
    }
  
    // ── Inject score badge next to a review ──
  
    function createBadge(score, label, flags) {
      const badge = document.createElement("div");
      badge.className = "genzon-badge";
  
      // Color based on score
      let colorClass = "genzon-uncertain";
      if (flags && flags.includes("rule_ml_divergence")) {
        colorClass = "genzon-flagged";
      } else if (score >= 8) {
        colorClass = "genzon-genuine";
      } else if (score < 5) {
        colorClass = "genzon-fake";
      }
  
      badge.classList.add(colorClass);
  
      // Score display
      badge.innerHTML = `
        <div class="genzon-badge-inner">
          <span class="genzon-score">${score.toFixed(1)}</span>
          <span class="genzon-label">${label}</span>
          <span class="genzon-brand">Genzon</span>
        </div>
      `;
  
      // Tooltip on hover
      badge.title = `Genzon Score: ${score}/10 — ${label}`;
  
      return badge;
    }
  
    function injectBadge(reviewElement, scoreData) {
      // Remove existing badge if re-running
      const existing = reviewElement.querySelector(".genzon-badge");
      if (existing) existing.remove();
  
      const badge = createBadge(
        scoreData.score,
        scoreData.label,
        scoreData.flags
      );
  
      // Insert at the top of the review
      const firstChild = reviewElement.querySelector(
        '[data-hook="review-star-rating"], .review-rating'
      );
      if (firstChild) {
        firstChild.parentNode.insertBefore(badge, firstChild.nextSibling);
      } else {
        reviewElement.prepend(badge);
      }
    }
  
    // ── Inject aggregate score at the top ──
  
    function injectAggregateScore(aggregateData) {
      // Remove existing
      const existing = document.querySelector(".genzon-aggregate");
      if (existing) existing.remove();
  
      let colorClass = "genzon-uncertain";
      if (aggregateData.aggregate_score >= 8) colorClass = "genzon-genuine";
      else if (aggregateData.aggregate_score < 5) colorClass = "genzon-fake";
  
      const container = document.createElement("div");
      container.className = `genzon-aggregate ${colorClass}`;
  
      container.innerHTML = `
        <div class="genzon-aggregate-inner">
          <div class="genzon-aggregate-header">
            <span class="genzon-aggregate-logo">🔍 Genzon Review Analysis</span>
          </div>
          <div class="genzon-aggregate-body">
            <div class="genzon-aggregate-score">
              <span class="genzon-big-score">${aggregateData.aggregate_score.toFixed(1)}</span>
              <span class="genzon-out-of">/10</span>
            </div>
            <div class="genzon-aggregate-details">
              <span class="genzon-aggregate-label">${aggregateData.aggregate_label}</span>
              <span class="genzon-aggregate-count">${aggregateData.total_reviews_analyzed} reviews analyzed</span>
            </div>
          </div>
        </div>
      `;
  
      // Insert before the reviews section
      const reviewSection = document.querySelector(
        '#cm_cr-review_list, [data-hook="top-customer-reviews-widget"], #reviewsMedley'
      );
      if (reviewSection) {
        reviewSection.parentNode.insertBefore(container, reviewSection);
      }
    }
  
    // ── Show loading state ──
  
    function showLoading(reviews) {
      reviews.forEach(({ element }) => {
        const existing = element.querySelector(".genzon-badge");
        if (existing) existing.remove();
  
        const loading = document.createElement("div");
        loading.className = "genzon-badge genzon-loading";
        loading.innerHTML = `
          <div class="genzon-badge-inner">
            <span class="genzon-label">Analyzing...</span>
            <span class="genzon-brand">Genzon</span>
          </div>
        `;
  
        const firstChild = element.querySelector(
          '[data-hook="review-star-rating"], .review-rating'
        );
        if (firstChild) {
          firstChild.parentNode.insertBefore(loading, firstChild.nextSibling);
        } else {
          element.prepend(loading);
        }
      });
    }
  
    // ── Show error state ──
  
    function showError(reviews, message) {
      reviews.forEach(({ element }) => {
        const existing = element.querySelector(".genzon-badge");
        if (existing) existing.remove();
  
        const errorBadge = document.createElement("div");
        errorBadge.className = "genzon-badge genzon-error";
        errorBadge.innerHTML = `
          <div class="genzon-badge-inner">
            <span class="genzon-label">${message}</span>
            <span class="genzon-brand">Genzon</span>
          </div>
        `;
        errorBadge.title = message;
  
        const firstChild = element.querySelector(
          '[data-hook="review-star-rating"], .review-rating'
        );
        if (firstChild) {
          firstChild.parentNode.insertBefore(errorBadge, firstChild.nextSibling);
        } else {
          element.prepend(errorBadge);
        }
      });
    }
  
    // ── Main: scrape → send → inject ──
  
    async function analyzeReviews() {
      const reviews = scrapeReviews();
  
      if (reviews.length === 0) {
        console.log("[Genzon] No reviews found on this page");
        return;
      }
  
      // Show loading
      showLoading(reviews);
  
      // Check API health
      const healthy = await GenzonAPI.healthCheck();
      if (!healthy) {
        showError(reviews, "API offline");
        console.error("[Genzon] API is not reachable at", GenzonAPI.BASE_URL);
        return;
      }
  
      try {
        // Send to API
        const reviewData = reviews.map((r) => r.data);
        console.log(`[Genzon] Sending ${reviewData.length} reviews to API...`);
  
        const result = await GenzonAPI.predict(reviewData);
        console.log("[Genzon] Got scores:", result);
  
        // Inject per-review badges
        result.review_scores.forEach((score, i) => {
          if (reviews[i]) {
            injectBadge(reviews[i].element, score);
          }
        });
  
        // Inject aggregate score
        injectAggregateScore(result);
  
        console.log("[Genzon] Scores injected successfully");
      } catch (error) {
        console.error("[Genzon] Error:", error.message);
        showError(reviews, "Error");
      }
    }
  
    // ── Run ──
  
    // Wait for reviews to load (Amazon lazy-loads them)
    function waitForReviews() {
      const reviewSection = document.querySelector(
        '[data-hook="review"], .review, #cm_cr-review_list'
      );
  
      if (reviewSection) {
        analyzeReviews();
      } else {
        console.log("[Genzon] Waiting for reviews to load...");
        setTimeout(waitForReviews, RETRY_DELAY);
      }
    }
  
    // Start
    if (document.readyState === "complete") {
      waitForReviews();
    } else {
      window.addEventListener("load", waitForReviews);
    }
  
    // Re-analyze when user navigates to a different page (SPA behavior)
    let lastUrl = location.href;
    new MutationObserver(() => {
      if (location.href !== lastUrl) {
        lastUrl = location.href;
        console.log("[Genzon] URL changed, re-analyzing...");
        setTimeout(waitForReviews, RETRY_DELAY);
      }
    }).observe(document.body, { childList: true, subtree: true });
  })();