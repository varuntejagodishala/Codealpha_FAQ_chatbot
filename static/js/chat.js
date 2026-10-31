/**
 * NovaTech FAQ Chatbot Frontend Controller
 * Handles real-time messaging, API communication, rendering, and modal explorer.
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const chatForm = document.getElementById("chatForm");
  const chatInput = document.getElementById("chatInput");
  const messagesContainer = document.getElementById("messagesContainer");
  const typingIndicator = document.getElementById("typingIndicator");
  const themeToggle = document.getElementById("themeToggle");
  const btnClearChat = document.getElementById("btnClearChat");
  const btnBrowseFaqs = document.getElementById("btnBrowseFaqs");
  const sidebar = document.getElementById("sidebar");
  const sidebarToggle = document.getElementById("sidebarToggle");

  // Modal Elements
  const faqModal = document.getElementById("faqModal");
  const modalClose = document.getElementById("modalClose");
  const modalFaqList = document.getElementById("modalFaqList");
  const modalSearchInput = document.getElementById("modalSearchInput");

  let allFaqsCache = [];

  // --------------------------------------------------------------------------
  // Theme Toggle
  // --------------------------------------------------------------------------
  const savedTheme = localStorage.getItem("novatech_theme") || "theme-light";
  document.body.className = savedTheme;
  themeToggle.textContent = savedTheme === "theme-dark" ? "☀️" : "🌙";

  themeToggle.addEventListener("click", () => {
    if (document.body.classList.contains("theme-dark")) {
      document.body.className = "theme-light";
      themeToggle.textContent = "🌙";
      localStorage.setItem("novatech_theme", "theme-light");
    } else {
      document.body.className = "theme-dark";
      themeToggle.textContent = "☀️";
      localStorage.setItem("novatech_theme", "theme-dark");
    }
  });

  // Mobile sidebar toggle
  if (sidebarToggle) {
    sidebarToggle.addEventListener("click", () => {
      sidebar.classList.toggle("open");
    });
  }

  // --------------------------------------------------------------------------
  // Helpers
  // --------------------------------------------------------------------------
  function getTimeString() {
    const now = new Date();
    return now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  function escapeHtml(str) {
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function formatMarkdown(text) {
    if (!text) return "";
    let safe = escapeHtml(text);
    // Bold: **text**
    safe = safe.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    // Italics: *text*
    safe = safe.replace(/\*(.*?)\*/g, "<em>$1</em>");
    // Links: [text](url) or naked URLs
    safe = safe.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
    // Newlines to breaks
    safe = safe.replace(/\n\n/g, "</p><p>");
    safe = safe.replace(/\n/g, "<br>");
    return `<p>${safe}</p>`;
  }

  // --------------------------------------------------------------------------
  // Message Rendering
  // --------------------------------------------------------------------------
  function appendUserMessage(text) {
    const row = document.createElement("div");
    row.className = "message-row user-row";
    row.innerHTML = `
      <div class="msg-avatar">You</div>
      <div class="message-bubble user-bubble">
        <div class="bubble-header">
          <span class="sender-name">You</span>
          <span class="msg-time">${getTimeString()}</span>
        </div>
        <div class="bubble-text">
          <p>${escapeHtml(text)}</p>
        </div>
      </div>
    `;
    messagesContainer.appendChild(row);
    scrollToBottom();
  }

  function appendBotMessage(data) {
    const row = document.createElement("div");
    row.className = "message-row bot-row";

    const answerHtml = formatMarkdown(data.answer);

    // Metadata badge (Confidence + Category)
    let metaHtml = "";
    if (data.type === "matched") {
      const confLevel = data.confidence_level || "high";
      const badgeIcon = confLevel === "high" ? "✓" : "⚡";
      const badgeClass = confLevel === "high" ? "high" : "moderate";
      const confText = data.confidence_percent ? `${data.confidence_percent}% Match` : "Matched";

      metaHtml = `
        <div class="match-meta">
          <span class="confidence-badge ${badgeClass}">${badgeIcon} ${confText}</span>
          ${data.category ? `<span class="category-tag">${escapeHtml(data.category)}</span>` : ""}
        </div>
      `;
    }

    // Related suggestions
    let suggestionsHtml = "";
    if (data.suggestions && data.suggestions.length > 0) {
      const chipButtons = data.suggestions
        .map(
          (s) =>
            `<button class="suggestion-chip" data-query="${escapeHtml(s)}">${escapeHtml(s)}</button>`
        )
        .join("");
      suggestionsHtml = `
        <div class="suggestions-box">
          <span class="suggestions-title">Related questions:</span>
          <div class="quick-chips">${chipButtons}</div>
        </div>
      `;
    }

    row.innerHTML = `
      <div class="msg-avatar">🤖</div>
      <div class="message-bubble bot-bubble">
        <div class="bubble-header">
          <span class="sender-name">NovaTech Assistant</span>
          <span class="msg-time">${getTimeString()}</span>
        </div>
        <div class="bubble-text">
          ${answerHtml}
        </div>
        ${metaHtml}
        ${suggestionsHtml}
      </div>
    `;

    messagesContainer.appendChild(row);
    scrollToBottom();

    // Attach click handlers to newly created suggestion chips
    row.querySelectorAll(".suggestion-chip").forEach((chip) => {
      chip.addEventListener("click", () => {
        sendUserQuery(chip.getAttribute("data-query"));
      });
    });
  }

  // --------------------------------------------------------------------------
  // Chat Communication (POST /api/chat)
  // --------------------------------------------------------------------------
  async function sendUserQuery(query) {
    const text = query.trim();
    if (!text) return;

    appendUserMessage(text);
    chatInput.value = "";
    chatInput.disabled = true;

    // Show typing animation
    typingIndicator.style.display = "flex";
    scrollToBottom();

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      });

      const result = await response.json();
      typingIndicator.style.display = "none";

      if (response.ok && result.response) {
        appendBotMessage(result.response);
      } else {
        appendBotMessage({
          type: "fallback",
          answer: "Sorry, I encountered a temporary problem answering your query. Please try again.",
          suggestions: ["How can I track my order status?", "What is your return policy?"]
        });
      }
    } catch (err) {
      console.error("Chat API error:", err);
      typingIndicator.style.display = "none";
      appendBotMessage({
        type: "fallback",
        answer: "Connection error: Unable to reach the FAQ server. Please make sure the service is running.",
        suggestions: []
      });
    } finally {
      chatInput.disabled = false;
      chatInput.focus();
      scrollToBottom();
    }
  }

  // Form submission
  chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    sendUserQuery(chatInput.value);
  });

  // Attach click listeners to static chips & topic pills
  document.querySelectorAll(".chip-btn, .topic-pill").forEach((btn) => {
    btn.addEventListener("click", () => {
      sendUserQuery(btn.getAttribute("data-query"));
    });
  });

  // --------------------------------------------------------------------------
  // Clear Chat Conversation
  // --------------------------------------------------------------------------
  btnClearChat.addEventListener("click", () => {
    if (confirm("Clear conversation history?")) {
      messagesContainer.innerHTML = `
        <div class="message-row bot-row">
          <div class="msg-avatar">🤖</div>
          <div class="message-bubble bot-bubble">
            <div class="bubble-header">
              <span class="sender-name">NovaTech Assistant</span>
              <span class="msg-time">${getTimeString()}</span>
            </div>
            <div class="bubble-text">
              <p>Conversation cleared! What else can I help you find today?</p>
            </div>
          </div>
        </div>
      `;
    }
  });

  // --------------------------------------------------------------------------
  // Knowledge Base Explorer Modal
  // --------------------------------------------------------------------------
  async function loadKnowledgeBase(categoryFilter = null) {
    try {
      if (allFaqsCache.length === 0) {
        const res = await fetch("/api/faqs");
        const data = await res.json();
        allFaqsCache = data.faqs || [];
      }

      renderModalFaqs(allFaqsCache, categoryFilter);
      faqModal.style.display = "flex";
    } catch (err) {
      console.error("Failed to load FAQs:", err);
    }
  }

  function renderModalFaqs(faqs, categoryFilter = null, searchQuery = "") {
    let filtered = faqs;

    if (categoryFilter) {
      filtered = filtered.filter((f) => f.category === categoryFilter);
    }

    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      filtered = filtered.filter(
        (f) =>
          f.question.toLowerCase().includes(q) ||
          f.answer.toLowerCase().includes(q) ||
          (f.tags && f.tags.some((t) => t.toLowerCase().includes(q)))
      );
    }

    if (filtered.length === 0) {
      modalFaqList.innerHTML = `<p style="color: var(--text-muted); text-align: center; padding: 2rem;">No matching FAQs found.</p>`;
      return;
    }

    modalFaqList.innerHTML = filtered
      .map(
        (faq) => `
        <div class="faq-card-item" data-question="${escapeHtml(faq.question)}">
          <div class="faq-card-cat">${escapeHtml(faq.category || "General")}</div>
          <div class="faq-card-q">${escapeHtml(faq.question)}</div>
          <div class="faq-card-a">${escapeHtml(faq.answer)}</div>
        </div>
      `
      )
      .join("");

    // Clicking an FAQ card in the modal asks that question in chat
    modalFaqList.querySelectorAll(".faq-card-item").forEach((card) => {
      card.addEventListener("click", () => {
        const q = card.getAttribute("data-question");
        faqModal.style.display = "none";
        sendUserQuery(q);
      });
    });
  }

  btnBrowseFaqs.addEventListener("click", () => {
    modalSearchInput.value = "";
    loadKnowledgeBase();
  });

  modalClose.addEventListener("click", () => {
    faqModal.style.display = "none";
  });

  window.addEventListener("click", (e) => {
    if (e.target === faqModal) {
      faqModal.style.display = "none";
    }
  });

  modalSearchInput.addEventListener("input", (e) => {
    renderModalFaqs(allFaqsCache, null, e.target.value.trim());
  });

  // Sidebar category buttons
  document.querySelectorAll(".category-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const category = btn.getAttribute("data-category");
      modalSearchInput.value = "";
      loadKnowledgeBase(category);
      if (window.innerWidth <= 900) {
        sidebar.classList.remove("open");
      }
    });
  });
});
