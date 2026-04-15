/* ─────────────────────────────────────────────────────────
   sLLMates · app.js
   Flask API와 통신하는 프론트엔드 로직 전체
───────────────────────────────────────────────────────── */

// ── State ───────────────────────────────────────────────
const state = {
  currentThreadId: null,
  currentChatName: null,
  selectedBioId: null,
  isStreaming: false,
};

// ── DOM refs ─────────────────────────────────────────────
const $ = (id) => document.getElementById(id);

const els = {
  chatList:         $("chat-list"),
  messages:         $("messages"),
  emptyState:       $("empty-state"),
  msgInput:         $("msg-input"),
  sendBtn:          $("send-btn"),
  currentChatName:  $("current-chat-name"),
  currentChatDisp:  $("current-chat-display"),
  renameInput:      $("rename-input"),
  renameBtn:        $("rename-btn"),
  deleteChatBtn:    $("delete-chat-btn"),
  newChatBtn:       $("new-chat-btn"),
  bioList:          $("bio-list"),
  bioSelect:        $("bio-select"),
  addBioText:       $("add-bio-text"),
  addBioImportance: $("add-bio-importance"),
  addImportanceVal: $("add-importance-val"),
  addBioBtn:        $("add-bio-btn"),
  editBioText:      $("edit-bio-text"),
  editBioImportance:$("edit-bio-importance"),
  editImportanceVal:$("edit-importance-val"),
  updateBioBtn:     $("update-bio-btn"),
  deleteBioBtn:     $("delete-bio-btn"),
  refreshBioBtn:    $("refresh-bio-btn"),
  toast:            $("toast"),
};

// ── Toast ────────────────────────────────────────────────
let toastTimer = null;
function showToast(msg, duration = 2400) {
  els.toast.textContent = msg;
  els.toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => els.toast.classList.remove("show"), duration);
}

// ── Tab switching ────────────────────────────────────────
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    $(`panel-${btn.dataset.tab}`).classList.add("active");
  });
});

document.querySelectorAll(".sub-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const parent = btn.closest(".panel");
    parent.querySelectorAll(".sub-btn").forEach((b) => b.classList.remove("active"));
    parent.querySelectorAll(".sub-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    $(`sub-${btn.dataset.sub}`).classList.add("active");
  });
});

// ── API helpers ──────────────────────────────────────────
async function api(method, path, body) {
  const opts = {
    method,
    headers: { "Content-Type": "application/json" },
  };
  if (body) opts.body = JSON.stringify(body);
  const res = await fetch(path, opts);
  return res.json();
}

// ── Render chat list ─────────────────────────────────────
function renderChatList(chats) {
  els.chatList.innerHTML = "";
  if (!chats.length) {
    els.chatList.innerHTML = `<div style="padding:10px 14px;font-size:12px;color:var(--g400)">채팅이 없습니다</div>`;
    return;
  }
  chats.forEach((c) => {
    const el = document.createElement("div");
    el.className = "chat-item" + (c.id === state.currentThreadId ? " active" : "");
    el.dataset.id = c.id;
    el.innerHTML = `<span>${escHtml(c.name)}</span>`;
    el.addEventListener("click", () => loadChat(c.id));
    els.chatList.appendChild(el);
  });
}

async function refreshChatList() {
  const chats = await api("GET", "/api/chats");
  renderChatList(chats);
}

// ── Load a chat ──────────────────────────────────────────
async function loadChat(threadId) {
  const data = await api("GET", `/api/chats/${threadId}`);
  if (data.error) { showToast("❌ " + data.error); return; }

  state.currentThreadId = threadId;
  state.currentChatName = data.name;

  // Sidebar highlight
  document.querySelectorAll(".chat-item").forEach((el) => {
    el.classList.toggle("active", el.dataset.id === threadId);
  });

  els.currentChatName.textContent = data.name;
  els.currentChatDisp.value = data.name;

  // Render history
  clearMessages();
  if (data.history.length === 0) {
    showEmpty(true);
  } else {
    showEmpty(false);
    data.history.forEach((m) => {
      if (!m.content || m.content.trim() === "") {
        return; // Ignore any empty messages in history
      }
      appendMessage(m.role, m.content);
    });
    scrollToBottom();
  }
}

// ── New chat ─────────────────────────────────────────────
els.newChatBtn.addEventListener("click", async () => {
  const data = await api("POST", "/api/chats");
  if (data.error) { showToast("❌ " + data.error); return; }

  state.currentThreadId = data.id;
  state.currentChatName = data.name;
  els.currentChatName.textContent = data.name;
  els.currentChatDisp.value = data.name;

  clearMessages();
  showEmpty(false);
  appendMessage("assistant", "안녕하세요! 무엇을 도와드릴까요?");
  scrollToBottom();

  await refreshChatList();

  // Activate correct chat item
  document.querySelectorAll(".chat-item").forEach((el) => {
    el.classList.toggle("active", el.dataset.id === data.id);
  });
});

// ── Delete chat ──────────────────────────────────────────
els.deleteChatBtn.addEventListener("click", async () => {
  if (!state.currentThreadId) { showToast("⚠️ 채팅방을 선택하세요"); return; }
  if (!confirm("이 채팅방을 삭제하시겠습니까?")) return;

  const res = await api("DELETE", `/api/chats/${state.currentThreadId}`);
  if (res.error) { showToast("❌ " + res.error); return; }

  state.currentThreadId = null;
  state.currentChatName = null;
  els.currentChatName.textContent = "채팅방을 선택하세요";
  els.currentChatDisp.value = "";
  clearMessages();
  showEmpty(true);
  showToast("🗑 채팅방이 삭제되었습니다");
  await refreshChatList();
});

// ── Rename chat ──────────────────────────────────────────
els.renameBtn.addEventListener("click", async () => {
  if (!state.currentThreadId) { showToast("⚠️ 채팅방을 선택하세요"); return; }
  const newName = els.renameInput.value.trim();
  if (!newName) { showToast("⚠️ 이름을 입력하세요"); return; }

  const res = await api("PATCH", `/api/chats/${state.currentThreadId}`, { name: newName });
  if (res.error) { showToast("❌ " + res.error); return; }

  state.currentChatName = newName;
  els.currentChatName.textContent = newName;
  els.currentChatDisp.value = newName;
  els.renameInput.value = "";
  showToast("✅ 이름이 변경되었습니다");
  await refreshChatList();
});

// ── Send message (SSE streaming) ─────────────────────────
async function sendMessage() {
    const message = els.msgInput.value.trim();
    if (!message || state.isStreaming) return;
    if (!state.currentThreadId) { showToast("⚠️ 먼저 채팅방을 선택하세요"); return; }

    state.isStreaming = true;
    els.sendBtn.disabled = true;
    els.msgInput.value = "";
    autoResize(els.msgInput);

    showEmpty(false);
    appendMessage("user", message);

    const aiRow = createAIRow();
    const bubble = aiRow.querySelector(".bubble");
    els.messages.appendChild(aiRow);
    scrollToBottom();

    let accumulated = "";
    let isStatus = true;

    try {
        // 1단계: 메시지 등록
        const postRes = await fetch(`/api/chats/${state.currentThreadId}/message`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message }),
        });
        const { stream_id, error } = await postRes.json();
        if (error) throw new Error(error);

        // 2단계: EventSource로 스트림 수신
        await new Promise((resolve, reject) => {
            const es = new EventSource(`/api/stream/${stream_id}`);

            es.onmessage = (e) => {
                let chunk;
                try { chunk = JSON.parse(e.data); } catch { return; }

                if (chunk.type === "rename") {
                    state.currentChatName = chunk.name;
                    els.currentChatName.textContent = chunk.name;
                    els.currentChatDisp.value = chunk.name;
                    refreshChatList();

                } else if (chunk.type === "status") {
                    isStatus = true;
                    bubble.className = "bubble status";
                    bubble.innerHTML = `<div class="typing-dots"><span></span><span></span><span></span></div> ${escHtml(chunk.content)}`;
                    scrollToBottom();

                } else if (chunk.type === "token") {
                    if (isStatus) {
                        isStatus = false;
                        accumulated = "";
                        bubble.className = "bubble ai";
                        bubble.innerHTML = "";
                    }
                    accumulated += chunk.content;
                    bubble.innerHTML = renderMarkdown(accumulated);
                    scrollToBottom();

                } else if (chunk.type === "done") {
                    es.close();
                    refreshChatList();
                    resolve();

                } else if (chunk.type === "error") {
                    es.close();
                    bubble.className = "bubble ai";
                    bubble.innerHTML = `<span style="color:#ef4444">❌ ${escHtml(chunk.content)}</span>`;
                    reject(new Error(chunk.content));
                }
            };

            es.onerror = (e) => {
                es.close();
                reject(new Error("EventSource 연결 오류"));
            };
        });

    } catch (err) {
        bubble.className = "bubble ai";
        bubble.innerHTML = `<span style="color:#ef4444">❌ 연결 오류: ${escHtml(err.message)}</span>`;
    } finally {
        state.isStreaming = false;
        els.sendBtn.disabled = false;
        els.msgInput.focus();
        scrollToBottom();
    }
}

els.sendBtn.addEventListener("click", sendMessage);
els.msgInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// ── Auto resize textarea ──────────────────────────────────
function autoResize(el) {
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 140) + "px";
}
els.msgInput.addEventListener("input", () => autoResize(els.msgInput));

// ── Message rendering helpers ─────────────────────────────
function clearMessages() {
  // empty-state는 남기고 다른 메시지만 제거
  const rows = els.messages.querySelectorAll(".msg-row");
  rows.forEach((r) => r.remove());
}

function showEmpty(show) {
  $("empty-state").style.display = show ? "flex" : "none";
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderMarkdown(text) {
  if (typeof marked === "undefined") return escHtml(text).replace(/\n/g, "<br>");
  return marked.parse(text);
}

function appendMessage(role, content) {
  const row = document.createElement("div");
  row.className = `msg-row ${role === "user" ? "user" : "ai"}`;

  const avatar = document.createElement("div");
  avatar.className = `avatar ${role === "user" ? "user" : "ai"}`;
  avatar.textContent = role === "user" ? "👤" : "🤖";

  const bubble = document.createElement("div");
  bubble.className = `bubble ${role === "user" ? "user" : "ai"}`;
  bubble.innerHTML = role === "user" ? escHtml(content).replace(/\n/g, "<br>") : renderMarkdown(content);

  row.appendChild(avatar);
  row.appendChild(bubble);
  els.messages.appendChild(row);
  return row;
}

function createAIRow() {
  const row = document.createElement("div");
  row.className = "msg-row ai";

  const avatar = document.createElement("div");
  avatar.className = "avatar ai";
  avatar.textContent = "🤖";

  const bubble = document.createElement("div");
  bubble.className = "bubble status";
  bubble.innerHTML = `<div class="typing-dots"><span></span><span></span><span></span></div> 처리 중...`;

  row.appendChild(avatar);
  row.appendChild(bubble);
  return row;
}

function scrollToBottom() {
  els.messages.scrollTop = els.messages.scrollHeight;
}

// ── Bio ───────────────────────────────────────────────────

// Slider live value display
els.addBioImportance.addEventListener("input", () => {
  els.addImportanceVal.textContent = els.addBioImportance.value;
});
els.editBioImportance.addEventListener("input", () => {
  els.editImportanceVal.textContent = els.editBioImportance.value;
});

async function loadBios() {
  const bios = await api("GET", "/api/bios");
  renderBioList(bios);
  renderBioSelect(bios);
}

function renderBioList(bios) {
  els.bioList.innerHTML = "";
  if (!bios.length) {
    els.bioList.innerHTML = `<div style="padding:10px 14px;font-size:12px;color:var(--g400)">저장된 Bio가 없습니다</div>`;
    return;
  }
  bios.forEach((b) => {
    const el = document.createElement("div");
    el.className = "bio-item" + (b.id === state.selectedBioId ? " active" : "");
    el.dataset.id = b.id;
    el.innerHTML = `
      <span class="bio-badge">${b.importance}</span>
      <span class="bio-text">${escHtml(b.document)}</span>
    `;
    el.addEventListener("click", () => {
      state.selectedBioId = b.id;
      document.querySelectorAll(".bio-item").forEach((i) => i.classList.toggle("active", i.dataset.id === b.id));
    });
    els.bioList.appendChild(el);
  });
}

function renderBioSelect(bios) {
  els.bioSelect.innerHTML = `<option value="">-- 선택하세요 --</option>`;
  bios.forEach((b) => {
    const opt = document.createElement("option");
    opt.value = b.id;
    opt.textContent = `[${b.importance}] ${b.document.slice(0, 40)}${b.document.length > 40 ? "..." : ""}`;
    els.bioSelect.appendChild(opt);
  });
}

els.bioSelect.addEventListener("change", async () => {
  const bioId = els.bioSelect.value;
  if (!bioId) { els.editBioText.value = ""; return; }

  const bios = await api("GET", "/api/bios");
  const bio = bios.find((b) => b.id === bioId);
  if (bio) {
    state.selectedBioId = bioId;
    els.editBioText.value = bio.document;
    els.editBioImportance.value = bio.importance;
    els.editImportanceVal.textContent = bio.importance;
  }
});

els.addBioBtn.addEventListener("click", async () => {
  const text = els.addBioText.value.trim();
  const importance = parseInt(els.addBioImportance.value);
  if (!text) { showToast("⚠️ 내용을 입력하세요"); return; }

  const res = await api("POST", "/api/bios", { text, importance });
  if (res.error) { showToast("❌ " + res.error); return; }

  els.addBioText.value = "";
  showToast("✅ Bio가 추가되었습니다");
  await loadBios();
});

els.updateBioBtn.addEventListener("click", async () => {
  const bioId = els.bioSelect.value;
  if (!bioId) { showToast("⚠️ Bio를 선택하세요"); return; }
  const text = els.editBioText.value.trim();
  const importance = parseInt(els.editBioImportance.value);
  if (!text) { showToast("⚠️ 내용을 입력하세요"); return; }

  const res = await api("PATCH", `/api/bios/${bioId}`, { text, importance });
  if (res.error) { showToast("❌ " + res.error); return; }

  showToast("✅ Bio가 업데이트되었습니다");
  await loadBios();
});

els.deleteBioBtn.addEventListener("click", async () => {
  const bioId = els.bioSelect.value;
  if (!bioId) { showToast("⚠️ Bio를 선택하세요"); return; }
  if (!confirm("이 Bio를 삭제하시겠습니까?")) return;

  const res = await api("DELETE", `/api/bios/${bioId}`);
  if (res.error) { showToast("❌ " + res.error); return; }

  els.editBioText.value = "";
  els.bioSelect.value = "";
  state.selectedBioId = null;
  showToast("🗑 Bio가 삭제되었습니다");
  await loadBios();
});

els.refreshBioBtn.addEventListener("click", async () => {
  await loadBios();
  showToast("↺ 새로고침 완료");
});

// ── Init ──────────────────────────────────────────────────
(async () => {
  await refreshChatList();
  await loadBios();

  // 앱 시작 시 새 채팅 자동 생성
  els.newChatBtn.click();
})();