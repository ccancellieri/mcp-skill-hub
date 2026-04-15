// skill-hub webapp entry — reserved for Phase 2+ client hooks.
document.addEventListener("DOMContentLoaded", () => {
  initTooltips();
  initExpandableCards();
});

/**
 * Tooltip system — single body-attached popup, position:fixed.
 * Works inside overflow:hidden containers (e.g. settings panels).
 *
 * Usage: <a class="tooltip-help" href="#"
 *            data-tooltip="Short description"
 *            data-help-page="settings-router"   ← optional, opens modal on click
 *            tabindex="0">?</a>
 */
function initTooltips() {
  // Shared singleton popup
  const popup = document.createElement('div');
  popup.className = 'tooltip-popup';
  document.body.appendChild(popup);

  let hideTimer = null;
  let currentTrigger = null;

  function buildContent(el) {
    popup.innerHTML = '';
    const textP = document.createElement('p');
    textP.className = 'tooltip-text';
    textP.textContent = el.dataset.tooltip;
    popup.appendChild(textP);

    const hp = el.dataset.helpPage;
    if (hp) {
      const a = document.createElement('a');
      a.href = '#';
      a.className = 'tooltip-link';
      a.textContent = 'Full details →';
      a.onclick = ev => { ev.preventDefault(); hidePopup(); openHelpModal(hp); };
      popup.appendChild(a);
    }
  }

  function positionPopup(el) {
    const r = el.getBoundingClientRect();
    // Measure popup size (it must be briefly visible for this)
    popup.style.top = '-9999px';
    popup.style.left = '-9999px';
    popup.classList.add('visible');

    const pw = popup.offsetWidth;
    const ph = popup.offsetHeight;
    const gap = 8;
    const vw = window.innerWidth;

    // Prefer above; fall back to below if not enough room
    let top = r.top - ph - gap;
    if (top < 6) top = r.bottom + gap;

    // Center horizontally on trigger, clamp to viewport
    let left = r.left + r.width / 2 - pw / 2;
    left = Math.max(6, Math.min(left, vw - pw - 6));

    // Arrow X offset relative to popup
    const arrowX = Math.round(r.left + r.width / 2 - left);
    popup.style.setProperty('--arrow-x', arrowX + 'px');
    popup.style.top  = top  + 'px';
    popup.style.left = left + 'px';
  }

  function showPopup(el) {
    clearTimeout(hideTimer);
    currentTrigger = el;
    buildContent(el);
    positionPopup(el);
  }

  function hidePopup() {
    clearTimeout(hideTimer);
    popup.classList.remove('visible');
    currentTrigger = null;
  }

  function scheduleHide() {
    hideTimer = setTimeout(hidePopup, 120);
  }

  // Popup itself keeps tooltip alive when cursor moves from icon to popup
  popup.addEventListener('mouseenter', () => clearTimeout(hideTimer));
  popup.addEventListener('mouseleave', scheduleHide);

  // Hide on scroll / resize
  window.addEventListener('scroll', hidePopup, { passive: true });
  window.addEventListener('resize', hidePopup, { passive: true });

  document.querySelectorAll('.tooltip-help').forEach(el => {
    if (!el.dataset.tooltip) return;

    el.addEventListener('mouseenter', () => showPopup(el));
    el.addEventListener('mouseleave', scheduleHide);
    el.addEventListener('focus',      () => showPopup(el));
    el.addEventListener('blur',       scheduleHide);

    el.addEventListener('click', ev => {
      ev.preventDefault();
      const hp = el.dataset.helpPage;
      if (hp) {
        hidePopup();
        openHelpModal(hp);
      } else {
        // No modal: toggle tooltip on click (touch devices)
        popup.classList.contains('visible') && currentTrigger === el
          ? hidePopup()
          : showPopup(el);
      }
    });
  });
}

/**
 * Initialize expandable cards on dashboard.
 * Cards with .expandable class can be clicked to show details.
 */
function initExpandableCards() {
  document.querySelectorAll('.db-card.expandable').forEach(card => {
    const header = card.querySelector('.db-card-header');
    if (!header) return;

    const content = card.querySelector('.db-card-expanded');
    if (!content) return;

    // Create toggle button
    const toggle = document.createElement('button');
    toggle.className = 'expand-toggle';
    toggle.innerHTML = '▼';
    toggle.setAttribute('aria-expanded', 'false');
    toggle.onclick = (e) => {
      e.stopPropagation();
      const isExpanded = card.classList.toggle('expanded');
      toggle.setAttribute('aria-expanded', isExpanded);
      toggle.style.transform = isExpanded ? 'rotate(180deg)' : 'rotate(0deg)';
    };

    header.appendChild(toggle);
    header.style.cursor = 'pointer';
    header.style.userSelect = 'none';

    // Click header to toggle
    header.onclick = () => toggle.click();
  });
}

/**
 * Open help modal with detailed information about a metric or feature.
 */
function openHelpModal(pageId) {
  // Build help content based on page ID
  const helpContent = getHelpContent(pageId);

  // Create modal
  const modal = document.createElement('div');
  modal.className = 'help-modal';
  modal.innerHTML = `
    <div class="help-modal-overlay"></div>
    <div class="help-modal-content">
      <div class="help-modal-header">
        <h2>${helpContent.title}</h2>
        <button class="help-modal-close" onclick="this.closest('.help-modal').remove()">&times;</button>
      </div>
      <div class="help-modal-body">
        ${helpContent.html}
      </div>
    </div>
  `;

  document.body.appendChild(modal);

  // Close on overlay click
  modal.querySelector('.help-modal-overlay').onclick = () => modal.remove();
}

/**
 * Get help content for a specific metric.
 */
function getHelpContent(pageId) {
  // Claude API pricing (as of 2024)
  const pricing = {
    haiku: { input: 0.000080, output: 0.0004 },
    sonnet: { input: 0.000300, output: 0.0015 },
    opus: { input: 0.0015, output: 0.0075 }
  };

  const content = {
    'tokens-saved': {
      title: 'Tokens Saved',
      html: `
        <p><strong>How we calculate token savings:</strong></p>
        <ul>
          <li><strong>Tier selection:</strong> Routing simple tasks to Haiku instead of Opus saves ~70% cost per prompt</li>
          <li><strong>Enrichment:</strong> Context reuse from prior sessions avoids redundant work, reducing total prompt tokens</li>
          <li><strong>Plan mode optimization:</strong> Avoiding unnecessary architectural exploration saves rework tokens</li>
        </ul>
        <p><strong>Example:</strong> A 1000-token prompt routed to Haiku costs ~$0.08. The same prompt on Opus costs ~$1.50. That's a $1.42 savings per single prompt.</p>
        <p style="color: var(--color-muted); font-size: 11px;">This is an estimate based on Anthropic Claude API token counts and current pricing.</p>
      `
    },
    'routing-distributions': {
      title: 'Model Routing',
      html: `
        <p><strong>How we choose which model to use:</strong></p>
        <ul>
          <li><strong>🟢 Haiku:</strong> Routine tasks, simple code changes, documentation reading (fast, $0.08/M input tokens)</li>
          <li><strong>🔵 Sonnet:</strong> Feature work, debugging, analysis, moderate complexity ($0.30/M input tokens)</li>
          <li><strong>🔴 Opus:</strong> Complex architecture, planning mode, novel research ($1.50/M input tokens)</li>
        </ul>
        <p><strong>The goal:</strong> Match task complexity to model capability. Using Opus for every task is like hiring a senior architect to fix typos.</p>
        <p style="color: var(--color-muted); font-size: 11px;">Better routing means faster responses AND lower costs.</p>
      `
    },
    'verdict-cache': {
      title: 'Verdict Cache',
      html: `
        <p><strong>What is verdict caching?</strong></p>
        <p>Tool approval/denial verdicts are cached. When you run the same command again, we serve the cached verdict instead of asking the LLM, saving both time and cost.</p>
        <ul>
          <li><strong>Total verdicts:</strong> Unique command approval decisions cached</li>
          <li><strong>Cache hits:</strong> Times we reused a cached decision (cost avoided)</li>
          <li><strong>LLM samples:</strong> New verdicts that required LLM evaluation</li>
        </ul>
        <p><strong>Cost impact:</strong> Each cache hit saves ~$0.0003-0.0010 in LLM costs (depending on verdict length).</p>
        <p style="color: var(--color-muted); font-size: 11px;">Higher cache hit rate = better efficiency and faster decisions.</p>
      `
    },
    'net-tokens': {
      title: 'Net Tokens Saved',
      html: `
        <p><strong>The full ROI calculation:</strong></p>
        <p>Skill Hub saves tokens through smart routing and caching, but it also costs tokens to run the system:</p>
        <ul>
          <li><strong>Tokens saved:</strong> From routing optimization and enrichment</li>
          <li><strong>System overhead:</strong> Tokens spent running Skill Hub itself (routing decisions, enrichment, caching)</li>
          <li><strong>Net:</strong> The difference (positive = you're making money)</li>
        </ul>
        <p><strong>Claude API pricing reference:</strong></p>
        <ul>
          <li>Haiku: $0.08/M input, $0.40/M output tokens</li>
          <li>Sonnet: $0.30/M input, $1.50/M output tokens</li>
          <li>Opus: $1.50/M input, $7.50/M output tokens</li>
        </ul>
        <p style="color: var(--color-good); font-weight: bold;">If this number is positive, Skill Hub is paying for itself and saving you money.</p>
      `
    },
    'enrichment-rate': {
      title: 'Enrichment Rate',
      html: `
        <p><strong>What is enrichment?</strong></p>
        <p>When you submit a vague prompt like "fix the bug," Skill Hub automatically adds context from prior sessions to create a richer prompt:</p>
        <p style="background: rgba(58,160,255,0.08); padding: 8px; border-radius: 3px; font-size: 12px; margin: 10px 0;">
          <strong>Before:</strong> "fix the bug"<br>
          <strong>After:</strong> "fix the deadlock bug in /src/utils/cache.ts (line 47, previously identified in task #14, reproduction: run tests with --race)"
        </p>
        <p><strong>Benefits:</strong></p>
        <ul>
          <li>Better routing decisions (system knows if it's complex)</li>
          <li>Fewer wrong answers (model has full context)</li>
          <li>Less back-and-forth clarification (saves tokens)</li>
          <li>Faster resolution (fewer retries needed)</li>
        </ul>
        <p>Higher enrichment rate = higher quality responses and lower overall costs.</p>
      `
    }
  };

  const extra = {
    'avg-latency': {
      title: 'Routing Latency',
      html: `
        <p><strong>What is measured:</strong> Time from when a prompt is submitted to when the router returns a decision (model selection + context enrichment).</p>
        <p><strong>Why it matters:</strong> Skill Hub runs in the critical path of every LLM call. Low latency means negligible overhead for the user.</p>
        <ul>
          <li><strong>&lt;50 ms:</strong> Excellent — SQLite vector search only</li>
          <li><strong>50–200 ms:</strong> Normal — includes enrichment or LLM verdict</li>
          <li><strong>&gt;500 ms:</strong> Investigate — may indicate local LLM bottleneck</li>
        </ul>
        <p>Vector search (sqlite-vec) is typically 3–8 ms. LLM verdict adds ~100–300 ms on first call; cached = instant.</p>
      `
    },
    'plan-mode': {
      title: 'Plan Mode',
      html: `
        <p><strong>What is plan mode:</strong> When Skill Hub detects an ambiguous or architecturally complex prompt, it fires <code>EnterPlanMode</code> so the model brainstorms a solution space before writing any code.</p>
        <p><strong>Why it saves tokens:</strong> Jumping straight to implementation on a complex task often produces code that needs to be rewritten. Plan mode spends ~500 tokens upfront to avoid wasting 5,000+ on the wrong approach.</p>
        <p><strong>Activation signal:</strong> Router confidence &lt; 0.65 on a Sonnet/Opus prompt, or keywords like "design", "architecture", "refactor entire".</p>
      `
    },
    'forced-switches': {
      title: 'Forced Model Switches',
      html: `
        <p><strong>What this means:</strong> Times the router's model choice was overridden — either by the user manually or by a confidence threshold rule (conf ≥ 0.9 in the opposite direction).</p>
        <p><strong>High count is a warning sign:</strong> Many forced switches means the router is miscalibrated for your workflow. You can correct it via:</p>
        <ul>
          <li>Teaching rules: <code>teach("when I ask X, use Opus")</code></li>
          <li>Adjusting tier thresholds in Settings → Router</li>
          <li>Providing feedback with <code>record_feedback()</code></li>
        </ul>
      `
    },
    'tasks-closed': {
      title: 'Tasks Closed',
      html: `
        <p><strong>Tasks</strong> are threads of work created with <code>save_task()</code> and closed with <code>close_task()</code>. They persist context across sessions.</p>
        <p><strong>Closure rate</strong> shows how productive your work tracking is. A healthy ratio is 60–80% closed tasks. Very low closure can mean tasks are abandoned; very high can mean they're closed too early.</p>
        <p>Closed tasks remain searchable. You can reopen them at any time with <code>reopen_task(id)</code>.</p>
      `
    },
    'avg-task-duration': {
      title: 'Average Task Duration',
      html: `
        <p><strong>Duration</strong> is measured from task creation to close (wall-clock time, not active working time).</p>
        <p>Short durations (&lt;1h) indicate quick, well-scoped tasks. Long durations may indicate complex multi-session work or abandoned tasks.</p>
        <p>The router correlates task duration with token usage to identify which types of work are most expensive.</p>
      `
    },
    'skills-indexed': {
      title: 'Skills Indexed',
      html: `
        <p><strong>Skills</strong> are markdown files that instruct the model how to handle specific types of work. They're discovered from enabled plugins and indexed into a vector database for semantic search.</p>
        <p><strong>How routing uses them:</strong> When you submit a prompt, the router runs a semantic search across all indexed skills and selects the most relevant one to inject as context — guiding the model without consuming your context window.</p>
        <p>More skills = better coverage. Use <code>index_skills()</code> after adding a new plugin.</p>
      `
    },
    'teachings': {
      title: 'Teachings',
      html: `
        <p><strong>Teachings</strong> are explicit rules you've created with <code>teach()</code>. They are the highest-priority routing signal — they override vector search and model defaults.</p>
        <p><strong>Examples:</strong></p>
        <ul>
          <li><code>teach("when I give a URL, suggest chrome-devtools")</code></li>
          <li><code>teach("for Python code always use Sonnet, not Haiku")</code></li>
          <li><code>teach("security reviews always use Opus")</code></li>
        </ul>
        <p>Teachings persist across all sessions and are stored in the local database. View and manage them on the <a href="/teachings" style="color:var(--color-accent)">Teachings page</a>.</p>
      `
    },
    'auto-approve-hook': {
      title: 'Auto-Approve Hook',
      html: `
        <p><strong>What is the auto-approve hook:</strong> A Claude Code hook that intercepts tool calls (Bash, Edit, Read, Write…) and decides whether to approve or deny them without asking you.</p>
        <p><strong>How it works:</strong></p>
        <ol>
          <li>Tool call arrives at the hook</li>
          <li>Router checks the verdict cache (fast path)</li>
          <li>If not cached, runs LLM inference to classify: allow / deny / pass-through</li>
          <li>Verdict is cached for future identical calls</li>
        </ol>
        <p><strong>Pass-through</strong> means the decision is forwarded to you to approve interactively. <strong>Auto-proceed</strong> fires when Skill Hub detects the model is waiting for your input and can safely continue.</p>
      `
    },
    'live-queues': {
      title: 'Live Hook Queues',
      html: `
        <p><strong>Real-time state of the hook pipeline:</strong></p>
        <ul>
          <li><strong>Intents pending:</strong> Tool calls waiting for a verdict (should normally be 0 or 1)</li>
          <li><strong>Questions open:</strong> User prompts awaiting a routing decision or response</li>
          <li><strong>Intercept errors:</strong> Failures in the hook interception logic — these cause tool calls to be passed through unprocessed</li>
        </ul>
        <p>Non-zero intercept errors indicate a bug in the hook or a model mismatch. Check <a href="/logs" style="color:var(--color-accent)">Logs</a> for details.</p>
      `
    },
    'top-intercept-types': {
      title: 'Top Intercept Types',
      html: `
        <p><strong>What is intercepted:</strong> Every Claude Code tool call passes through the hook. This card shows which tool types appear most frequently in your workflow.</p>
        <p>High <code>Bash</code> volume = lots of shell execution (scripting, builds). High <code>Edit</code> = code editing sessions. High <code>Read</code> = exploration/analysis work.</p>
        <p>Token counts show how many tokens were consumed by the <em>content</em> of those calls (file content, command output) — useful for identifying expensive tools.</p>
      `
    },
    'feedback-helpful': {
      title: 'Feedback: Helpful Rate',
      html: `
        <p><strong>What is measured:</strong> When you call <code>record_feedback(skill_id, helpful=True/False)</code> after a skill is invoked, that vote is stored here.</p>
        <p><strong>How it drives learning:</strong></p>
        <ul>
          <li>Helpful votes increase the skill's bandit score → it gets surfaced more often</li>
          <li>Unhelpful votes decrease the score → skill appears less often for similar prompts</li>
          <li>After ~10 votes per skill, the router shifts from exploration to exploitation</li>
        </ul>
        <p>A rate below 40% with &gt;20 total votes suggests the skill-matching quality is low — consider reindexing with <code>index_skills()</code>.</p>
      `
    },
    'tasks-open': {
      title: 'Tasks Open',
      html: `
        <p><strong>Open tasks</strong> are threads of work you've saved with <code>save_task()</code> that haven't been closed yet.</p>
        <p>Each task stores a title, description, tags, and a log of relevant events. Tasks survive session restarts and appear in the sidebar of the Tasks page for quick context recovery.</p>
        <p>Use <code>update_task(id, ...)</code> to add notes as work progresses, and <code>close_task(id)</code> when done.</p>
      `
    }
  };

  const settings = {
    'settings-auto_approve': {
      title: 'Auto-approve',
      html: `
        <p>Controls which tool calls (Bash, Edit, Write…) the hook approves automatically without asking you.</p>
        <p><strong>Key fields:</strong></p>
        <ul>
          <li><code>auto_approve_enabled</code> — master on/off switch</li>
          <li><code>auto_approve_verdict_ttl</code> — how long a cached verdict is valid</li>
          <li><code>auto_approve_max_llm_samples</code> — cap LLM calls before falling back to ask</li>
        </ul>
        <p>Verdicts are cached in the local SQLite DB. The first time you approve a pattern, it trains the cache. Subsequent identical patterns are approved in &lt;1 ms.</p>
      `
    },
    'settings-auto_proceed': {
      title: 'Auto-proceed',
      html: `
        <p>Multi-signal auto-proceed fires after the Stop hook to continue stalled sessions without manual intervention.</p>
        <p><strong>Signals checked (in order):</strong></p>
        <ol>
          <li>Pending clarifying questions — if one is answered, resume</li>
          <li>Timer expiry — if set, fires after N seconds of silence</li>
          <li>Intent queue drain — if intents are all resolved</li>
        </ol>
        <p>Set <code>auto_proceed_enabled = false</code> to fully manual mode.</p>
      `
    },
    'settings-hook': {
      title: 'Hook behavior',
      html: `
        <p>Core hook toggles and safety guards.</p>
        <ul>
          <li><code>hook_enabled</code> — if false, all prompts pass through unprocessed</li>
          <li><code>hook_timeout_seconds</code> — how long the hook waits for LLM verdict before giving up</li>
          <li><code>hook_semantic_threshold</code> — min similarity to even attempt LLM classify (saves latency for unrelated content)</li>
          <li><code>always_forward_to_claude</code> — safety override: never block, always forward</li>
        </ul>
        <p><strong>Tip:</strong> If the hook feels slow, increase <code>hook_semantic_threshold</code> to 0.35–0.45. Prompts below this skip LLM entirely.</p>
      `
    },
    'settings-hook_context': {
      title: 'Context injection',
      html: `
        <p>RAG context injection enriches every prompt with relevant skills and task context before it reaches Claude.</p>
        <ul>
          <li><code>hook_context_injection</code> — master toggle</li>
          <li><code>hook_context_max_chars</code> — total budget (~10K chars ≈ 2.5K tokens)</li>
          <li><code>hook_context_top_k_skills</code> — max skills injected per message</li>
          <li><code>hook_precompact_threshold</code> — long contexts get LLM-compacted first</li>
        </ul>
        <p>Context injection is the <strong>primary source of token savings</strong> — it gives Claude the right context upfront, avoiding follow-up questions.</p>
      `
    },
    'settings-hook_llm': {
      title: 'LLM triage',
      html: `
        <p>Optional local LLM pre-triage classifies prompts <em>before</em> they reach Claude. Disabled by default because small local models are unreliable for this task.</p>
        <p>When enabled, the local model runs a lightweight intent classification. If it's confident enough (above <code>hook_llm_triage_min_confidence</code>), the result gates routing.</p>
        <p><strong>Recommended for:</strong> Users running Ollama with Llama 3.1 8B+ or equivalent. Not recommended with models below 3B parameters.</p>
      `
    },
    'settings-router': {
      title: 'Router core',
      html: `
        <p>The three-tier router selects which LLM handles each prompt:</p>
        <ol>
          <li><strong>Tier 1:</strong> Heuristics (regex, keyword) — instant</li>
          <li><strong>Tier 2:</strong> Local Ollama model — ~50–200 ms</li>
          <li><strong>Tier 3:</strong> Haiku 4.5 API — ~300–600 ms, highest accuracy</li>
        </ol>
        <p>Each tier escalates to the next if confidence is below the threshold. Most prompts resolve at Tier 1 or 2.</p>
        <p><code>router_enrich_thin_prompts</code> — prepends task context to short prompts (&lt;60 chars) so they route correctly.</p>
      `
    },
    'settings-router_bandit': {
      title: 'Router bandit',
      html: `
        <p>ε-greedy bandit continuously optimizes the tier selection over cheap / mid / smart model groups.</p>
        <ul>
          <li><strong>ε (epsilon):</strong> Exploration rate. 0.1 = 10% random exploration, 90% exploit current best</li>
          <li><strong>Reward signal:</strong> Explicit feedback (<code>record_feedback</code>) + implicit session signals</li>
          <li><strong>Warmup:</strong> Exploration-heavy until each arm has 10+ samples</li>
        </ul>
        <p>Disable if you want deterministic routing based solely on confidence thresholds.</p>
      `
    },
    'settings-improve_prompt': {
      title: 'Prompt rewriters',
      html: `
        <p>Prompt rewriters run before the router to enrich prompts with context. The chain is configurable.</p>
        <p><strong>Built-in rewriters:</strong></p>
        <ul>
          <li><code>add_skill_context</code> — appends the top-k matched skill descriptions</li>
          <li><code>add_recent_tasks</code> — appends summaries of the N most recent tasks</li>
          <li><code>add_session_memory</code> — appends the stored session memory summary</li>
        </ul>
        <p>Enriched prompts route more accurately and get better responses without additional back-and-forth.</p>
      `
    },
    'settings-llm': {
      title: 'LLM providers',
      html: `
        <p>Maps tier names to actual model endpoints using litellm syntax.</p>
        <p><strong>Tier mapping:</strong></p>
        <ul>
          <li><code>tier_cheap</code> → e.g. <code>ollama/phi3:mini</code> — quick single-turn tasks</li>
          <li><code>tier_mid</code>   → e.g. <code>ollama/llama3.1:8b</code> — medium complexity</li>
          <li><code>tier_smart</code> → e.g. <code>ollama/llama3.1:70b</code> or <code>anthropic/claude-haiku-4-5</code></li>
          <li><code>tier_embed</code> → embedding model for vector search</li>
        </ul>
        <p>Use <code>ollama/&lt;model&gt;</code> for local, <code>anthropic/&lt;model&gt;</code> for cloud, <code>openai/&lt;model&gt;</code> for OpenAI.</p>
      `
    },
    'settings-vec': {
      title: 'Vector & embeddings',
      html: `
        <p>Controls the vector search engine used for skill/task/teaching retrieval.</p>
        <ul>
          <li><code>vec_engine = "sqlite-vec"</code> — ANN search with binary quantization, ~7× faster than brute-force</li>
          <li><code>binary_quant_enabled</code> — enables 32-bit → 1-bit quantization (32× compression, ~97% recall@5)</li>
          <li><code>rerank_top_k</code> — how many binary candidates to rerank with float32 (default 60)</li>
        </ul>
        <p>Binary quantization is the key performance optimization. Disable only if you see recall degradation.</p>
      `
    },
    'settings-session_memory': {
      title: 'Session memory',
      html: `
        <p>Per-session transcript compaction: every N messages, the full transcript is summarized into a 6-section memory object and stored in SQLite.</p>
        <p>On session resume, the stored memory is injected as a system message so Claude has immediate context without replaying the full conversation.</p>
        <p><strong>Tune for your hardware:</strong></p>
        <ul>
          <li><code>session_memory_tier</code> — use <code>tier_cheap</code> for local Ollama, <code>tier_smart</code> for Claude API</li>
          <li><code>session_memory_min_messages</code> — don't summarize very short sessions</li>
          <li><code>session_memory_inject_max_chars</code> — cap injected memory size</li>
        </ul>
      `
    },
    'settings-skill_evolution': {
      title: 'Skill evolution',
      html: `
        <p>Shadow-learning: Skill Hub observes how Claude solves tasks, then proposes improvements to your local skills.</p>
        <ul>
          <li><code>skill_evolution_enabled</code> — master toggle (disabled by default)</li>
          <li><code>skill_evolution_auto</code> — auto-apply to all skills (only shadow-flagged skills if false)</li>
          <li><code>skill_evolution_cross_pollinate</code> — reference official Claude skill definitions during evolution</li>
        </ul>
        <p>Skills with <code>shadow: true</code> in their frontmatter are candidates for auto-evolution. Start with a single trusted skill to test the feature.</p>
      `
    },
    'settings-services': {
      title: 'Services & monitor',
      html: `
        <p>Background service configuration and resource gating.</p>
        <p><strong>Services managed:</strong> Ollama health check, SearXNG, file watcher (index on change), Haiku router.</p>
        <p><strong>Resource monitor:</strong> When RAM/CPU exceeds thresholds for <code>resource_gating_sustain</code> seconds, expensive operations (LLM calls, vector rebuild) are deferred.</p>
        <p>Useful on laptops or constrained environments to prevent skill-hub from competing with Claude itself.</p>
      `
    },
  };

  return content[pageId] || extra[pageId] || settings[pageId] || {
    title: 'Help',
    html: '<p>No detailed help available for this topic yet.</p>'
  };
}
