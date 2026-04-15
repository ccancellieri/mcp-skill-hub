// skill-hub webapp entry — reserved for Phase 2+ client hooks.
document.addEventListener("DOMContentLoaded", () => {
  initTooltips();
  initExpandableCards();
});

/**
 * Initialize tooltip system.
 *
 * Usage:
 *   <a class="tooltip-help" href="#"
 *      data-tooltip="Description text"
 *      data-tooltip-link="optional-link-url"
 *      data-tooltip-link-text="Learn more"
 *      tabindex="0">?</a>
 */
function initTooltips() {
  document.querySelectorAll('.tooltip-help').forEach(el => {
    const tooltipText = el.dataset.tooltip;
    if (!tooltipText) return;

    const helpPage = el.dataset.helpPage;

    // Create popup element
    const popup = document.createElement('div');
    popup.className = 'tooltip-popup';

    const textP = document.createElement('p');
    textP.className = 'tooltip-text';
    textP.textContent = tooltipText;
    popup.appendChild(textP);

    if (helpPage) {
      const linkEl = document.createElement('a');
      linkEl.href = '#';
      linkEl.className = 'tooltip-link';
      linkEl.textContent = 'Full details →';
      linkEl.onclick = (e) => { e.preventDefault(); openHelpModal(helpPage); };
      popup.appendChild(linkEl);
    }

    // Attach popup to wrapper (the positioned parent)
    const wrapper = el.closest('.tooltip-wrapper') || el.parentNode;
    wrapper.style.position = 'relative';
    wrapper.appendChild(popup);

    // Hover on the WRAPPER keeps popup alive when moving from icon to popup
    let hideTimer = null;

    const show = () => {
      clearTimeout(hideTimer);
      popup.classList.add('visible');
    };
    const scheduleHide = () => {
      hideTimer = setTimeout(() => popup.classList.remove('visible'), 120);
    };

    wrapper.addEventListener('mouseenter', show);
    wrapper.addEventListener('mouseleave', scheduleHide);
    popup.addEventListener('mouseenter', show);   // stay open when cursor moves to popup
    popup.addEventListener('mouseleave', scheduleHide);

    // Click on "?" opens help modal directly (most discoverable UX)
    el.addEventListener('click', (e) => {
      e.preventDefault();
      if (helpPage) {
        openHelpModal(helpPage);
      } else {
        // No help page: toggle tooltip visibility on click for touch devices
        popup.classList.toggle('visible');
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

  return content[pageId] || extra[pageId] || {
    title: 'Help',
    html: '<p>No detailed help available for this topic yet.</p>'
  };
}
