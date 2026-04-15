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
    const tooltip = el.dataset.tooltip;
    if (!tooltip) return;

    // Create popup element
    const popup = document.createElement('div');
    popup.className = 'tooltip-popup';

    const textP = document.createElement('p');
    textP.className = 'tooltip-text';
    textP.textContent = tooltip;
    popup.appendChild(textP);

    // Add optional help page link
    const helpPage = el.dataset.helpPage;
    if (helpPage) {
      const linkEl = document.createElement('a');
      linkEl.href = '#';
      linkEl.className = 'tooltip-link';
      linkEl.textContent = 'Learn more';
      linkEl.onclick = (e) => {
        e.preventDefault();
        openHelpModal(helpPage);
      };
      popup.appendChild(linkEl);
    }

    // Find tooltip wrapper and ensure it has position: relative
    const wrapper = el.closest('.tooltip-wrapper');
    if (wrapper) {
      wrapper.style.position = 'relative';
      wrapper.appendChild(popup);
    } else {
      // Fallback: append to parent
      el.parentNode.appendChild(popup);
    }

    // Show/hide on hover and focus
    ['mouseenter', 'focus'].forEach(ev => {
      el.addEventListener(ev, () => {
        popup.classList.add('visible');
      });
    });
    ['mouseleave', 'blur'].forEach(ev => {
      el.addEventListener(ev, () => {
        popup.classList.remove('visible');
      });
    });

    // Prevent default link behavior for the help icon itself
    el.addEventListener('click', (e) => {
      e.preventDefault();
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

  return content[pageId] || {
    title: 'Help',
    html: '<p>No detailed help available for this topic yet.</p>'
  };
}
