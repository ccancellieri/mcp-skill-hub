// skill-hub webapp entry — reserved for Phase 2+ client hooks.
document.addEventListener("DOMContentLoaded", () => {
  initTooltips();
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

    // Add optional link
    const link = el.dataset.tooltipLink;
    const linkText = el.dataset.tooltipLinkText || 'Learn more';
    if (link) {
      const linkEl = document.createElement('a');
      linkEl.href = link;
      linkEl.className = 'tooltip-link';
      linkEl.textContent = linkText;
      linkEl.target = '_blank';
      linkEl.rel = 'noopener noreferrer';
      popup.appendChild(linkEl);
    }

    // Insert popup after the help icon
    el.parentNode.appendChild(popup);

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
