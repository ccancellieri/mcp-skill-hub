# Tooltips & Help System — Quick Reference

## 🎯 What's New

Interactive tooltip system with three levels of help:
1. **Hover tooltips** — Brief explanation + "Learn more" button
2. **Help modals** — Detailed information with pricing and examples
3. **Expandable cards** — Click KPI cards to see cost breakdowns

---

## 📍 Where to Find Them

### Dashboard (`/dashboard`)
Hover over **?** icons next to:
- 🟢 **Net tokens saved** ← Click KPI to expand breakdown
- 📊 **Skills indexed**
- 👥 **Teachings**
- ✨ **Feedback helpful**
- 🔒 **Verdict cache** ← Click KPI to expand efficiency metrics
- 📡 **Router** ← Click KPI to expand model distribution

### Report (`/report`)
Hover over **?** icons next to:
- 🟢 **Tokens saved** ← Click KPI to expand cost breakdown
- 📨 **Prompts routed** ← Click KPI to expand routing efficiency
- 📈 **Enrichment rate** ← Click KPI to expand impact metrics
- ... and all other metrics

---

## 🔍 How to Use

### 1. **Hover for Quick Help**
```
Hover over "?" icon
    ↓
Tooltip popup appears above icon
    ↓
Read brief description
```

### 2. **Click "Learn more" for Details**
```
Click "Learn more" in tooltip
    ↓
Help modal opens (dark overlay + centered box)
    ↓
Read full explanation with:
  • Pricing reference (Haiku/Sonnet/Opus costs)
  • Examples and use cases
  • Cost impact calculations
    ↓
Close: Click × button or click outside modal
```

### 3. **Click KPI Cards to Expand**
```
Click on any KPI card (e.g., "Net tokens saved")
    ↓
Card expands with smooth animation
    ↓
See breakdown of where savings come from
    ↓
Click again to collapse
```

---

## 💰 Pricing Reference

Included in all help modals:

| Model | Input Cost | Output Cost | Use Case |
|-------|-----------|-------------|----------|
| 🟢 Haiku | $0.000080 per token | $0.0004 per token | Routine tasks, docs, simple code |
| 🔵 Sonnet | $0.000300 per token | $0.0015 per token | Standard features, debugging |
| 🔴 Opus | $0.0015 per token | $0.0075 per token | Complex architecture, planning |

**Example:** A 1,000-token prompt costs:
- Haiku: ~$0.08 (40 input + 40 output tokens avg)
- Sonnet: ~$0.30
- Opus: ~$1.50

**The benefit:** Smart routing can save 70% on simple tasks.

---

## 📊 Expandable Cards — What You'll See

### Net Tokens Saved
```
├─ Where savings come from
├─ Tier selection (60% of savings)
├─ Enrichment/context reuse (40%)
└─ Total cost impact in USD
```

### Verdict Cache
```
├─ Cache hit rate percentage
├─ Total decisions cached
├─ LLM samples (actual calls made)
└─ Cost savings from reused verdicts
```

### Router Statistics
```
├─ Model distribution (🟢 Haiku %, 🔵 Sonnet %, 🔴 Opus %)
├─ Prompts enriched with context
├─ Plan mode activations
└─ Explanation of each tier
```

### Enrichment Rate
```
├─ Number of prompts auto-filled with context
├─ Quality improvement estimate
├─ Error reduction from enrichment
└─ Cost savings from better routing
```

---

## 🎨 Visual Design

- **"?" icon**: 16px circular button, blue hover state
- **Tooltip popup**: Max 280px width, positioned above icon, with arrow pointer
- **Help modal**: Centered box with dark overlay, smooth transitions
- **Expandable cards**: Slide-down animation, colored borders on hover
- **Theme**: Dark mode (blue accent, green for good, orange for warnings, red for errors)

---

## ⌨️ Keyboard Navigation

- **Tab**: Move focus to "?" icons
- **Enter**: Opens help modal from focused "?" icon
- **Escape**: Closes help modal (or click × button)
- **Click card**: Toggles expand/collapse

All keyboard interactions work without mouse!

---

## 🔧 For Developers

### Adding a New Tooltip

Edit the HTML template (dashboard.html or report.html):

```html
<span class="tooltip-wrapper">
  <span>Your Metric Name</span>
  <a class="tooltip-help" href="#" 
     data-tooltip="Short description"
     data-help-page="my-metric-id"
     tabindex="0">?</a>
</span>
```

### Adding Help Content

Edit `app.js`, in the `getHelpContent()` function:

```javascript
'my-metric-id': {
  title: 'My Metric Title',
  html: `
    <p><strong>What it measures:</strong></p>
    <ul>
      <li>First component</li>
      <li>Second component</li>
    </ul>
    <p>Additional explanation.</p>
    <p style="color: var(--color-muted); font-size: 11px;">Tip: Use Markdown patterns like <code>code</code> for formatting.</p>
  `
}
```

### Making a Card Expandable

Add `onclick="this.classList.toggle('expanded')"` to the card and include an `.expanded-content` div:

```html
<div class="db-card" onclick="this.classList.toggle('expanded')" 
     style="cursor:pointer">
  <!-- Card content -->
  <div style="display:none" class="expanded-content">
    <!-- Expanded details -->
  </div>
</div>
```

---

## 📱 Mobile Support

- Tooltips work on touch (long-press to show, tap outside to hide)
- Help modals are responsive (max-height: 80vh)
- Expandable cards work with touch (single tap to expand)
- All content is readable on mobile devices

---

## 🚀 Future Ideas

- [ ] Export cost analysis as PDF or CSV
- [ ] Show 30-day cost trends (chart)
- [ ] Support for other LLM providers (OpenAI, Mistral)
- [ ] Video tutorials in help modals
- [ ] Glossary of terms
- [ ] Comparison tool (Haiku vs Opus cost per task)

---

## 📝 Files Modified

- `app.css` — Tooltip, modal, expandable card styles
- `app.js` — Tooltip initialization, help modal logic
- `dashboard.html` — Added help-page attributes and expandable content
- `report.html` — Added help-page attributes and expandable content

All changes are backward compatible. Existing functionality is unchanged.

---

## 💡 Tips

1. **For users exploring the system:** Start by hovering over "?" next to any metric
2. **For cost analysis:** Click on KPI cards to see where savings come from
3. **For detailed learning:** Click "Learn more" to read full explanations with pricing
4. **For mobile:** Long-press "?" icons on touch devices to see tooltips
5. **For developers:** See the "For Developers" section above to extend the system

---

## ✅ Testing Checklist

After deployment, verify:
- [ ] Hovering over "?" shows tooltip popup
- [ ] "Learn more" button opens help modal
- [ ] Help modal closes with × button and outside click
- [ ] Clicking KPI cards expands them
- [ ] Expanded content shows with animation
- [ ] Tab navigation focuses "?" icons
- [ ] Help modals are keyboard-closeable (Escape key)
- [ ] All links in help modals work
- [ ] Mobile tooltips work (long-press)
- [ ] Dark mode theme is consistent

---

Generated: 2026-04-15  
System: mcp-skill-hub v0.3.36+  
Author: Claude Code
