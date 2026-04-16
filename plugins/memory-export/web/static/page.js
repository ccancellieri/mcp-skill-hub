/* memory-export plugin — page interactivity (vanilla JS, no framework). */
(function () {
  "use strict";

  const tablesTable   = document.getElementById("me-tables");
  const projectsTable = document.getElementById("me-projects");
  const previewEl     = document.getElementById("me-preview");
  const piiWarnEl     = document.getElementById("me-pii-warn");
  const downloadEl    = document.getElementById("me-download");
  const localSkillsEl = document.getElementById("me-include-local-skills");
  const importForm    = document.getElementById("me-import-form");
  const importReport  = document.getElementById("me-import-report");
  const tablesAll     = document.getElementById("me-tables-all");
  const projectsAll   = document.getElementById("me-projects-all");

  // --------------------------------------------------------------- selection
  function selectedTables() {
    return Array.from(tablesTable.querySelectorAll("tr[data-table]"))
      .filter(r => r.querySelector(".me-tbl-include").checked)
      .map(r => r.dataset.table);
  }
  function selectedProjects() {
    return Array.from(projectsTable.querySelectorAll("tr[data-project]"))
      .filter(r => r.querySelector(".me-proj-include").checked)
      .map(r => r.dataset.project);
  }

  // ------------------------------------------------------------- mode + tier
  function syncModeTier(rowSelector, modeClass, tierClass) {
    document.querySelectorAll(rowSelector).forEach(row => {
      const mode = row.querySelector(modeClass);
      const tier = row.querySelector(tierClass);
      if (!mode || !tier) return;
      mode.addEventListener("change", () => {
        tier.disabled = mode.value !== "llm";
      });
    });
  }
  syncModeTier("tr[data-table]",   ".me-tbl-mode",  ".me-tbl-tier");
  syncModeTier("tr[data-project]", ".me-proj-mode", ".me-proj-tier");

  // ----------------------------------------------------------------- preview
  let previewTimer = null;
  async function refreshPreview() {
    const tables   = selectedTables();
    const projects = selectedProjects();
    if (!tables.length && !projects.length) {
      previewEl.textContent = "(select something to preview)";
      piiWarnEl.hidden = true;
      return;
    }
    const params = new URLSearchParams();
    if (tables.length)   params.set("tables", tables.join(","));
    if (projects.length) params.set("projects", projects.join(","));
    if (localSkillsEl.checked) params.set("include_local_skills", "true");
    try {
      const r = await fetch(`/memory-export/preview?${params.toString()}`);
      const j = await r.json();
      previewEl.textContent = JSON.stringify(j, null, 2);
      if (j.pii_offenders && j.pii_offenders.length) {
        piiWarnEl.textContent =
          `⚠ PII tokens detected in ${j.pii_offenders.length} file(s) — review before exporting.`;
        piiWarnEl.hidden = false;
      } else {
        piiWarnEl.hidden = true;
      }
      updateDownloadLink(params);
    } catch (e) {
      previewEl.textContent = `preview error: ${e.message}`;
    }
  }
  function schedulePreview() {
    clearTimeout(previewTimer);
    previewTimer = setTimeout(refreshPreview, 150);
  }

  function updateDownloadLink(params) {
    downloadEl.href = `/memory-export/snapshot.tar.gz?${params.toString()}`;
  }

  // -------------------------------------------------------- bulk-select all
  function bindBulk(allBox, rowSel, includeSel) {
    if (!allBox) return;
    allBox.addEventListener("change", () => {
      document.querySelectorAll(`${rowSel} ${includeSel}`).forEach(c => {
        c.checked = allBox.checked;
      });
      schedulePreview();
    });
  }
  bindBulk(tablesAll,   "tr[data-table]",   ".me-tbl-include");
  bindBulk(projectsAll, "tr[data-project]", ".me-proj-include");

  // -------------------------------------------------------- change wiring
  document.querySelectorAll(".me-tbl-include, .me-proj-include").forEach(c => {
    c.addEventListener("change", schedulePreview);
  });
  localSkillsEl.addEventListener("change", schedulePreview);

  // ---------------------------------------------------------- build plan
  function buildPlan() {
    const hub_modes = {};
    const llm_per_target = {};
    document.querySelectorAll("tr[data-table]").forEach(r => {
      const t = r.dataset.table;
      const mode = r.querySelector(".me-tbl-mode").value;
      hub_modes[t] = mode;
      if (mode === "llm") llm_per_target[t] = r.querySelector(".me-tbl-tier").value;
    });
    const project_modes = {};
    document.querySelectorAll("tr[data-project]").forEach(r => {
      const k = r.dataset.project;
      const mode = r.querySelector(".me-proj-mode").value;
      const tier = r.querySelector(".me-proj-tier").value;
      project_modes[k] = { mode, llm_tier: tier };
    });
    return { hub_modes, project_modes, llm_per_target,
             default_mode: "skip", default_llm_tier: "tier_cheap",
             max_llm_calls: 200 };
  }

  // ----------------------------------------------------------- import
  importForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const fd = new FormData(importForm);
    fd.set("plan", JSON.stringify(buildPlan()));
    importReport.hidden = false;
    importReport.textContent = "Importing…";
    try {
      const r = await fetch("/memory-export/import", { method: "POST", body: fd });
      const j = await r.json();
      importReport.textContent = JSON.stringify(j, null, 2);
    } catch (err) {
      importReport.textContent = `import error: ${err.message}`;
    }
  });

  // initial preview tick
  schedulePreview();
})();
