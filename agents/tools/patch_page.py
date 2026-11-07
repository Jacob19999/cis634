"""Patch the Precedent Graph page for corpus v2 (ASRS + NTSB). Each edit must match exactly."""
import sys
from pathlib import Path

p = Path(sys.argv[1])
s = p.read_text(encoding="utf-8")

EDITS = [
    # title and brand
    ("<title>ASRS Precedent Graph</title>", "<title>ASRS &amp; NTSB Precedent Graph</title>", 1),
    ("<h1>ASRS Precedent Graph</h1>", "<h1>ASRS &amp; NTSB Precedent Graph</h1>", 1),
    ("`${N.toLocaleString()} NASA ASRS reports · all flight phases`",
     "META.sources ? `${N.toLocaleString()} reports · ${META.sources.map(x => `${x.label} ${x.n.toLocaleString()}`).join(\" · \")} · 2010–2026` : `${N.toLocaleString()} NASA ASRS reports · all flight phases`", 1),
    # CSS: NTSB badge in report rows
    (".row .acn { font: 500 var(--fs-xs)/1.5 var(--font-mono); color: var(--ink-2); }",
     ".row .acn { font: 500 var(--fs-xs)/1.5 var(--font-mono); color: var(--ink-2); }\n"
     ".row .acn.ntsb::before { content: \"NTSB \"; color: var(--muted); }", 1),
    # data: reports in shards
    ("""    D = await res.json();
""", """    D = await res.json();
    // Corpus v2 keeps the reports in shards, each well under GitHub's 100 MB file limit.
    if (!D.reports) {
      const parts = await Promise.all([...Array(D.meta.n_report_shards).keys()].map(k =>
        fetch(`data/reports/${k}.json`).then(r => { if (!r.ok) throw new Error(r.status); return r.json(); })));
      D.reports = parts.flat();
    }
""", 1),
    ("""    r.i = i;
    r.fs = new Set(r.f);""", """    r.i = i;
    r.src = r.src || 0;
    r.fs = new Set(r.f);""", 1),
    ("r.hay = ` ${r.acn} ${r.aircraft}", "r.hay = ` ${r.acn} ${r.src === 1 ? \"ntsb\" : \"asrs\"} ${r.aircraft}", 1),
    # helpers
    ("""  const clip = (s, n) => s.length > n ? s.slice(0, n - 1) + "…" : s;
""", """  const clip = (s, n) => s.length > n ? s.slice(0, n - 1) + "…" : s;
  // Sources: 0 = ASRS with analyst codes, 1 = NTSB, 2 = ASRS narrative only (date estimated).
  const SOURCES = META.sources || [{ id: 0, label: "ASRS" }];
  const idLabel = r => r.src === 1 ? `NTSB ${r.acn}` : `ACN ${r.acn}`;
  const dateLabel = r => (r.de ? "≈" : "") + r.date;
  const SYN_LABEL = ["ASRS analyst synopsis", "NTSB probable cause", "Summary · Qwen3.5-27B (local LLM)"];
  const NARR_LABEL = ["Read the reporter's narrative", "Read the NTSB factual narrative", "Read the reporter's narrative"];
""", 1),
    # filters: source
    ('const FILTER_KINDS = new Set(["date", "tod", "state", "apt"]);',
     'const FILTER_KINDS = new Set(["src", "date", "tod", "state", "apt"]);', 1),
    ("""        : c.kind === "tod" ? r.t != null && c.ids.includes(r.t)
""", """        : c.kind === "tod" ? r.t != null && c.ids.includes(r.t)
        : c.kind === "src" ? c.ids.includes(r.src)
""", 1),
    ("""    if (c.kind === "date") return c.from === c.to""",
     """    if (c.kind === "src") return c.ids.slice().sort().map(k => SOURCES[k].label).join(" + ");
    if (c.kind === "date") return c.from === c.to""", 1),
    ("""          <div class="frow"><span id="f-date-l">Date</span>""",
     """          <div class="frow"><span id="f-src-l">Source</span><div class="tods" id="f-src" role="group" aria-labelledby="f-src-l"></div></div>
          <div class="frow"><span id="f-date-l">Date</span>""", 1),
    ('const fFrom = $("#f-from"), fTo = $("#f-to"),', 'const fSrc = $("#f-src"), fFrom = $("#f-from"), fTo = $("#f-to"),', 1),
    ("""    fFrom.innerHTML = yrs; fTo.innerHTML = yrs;
""", """    fFrom.innerHTML = yrs; fTo.innerHTML = yrs;
    fSrc.innerHTML = SOURCES.map(x => `<button type="button" data-src="${x.id}" aria-pressed="false" title="${esc(x.label)}${x.n ? ` · ${x.n.toLocaleString()} reports` : ""}">${esc(x.label.replace(" (narrative only)", "*"))}</button>`).join("");
    fSrc.onclick = e => {
      const b = e.target.closest("button[data-src]");
      if (!b) return;
      const cur = state.chips.find(x => x.kind === "src")?.ids || [];
      const k = +b.dataset.src;
      const ids = cur.includes(k) ? cur.filter(x => x !== k) : [...cur, k];
      setFilter("src", ids.length && ids.length < SOURCES.length ? { kind: "src", ids } : null);
    };
""", 1),
    ("""    const d = get("date"), t = get("tod"), s = get("state"), a = get("apt");
""", """    const d = get("date"), t = get("tod"), s = get("state"), a = get("apt"), so = get("src");
    fSrc.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", String(!!so && so.ids.includes(+b.dataset.src))));
""", 1),
    ("""    $("#f-clear").hidden = !(d || t || s || a);""", """    $("#f-clear").hidden = !(d || t || s || a || so);""", 1),
    ("Local time of day. ASRS withholds the airport in most reports, so the airport filter covers only those that name one.",
     "Local time of day. ASRS withholds the airport in most reports, so the airport filter covers only those that name one. * ASRS reports 2021–2025 without analyst codes; their month is estimated from the ACN.", 2),
    # search by NTSB case number as well as ACN
    ("if (/^\\d{3,}$/.test(q)) { for (const r of R) if (r.acn.startsWith(q)) out.push(r.i); }",
     "if (/^\\d{3,}$/.test(q) || /^[a-z]{3}\\d{2}[a-z]{0,2}\\d*$/i.test(q)) { const Q = q.toUpperCase(); for (const r of R) if (r.acn.toUpperCase().startsWith(Q)) out.push(r.i); }", 1),
    # ids and dates in labels
    ("<div class=\"th\">ACN ${esc(r.acn)} · ${esc(r.date)}</div>", "<div class=\"th\">${esc(idLabel(r))} · ${esc(dateLabel(r))}</div>", 1),
    ("<span class=\"s mono\">ACN ${esc(x.acn)} · ${esc(R[i].date)} · ${esc(R[i].aircraft)}</span>",
     "<span class=\"s mono\">${esc(idLabel(R[i]))} · ${esc(dateLabel(R[i]))} · ${esc(R[i].aircraft)}</span>", 1),
    ("<span class=\"t mono\">${esc(R[i].acn)} · ${esc(R[i].date)} · ${esc(R[i].aircraft)}</span>",
     "<span class=\"t mono\">${esc(idLabel(R[i]))} · ${esc(dateLabel(R[i]))} · ${esc(R[i].aircraft)}</span>", 1),
    ("Why it links to ACN ${esc(s.acn)}", "Why it links to ${esc(idLabel(s))}", 1),
    ("data-i=\"${j}\">ACN ${esc(R[j].acn)}</button>", "data-i=\"${j}\">${esc(idLabel(R[j]))}</button>", 1),
    ("<h2 class=\"mono\">ACN ${esc(r.acn)}</h2><div class=\"meta\">${esc(r.date)} · ${esc(r.aircraft)}</div>",
     "<h2 class=\"mono\">${esc(idLabel(r))}</h2><div class=\"meta\">${esc(dateLabel(r))}${r.de ? \" (estimated from the ACN)\" : \"\"}${r.aircraft ? ` · ${esc(r.aircraft)}` : \"\"}</div>", 1),
    ("<span class=\"acn\">${esc(R[i].acn)}</span><span class=\"rt\">${esc(R[i].synopsis)}</span><span class=\"rm\">${esc(R[i].date)} · ${esc(R[i].aircraft)}</span>",
     "<span class=\"acn${R[i].src === 1 ? \" ntsb\" : \"\"}\">${esc(R[i].acn)}</span><span class=\"rt\">${esc(R[i].synopsis)}</span><span class=\"rm\">${esc(dateLabel(R[i]))} · ${esc(R[i].aircraft)}</span>", 1),
    ("<span class=\"acn\">${esc(R[c.j].acn)}</span><span class=\"rt\">${esc(reasonText(c))}</span><span class=\"rm\">${esc(R[c.j].date)}",
     "<span class=\"acn${R[c.j].src === 1 ? \" ntsb\" : \"\"}\">${esc(R[c.j].acn)}</span><span class=\"rt\">${esc(reasonText(c))}</span><span class=\"rm\">${esc(dateLabel(R[c.j]))}", 1),
    # inspector wording per source
    ("<p class=\"label\">ASRS analyst synopsis</p>", "<p class=\"label\">${SYN_LABEL[r.src]}</p>", 1),
    ("<summary>Read the reporter's narrative</summary>", "<summary>${NARR_LABEL[r.src]}</summary>", 1),
    ("""x.src === "llm" ? " · coded by the local LLM" : " · ASRS analyst code\"""",
     """x.src === "llm" ? " · coded by the local LLM" : x.src === "ntsb" ? " · NTSB coded field" : " · ASRS analyst code\"""", 1),
    # narrative shards: {n: narrative, s: summary}
    ("""      el.textContent = (await shards.get(k))[acn] || "No narrative on file.";""",
     """      const v = (await shards.get(k))[acn], r = R[byAcn.get(acn)];
      if (v && typeof v === "object") {
        el.innerHTML = (v.s && r.src !== 2 ? `<p class="hint"><b>Summary · Qwen3.5-27B:</b> ${esc(v.s)}</p>` : "")
          + `<p>${esc(v.n || "No narrative on file.")}</p>`;
      } else el.textContent = v || "No narrative on file.";""", 1),
    # colour by source; year bands for 2010–2026
    ("""    none: { label: "Nothing (grey)" },""", """    source: { label: "Source", other: "",
      groups: SOURCES.map((x, k) => ({ label: x.label, c: SLOT[k] || KEY_OTHER })), of: r => r.src || 0 },
    none: { label: "Nothing (grey)" },""", 1),
    ("""      groups: [["2012–13", "#184f95"], ["2014–15", "#256abf"], ["2016–17", "#3987e5"], ["2018–19", "#6da7ec"], ["2020–22", "#b7d3f6"]].map(([label, c]) => ({ label, c })),
      of: r => r.yr <= 2013 ? 0 : r.yr <= 2015 ? 1 : r.yr <= 2017 ? 2 : r.yr <= 2019 ? 3 : 4 },""",
     """      groups: [["2010–12", "#184f95"], ["2013–15", "#256abf"], ["2016–18", "#3987e5"], ["2019–21", "#6da7ec"], ["2022–26", "#b7d3f6"]].map(([label, c]) => ({ label, c })),
      of: r => r.yr <= 2012 ? 0 : r.yr <= 2015 ? 1 : r.yr <= 2018 ? 2 : r.yr <= 2021 ? 3 : 4 },""", 1),
]

bad = []
for old, new, want in EDITS:
    got = s.count(old)
    if got != want:
        bad.append((got, want, old[:90]))
        continue
    s = s.replace(old, new)
if bad:
    for b in bad:
        print("MISMATCH found %d want %d: %r" % b)
    sys.exit(1)
p.write_text(s, encoding="utf-8")
print(f"applied {len(EDITS)} edits")
