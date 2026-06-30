"""
Singleton / low-CA annotation review app.

Usage:
    python singleton_review_app.py --db <path_to_singletons.db> --images <path_to_images_folder> [--port 5555]
"""

import argparse
import sqlite3
import os
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__)

DB_PATH = None
IMAGES_DIR = None


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@app.route("/")
def index():
    return INDEX_HTML


@app.route("/api/stats")
def stats():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM annotations")
    total = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM annotations WHERE decision IS NOT NULL")
    reviewed = cur.fetchone()[0]
    conn.close()
    return jsonify({"total": total, "reviewed": reviewed})


@app.route("/api/annotations")
def get_annotations():
    """Return all annotations with current decision state."""
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT uuid, filename, decision FROM annotations
        ORDER BY CAST(
            SUBSTR(filename,
                   INSTR(filename, 'CA') + 2,
                   INSTR(SUBSTR(filename, INSTR(filename, 'CA') + 2), '_') - 1
            ) AS REAL
        ) ASC
    """)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return jsonify(rows)


@app.route("/api/update", methods=["POST"])
def update_decision():
    data = request.json
    uuid = data.get("uuid")
    decision = data.get("decision")  # 'not_census' or None

    conn = get_db()
    cur = conn.cursor()
    cur.execute("UPDATE annotations SET decision=? WHERE uuid=?", (decision, uuid))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})


@app.route("/images/<path:filename>")
def serve_image(filename):
    return send_from_directory(IMAGES_DIR, filename)


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Singleton Review</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #1a1a2e; color: #e0e0e0; }
  .header { background: #16213e; padding: 12px 20px; display: flex;
            justify-content: space-between; align-items: center;
            border-bottom: 2px solid #0f3460; position: sticky; top: 0; z-index: 10; }
  .header h1 { font-size: 18px; color: #e94560; }
  .progress { font-size: 14px; color: #a0a0a0; }
  .progress span { color: #e94560; font-weight: bold; }
  .filters { background: #16213e; padding: 10px 20px; display: flex; gap: 12px;
             align-items: center; border-bottom: 1px solid #0f3460; }
  .filters label { font-size: 13px; cursor: pointer; }
  .filters input, .filters select { margin-right: 4px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
          gap: 12px; padding: 16px; }
  .card { background: #16213e; border-radius: 8px; overflow: hidden;
          border: 2px solid transparent; transition: border-color 0.2s; }
  .card.marked { border-color: #e94560; }
  .card img { width: 100%; height: 220px; object-fit: contain; background: #0a0a1a;
              cursor: pointer; }
  .card-info { padding: 8px 12px; font-size: 12px; display: flex;
               justify-content: space-between; align-items: center; }
  .card-info .meta { display: flex; flex-direction: column; gap: 2px; }
  .card-info .tag { display: inline-block; padding: 2px 6px; border-radius: 3px;
                    font-size: 11px; font-weight: bold; }
  .tag.singleton { background: #e94560; color: white; }
  .tag.cluster { background: #0f3460; color: #a0d0ff; }
  .btn-mark { padding: 6px 12px; border-radius: 4px; border: none; cursor: pointer;
              font-size: 12px; font-weight: bold; transition: all 0.2s; }
  .btn-mark.unmarked { background: #2a2a4a; color: #a0a0a0; }
  .btn-mark.unmarked:hover { background: #e94560; color: white; }
  .btn-mark.is-marked { background: #e94560; color: white; }
  .btn-mark.is-marked:hover { background: #2a2a4a; color: #a0a0a0; }
  .overlay { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
             background: rgba(0,0,0,0.9); z-index: 100; justify-content: center;
             align-items: center; cursor: pointer; }
  .overlay.active { display: flex; }
  .overlay img { max-width: 90%; max-height: 90%; object-fit: contain; }
</style>
</head>
<body>

<div class="header">
  <h1>Singleton & Low-CA Review</h1>
  <div class="progress">
    Reviewed: <span id="reviewed">0</span> / <span id="total">0</span>
    &nbsp; | &nbsp; Marked not-census: <span id="marked">0</span>
  </div>
</div>

<div class="filters">
  <label>Show:
    <select id="filterShow">
      <option value="all">All</option>
      <option value="unreviewed">Unreviewed only</option>
      <option value="marked">Marked not-census</option>
    </select>
  </label>
  <label>Type:
    <select id="filterType">
      <option value="all">All</option>
      <option value="singleton">Singletons only</option>
      <option value="cluster">Cluster only</option>
    </select>
  </label>
  <label>Sort:
    <select id="filterSort">
      <option value="ca_asc">CA score (low first)</option>
      <option value="ca_desc">CA score (high first)</option>
      <option value="name">Filename</option>
    </select>
  </label>
</div>

<div class="grid" id="grid"></div>

<div class="overlay" id="overlay" onclick="this.classList.remove('active')">
  <img id="overlayImg" src="">
</div>

<script>
let annotations = [];

async function loadData() {
  const resp = await fetch('/api/annotations');
  annotations = await resp.json();
  render();
}

function parseCard(a) {
  const fn = a.filename;
  const isSingleton = fn.startsWith('singleton_');
  const caMatch = fn.match(/CA([0-9.]+)/);
  const ca = caMatch ? parseFloat(caMatch[1]) : null;
  return { ...a, isSingleton, ca };
}

function render() {
  const showFilter = document.getElementById('filterShow').value;
  const typeFilter = document.getElementById('filterType').value;
  const sortBy = document.getElementById('filterSort').value;

  let items = annotations.map(parseCard);

  // Filter
  if (showFilter === 'unreviewed') items = items.filter(a => !a.decision);
  else if (showFilter === 'marked') items = items.filter(a => a.decision === 'not_census');

  if (typeFilter === 'singleton') items = items.filter(a => a.isSingleton);
  else if (typeFilter === 'cluster') items = items.filter(a => !a.isSingleton);

  // Sort
  if (sortBy === 'ca_asc') items.sort((a, b) => (a.ca ?? 99) - (b.ca ?? 99));
  else if (sortBy === 'ca_desc') items.sort((a, b) => (b.ca ?? -1) - (a.ca ?? -1));
  else items.sort((a, b) => a.filename.localeCompare(b.filename));

  const grid = document.getElementById('grid');
  grid.innerHTML = items.map(a => {
    const isMarked = a.decision === 'not_census';
    const tagClass = a.isSingleton ? 'singleton' : 'cluster';
    const tagText = a.isSingleton ? 'Singleton' : 'Cluster';
    const caStr = a.ca !== null ? 'CA: ' + a.ca.toFixed(3) : '';
    return `
      <div class="card ${isMarked ? 'marked' : ''}" data-uuid="${a.uuid}">
        <img src="/images/${a.filename}" onclick="showOverlay(this.src)" loading="lazy">
        <div class="card-info">
          <div class="meta">
            <span class="tag ${tagClass}">${tagText}</span>
            <span>${caStr}</span>
          </div>
          <button class="btn-mark ${isMarked ? 'is-marked' : 'unmarked'}"
                  onclick="toggleMark('${a.uuid}')">
            ${isMarked ? 'Not Census ✓' : 'Not Census'}
          </button>
        </div>
      </div>`;
  }).join('');

  updateStats();
}

async function toggleMark(uuid) {
  const ann = annotations.find(a => a.uuid === uuid);
  const newDecision = ann.decision === 'not_census' ? null : 'not_census';
  ann.decision = newDecision;

  await fetch('/api/update', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ uuid, decision: newDecision })
  });

  render();
}

function updateStats() {
  const total = annotations.length;
  const reviewed = annotations.filter(a => a.decision !== null).length;
  const marked = annotations.filter(a => a.decision === 'not_census').length;
  document.getElementById('total').textContent = total;
  document.getElementById('reviewed').textContent = reviewed;
  document.getElementById('marked').textContent = marked;
}

function showOverlay(src) {
  document.getElementById('overlayImg').src = src;
  document.getElementById('overlay').classList.add('active');
}

document.getElementById('filterShow').addEventListener('change', render);
document.getElementById('filterType').addEventListener('change', render);
document.getElementById('filterSort').addEventListener('change', render);

loadData();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Singleton annotation review UI")
    parser.add_argument("--db", required=True, help="Path to singletons.db")
    parser.add_argument("--images", required=True, help="Path to folder with cropped images")
    parser.add_argument("--port", type=int, default=5555, help="Port to run on")
    args = parser.parse_args()

    DB_PATH = args.db
    IMAGES_DIR = args.images

    if not os.path.exists(DB_PATH):
        print(f"Error: DB not found at {DB_PATH}")
        exit(1)
    if not os.path.isdir(IMAGES_DIR):
        print(f"Error: Images dir not found at {IMAGES_DIR}")
        exit(1)

    print(f"Starting review app on port {args.port}")
    print(f"  DB: {DB_PATH}")
    print(f"  Images: {IMAGES_DIR}")
    app.run(host="0.0.0.0", port=args.port, debug=False)
