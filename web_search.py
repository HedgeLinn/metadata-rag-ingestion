"""单题检索测试接口 — Flask 后端 + HTML 前端。"""
from __future__ import annotations

from flask import Flask, request, jsonify, send_from_directory

from config import DEFAULT_KB
from core.registry import factory
from search import search as strategy_search

app = Flask(__name__)

HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RAG 检索测试</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, "Microsoft YaHei", sans-serif; background: #f5f5f5; color: #333; }
  .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
  h1 { font-size: 20px; margin-bottom: 16px; color: #1a1a1a; }
  .panel { background: #fff; border-radius: 8px; padding: 20px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
  .row { display: flex; gap: 12px; align-items: flex-end; flex-wrap: wrap; }
  .field { display: flex; flex-direction: column; gap: 4px; }
  .field label { font-size: 12px; color: #666; font-weight: 600; }
  .field input, .field select { padding: 8px 12px; border: 1px solid #d0d0d0; border-radius: 6px; font-size: 14px; }
  .field input:focus, .field select:focus { outline: none; border-color: #4a90d9; box-shadow: 0 0 0 2px rgba(74,144,217,0.15); }
  #question { width: 500px; }
  #btn-search { padding: 8px 24px; background: #4a90d9; color: #fff; border: none; border-radius: 6px; font-size: 14px; cursor: pointer; font-weight: 600; }
  #btn-search:hover { background: #3a7bc8; }
  #btn-search:disabled { opacity: 0.5; cursor: not-allowed; }
  .chunk { background: #fafafa; border: 1px solid #e8e8e8; border-radius: 6px; padding: 14px; margin-bottom: 10px; }
  .chunk-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; font-size: 13px; color: #888; }
  .chunk-score { font-weight: 700; color: #4a90d9; }
  .chunk-source { font-size: 12px; color: #999; }
  .chunk-content { font-size: 14px; line-height: 1.7; white-space: pre-wrap; word-break: break-all; }
  .chunk-index { font-size: 12px; color: #aaa; margin-top: 6px; }
  .meta-row { display: flex; gap: 24px; font-size: 13px; color: #666; margin-top: 8px; padding-top: 8px; border-top: 1px solid #eee; }
  .meta-row span { display: flex; gap: 4px; }
  .meta-label { color: #999; }
  .stats { font-size: 13px; color: #888; margin-bottom: 12px; }
  .empty { text-align: center; padding: 60px; color: #bbb; font-size: 15px; }
</style>
</head>
<body>
<div class="container">
  <h1>RAG 检索测试 <span id="status" style="font-size:12px;color:#999;font-weight:400"></span></h1>
  <div class="panel">
    <div class="row">
      <div class="field">
        <label for="question">问题</label>
        <input id="question" type="text" placeholder="输入要检索的问题...">
      </div>
      <div class="field">
        <label for="kb">知识库</label>
        <select id="kb"></select>
      </div>
      <div class="field">
        <label for="chunker">切分策略</label>
        <select id="chunker"></select>
      </div>
      <div class="field">
        <label for="retriever">检索器</label>
        <select id="retriever">
          <option value="hybrid">hybrid (RRF)</option>
          <option value="vector">vector (dense)</option>
          <option value="bm25">bm25 (sparse)</option>
        </select>
      </div>
      <div class="field">
        <label for="top_k">Top-K</label>
        <select id="top_k">
          <option value="5">5</option>
          <option value="10" selected>10</option>
          <option value="20">20</option>
        </select>
      </div>
      <button id="btn-search" onclick="doSearch()">检索</button>
    </div>
  </div>
  <div class="stats" id="stats"></div>
  <div id="results" class="empty">输入问题并选择策略后点击"检索"</div>
</div>
<script>
async function loadKbs() {
  try {
    const resp = await fetch('/api/kbs');
    const data = await resp.json();
    const sel = document.getElementById('kb');
    sel.innerHTML = '';
    data.kbs.forEach(kb => {
      const opt = document.createElement('option');
      opt.value = kb;
      opt.textContent = kb;
      sel.appendChild(opt);
    });
    if (data.kbs.length) { sel.value = data.kbs[0]; await loadStrategies(); }
  } catch(e) {
    document.getElementById('status').textContent = '加载知识库失败: ' + e.message;
  }
}
async function loadStrategies() {
  try {
    const kb = document.getElementById('kb').value;
    const resp = await fetch('/api/strategies?kb=' + kb);
    const data = await resp.json();
    const sel = document.getElementById('chunker');
    sel.innerHTML = '';
    data.chunkers.forEach(c => {
      const opt = document.createElement('option');
      opt.value = c;
      opt.textContent = c;
      sel.appendChild(opt);
    });
    document.getElementById('status').textContent = `已加载 ${data.chunkers.length} 个策略`;
  } catch(e) {
    document.getElementById('status').textContent = '加载策略失败: ' + e.message;
  }
}
// KB 切换时重新加载策略列表
document.getElementById('kb').addEventListener('change', loadStrategies);
async function doSearch() {
  const q = document.getElementById('question').value.trim();
  if (!q) return;
  const btn = document.getElementById('btn-search');
  btn.disabled = true; btn.textContent = '检索中...';
  const resultsDiv = document.getElementById('results');
  resultsDiv.innerHTML = '<div class="empty">检索中...</div>';
  document.getElementById('stats').textContent = '';
  try {
    const body = JSON.stringify({
      question: q,
      kb: document.getElementById('kb').value,
      chunker: document.getElementById('chunker').value,
      retriever: document.getElementById('retriever').value,
      top_k: parseInt(document.getElementById('top_k').value),
    });
    console.log('Search request:', body);
    const resp = await fetch('/api/search', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: body,
    });
    console.log('Response status:', resp.status);
    const data = await resp.json();
    console.log('Response data:', data);
    if (data.error) {
      resultsDiv.innerHTML = `<div class="empty" style="color:#e55">${esc(data.error)}</div>`;
      return;
    }
    document.getElementById('stats').textContent = `${data.results.length} 条结果，耗时 ${data.elapsed_ms} ms`;
    if (data.results.length === 0) {
      resultsDiv.innerHTML = '<div class="empty">无结果</div>';
      return;
    }
    let html = '';
    data.results.forEach((r, i) => {
      html += `<div class="chunk">
        <div class="chunk-header">
          <span>#${i+1} <span class="chunk-score">score=${r.score.toFixed(4)}</span></span>
          <span class="chunk-source">${esc(r.source)}</span>
        </div>
        <div class="chunk-content">${esc(r.content_text)}</div>
        <div class="meta-row">
          <span><span class="meta-label">文件</span>${esc(r.file_name)}</span>
          <span><span class="meta-label">类型</span>${esc(r.file_type)}</span>
          <span><span class="meta-label">章节</span>${esc(r.section_title)}</span>
        </div>
      </div>`;
    });
    resultsDiv.innerHTML = html;
  } catch(e) {
    console.error('Search error:', e);
    resultsDiv.innerHTML = `<div class="empty" style="color:#e55">请求失败: ${esc(e.message)}</div>`;
  } finally {
    btn.disabled = false; btn.textContent = '检索';
  }
}
function esc(s) { return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }
document.addEventListener('DOMContentLoaded', loadKbs);
document.getElementById('question').addEventListener('keydown', e => { if (e.key==='Enter') doSearch(); });
</script>
</body>
</html>"""


@app.route("/")
def index():
    return HTML


@app.route("/api/kbs")
def api_kbs():
    from config import COLLECTION_SEP
    from core.qdrant import get_client
    import re
    client = get_client()
    all_cols = {c.name for c in client.get_collections().collections}
    client.close()
    # 从 {kb}__{strategy} 集合名提取 kb 集合，剔除内部 collection（如 langchain 相关）
    kbs = set()
    for cname in all_cols:
        if COLLECTION_SEP in cname:
            kbs.add(cname.split(COLLECTION_SEP)[0])
    kbs = sorted(kbs)
    if not kbs:
        kbs = [DEFAULT_KB]
    return jsonify({"kbs": kbs})


@app.route("/api/strategies")
def api_strategies():
    from core.qdrant import get_client
    kb = request.args.get("kb", DEFAULT_KB)
    client = get_client()
    all_cols = {c.name for c in client.get_collections().collections}
    client.close()
    prefix = f"{kb}__"
    available = sorted([cname[len(prefix):] for cname in all_cols if cname.startswith(prefix)])
    return jsonify({"chunkers": available, "kb": kb, "total": len(available)})


@app.route("/api/search", methods=["POST"])
def api_search():
    data = request.get_json()
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "问题不能为空"}), 400

    kb_name = data.get("kb", DEFAULT_KB)
    chunker = data.get("chunker", "fixed_1100")
    retriever = data.get("retriever", "hybrid")
    top_k = int(data.get("top_k", 10))

    import time
    t0 = time.time()
    try:
        docs = strategy_search(
            query=question, kb_name=kb_name,
            chunker_name=chunker, top_k=top_k,
            retriever=retriever,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    elapsed = round((time.time() - t0) * 1000)
    results = []
    for d in docs:
        results.append({
            "score": d.get("score", 0),
            "content_text": str(d.get("content_text", "")),
            "source": str(d.get("source", "")),
            "file_name": str(d.get("file_name", "")),
            "file_type": str(d.get("file_type", "")),
            "section_title": str(d.get("section_title", "")),
            "section_summary": str(d.get("section_summary", "")),
        })

    return jsonify({"results": results, "elapsed_ms": elapsed})


if __name__ == "__main__":
    print(f"RAG 检索测试接口: http://127.0.0.1:5050")
    print(f"知识库: {DEFAULT_KB}")
    app.run(host="127.0.0.1", port=5050, debug=False)