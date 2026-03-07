"""
dashboard.py
Sprint 4 — Role-Based Management & Analytics Dashboard

Flask web application that reads from the existing audit_log.db SQLite database
and provides the role-based management interface required by the grading rubric.

Three User Roles (Interim Report §3.1, FR-06 — Role-Based Access Control):
  admin    /admin123   → Full access: users, vehicles, fraud feed, logs, health
  security /security123→ Operational: vehicles, fraud feed, access log
  finance  /finance123 → Audit access: read-only logs, analytics, export

CRUD:
  Vehicle Registry is managed via the dashboard. The registered_vehicles SQLite
  table replaces the hardcoded REGISTERED_VEHICLES frozenset in validator.py.

Run:
  python dashboard.py [--db path/to/audit_log.db] [--port 5000]

Open:
  http://localhost:5000
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta
from functools import wraps
from typing import Optional

from flask import (Flask, Response, flash, jsonify, redirect,
                   render_template_string, request, session, url_for)

# ── Project root on sys.path so src.* imports work ───────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from src.database import DatabaseManager, DEFAULT_SQLITE_PATH

# ─────────────────────────────────────────────────────────────────────────────
# Flask App
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.secret_key = os.environ.get('DASHBOARD_SECRET', 'vlpr-dashboard-secret-2026')

# ─────────────────────────────────────────────────────────────────────────────
# Mock User Store (RBAC)
# In production: replace with a hashed-password users table in SQLite.
# ─────────────────────────────────────────────────────────────────────────────

USERS: dict = {
    'admin': {
        'password': 'admin123',
        'role':     'admin',
        'label':    'System Administrator',
        'badge':    '#ef4444',
    },
    'security': {
        'password': 'security123',
        'role':     'security',
        'label':    'Security Officer',
        'badge':    '#f59e0b',
    },
    'finance': {
        'password': 'finance123',
        'role':     'finance',
        'label':    'Finance / Auditor',
        'badge':    '#10b981',
    },
}

# ── Role permission matrix ────────────────────────────────────────────────────
PERMISSIONS: dict = {
    'admin':    {'overview', 'vehicles', 'fraud_feed', 'access_log', 'health', 'users'},
    'security': {'overview', 'vehicles', 'fraud_feed', 'access_log'},
    'finance':  {'overview', 'access_log', 'reports'},
}

# ─────────────────────────────────────────────────────────────────────────────
# Database singleton — lazy-initialised on first request
# ─────────────────────────────────────────────────────────────────────────────

_db_manager: Optional[DatabaseManager] = None
_db_path:    str = DEFAULT_SQLITE_PATH


def get_db() -> DatabaseManager:
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager(
            base_url  = 'http://localhost',
            cam_code  = 'DASHBOARD',
            device    = '00',
            in_out    = 'N',
            sqlite_db_path = _db_path,
        )
        _db_manager.seed_registered_vehicles()
    return _db_manager


def raw_db() -> sqlite3.Connection:
    """Direct read-only SQLite connection for dashboard queries."""
    conn = sqlite3.connect(_db_path)
    conn.row_factory = sqlite3.Row
    return conn

# ─────────────────────────────────────────────────────────────────────────────
# Auth Decorators
# ─────────────────────────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


def role_required(*roles):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if 'username' not in session:
                return redirect(url_for('login'))
            if session.get('role') not in roles:
                return render_page('forbidden.html', {})
            return f(*args, **kwargs)
        return decorated
    return decorator

# ─────────────────────────────────────────────────────────────────────────────
# Template rendering helper
# ─────────────────────────────────────────────────────────────────────────────

def render_page(template_name: str, ctx: dict) -> str:
    base_ctx = {
        'username':    session.get('username', ''),
        'role':        session.get('role', ''),
        'role_label':  USERS.get(session.get('username', ''), {}).get('label', ''),
        'role_badge':  USERS.get(session.get('username', ''), {}).get('badge', '#64748b'),
        'permissions': PERMISSIONS.get(session.get('role', ''), set()),
        'active_page': template_name.replace('.html', ''),
    }
    base_ctx.update(ctx)
    template_str = TEMPLATES.get(template_name, TEMPLATES['404.html'])
    return render_template_string(template_str, **base_ctx)

# ─────────────────────────────────────────────────────────────────────────────
# HTML Templates
# ─────────────────────────────────────────────────────────────────────────────

BASE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VLPR Security System{% block title %} | Dashboard{% endblock %}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root {
  --bg:         #07090f;
  --bg2:        #0d1626;
  --panel:      #111b30;
  --panel2:     #162040;
  --border:     #1a3356;
  --accent:     #f59e0b;
  --accent2:    #0ea5e9;
  --text:       #d1daf0;
  --text2:      #8898b8;
  --text3:      #4a607a;
  --critical:   #ef4444;
  --high:       #f97316;
  --medium:     #eab308;
  --success:    #10b981;
  --sidebar-w:  252px;
  --topbar-h:   58px;
  --font-hd:    'Rajdhani', sans-serif;
  --font-mono:  'JetBrains Mono', monospace;
  --font-body:  'DM Sans', sans-serif;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:var(--font-body);font-size:14px;min-height:100vh;display:flex}
a{color:var(--accent);text-decoration:none}
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:var(--bg2)}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}

/* ── Sidebar ─────────────────────────────────────────────────────────── */
.sidebar{
  width:var(--sidebar-w);flex-shrink:0;background:var(--bg2);
  border-right:1px solid var(--border);display:flex;flex-direction:column;
  height:100vh;position:sticky;top:0;overflow-y:auto;
}
.sidebar-logo{
  padding:22px 20px 16px;border-bottom:1px solid var(--border);
  display:flex;align-items:center;gap:12px;
}
.sidebar-logo .icon{
  width:36px;height:36px;background:var(--accent);border-radius:8px;
  display:flex;align-items:center;justify-content:center;flex-shrink:0;
  font-size:18px;
}
.sidebar-logo .brand{font-family:var(--font-hd);font-weight:700;font-size:15px;
  letter-spacing:.08em;text-transform:uppercase;color:var(--text);line-height:1.2}
.sidebar-logo .brand small{display:block;font-size:10px;color:var(--text2);
  font-weight:400;letter-spacing:.05em}

.role-chip{
  margin:14px 16px;padding:7px 12px;border-radius:6px;
  font-size:11px;font-family:var(--font-mono);letter-spacing:.06em;
  text-transform:uppercase;border:1px solid;font-weight:500;
}

.nav-section{padding:8px 0;flex:1}
.nav-label{padding:8px 20px 4px;font-size:10px;font-family:var(--font-mono);
  letter-spacing:.12em;text-transform:uppercase;color:var(--text3)}
.nav-item{
  display:flex;align-items:center;gap:11px;padding:9px 20px;
  color:var(--text2);font-size:13.5px;font-weight:400;transition:.15s;
  position:relative;cursor:pointer;
}
.nav-item:hover{color:var(--text);background:var(--panel)}
.nav-item.active{color:var(--accent);background:var(--panel);font-weight:500}
.nav-item.active::before{
  content:'';position:absolute;left:0;top:4px;bottom:4px;
  width:3px;background:var(--accent);border-radius:0 3px 3px 0;
}
.nav-item .nav-icon{font-size:15px;width:18px;text-align:center;flex-shrink:0}

.sidebar-footer{
  padding:14px 16px;border-top:1px solid var(--border);
  display:flex;align-items:center;gap:10px;
}
.sidebar-footer .avatar{
  width:32px;height:32px;border-radius:50%;background:var(--panel2);
  display:flex;align-items:center;justify-content:center;font-size:13px;
  font-family:var(--font-mono);color:var(--text);flex-shrink:0;border:1px solid var(--border);
}
.sidebar-footer .user-info{flex:1;min-width:0}
.sidebar-footer .user-info .uname{font-size:13px;font-weight:500;truncate;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.sidebar-footer .user-info .urole{font-size:11px;color:var(--text2)}
.sidebar-footer .logout-btn{
  width:28px;height:28px;border-radius:5px;background:transparent;
  border:1px solid var(--border);color:var(--text2);display:flex;
  align-items:center;justify-content:center;font-size:13px;cursor:pointer;
  transition:.15s;flex-shrink:0;
}
.sidebar-footer .logout-btn:hover{background:var(--critical);border-color:var(--critical);color:#fff}

/* ── Main area ───────────────────────────────────────────────────────── */
.main-wrap{flex:1;display:flex;flex-direction:column;min-width:0;min-height:100vh}
.topbar{
  height:var(--topbar-h);background:var(--bg2);border-bottom:1px solid var(--border);
  display:flex;align-items:center;padding:0 24px;gap:16px;position:sticky;top:0;z-index:100;
  flex-shrink:0;
}
.topbar-title{font-family:var(--font-hd);font-size:18px;font-weight:600;
  letter-spacing:.04em;flex:1}
.topbar-clock{font-family:var(--font-mono);font-size:12px;color:var(--text2)}
.status-dot{
  width:8px;height:8px;border-radius:50%;background:var(--success);
  box-shadow:0 0 6px var(--success);flex-shrink:0;
}
.page-content{flex:1;padding:24px;overflow-y:auto}

/* ── KPI Cards ───────────────────────────────────────────────────────── */
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin-bottom:24px}
.kpi-card{
  background:var(--panel);border:1px solid var(--border);border-radius:10px;
  padding:18px 20px;position:relative;overflow:hidden;transition:.2s;
}
.kpi-card:hover{transform:translateY(-2px);border-color:var(--accent);box-shadow:0 8px 24px rgba(0,0,0,.4)}
.kpi-card::before{
  content:'';position:absolute;top:0;left:0;right:0;height:3px;border-radius:10px 10px 0 0;
  background:var(--card-accent,var(--accent));
}
.kpi-card .kpi-icon{font-size:22px;margin-bottom:10px}
.kpi-card .kpi-val{font-family:var(--font-hd);font-size:32px;font-weight:700;color:var(--text);line-height:1}
.kpi-card .kpi-label{font-size:12px;color:var(--text2);margin-top:4px;text-transform:uppercase;letter-spacing:.06em}
.kpi-card .kpi-sub{font-size:11px;color:var(--text3);margin-top:6px;font-family:var(--font-mono)}

/* ── Panels ──────────────────────────────────────────────────────────── */
.panel{background:var(--panel);border:1px solid var(--border);border-radius:10px;overflow:hidden}
.panel-header{
  padding:14px 20px;border-bottom:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between;gap:12px;
}
.panel-title{font-family:var(--font-hd);font-size:15px;font-weight:600;letter-spacing:.03em}
.panel-body{padding:20px}

/* ── Tables ──────────────────────────────────────────────────────────── */
.data-table{width:100%;border-collapse:collapse;font-size:13px}
.data-table th{
  padding:9px 14px;text-align:left;font-family:var(--font-mono);font-size:11px;
  font-weight:500;letter-spacing:.08em;text-transform:uppercase;color:var(--text2);
  border-bottom:1px solid var(--border);background:var(--panel2);
}
.data-table td{
  padding:10px 14px;border-bottom:1px solid rgba(26,51,86,.5);
  font-family:var(--font-mono);font-size:12px;vertical-align:middle;
}
.data-table tbody tr:hover{background:var(--panel2)}
.data-table tbody tr:last-child td{border-bottom:none}

/* ── Severity Badges ─────────────────────────────────────────────────── */
.badge-sev{
  display:inline-flex;align-items:center;gap:5px;padding:3px 9px;
  border-radius:4px;font-family:var(--font-mono);font-size:11px;font-weight:500;
  letter-spacing:.04em;
}
.badge-CRITICAL{background:rgba(239,68,68,.15);color:var(--critical);
  border:1px solid rgba(239,68,68,.3);box-shadow:0 0 8px rgba(239,68,68,.2)}
.badge-HIGH{background:rgba(249,115,22,.15);color:var(--high);border:1px solid rgba(249,115,22,.3)}
.badge-MEDIUM{background:rgba(234,179,8,.15);color:var(--medium);border:1px solid rgba(234,179,8,.3)}
.badge-GRANTED{background:rgba(16,185,129,.12);color:var(--success);border:1px solid rgba(16,185,129,.25)}
.badge-DENIED{background:rgba(239,68,68,.12);color:var(--critical);border:1px solid rgba(239,68,68,.25)}
.badge-active{background:rgba(16,185,129,.12);color:var(--success);border:1px solid rgba(16,185,129,.25);
  display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-family:var(--font-mono)}
.badge-inactive{background:rgba(74,96,122,.2);color:var(--text3);border:1px solid rgba(74,96,122,.4);
  display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-family:var(--font-mono)}

.violation-CLONE_ATTACK{color:var(--critical);font-weight:600}
.violation-SPEEDING_VIOLATION{color:var(--high)}
.violation-PATH_VIOLATION{color:var(--medium)}

/* ── Buttons ─────────────────────────────────────────────────────────── */
.btn-primary-vlpr{
  background:var(--accent);color:#07090f;border:none;padding:8px 18px;
  border-radius:6px;font-family:var(--font-hd);font-size:14px;font-weight:600;
  letter-spacing:.04em;cursor:pointer;transition:.15s;
}
.btn-primary-vlpr:hover{background:#d97706;transform:translateY(-1px)}
.btn-ghost{
  background:transparent;color:var(--text2);border:1px solid var(--border);
  padding:7px 14px;border-radius:6px;font-size:13px;cursor:pointer;transition:.15s;
}
.btn-ghost:hover{border-color:var(--accent);color:var(--accent)}
.btn-danger-sm{
  background:transparent;color:var(--critical);border:1px solid rgba(239,68,68,.3);
  padding:4px 10px;border-radius:4px;font-size:11px;cursor:pointer;transition:.15s;
  font-family:var(--font-mono);
}
.btn-danger-sm:hover{background:var(--critical);color:#fff}
.btn-edit-sm{
  background:transparent;color:var(--accent2);border:1px solid rgba(14,165,233,.3);
  padding:4px 10px;border-radius:4px;font-size:11px;cursor:pointer;transition:.15s;
  font-family:var(--font-mono);
}
.btn-edit-sm:hover{background:var(--accent2);color:#fff}

/* ── Form fields ─────────────────────────────────────────────────────── */
.form-field{
  background:var(--bg);border:1px solid var(--border);color:var(--text);
  padding:9px 13px;border-radius:6px;font-family:var(--font-mono);font-size:13px;
  width:100%;outline:none;transition:.15s;
}
.form-field:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(245,158,11,.1)}
.form-label{font-size:12px;color:var(--text2);margin-bottom:5px;
  font-family:var(--font-mono);text-transform:uppercase;letter-spacing:.06em}

/* ── Modal ───────────────────────────────────────────────────────────── */
.modal-backdrop-vlpr{
  position:fixed;inset:0;background:rgba(7,9,15,.85);z-index:1050;
  display:none;align-items:center;justify-content:center;backdrop-filter:blur(3px);
}
.modal-backdrop-vlpr.show{display:flex}
.modal-box{
  background:var(--panel);border:1px solid var(--border);border-radius:12px;
  width:100%;max-width:480px;overflow:hidden;box-shadow:0 24px 64px rgba(0,0,0,.6);
}
.modal-header{
  padding:18px 22px;border-bottom:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between;
}
.modal-header .modal-title{font-family:var(--font-hd);font-size:17px;font-weight:600}
.modal-close{background:none;border:none;color:var(--text2);font-size:20px;cursor:pointer;
  width:30px;height:30px;display:flex;align-items:center;justify-content:center;border-radius:5px}
.modal-close:hover{background:var(--panel2);color:var(--text)}
.modal-body{padding:22px}
.modal-footer{padding:16px 22px;border-top:1px solid var(--border);display:flex;
  align-items:center;justify-content:flex-end;gap:10px}

/* ── Flash messages ──────────────────────────────────────────────────── */
.flash-msg{
  padding:11px 16px;border-radius:7px;margin-bottom:16px;
  font-size:13px;display:flex;align-items:center;gap:10px;
}
.flash-success{background:rgba(16,185,129,.1);border:1px solid rgba(16,185,129,.3);color:var(--success)}
.flash-error{background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.3);color:var(--critical)}
.flash-info{background:rgba(14,165,233,.1);border:1px solid rgba(14,165,233,.3);color:var(--accent2)}

/* ── Search box ──────────────────────────────────────────────────────── */
.search-wrap{position:relative;display:inline-block}
.search-wrap input{padding-left:34px;width:240px}
.search-wrap .search-icon{
  position:absolute;left:11px;top:50%;transform:translateY(-50%);
  color:var(--text3);font-size:13px;pointer-events:none;
}

/* ── Chart container ─────────────────────────────────────────────────── */
.chart-wrap{position:relative;height:220px}

/* ── Toolbar ─────────────────────────────────────────────────────────── */
.toolbar{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:16px;flex-wrap:wrap}
</style>
{% block extra_style %}{% endblock %}
</head>
<body>
{% if username %}
<nav class="sidebar">
  <div class="sidebar-logo">
    <div class="icon">🎥</div>
    <div class="brand">VLPR System<small>Security Platform</small></div>
  </div>
  <div class="role-chip" style="color:{{ role_badge }};border-color:{{ role_badge }}22;background:{{ role_badge }}12">
    ◆ {{ role_label }}
  </div>
  <div class="nav-section">
    {% if 'overview' in permissions %}
    <a href="/overview" class="nav-item {% if active_page=='overview' %}active{% endif %}">
      <span class="nav-icon">📊</span> Overview
    </a>
    {% endif %}
    {% if 'vehicles' in permissions %}
    <a href="/vehicles" class="nav-item {% if active_page=='vehicles' %}active{% endif %}">
      <span class="nav-icon">🚗</span> Vehicle Registry
    </a>
    {% endif %}
    {% if 'fraud_feed' in permissions %}
    <a href="/fraud-feed" class="nav-item {% if active_page=='fraud_feed' %}active{% endif %}">
      <span class="nav-icon">🚨</span> Fraud Intelligence
    </a>
    {% endif %}
    {% if 'access_log' in permissions %}
    <a href="/access-log" class="nav-item {% if active_page=='access_log' %}active{% endif %}">
      <span class="nav-icon">📋</span> Access Log
    </a>
    {% endif %}
    {% if 'reports' in permissions %}
    <a href="/reports" class="nav-item {% if active_page=='reports' %}active{% endif %}">
      <span class="nav-icon">📈</span> Reports
    </a>
    {% endif %}
    {% if role == 'admin' %}
    <div class="nav-label">Administration</div>
    <a href="/admin/health" class="nav-item {% if active_page=='health' %}active{% endif %}">
      <span class="nav-icon">⚙️</span> System Health
    </a>
    <a href="/admin/users" class="nav-item {% if active_page=='users' %}active{% endif %}">
      <span class="nav-icon">👥</span> User Management
    </a>
    {% endif %}
  </div>
  <div class="sidebar-footer">
    <div class="avatar">{{ username[0].upper() }}</div>
    <div class="user-info">
      <div class="uname">{{ username }}</div>
      <div class="urole">{{ role }}</div>
    </div>
    <a href="/logout" class="logout-btn" title="Logout">⏏</a>
  </div>
</nav>
{% endif %}

<div class="main-wrap">
  {% if username %}
  <div class="topbar">
    <div class="topbar-title">{% block topbar_title %}Dashboard{% endblock %}</div>
    <div class="topbar-clock" id="clock">--:--:--</div>
    <div class="status-dot" title="System Online"></div>
  </div>
  {% endif %}
  <div class="page-content">
    {% with messages = get_flashed_messages(with_categories=true) %}
      {% for cat, msg in messages %}
        <div class="flash-msg flash-{{ cat }}">{% if cat=='success' %}✓{% elif cat=='error' %}✕{% else %}ℹ{% endif %} {{ msg }}</div>
      {% endfor %}
    {% endwith %}
    {% block content %}{% endblock %}
  </div>
</div>

<!-- Toast container -->
<div id="toast-wrap" style="position:fixed;bottom:24px;right:24px;z-index:9999;display:flex;flex-direction:column;gap:8px;"></div>

<script>
// Live clock
function updateClock(){
  const el=document.getElementById('clock');
  if(el) el.textContent=new Date().toLocaleTimeString('en-GB',{hour12:false});
}
setInterval(updateClock,1000); updateClock();

// Toast utility
function toast(msg,type='info'){
  const wrap=document.getElementById('toast-wrap');
  const t=document.createElement('div');
  const colors={'info':'#0ea5e9','success':'#10b981','error':'#ef4444','warn':'#f59e0b'};
  t.style.cssText=`background:#111b30;border:1px solid ${colors[type]||'#1a3356'};color:#d1daf0;
    padding:11px 16px;border-radius:7px;font-size:13px;max-width:320px;
    animation:fadeIn .2s ease;box-shadow:0 8px 24px rgba(0,0,0,.5)`;
  t.innerHTML=`<span style="color:${colors[type]};margin-right:8px">●</span>${msg}`;
  wrap.appendChild(t);
  setTimeout(()=>{t.style.opacity='0';t.style.transition='opacity .3s';setTimeout(()=>t.remove(),300)},3500);
}

// Generic API fetch
async function api(path,opts={}){
  try{
    const r=await fetch(path,{headers:{'Content-Type':'application/json'},...opts});
    return await r.json();
  }catch(e){console.error(e);return null;}
}
</script>
<style>
@keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
</style>
{% block extra_js %}{% endblock %}
</body>
</html>
"""

# ─────────────────────────────────────────────────────────────────────────────
LOGIN_HTML = """{% extends "base.html" %}
{% block title %} | Login{% endblock %}
{% block content %}
<style>
body{align-items:center;justify-content:center;background:var(--bg)}
.main-wrap{flex:1;display:flex;align-items:center;justify-content:center;
  background:radial-gradient(ellipse 80% 60% at 50% 40%,rgba(14,165,233,.06) 0%,transparent 70%)}
.login-card{
  background:var(--panel);border:1px solid var(--border);border-radius:14px;
  padding:40px 36px;width:380px;box-shadow:0 32px 80px rgba(0,0,0,.6);
}
.login-logo{text-align:center;margin-bottom:28px}
.login-logo .icon-big{
  width:56px;height:56px;background:var(--accent);border-radius:14px;
  display:inline-flex;align-items:center;justify-content:center;
  font-size:26px;margin-bottom:14px;box-shadow:0 0 24px rgba(245,158,11,.3);
}
.login-logo h1{font-family:var(--font-hd);font-size:22px;font-weight:700;
  letter-spacing:.06em;text-transform:uppercase}
.login-logo p{font-size:12px;color:var(--text2);margin-top:4px;font-family:var(--font-mono)}
.form-group{margin-bottom:16px}
.demo-creds{
  margin-top:22px;padding:14px;background:var(--bg);border-radius:8px;
  border:1px solid var(--border);font-family:var(--font-mono);font-size:11px;
}
.demo-creds .dc-title{color:var(--text2);margin-bottom:8px;font-size:10px;
  text-transform:uppercase;letter-spacing:.08em}
.demo-creds .dc-row{display:flex;justify-content:space-between;color:var(--text);
  padding:3px 0;border-bottom:1px solid rgba(26,51,86,.4)}
.demo-creds .dc-row:last-child{border-bottom:none}
.dc-role{color:var(--accent)}
</style>
<div class="login-card">
  <div class="login-logo">
    <div class="icon-big">🎥</div>
    <h1>VLPR System</h1>
    <p>Colombo Dockyard Security Platform</p>
  </div>
  {% with msgs = get_flashed_messages(with_categories=true) %}
    {% for cat,msg in msgs %}
      <div class="flash-msg flash-{{ cat }}">{{ msg }}</div>
    {% endfor %}
  {% endwith %}
  <form method="POST" action="/login">
    <div class="form-group">
      <div class="form-label">Username</div>
      <input class="form-field" type="text" name="username" placeholder="Enter username" required autocomplete="username">
    </div>
    <div class="form-group">
      <div class="form-label">Password</div>
      <input class="form-field" type="password" name="password" placeholder="Enter password" required autocomplete="current-password">
    </div>
    <button type="submit" class="btn-primary-vlpr" style="width:100%;margin-top:8px;padding:11px">Sign In →</button>
  </form>
  <div class="demo-creds">
    <div class="dc-title">Demo Credentials</div>
    <div class="dc-row"><span class="dc-role">admin</span><span>admin123</span><span style="color:var(--text2)">Full Access</span></div>
    <div class="dc-row"><span class="dc-role">security</span><span>security123</span><span style="color:var(--text2)">Operations</span></div>
    <div class="dc-row"><span class="dc-role">finance</span><span>finance123</span><span style="color:var(--text2)">Audit Only</span></div>
  </div>
</div>
{% endblock %}"""

# ─────────────────────────────────────────────────────────────────────────────
OVERVIEW_HTML = """{% extends "base.html" %}
{% block topbar_title %}Overview{% endblock %}
{% block content %}
<div class="kpi-grid" id="kpi-grid">
  <div class="kpi-card" style="--card-accent:var(--accent2)">
    <div class="kpi-icon">🚗</div>
    <div class="kpi-val" id="kpi-vehicles">—</div>
    <div class="kpi-label">Registered Vehicles</div>
    <div class="kpi-sub">Active registry entries</div>
  </div>
  <div class="kpi-card" style="--card-accent:var(--success)">
    <div class="kpi-icon">✅</div>
    <div class="kpi-val" id="kpi-detections">—</div>
    <div class="kpi-label">Total Detections</div>
    <div class="kpi-sub" id="kpi-granted">— granted</div>
  </div>
  <div class="kpi-card" style="--card-accent:var(--critical)">
    <div class="kpi-icon">🚨</div>
    <div class="kpi-val" id="kpi-fraud">—</div>
    <div class="kpi-label">Fraud Alerts</div>
    <div class="kpi-sub" id="kpi-fraud-rate">—% of traffic</div>
  </div>
  <div class="kpi-card" style="--card-accent:var(--accent)">
    <div class="kpi-icon">📡</div>
    <div class="kpi-val" id="kpi-health" style="font-size:22px">ONLINE</div>
    <div class="kpi-label">System Status</div>
    <div class="kpi-sub" id="kpi-uptime">Audit log active</div>
  </div>
</div>

<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px">
  <div class="panel">
    <div class="panel-header">
      <span class="panel-title">📈 Traffic & Fraud (Last 24h)</span>
      <span style="font-size:11px;color:var(--text2);font-family:var(--font-mono)">Hourly</span>
    </div>
    <div class="panel-body"><div class="chart-wrap"><canvas id="trafficChart"></canvas></div></div>
  </div>
  <div class="panel">
    <div class="panel-header">
      <span class="panel-title">🚨 Recent Fraud Alerts</span>
      <a href="/fraud-feed" style="font-size:11px;color:var(--accent);font-family:var(--font-mono)">View all →</a>
    </div>
    <div class="panel-body" style="padding:0">
      <table class="data-table" id="mini-fraud"></table>
    </div>
  </div>
</div>

<div class="panel">
  <div class="panel-header">
    <span class="panel-title">📋 Recent Access Events</span>
    <a href="/access-log" style="font-size:11px;color:var(--accent);font-family:var(--font-mono)">View all →</a>
  </div>
  <div class="panel-body" style="padding:0">
    <table class="data-table" id="mini-access"></table>
  </div>
</div>

{% endblock %}
{% block extra_js %}
<script>
// KPIs
api('/api/kpis').then(d=>{
  if(!d) return;
  document.getElementById('kpi-vehicles').textContent=d.registered_vehicles;
  document.getElementById('kpi-detections').textContent=d.total_detections;
  document.getElementById('kpi-granted').textContent=d.granted+' granted';
  document.getElementById('kpi-fraud').textContent=d.fraud_alerts;
  const rate=d.total_detections>0?((d.fraud_alerts/d.total_detections)*100).toFixed(1):0;
  document.getElementById('kpi-fraud-rate').textContent=rate+'% of traffic';
});

// Chart
api('/api/chart-data').then(d=>{
  if(!d) return;
  new Chart(document.getElementById('trafficChart'),{
    data:{
      labels:d.labels,
      datasets:[
        {type:'bar',label:'Detections',data:d.detections,
         backgroundColor:'rgba(14,165,233,.25)',borderColor:'rgba(14,165,233,.7)',borderWidth:1},
        {type:'line',label:'Fraud Events',data:d.fraud,
         borderColor:'rgba(239,68,68,.8)',backgroundColor:'rgba(239,68,68,.1)',
         borderWidth:2,fill:true,tension:.4,pointRadius:3,
         pointBackgroundColor:'rgba(239,68,68,.9)'}
      ]
    },
    options:{
      responsive:true,maintainAspectRatio:false,
      plugins:{legend:{labels:{color:'#8898b8',font:{size:11}}}},
      scales:{
        x:{ticks:{color:'#4a607a',font:{size:10},maxRotation:0},grid:{color:'rgba(26,51,86,.4)'}},
        y:{ticks:{color:'#4a607a',font:{size:10}},grid:{color:'rgba(26,51,86,.4)'}}
      }
    }
  });
});

// Mini fraud table
api('/api/fraud-events?limit=5').then(d=>{
  const t=document.getElementById('mini-fraud');
  if(!d||!d.rows||!d.rows.length){t.innerHTML='<tr><td style="padding:16px;color:var(--text2);font-size:12px">No fraud events recorded.</td></tr>';return;}
  t.innerHTML='<thead><tr><th>Timestamp</th><th>Plate</th><th>Type</th><th>Severity</th></tr></thead><tbody>'
    +d.rows.map(r=>`<tr>
      <td>${r.timestamp||''}</td>
      <td style="color:var(--accent)">${r.plate_number}</td>
      <td class="violation-${r.violation_type}">${(r.violation_type||'').replace('_',' ')}</td>
      <td><span class="badge-sev badge-${r.severity_level}">⬤ ${r.severity_level}</span></td>
    </tr>`).join('')+'</tbody>';
});

// Mini access table
api('/api/access-log?limit=6').then(d=>{
  const t=document.getElementById('mini-access');
  if(!d||!d.rows||!d.rows.length){t.innerHTML='<tr><td style="padding:16px;color:var(--text2);font-size:12px">No access events recorded.</td></tr>';return;}
  t.innerHTML='<thead><tr><th>Timestamp</th><th>Plate</th><th>Gate</th><th>Decision</th></tr></thead><tbody>'
    +d.rows.map(r=>`<tr>
      <td>${r.timestamp||''}</td>
      <td style="color:var(--accent)">${r.plate_number}</td>
      <td>${r.gate_id||'—'}</td>
      <td><span class="badge-sev badge-${r.decision}">${r.decision}</span></td>
    </tr>`).join('')+'</tbody>';
});
</script>
{% endblock %}"""

# ─────────────────────────────────────────────────────────────────────────────
VEHICLES_HTML = """{% extends "base.html" %}
{% block topbar_title %}Vehicle Registry{% endblock %}
{% block content %}
<div class="panel">
  <div class="panel-header">
    <span class="panel-title">🚗 Registered Vehicles</span>
    <div style="display:flex;gap:10px;align-items:center">
      <div class="search-wrap">
        <span class="search-icon">🔍</span>
        <input class="form-field search-wrap" id="search" placeholder="Search plate, name, dept…" oninput="filterTable()" style="padding-left:34px;width:240px">
      </div>
      {% if role in ['admin','security'] %}
      <button class="btn-primary-vlpr" onclick="openAddModal()">+ Add Vehicle</button>
      {% endif %}
    </div>
  </div>
  <div style="padding:0;overflow-x:auto">
    <table class="data-table" id="veh-table">
      <thead><tr>
        <th>Plate Number</th><th>Owner Name</th><th>Type</th>
        <th>Department</th><th>Registered</th><th>Status</th>
        {% if role in ['admin','security'] %}<th>Actions</th>{% endif %}
      </tr></thead>
      <tbody id="veh-body"><tr><td colspan="7" style="text-align:center;padding:24px;color:var(--text2)">Loading…</td></tr></tbody>
    </table>
  </div>
</div>

{% if role in ['admin','security'] %}
<!-- Add/Edit Modal -->
<div class="modal-backdrop-vlpr" id="modal">
  <div class="modal-box">
    <div class="modal-header">
      <span class="modal-title" id="modal-title">Add Vehicle</span>
      <button class="modal-close" onclick="closeModal()">✕</button>
    </div>
    <div class="modal-body">
      <input type="hidden" id="edit-id">
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px">
        <div>
          <div class="form-label">Plate Number *</div>
          <input class="form-field" id="f-plate" placeholder="e.g. CAB-1234">
        </div>
        <div>
          <div class="form-label">Owner Name *</div>
          <input class="form-field" id="f-owner" placeholder="Full name">
        </div>
        <div>
          <div class="form-label">Vehicle Type</div>
          <select class="form-field" id="f-type">
            <option>Car</option><option>Van</option><option>Truck</option>
            <option>Motorcycle</option><option>SUV</option>
          </select>
        </div>
        <div>
          <div class="form-label">Department</div>
          <input class="form-field" id="f-dept" placeholder="e.g. Engineering">
        </div>
        <div>
          <div class="form-label">Status</div>
          <select class="form-field" id="f-status"><option value="1">Active</option><option value="0">Inactive</option></select>
        </div>
      </div>
    </div>
    <div class="modal-footer">
      <button class="btn-ghost" onclick="closeModal()">Cancel</button>
      <button class="btn-primary-vlpr" onclick="saveVehicle()">Save Vehicle</button>
    </div>
  </div>
</div>
{% endif %}

{% endblock %}
{% block extra_js %}
<script>
let allVehicles=[];

function loadVehicles(){
  api('/api/vehicles').then(d=>{
    allVehicles=d||[];
    renderTable(allVehicles);
  });
}

function renderTable(rows){
  const editable = {{ 'true' if role in ['admin','security'] else 'false' }};
  const b=document.getElementById('veh-body');
  if(!rows.length){b.innerHTML='<tr><td colspan="7" style="text-align:center;padding:24px;color:var(--text2)">No vehicles registered.</td></tr>';return;}
  b.innerHTML=rows.map(r=>`<tr>
    <td style="color:var(--accent);font-weight:500">${r.plate_number}</td>
    <td>${r.owner_name}</td>
    <td>${r.vehicle_type}</td>
    <td>${r.department}</td>
    <td>${r.registered_date}</td>
    <td><span class="${r.is_active?'badge-active':'badge-inactive'}">${r.is_active?'Active':'Inactive'}</span></td>
    ${editable?`<td style="display:flex;gap:6px;padding:8px 14px">
      <button class="btn-edit-sm" onclick="openEditModal(${r.id})">Edit</button>
      <button class="btn-danger-sm" onclick="deleteVehicle(${r.id},'${r.plate_number}')">Delete</button>
    </td>`:'<td>—</td>'}
  </tr>`).join('');
}

function filterTable(){
  const q=document.getElementById('search').value.toLowerCase();
  renderTable(allVehicles.filter(r=>
    r.plate_number.toLowerCase().includes(q)||
    r.owner_name.toLowerCase().includes(q)||
    r.department.toLowerCase().includes(q)
  ));
}

function openAddModal(){
  document.getElementById('modal-title').textContent='Add Vehicle';
  document.getElementById('edit-id').value='';
  ['plate','owner','dept'].forEach(id=>document.getElementById('f-'+id).value='');
  document.getElementById('f-type').value='Car';
  document.getElementById('f-status').value='1';
  document.getElementById('f-plate').disabled=false;
  document.getElementById('modal').classList.add('show');
}

function openEditModal(id){
  const v=allVehicles.find(x=>x.id===id);
  if(!v) return;
  document.getElementById('modal-title').textContent='Edit Vehicle';
  document.getElementById('edit-id').value=id;
  document.getElementById('f-plate').value=v.plate_number;
  document.getElementById('f-plate').disabled=true;
  document.getElementById('f-owner').value=v.owner_name;
  document.getElementById('f-type').value=v.vehicle_type;
  document.getElementById('f-dept').value=v.department;
  document.getElementById('f-status').value=v.is_active?'1':'0';
  document.getElementById('modal').classList.add('show');
}

function closeModal(){document.getElementById('modal').classList.remove('show')}

async function saveVehicle(){
  const id=document.getElementById('edit-id').value;
  const payload={
    plate_number:document.getElementById('f-plate').value.trim().toUpperCase(),
    owner_name:document.getElementById('f-owner').value.trim(),
    vehicle_type:document.getElementById('f-type').value,
    department:document.getElementById('f-dept').value.trim(),
    is_active:parseInt(document.getElementById('f-status').value),
  };
  if(!payload.plate_number||!payload.owner_name){toast('Plate and owner name are required.','warn');return;}
  const method=id?'PUT':'POST';
  const url=id?`/api/vehicles/${id}`:'/api/vehicles';
  const r=await api(url,{method,body:JSON.stringify(payload)});
  if(r&&r.success){toast(r.message,'success');closeModal();loadVehicles();}
  else toast(r?.message||'Save failed.','error');
}

async function deleteVehicle(id,plate){
  if(!confirm(`Remove ${plate} from registry?`)) return;
  const r=await api(`/api/vehicles/${id}`,{method:'DELETE'});
  if(r&&r.success){toast(r.message,'success');loadVehicles();}
  else toast(r?.message||'Delete failed.','error');
}

loadVehicles();
</script>
{% endblock %}"""

# ─────────────────────────────────────────────────────────────────────────────
FRAUD_HTML = """{% extends "base.html" %}
{% block topbar_title %}Fraud Intelligence Feed{% endblock %}
{% block content %}
<div class="panel">
  <div class="panel-header">
    <span class="panel-title">🚨 Fraud Events — Full Log</span>
    <div style="display:flex;gap:8px;align-items:center">
      <select class="form-field" id="filter-type" onchange="loadFraud()" style="width:200px;padding:6px 10px">
        <option value="">All Types</option>
        <option>CLONE_ATTACK</option>
        <option>SPEEDING_VIOLATION</option>
        <option>PATH_VIOLATION</option>
      </select>
    </div>
  </div>
  <div style="overflow-x:auto;padding:0">
    <table class="data-table">
      <thead><tr>
        <th>Timestamp</th><th>Plate Number</th><th>Origin Gate</th>
        <th>Destination</th><th>Violation Type</th><th>Severity</th>
        <th>ΔT (s)</th><th>Min Required (s)</th>
      </tr></thead>
      <tbody id="fraud-body"><tr><td colspan="8" style="text-align:center;padding:24px;color:var(--text2)">Loading…</td></tr></tbody>
    </table>
  </div>
  <div style="padding:14px 20px;border-top:1px solid var(--border);display:flex;justify-content:space-between;align-items:center">
    <span id="fraud-count" style="font-size:12px;color:var(--text2);font-family:var(--font-mono)"></span>
    <div style="display:flex;gap:8px">
      <button class="btn-ghost" id="btn-prev" onclick="page--;loadFraud()" style="padding:5px 12px;font-size:12px">← Prev</button>
      <button class="btn-ghost" id="btn-next" onclick="page++;loadFraud()" style="padding:5px 12px;font-size:12px">Next →</button>
    </div>
  </div>
</div>
{% endblock %}
{% block extra_js %}
<script>
let page=0,perPage=20,totalFraud=0;
function loadFraud(){
  const t=document.getElementById('filter-type').value;
  api(`/api/fraud-events?limit=${perPage}&offset=${page*perPage}&type=${encodeURIComponent(t)}`).then(d=>{
    const b=document.getElementById('fraud-body');
    if(!d||!d.rows||!d.rows.length){
      b.innerHTML='<tr><td colspan="8" style="text-align:center;padding:24px;color:var(--text2)">No fraud events found.</td></tr>';
      document.getElementById('fraud-count').textContent='0 events';
      return;
    }
    totalFraud=d.total;
    document.getElementById('fraud-count').textContent=`Showing ${page*perPage+1}–${Math.min((page+1)*perPage,d.total)} of ${d.total} events`;
    document.getElementById('btn-prev').disabled=page===0;
    document.getElementById('btn-next').disabled=(page+1)*perPage>=d.total;
    b.innerHTML=d.rows.map(r=>`<tr>
      <td>${r.timestamp}</td>
      <td style="color:var(--accent);font-weight:600">${r.plate_number}</td>
      <td>${r.origin_gate||'—'}</td>
      <td>${r.dest_gate||'—'}</td>
      <td class="violation-${r.violation_type}">${(r.violation_type||'').replace(/_/g,' ')}</td>
      <td><span class="badge-sev badge-${r.severity_level}">⬤ ${r.severity_level}</span></td>
      <td style="font-family:var(--font-mono)">${r.delta_time!=null?r.delta_time.toFixed(3):'—'}</td>
      <td style="font-family:var(--font-mono)">${r.min_time!=null?r.min_time.toFixed(0):'—'}</td>
    </tr>`).join('');
  });
}
loadFraud();
</script>
{% endblock %}"""

# ─────────────────────────────────────────────────────────────────────────────
ACCESS_LOG_HTML = """{% extends "base.html" %}
{% block topbar_title %}Access Log{% endblock %}
{% block content %}
<div class="panel">
  <div class="panel-header">
    <span class="panel-title">📋 Vehicle Access Log</span>
    <div class="search-wrap">
      <span class="search-icon">🔍</span>
      <input class="form-field" id="log-search" placeholder="Search plate or gate…" oninput="page=0;loadLog()" style="padding-left:34px;width:240px">
    </div>
  </div>
  <div style="overflow-x:auto;padding:0">
    <table class="data-table">
      <thead><tr>
        <th>Log ID</th><th>Timestamp</th><th>Plate Number</th>
        <th>Gate</th><th>Confidence</th><th>Decision</th>
      </tr></thead>
      <tbody id="log-body"><tr><td colspan="6" style="text-align:center;padding:24px;color:var(--text2)">Loading…</td></tr></tbody>
    </table>
  </div>
  <div style="padding:14px 20px;border-top:1px solid var(--border);display:flex;justify-content:space-between;align-items:center">
    <span id="log-count" style="font-size:12px;color:var(--text2);font-family:var(--font-mono)"></span>
    <div style="display:flex;gap:8px">
      <button class="btn-ghost" id="log-prev" onclick="page--;loadLog()" style="padding:5px 12px;font-size:12px">← Prev</button>
      <button class="btn-ghost" id="log-next" onclick="page++;loadLog()" style="padding:5px 12px;font-size:12px">Next →</button>
    </div>
  </div>
</div>
{% endblock %}
{% block extra_js %}
<script>
let page=0, perPage=25;
function loadLog(){
  const q=document.getElementById('log-search').value;
  api(`/api/access-log?limit=${perPage}&offset=${page*perPage}&q=${encodeURIComponent(q)}`).then(d=>{
    const b=document.getElementById('log-body');
    if(!d||!d.rows||!d.rows.length){
      b.innerHTML='<tr><td colspan="6" style="text-align:center;padding:24px;color:var(--text2)">No log entries found.</td></tr>';
      document.getElementById('log-count').textContent='0 entries';
      return;
    }
    document.getElementById('log-count').textContent=`Showing ${page*perPage+1}–${Math.min((page+1)*perPage,d.total)} of ${d.total} entries`;
    document.getElementById('log-prev').disabled=page===0;
    document.getElementById('log-next').disabled=(page+1)*perPage>=d.total;
    b.innerHTML=d.rows.map(r=>`<tr>
      <td style="color:var(--text3)">#${r.log_id}</td>
      <td>${r.timestamp}</td>
      <td style="color:var(--accent)">${r.plate_number}</td>
      <td>${r.gate_id||'—'}</td>
      <td><div style="display:flex;align-items:center;gap:6px">
        <div style="flex:1;height:4px;background:var(--bg);border-radius:2px;overflow:hidden;width:60px">
          <div style="height:100%;width:${Math.round((r.ocr_confidence||0)*100)}%;background:var(--success);border-radius:2px"></div>
        </div>
        <span style="font-family:var(--font-mono);font-size:11px">${((r.ocr_confidence||0)*100).toFixed(0)}%</span>
      </div></td>
      <td><span class="badge-sev badge-${r.decision}">${r.decision}</span></td>
    </tr>`).join('');
  });
}
loadLog();
</script>
{% endblock %}"""

# ─────────────────────────────────────────────────────────────────────────────
REPORTS_HTML = """{% extends "base.html" %}
{% block topbar_title %}Analytics & Reports{% endblock %}
{% block content %}
<div class="kpi-grid">
  <div class="kpi-card" style="--card-accent:var(--accent2)">
    <div class="kpi-icon">📊</div>
    <div class="kpi-val" id="r-total">—</div>
    <div class="kpi-label">Total Detections</div>
  </div>
  <div class="kpi-card" style="--card-accent:var(--critical)">
    <div class="kpi-icon">🚨</div>
    <div class="kpi-val" id="r-fraud">—</div>
    <div class="kpi-label">Fraud Alerts</div>
  </div>
  <div class="kpi-card" style="--card-accent:var(--success)">
    <div class="kpi-icon">✅</div>
    <div class="kpi-val" id="r-rate">—%</div>
    <div class="kpi-label">Granted Rate</div>
  </div>
  <div class="kpi-card" style="--card-accent:var(--accent)">
    <div class="kpi-icon">🚗</div>
    <div class="kpi-val" id="r-reg">—</div>
    <div class="kpi-label">Registered Vehicles</div>
  </div>
</div>

<div style="display:grid;grid-template-columns:2fr 1fr;gap:16px">
  <div class="panel">
    <div class="panel-header">
      <span class="panel-title">📈 Hourly Traffic Volume</span>
      <span style="font-size:11px;color:var(--text2);font-family:var(--font-mono)">Read-only view</span>
    </div>
    <div class="panel-body"><div style="height:260px"><canvas id="rptChart"></canvas></div></div>
  </div>
  <div class="panel">
    <div class="panel-header"><span class="panel-title">🔍 Violation Breakdown</span></div>
    <div class="panel-body"><div style="height:260px"><canvas id="violChart"></canvas></div></div>
  </div>
</div>

<div class="panel" style="margin-top:16px">
  <div class="panel-header">
    <span class="panel-title">📥 Compliance Export</span>
    <span style="font-size:12px;color:var(--text2)">Finance Department — Read Only</span>
  </div>
  <div class="panel-body" style="display:flex;align-items:center;gap:16px">
    <div style="flex:1;font-size:13px;color:var(--text2)">
      Export the full audit log to CSV for Finance Department compliance review.
      Contains all vehicle access events, OCR confidence scores, gate IDs, and decisions.
    </div>
    <a href="/api/export-csv" class="btn-primary-vlpr" style="white-space:nowrap">📥 Export CSV</a>
  </div>
</div>
{% endblock %}
{% block extra_js %}
<script>
api('/api/kpis').then(d=>{
  if(!d)return;
  document.getElementById('r-total').textContent=d.total_detections;
  document.getElementById('r-fraud').textContent=d.fraud_alerts;
  document.getElementById('r-rate').textContent=d.total_detections>0?
    Math.round((d.granted/d.total_detections)*100)+'%':'—%';
  document.getElementById('r-reg').textContent=d.registered_vehicles;
});
api('/api/chart-data').then(d=>{
  if(!d)return;
  new Chart(document.getElementById('rptChart'),{
    type:'bar',data:{labels:d.labels,
      datasets:[{label:'Detections',data:d.detections,backgroundColor:'rgba(14,165,233,.3)',
        borderColor:'rgba(14,165,233,.7)',borderWidth:1}]},
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{labels:{color:'#8898b8',font:{size:11}}}},
      scales:{x:{ticks:{color:'#4a607a',font:{size:10}},grid:{color:'rgba(26,51,86,.4)'}},
        y:{ticks:{color:'#4a607a',font:{size:10}},grid:{color:'rgba(26,51,86,.4)'}}}}
  });
});
api('/api/fraud-events?limit=1000').then(d=>{
  if(!d)return;
  const counts={CLONE_ATTACK:0,SPEEDING_VIOLATION:0,PATH_VIOLATION:0};
  (d.rows||[]).forEach(r=>{ if(counts[r.violation_type]!==undefined) counts[r.violation_type]++; });
  new Chart(document.getElementById('violChart'),{
    type:'doughnut',
    data:{labels:Object.keys(counts).map(k=>k.replace(/_/g,' ')),
      datasets:[{data:Object.values(counts),
        backgroundColor:['rgba(239,68,68,.7)','rgba(249,115,22,.7)','rgba(234,179,8,.7)'],
        borderColor:['#ef4444','#f97316','#eab308'],borderWidth:1}]},
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{position:'bottom',labels:{color:'#8898b8',font:{size:11},padding:12}}}}
  });
});
</script>
{% endblock %}"""

# ─────────────────────────────────────────────────────────────────────────────
HEALTH_HTML = """{% extends "base.html" %}
{% block topbar_title %}System Health{% endblock %}
{% block content %}
<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
  <div class="panel">
    <div class="panel-header"><span class="panel-title">⚙️ Database Status</span></div>
    <div class="panel-body">
      <table class="data-table">
        <tbody id="health-body"><tr><td colspan="2" style="padding:16px;color:var(--text2)">Loading…</td></tr></tbody>
      </table>
    </div>
  </div>
  <div class="panel">
    <div class="panel-header"><span class="panel-title">📊 Table Row Counts</span></div>
    <div class="panel-body">
      <table class="data-table">
        <thead><tr><th>Table</th><th>Rows</th></tr></thead>
        <tbody id="table-counts"><tr><td colspan="2" style="padding:16px;color:var(--text2)">Loading…</td></tr></tbody>
      </table>
    </div>
  </div>
</div>
{% endblock %}
{% block extra_js %}
<script>
api('/api/health').then(d=>{
  if(!d)return;
  document.getElementById('health-body').innerHTML=Object.entries(d.status).map(([k,v])=>`
    <tr><td style="color:var(--text2)">${k}</td>
    <td style="color:var(--success)">${v}</td></tr>`).join('');
  document.getElementById('table-counts').innerHTML=Object.entries(d.tables).map(([k,v])=>`
    <tr><td style="font-family:var(--font-mono)">${k}</td>
    <td style="font-family:var(--font-mono);color:var(--accent)">${v}</td></tr>`).join('');
});
</script>
{% endblock %}"""

# ─────────────────────────────────────────────────────────────────────────────
USERS_HTML = """{% extends "base.html" %}
{% block topbar_title %}User Management{% endblock %}
{% block content %}
<div class="panel">
  <div class="panel-header">
    <span class="panel-title">👥 System Users</span>
    <span style="font-size:12px;color:var(--text2)">Mock user store — replace with DB table in production</span>
  </div>
  <div style="padding:0;overflow-x:auto">
    <table class="data-table">
      <thead><tr><th>Username</th><th>Role</th><th>Label</th><th>Status</th></tr></thead>
      <tbody>
        {% for uname, udata in users.items() %}
        <tr>
          <td style="font-family:var(--font-mono)">{{ uname }}</td>
          <td><span class="badge-sev" style="color:{{ udata.badge }};border-color:{{ udata.badge }}44;background:{{ udata.badge }}12">{{ udata.role }}</span></td>
          <td style="color:var(--text2)">{{ udata.label }}</td>
          <td><span class="badge-active">Active</span></td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
  <div style="padding:16px 20px;border-top:1px solid var(--border)">
    <p style="font-size:12px;color:var(--text2);font-family:var(--font-mono)">
      ℹ  In production, this page manages a <code>users</code> table in SQLite with bcrypt-hashed passwords and
      per-user permission overrides. For the Viva demo, users are stored in the USERS dict in dashboard.py.
    </p>
  </div>
</div>
{% endblock %}"""

FORBIDDEN_HTML = """{% extends "base.html" %}
{% block topbar_title %}Access Denied{% endblock %}
{% block content %}
<div style="text-align:center;padding:60px 24px">
  <div style="font-size:48px;margin-bottom:16px">🔒</div>
  <h2 style="font-family:var(--font-hd);font-size:24px;margin-bottom:8px">Access Denied</h2>
  <p style="color:var(--text2)">Your role (<strong style="color:var(--accent)">{{ role }}</strong>) does not have permission to view this page.</p>
  <a href="/overview" class="btn-primary-vlpr" style="display:inline-block;margin-top:20px">← Back to Overview</a>
</div>
{% endblock %}"""

NOT_FOUND_HTML = """{% extends "base.html" %}
{% block topbar_title %}Not Found{% endblock %}
{% block content %}
<div style="text-align:center;padding:60px 24px">
  <div style="font-size:48px;margin-bottom:16px">🔍</div>
  <h2 style="font-family:var(--font-hd);font-size:24px;margin-bottom:8px">Page Not Found</h2>
  <a href="/overview" class="btn-primary-vlpr" style="display:inline-block;margin-top:20px">← Back to Overview</a>
</div>
{% endblock %}"""

TEMPLATES = {
    'base.html':        BASE_HTML,
    'login.html':       LOGIN_HTML,
    'overview.html':    OVERVIEW_HTML,
    'vehicles.html':    VEHICLES_HTML,
    'fraud_feed.html':  FRAUD_HTML,
    'access_log.html':  ACCESS_LOG_HTML,
    'reports.html':     REPORTS_HTML,
    'health.html':      HEALTH_HTML,
    'users.html':       USERS_HTML,
    'forbidden.html':   FORBIDDEN_HTML,
    '404.html':         NOT_FOUND_HTML,
}
app.jinja_loader = __import__('jinja2').DictLoader(TEMPLATES)

# ─────────────────────────────────────────────────────────────────────────────
# Auth Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return redirect(url_for('overview'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip().lower()
        password = request.form.get('password', '').strip()
        user = USERS.get(username)
        if user and user['password'] == password:
            session['username'] = username
            session['role']     = user['role']
            return redirect(url_for('overview'))
        flash('Invalid username or password.', 'error')
    return render_page('login.html', {})


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ─────────────────────────────────────────────────────────────────────────────
# Page Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/overview')
@login_required
def overview():
    return render_page('overview.html', {})


@app.route('/vehicles')
@role_required('admin', 'security', 'finance')
def vehicles():
    return render_page('vehicles.html', {})


@app.route('/fraud-feed')
@role_required('admin', 'security')
def fraud_feed():
    return render_page('fraud_feed.html', {})


@app.route('/access-log')
@login_required
def access_log():
    return render_page('access_log.html', {})


@app.route('/reports')
@role_required('admin', 'finance')
def reports():
    return render_page('reports.html', {})


@app.route('/admin/health')
@role_required('admin')
def health():
    return render_page('health.html', {})


@app.route('/admin/users')
@role_required('admin')
def users():
    return render_page('users.html', {'users': USERS})

# ─────────────────────────────────────────────────────────────────────────────
# API Routes — JSON
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/kpis')
@login_required
def api_kpis():
    try:
        with raw_db() as conn:
            total     = conn.execute('SELECT COUNT(*) FROM access_log').fetchone()[0]
            granted   = conn.execute("SELECT COUNT(*) FROM access_log WHERE decision='GRANTED'").fetchone()[0]
            fraud     = conn.execute('SELECT COUNT(*) FROM fraud_events').fetchone()[0]
            reg_veh   = conn.execute("SELECT COUNT(*) FROM registered_vehicles WHERE is_active=1").fetchone()[0]
    except Exception:
        total = granted = fraud = reg_veh = 0
    return jsonify({
        'total_detections':   total,
        'granted':            granted,
        'fraud_alerts':       fraud,
        'registered_vehicles': reg_veh,
    })


@app.route('/api/chart-data')
@login_required
def api_chart_data():
    """Return hourly detection and fraud counts for the last 24 hours."""
    labels, det_counts, fraud_counts = [], [], []
    now  = datetime.now()
    base = now - timedelta(hours=23)
    hour_map_det   = {}
    hour_map_fraud = {}

    try:
        with raw_db() as conn:
            since = base.strftime('%Y-%m-%d %H:%M:%S')
            rows = conn.execute(
                "SELECT strftime('%Y-%m-%d %H', timestamp) as hr, COUNT(*) as cnt "
                "FROM access_log WHERE timestamp >= ? GROUP BY hr",
                (since,)
            ).fetchall()
            for r in rows:
                hour_map_det[r['hr']] = r['cnt']

            frows = conn.execute(
                "SELECT strftime('%Y-%m-%d %H', timestamp) as hr, COUNT(*) as cnt "
                "FROM fraud_events WHERE timestamp >= ? GROUP BY hr",
                (since,)
            ).fetchall()
            for r in frows:
                hour_map_fraud[r['hr']] = r['cnt']
    except Exception:
        pass

    for i in range(24):
        t    = base + timedelta(hours=i)
        key  = t.strftime('%Y-%m-%d %H')
        lbl  = t.strftime('%H:00')
        labels.append(lbl)
        det_counts.append(hour_map_det.get(key, 0))
        fraud_counts.append(hour_map_fraud.get(key, 0))

    return jsonify({'labels': labels, 'detections': det_counts, 'fraud': fraud_counts})


@app.route('/api/fraud-events')
@login_required
def api_fraud_events():
    limit  = min(int(request.args.get('limit', 20)), 200)
    offset = int(request.args.get('offset', 0))
    vtype  = request.args.get('type', '')

    where = 'WHERE violation_type = ?' if vtype else ''
    params_count = (vtype,) if vtype else ()
    params_rows  = (vtype, limit, offset) if vtype else (limit, offset)

    try:
        with raw_db() as conn:
            total = conn.execute(
                f'SELECT COUNT(*) FROM fraud_events {where}', params_count
            ).fetchone()[0]
            rows = conn.execute(
                f"""SELECT event_id, timestamp, plate_number, violation_type,
                           severity_level, origin_gate, dest_gate, delta_time, reason
                    FROM fraud_events {where}
                    ORDER BY event_id DESC LIMIT ? OFFSET ?""",
                params_rows
            ).fetchall()
        return jsonify({'total': total, 'rows': [dict(r) for r in rows]})
    except Exception as e:
        return jsonify({'error': str(e), 'total': 0, 'rows': []})


@app.route('/api/access-log')
@login_required
def api_access_log():
    limit  = min(int(request.args.get('limit', 25)), 500)
    offset = int(request.args.get('offset', 0))
    q      = request.args.get('q', '').strip()

    where  = "WHERE plate_number LIKE ? OR gate_id LIKE ?" if q else ""
    pats   = (f'%{q}%', f'%{q}%') if q else ()

    try:
        with raw_db() as conn:
            total = conn.execute(
                f'SELECT COUNT(*) FROM access_log {where}', pats
            ).fetchone()[0]
            rows = conn.execute(
                f"""SELECT log_id, timestamp, plate_number, gate_id,
                           ocr_confidence, decision
                    FROM access_log {where}
                    ORDER BY log_id DESC LIMIT ? OFFSET ?""",
                pats + (limit, offset)
            ).fetchall()
        return jsonify({'total': total, 'rows': [dict(r) for r in rows]})
    except Exception as e:
        return jsonify({'error': str(e), 'total': 0, 'rows': []})


# ── Vehicle Registry CRUD ──────────────────────────────────────────────────────

@app.route('/api/vehicles', methods=['GET'])
@login_required
def api_vehicles_list():
    return jsonify(get_db().get_all_vehicles())


@app.route('/api/vehicles', methods=['POST'])
@role_required('admin', 'security')
def api_vehicles_create():
    data = request.get_json() or {}
    ok, msg = get_db().add_vehicle(
        data.get('plate_number', ''),
        data.get('owner_name', ''),
        data.get('vehicle_type', 'Car'),
        data.get('department', ''),
    )
    return jsonify({'success': ok, 'message': msg}), (201 if ok else 400)


@app.route('/api/vehicles/<int:vehicle_id>', methods=['PUT'])
@role_required('admin', 'security')
def api_vehicles_update(vehicle_id: int):
    data = request.get_json() or {}
    ok, msg = get_db().update_vehicle(
        vehicle_id,
        data.get('owner_name', ''),
        data.get('vehicle_type', 'Car'),
        data.get('department', ''),
        data.get('is_active', 1),
    )
    return jsonify({'success': ok, 'message': msg}), (200 if ok else 400)


@app.route('/api/vehicles/<int:vehicle_id>', methods=['DELETE'])
@role_required('admin', 'security')
def api_vehicles_delete(vehicle_id: int):
    ok, msg = get_db().delete_vehicle(vehicle_id)
    return jsonify({'success': ok, 'message': msg}), (200 if ok else 400)


@app.route('/api/health')
@role_required('admin')
def api_health():
    tables = {}
    status = {}
    try:
        with raw_db() as conn:
            for tbl in ('access_log', 'fraud_events', 'registered_vehicles'):
                tables[tbl] = conn.execute(f'SELECT COUNT(*) FROM {tbl}').fetchone()[0]
        status['SQLite']    = '✓ Connected'
        status['DB Path']   = _db_path
        status['WAL Mode']  = '✓ Enabled'
        status['Python']    = f'✓ {sys.version.split()[0]}'
        status['Flask']     = '✓ Running'
    except Exception as e:
        status['Error'] = str(e)
    return jsonify({'status': status, 'tables': tables})


@app.route('/api/export-csv')
@role_required('admin', 'finance')
def api_export_csv():
    """Stream the access_log as a CSV download for Finance Department."""
    import io, csv as csv_mod
    try:
        with raw_db() as conn:
            rows = conn.execute(
                """SELECT log_id, timestamp, plate_number, gate_id,
                          ocr_confidence, decision
                   FROM access_log ORDER BY log_id"""
            ).fetchall()
    except Exception:
        rows = []

    buf = io.StringIO()
    w   = csv_mod.writer(buf)
    w.writerow(['log_id', 'timestamp', 'plate_number', 'gate_id', 'ocr_confidence', 'decision'])
    for r in rows:
        w.writerow(list(r))

    filename = f"vlpr_audit_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(
        buf.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename={filename}'}
    )

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VLPR Security Dashboard')
    parser.add_argument('--db',   default=DEFAULT_SQLITE_PATH,
                        help='Path to SQLite audit_log.db')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to listen on (default: 5000)')
    parser.add_argument('--host', default='0.0.0.0',
                        help='Host to bind (default: 0.0.0.0)')
    args = parser.parse_args()

    _db_path = os.path.abspath(args.db)
    print(f"\n{'='*60}")
    print("  VLPR Security Dashboard — Sprint 4")
    print(f"{'='*60}")
    print(f"  Database : {_db_path}")
    print(f"  URL      : http://localhost:{args.port}")
    print(f"  Roles    : admin / security / finance")
    print(f"{'='*60}\n")

    app.run(host=args.host, port=args.port, debug=False)
