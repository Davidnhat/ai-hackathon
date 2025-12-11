import os
import re
import json
import base64
import requests
import pandas as pd
import gradio as gr

API_BASE_DEFAULT = os.getenv("API_BASE", "http://localhost:8000")
MODULES = ["Parsing", "Resume Analysis", "Matching", "CV Scoring", "Job Recommendation", "Playground"]

# ---------- utils ----------
def img_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def safe_dict(x):
    if x is None:
        return None
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return {"raw": x}
    return {"raw": str(x)}

def as_df(rows):
    if rows is None:
        return pd.DataFrame()
    if isinstance(rows, dict):
        return pd.DataFrame([rows])
    if isinstance(rows, list):
        return pd.DataFrame(rows) if rows and isinstance(rows[0], dict) else pd.DataFrame({"value": rows})
    return pd.DataFrame()

def pills(items):
    if not items:
        return "<div style='color:#666'>—</div>"
    html = ""
    for s in items[:40]:
        html += (
            "<span style='display:inline-block;border:1px solid #e5e7eb;"
            "border-radius:999px;padding:2px 10px;margin:3px 6px 0 0;font-size:12px;'>"
            f"{s}</span>"
        )
    return html

def highlight_text(text, keywords):
    if not text:
        return "<div style='color:#666'>—</div>"
    if not keywords:
        return f"<pre style='white-space:pre-wrap;line-height:1.45'>{text}</pre>"
    kws = [re.escape(k) for k in keywords[:30] if isinstance(k, str) and k.strip()]
    if not kws:
        return f"<pre style='white-space:pre-wrap;line-height:1.45'>{text}</pre>"
    pattern = re.compile("(" + "|".join(kws) + ")", flags=re.IGNORECASE)
    safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    highlighted = pattern.sub(
        r"<mark style='background:#22c55e33;padding:0 2px;border-radius:4px;'>\1</mark>",
        safe,
    )
    return (
        "<div style='border:1px solid #e5e7eb;border-radius:14px;padding:12px;max-height:360px;overflow:auto;'>"
        f"<div style='white-space:pre-wrap;line-height:1.45'>{highlighted}</div></div>"
    )

def html_progress(score_0_100):
    try:
        s = float(score_0_100)
    except Exception:
        s = 0.0
    s = max(0.0, min(100.0, s))
    return f"""
    <div style="width:100%; background:#222; border:1px solid #333; border-radius:999px; overflow:hidden; height:14px;">
      <div style="width:{s}%; background:#f97316; height:14px;"></div>
    </div>
    <div style="margin-top:6px; font-weight:800;">{s:.0f}/100</div>
    """

def html_stars(rating_0_5):
    try:
        r = float(rating_0_5)
    except Exception:
        r = 0.0
    r = max(0.0, min(5.0, r))
    full = int(round(r))
    stars = "★" * full + "☆" * (5 - full)
    return f"<div style='font-size:20px; letter-spacing:1px;'>{stars} <span style='font-size:13px;color:#aaa'>({r:.1f}/5)</span></div>"

def score_circle_html(score: int):
    score = 0 if score is None else int(score)
    pct = max(0, min(100, score))
    label = "Excellent" if pct >= 85 else "Good" if pct >= 70 else "OK" if pct >= 50 else "Needs work"
    return f"""
    <div style="display:flex;gap:16px;align-items:center;">
      <div style="
        width:120px;height:120px;border-radius:999px;
        background: conic-gradient(#f97316 {pct}%, #2a2a2a 0);
        display:flex;align-items:center;justify-content:center;
      ">
        <div style="
          width:92px;height:92px;border-radius:999px;background:#111;
          display:flex;flex-direction:column;align-items:center;justify-content:center;
          border:1px solid #2a2a2a;
        ">
          <div style="font-size:28px;font-weight:900;line-height:1">{pct}</div>
          <div style="font-size:12px;color:#9ca3af;margin-top:2px">/ 100</div>
        </div>
      </div>
      <div>
        <div style="font-size:16px;font-weight:800;">{label}</div>
        <div style="font-size:12px;color:#9ca3af;">Tổng quan chất lượng CV (demo scoring)</div>
      </div>
    </div>
    """

def checklist_df(items):
    if not items:
        return pd.DataFrame(columns=["Item", "Status", "Detail"])
    rows = []
    for it in items:
        rows.append({
            "Item": it.get("item", ""),
            "Status": "✅" if it.get("ok") else "❌",
            "Detail": it.get("detail", "")
        })
    return pd.DataFrame(rows)


# ---------- API (Thong's backend) ----------
def api_parse_cv(api_base, pdf_path, timeout=120):
    with open(pdf_path, "rb") as f:
        r = requests.post(
            f"{api_base}/parse/cv",
            files={"file": ("cv.pdf", f, "application/pdf")},
            timeout=timeout,
        )
    r.raise_for_status()
    return r.json()

def api_match(api_base, cv_json, jd_text, timeout=120):
    r = requests.post(f"{api_base}/match", json={"cv": cv_json, "jd_text": jd_text}, timeout=timeout)
    r.raise_for_status()
    return r.json()

def api_score_cv(api_base, cv_json, timeout=120):
    r = requests.post(f"{api_base}/score/cv", json={"cv": cv_json}, timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"Score API HTTP {r.status_code}: {r.text[:800]}")
    return r.json()

def api_recommend_jobs(api_base, cv_json, top_k=5, timeout=120):
    r = requests.post(f"{api_base}/recommend/jobs", json={"cv": cv_json, "top_k": top_k}, timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"Recommend API HTTP {r.status_code}: {r.text[:800]}")
    return r.json()


# ---------- mock ----------
def mock_cv():
    return {
        "name": "Demo Candidate",
        "email": "demo@mail.com",
        "phone": "090xxxxxxx",
        "skills": ["Python", "SQL", "FastAPI", "NLP", "Docker"],
        "experience": [
            {"company": "DemoCo", "role": "Backend Engineer", "years": 2, "summary": "Built APIs with FastAPI"},
            {"company": "OtherCo", "role": "Intern", "years": 1, "summary": "Data cleaning & reporting"},
        ],
        "education": [
            {"school": "HCM University", "major": "Computer Science", "year": 2023}
        ],
        "raw_text": "Experienced in Python, SQL, FastAPI. Built REST APIs. Familiar with Docker and NLP basics."
    }

def mock_match():
    return {
        "score_0_100": 78,
        "rating_0_5": 3.9,
        "explanation": "Strong Python/SQL/FastAPI. Missing Kubernetes/CI-CD. Overall good fit for backend role.",
        "skills_overlap": ["Python", "SQL", "FastAPI"],
        "skills_missing": ["Kubernetes", "CI/CD"],
    }

def mock_cv_score(cv: dict):
    skills = cv.get("skills", []) or []
    exp = cv.get("experience", []) or []
    edu = cv.get("education", []) or []
    email_ok = bool(cv.get("email"))
    phone_ok = bool(cv.get("phone"))
    raw_ok = bool((cv.get("raw_text") or "").strip())

    score = 0
    score += 10 if email_ok else 0
    score += 10 if phone_ok else 0
    score += min(25, len(skills) * 3)
    score += min(30, len(exp) * 12)
    score += 10 if len(edu) > 0 else 0
    score += 15 if raw_ok else 0
    score = max(0, min(100, int(score)))

    checklist = [
        {"item": "Có email", "ok": email_ok, "detail": cv.get("email", "")},
        {"item": "Có số điện thoại", "ok": phone_ok, "detail": cv.get("phone", "")},
        {"item": "Có danh sách skills", "ok": len(skills) > 0, "detail": f"{len(skills)} skills"},
        {"item": "Có kinh nghiệm (experience)", "ok": len(exp) > 0, "detail": f"{len(exp)} mục"},
        {"item": "Có học vấn (education)", "ok": len(edu) > 0, "detail": f"{len(edu)} mục"},
        {"item": "Có raw_text để highlight/QA", "ok": raw_ok, "detail": f"{len(cv.get('raw_text',''))} chars"},
    ]

    suggestions = []
    if not email_ok: suggestions.append("Bổ sung email rõ ràng ở phần thông tin liên hệ.")
    if not phone_ok: suggestions.append("Bổ sung số điện thoại để HR liên hệ nhanh.")
    if len(skills) < 6: suggestions.append("Bổ sung thêm kỹ năng cứng/mềm liên quan (>= 6).")
    if len(exp) == 0: suggestions.append("Thêm kinh nghiệm dự án/việc làm (ít nhất 1 mục, nêu kết quả/impact).")
    if len(edu) == 0: suggestions.append("Thêm học vấn/chứng chỉ.")
    if not raw_ok: suggestions.append("CV nên có nội dung text rõ (tránh ảnh scan mờ).")
    if not suggestions:
        suggestions.append("CV khá đầy đủ. Tối ưu thêm bằng cách nêu định lượng kết quả và liên kết portfolio/GitHub.")

    return {
        "score": score,
        "rating_0_5": round(min(5.0, max(0.0, score / 20.0)), 1),
        "checklist": checklist,
        "suggestions": suggestions,
        "summary": "CV Scoring (mock) dựa trên đủ thông tin + skills + experience."
    }

def mock_job_recommendations(cv: dict):
    skills = cv.get("skills", []) or []
    base_title = "Backend Engineer (Python)"
    jobs = [
        {
            "id": "JD-001",
            "title": base_title,
            "company": "SaaSCo",
            "location": "HCMC",
            "score": 92,
            "top_skills": ["Python", "FastAPI", "SQL"],
            "jd_snippet": "Build REST APIs, integrate databases, work with FastAPI and Docker.",
            "match_reason": "Khớp mạnh với kinh nghiệm Python/SQL/FastAPI hiện tại của bạn."
        },
        {
            "id": "JD-002",
            "title": "Data Engineer (Python/SQL)",
            "company": "DataCorp",
            "location": "Remote",
            "score": 80,
            "top_skills": ["Python", "SQL", "ETL"],
            "jd_snippet": "Thiết kế pipeline ETL, tối ưu truy vấn SQL, xử lý dữ liệu lớn.",
            "match_reason": "Phù hợp nếu bạn muốn chuyển hướng sang data pipeline, vẫn dùng Python/SQL."
        },
        {
            "id": "JD-003",
            "title": "ML Engineer (NLP)",
            "company": "AI Labs",
            "location": "Hanoi / HCMC",
            "score": 70,
            "top_skills": ["Python", "NLP", "ML"],
            "jd_snippet": "Xây dựng, triển khai mô hình NLP, làm việc với backend inference.",
            "match_reason": "Liên quan tới NLP bạn có đề cập; cần bổ sung thêm kinh nghiệm ML production."
        },
    ]
    for j in jobs:
        overlap = len(set(j["top_skills"]) & set(skills))
        j["score"] = min(100, j["score"] + overlap * 2)
    return {"jobs": jobs}


# ---------- styling ----------
CSS = """
#header {display:flex; align-items:center; gap:12px; padding:8px 2px 2px;}
#title {font-size:18px;font-weight:900;}
#sub {color:#6b7280;font-size:12px;margin-top:-2px;}
"""

LOGO_PATH = "logo.png"
try:
    LOGO_B64 = img_b64(LOGO_PATH)
    LOGO_HTML = f"<img src='data:image/png;base64,{LOGO_B64}' style='width:44px;height:44px;object-fit:contain;'/>"
except Exception:
    LOGO_HTML = "<div style='width:44px;height:44px;border-radius:12px;background:#16a34a;color:#fff;display:flex;align-items:center;justify-content:center;font-weight:900;'>F</div>"


with gr.Blocks(css=CSS, title="Fram^ AI Recruiter Assistant") as demo:
    gr.HTML(f"""
    <div id="header">
      <div style="width:44px;height:44px;border-radius:12px;overflow:hidden;display:flex;align-items:center;justify-content:center;background:#111;">
        {LOGO_HTML}
      </div>
      <div>
        <div id="title">Fram^ AI Recruiter Assistant</div>
        <div id="sub">Upload CV/JD → Resume Analysis → CV ↔ JD Matching → CV Scoring → Job Recommendation → Playground</div>
      </div>
    </div>
    """)

    # controls chung
    with gr.Row():
        api_base = gr.Textbox(label="API Base URL", value=API_BASE_DEFAULT, scale=3)
        mock_mode = gr.Checkbox(label="Mock mode (backend chưa xong vẫn demo)", value=True, scale=1)

    # state chia sẻ
    cv_state = gr.State(None)
    jd_state = gr.State("")
    jobs_state = gr.State({})

    # ====================== TABS ======================
    with gr.Tabs():

        # -------- Tab 1: Upload CV / JD --------
        with gr.Tab("1) Upload CV / JD"):
            with gr.Row():
                with gr.Column(scale=1):
                    cv_pdf = gr.File(label="Upload CV (PDF)", file_types=[".pdf"], type="filepath")
                    jd_text = gr.Textbox(label="Dán JD (text)", lines=8, placeholder="(Optional) dán JD...")
                    module = gr.Dropdown(label="Chọn module", choices=MODULES, value="Parsing")
                    with gr.Row():
                        btn_parse = gr.Button("Parse CV → Resume Analysis", variant="primary")
                        btn_clear = gr.Button("Xoá hết")
                with gr.Column(scale=1):
                    preview = gr.JSON(label="Preview (đầu vào đã nhận)")

            def summarize_inputs(module, cv_pdf_path, jd_text_val):
                has_cv = bool(cv_pdf_path)
                jd_len = len(jd_text_val.strip()) if jd_text_val else 0
                return {
                    "selected_module": module,
                    "cv_uploaded": has_cv,
                    "cv_file_path": cv_pdf_path if has_cv else None,
                    "jd_provided": jd_len > 0,
                    "jd_characters": jd_len,
                }

            def clear_all():
                return None, "", "Parsing", {}, None, ""

            def do_parse(api, pdf_path, use_mock, jd_txt):
                if use_mock:
                    cv = mock_cv()
                    return cv, cv, jd_txt or ""
                if not pdf_path:
                    return {"error": "Chưa upload CV PDF"}, None, jd_txt or ""
                cv = api_parse_cv(api, pdf_path)
                return cv, cv, jd_txt or ""

            cv_pdf.change(summarize_inputs, inputs=[module, cv_pdf, jd_text], outputs=[preview])
            jd_text.change(summarize_inputs, inputs=[module, cv_pdf, jd_text], outputs=[preview])
            module.change(summarize_inputs, inputs=[module, cv_pdf, jd_text], outputs=[preview])

            btn_parse.click(
                do_parse,
                inputs=[api_base, cv_pdf, mock_mode, jd_text],
                outputs=[preview, cv_state, jd_state]
            )
            btn_clear.click(
                clear_all,
                inputs=[],
                outputs=[cv_pdf, jd_text, module, preview, cv_state, jd_state]
            )

        # -------- Tab 2: Resume Analysis --------
        with gr.Tab("2) Resume Analysis"):
            gr.Markdown("### Resume Analysis")
            gr.Markdown("Hiển thị **JSON Viewer + Table View + Highlighted CV Content + Skill Tag Chips**")

            with gr.Row():
                with gr.Column(scale=1):
                    cv_json_view = gr.JSON(label="CV JSON")
                    btn_refresh = gr.Button("Refresh từ CV state")
                with gr.Column(scale=1):
                    skills_html = gr.HTML(label="Skill Tag Chips")
                    highlight_html = gr.HTML(label="Highlighted CV Content")

            with gr.Accordion("Table View (Experience / Education)", open=True):
                exp_table = gr.Dataframe(label="Experience", interactive=False)
                edu_table = gr.Dataframe(label="Education", interactive=False)

            def render_resume(cv_json):
                cv = safe_dict(cv_json)
                if not cv or "error" in cv:
                    return cv or {}, "<div style='color:#666'>Chưa có CV JSON</div>", "<div style='color:#666'>—</div>", pd.DataFrame(), pd.DataFrame()

                skills = cv.get("skills", [])
                raw = cv.get("raw_text", "")
                exp = cv.get("experience", [])
                edu = cv.get("education", [])

                return (
                    cv,
                    pills(skills),
                    highlight_text(raw, skills),
                    as_df(exp),
                    as_df(edu),
                )

            btn_refresh.click(render_resume, inputs=[cv_state], outputs=[cv_json_view, skills_html, highlight_html, exp_table, edu_table])
            cv_state.change(render_resume, inputs=[cv_state], outputs=[cv_json_view, skills_html, highlight_html, exp_table, edu_table])

        # -------- Tab 3: CV ↔ JD Matching --------
        with gr.Tab("3) CV ↔ JD Matching"):
            gr.Markdown("### CV ↔ JD Matching")
            gr.Markdown("Hiển thị **Progress Bar + Star Rating + Matching Explanation + Skills Comparison Table**")

            with gr.Row():
                with gr.Column(scale=1):
                    jd_text_match = gr.Textbox(label="Dán JD (text)", lines=10, placeholder="Dán Job Description vào đây...")
                    with gr.Row():
                        btn_copy_jd = gr.Button("Copy JD từ tab Upload", size="sm")
                        btn_run_match = gr.Button("Run Matching", variant="primary")

                with gr.Column(scale=1):
                    match_progress = gr.HTML(label="Matching Score (Progress)")
                    match_stars = gr.HTML(label="Star Rating")
                    match_expl = gr.Textbox(label="Matching Explanation", lines=6)
                    match_table = gr.Dataframe(label="Skills Comparison Table", interactive=False)
                    match_raw = gr.JSON(label="Raw Match JSON")

            btn_copy_jd.click(lambda x: x or "", inputs=[jd_state], outputs=[jd_text_match])

            def do_match(api, use_mock, cv_json, jd_txt):
                if cv_json is None:
                    return "", "", "Chưa có CV. Qua tab 1 bấm Parse CV trước.", pd.DataFrame(), {"error": "no_cv"}
                if not jd_txt or not jd_txt.strip():
                    return "", "", "Chưa có JD text.", pd.DataFrame(), {"error": "no_jd"}

                out = mock_match() if use_mock else api_match(api, cv_json, jd_txt)

                score = out.get("score_0_100", 0)
                rating = out.get("rating_0_5", float(score) / 20 if score is not None else 0)
                expl = out.get("explanation", "—")

                overlap = out.get("skills_overlap", []) or []
                missing = out.get("skills_missing", []) or []
                n = max(len(overlap), len(missing), 1)
                df = pd.DataFrame({
                    "Overlap skills": overlap + [""] * (n - len(overlap)),
                    "Missing skills": missing + [""] * (n - len(missing)),
                })

                return html_progress(score), html_stars(rating), expl, df, out

            btn_run_match.click(
                do_match,
                inputs=[api_base, mock_mode, cv_state, jd_text_match],
                outputs=[match_progress, match_stars, match_expl, match_table, match_raw]
            )

        # -------- Tab 4: CV Scoring --------
        with gr.Tab("4) CV Scoring"):
            gr.Markdown("### CV Scoring")
            gr.Markdown("Hiển thị **Score Circle/Progress + Checklist + Feedback Suggestion**")

            with gr.Row():
                with gr.Column(scale=1):
                    score_html = gr.HTML()
                    score_bar = gr.Slider(label="Score (0-100)", minimum=0, maximum=100, value=0, interactive=False)
                    score_stars = gr.HTML(label="Star Rating")
                    btn_score = gr.Button("Run CV Scoring", variant="primary")
                with gr.Column(scale=1):
                    score_json = gr.JSON(label="Raw CV Score JSON")

            with gr.Accordion("Checklist", open=True):
                checklist_table = gr.Dataframe(label="Checklist", interactive=False)

            with gr.Accordion("Feedback Suggestion", open=True):
                suggestions_md = gr.Markdown()

            def do_score(api, use_mock, cv_json):
                cv = safe_dict(cv_json)
                if not cv:
                    empty = {"error": "Chưa có CV. Hãy Parse CV ở tab 1 trước."}
                    return score_circle_html(0), 0, html_stars(0), checklist_df([]), "⚠️ Chưa có CV state.", empty

                try:
                    result = mock_cv_score(cv) if use_mock else api_score_cv(api, cv)
                except Exception as e:
                    result = {"error": str(e), "hint": "Backend /score/cv chưa chạy hoặc sai endpoint."}

                if "error" in result:
                    return score_circle_html(0), 0, html_stars(0), checklist_df([]), f"⚠️ {result['error']}", result

                s = int(result.get("score", 0))
                r = result.get("rating_0_5", round(s / 20.0, 1))
                ck = result.get("checklist", [])
                sug = result.get("suggestions", [])

                sug_md = "\n".join([f"- {x}" for x in sug]) if sug else "- —"
                return score_circle_html(s), s, html_stars(r), checklist_df(ck), sug_md, result

            btn_score.click(
                do_score,
                inputs=[api_base, mock_mode, cv_state],
                outputs=[score_html, score_bar, score_stars, checklist_table, suggestions_md, score_json]
            )

            cv_state.change(
                do_score,
                inputs=[api_base, mock_mode, cv_state],
                outputs=[score_html, score_bar, score_stars, checklist_table, suggestions_md, score_json]
            )

        # -------- Tab 5: Job Recommendation --------
        with gr.Tab("5) Job Recommendation"):
            gr.Markdown("### Job Recommendation")
            gr.Markdown(
                "Gợi ý các JD phù hợp dựa trên CV đã upload. "
                "Hiển thị **Card/Table danh sách JD, Matching Score, nút “Xem chi tiết JD”**."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    btn_reco = gr.Button("Generate Job Recommendations", variant="primary")
                    jobs_table = gr.Dataframe(label="Danh sách JD gợi ý", interactive=False)
                    jobs_msg = gr.Markdown()
                    jobs_select = gr.Dropdown(label="Chọn JD để xem chi tiết", choices=[], value=None)
                    btn_view = gr.Button("Xem chi tiết JD")
                with gr.Column(scale=1):
                    jd_score_html = gr.HTML(label="JD Matching Score")
                    jd_detail_md = gr.Markdown(label="JD Detail")
                    jobs_raw = gr.JSON(label="Raw Recommended Jobs JSON")

            def do_recommend(api, use_mock, cv_json):
                cv = safe_dict(cv_json)
                empty_df = pd.DataFrame(
                    columns=["ID", "Title", "Company", "Location", "Score", "Top skills"]
                )

                if not cv:
                    msg = "⚠️ Chưa có CV. Hãy Parse CV ở tab 1 trước."
                    return empty_df, gr.update(choices=[], value=None), {}, msg

                try:
                    result = mock_job_recommendations(cv) if use_mock else api_recommend_jobs(api, cv)
                except Exception as e:
                    result = {"error": str(e), "jobs": []}

                if "error" in result:
                    msg = f"⚠️ {result['error']}"
                    return empty_df, gr.update(choices=[], value=None), result, msg

                jobs = result.get("jobs", []) or []
                rows, choices = [], []
                for idx, job in enumerate(jobs):
                    rows.append({
                        "ID": job.get("id", ""),
                        "Title": job.get("title", ""),
                        "Company": job.get("company", ""),
                        "Location": job.get("location", ""),
                        "Score": job.get("score", 0),
                        "Top skills": ", ".join(job.get("top_skills", [])[:6]),
                    })
                    choices.append(f"{idx} - {job.get('title', '')}")

                df = pd.DataFrame(rows) if rows else empty_df
                msg = f"Found {len(rows)} recommended jobs." if rows else "Không có job recommendation."

                value = choices[0] if choices else None
                dd_update = gr.update(choices=choices, value=value)

                return df, dd_update, result, msg

            def view_job_detail(result, choice):
                jobs = (result or {}).get("jobs", []) or []
                if not jobs:
                    return html_progress(0), "Chưa có job recommendation."

                if not choice:
                    job = jobs[0]
                else:
                    try:
                        idx = int(str(choice).split(" - ", 1)[0])
                    except Exception:
                        idx = 0
                    if idx < 0 or idx >= len(jobs):
                        idx = 0
                    job = jobs[idx]

                score = job.get("score", 0)
                title = job.get("title", "")
                company = job.get("company", "")
                loc = job.get("location", "")
                snippet = job.get("jd_snippet", "")
                reason = job.get("match_reason", "")

                md = f"""#### {title}

**Company:** {company}  
**Location:** {loc}  

**Matching reason:** {reason or "—"}

**JD snippet:**

> {snippet or "—"}
"""
                return html_progress(score), md

            btn_reco.click(
                do_recommend,
                inputs=[api_base, mock_mode, cv_state],
                outputs=[jobs_table, jobs_select, jobs_state, jobs_msg],
            )

            btn_view.click(
                view_job_detail,
                inputs=[jobs_state, jobs_select],
                outputs=[jd_score_html, jd_detail_md],
            )

            jobs_select.change(
                view_job_detail,
                inputs=[jobs_state, jobs_select],
                outputs=[jd_score_html, jd_detail_md],
            )

            def _auto_show_first_jobs(r):
                return view_job_detail(r, None)

            jobs_state.change(
                _auto_show_first_jobs,
                inputs=[jobs_state],
                outputs=[jd_score_html, jd_detail_md],
            )

            jobs_state.change(lambda r: r or {}, inputs=[jobs_state], outputs=[jobs_raw])

        # -------- Tab 6: Playground / Demo tổng hợp --------
        with gr.Tab("6) Playground / Demo tổng hợp"):
            gr.Markdown("### Playground / Demo tổng hợp")
            gr.Markdown(
                "Giao diện gom các chức năng chính để demo nhanh: Parse CV → Resume → Matching → Scoring → Job Recommendation."
            )

            with gr.Row():
                # --- cột trái: Input ---
                with gr.Column(scale=1):
                    gr.Markdown("#### Input")

                    with gr.Accordion("CV & JD Input", open=True):
                        pg_cv_summary = gr.HTML(label="Tóm tắt CV")
                        pg_jd_text = gr.Textbox(
                            label="JD text (dùng cho Matching / Playground)",
                            lines=8,
                            placeholder="Dán JD ở đây hoặc dùng nút Copy từ tab Upload..."
                        )
                        with gr.Row():
                            btn_pg_copy_jd = gr.Button("Copy JD từ tab Upload", size="sm")
                            btn_pg_run = gr.Button("Run demo pipeline", variant="primary")

                    with gr.Accordion("Ghi chú", open=False):
                        gr.Markdown(
                            "- Tab này dùng chung **API Base URL** và **Mock mode** ở trên.\n"
                            "- Bấm Parse CV ở tab 1 trước, sau đó qua đây bấm **Run demo pipeline**."
                        )

                # --- cột phải: Output ---
                with gr.Column(scale=2):
                    gr.Markdown("#### Output")

                    with gr.Tabs():
                        with gr.Tab("Resume View"):
                            pg_resume_skills = gr.HTML(label="Skill Tag Chips")
                            pg_resume_highlight = gr.HTML(label="Highlighted CV Content")
                        with gr.Tab("Matching"):
                            pg_match_progress = gr.HTML(label="Matching Score")
                            pg_match_reason = gr.Textbox(label="Matching Explanation", lines=4)
                            pg_match_table = gr.Dataframe(label="Skills Comparison", interactive=False)
                        with gr.Tab("CV Scoring"):
                            pg_score_circle = gr.HTML(label="CV Score")
                            pg_score_stars = gr.HTML(label="Stars")
                            pg_score_suggest = gr.Markdown(label="Feedback")
                        with gr.Tab("Job Reco"):
                            pg_jobs_table = gr.Dataframe(label="JD gợi ý", interactive=False)
                            pg_jobs_select = gr.Dropdown(label="Chọn JD", choices=[], value=None)
                            pg_jobs_score = gr.HTML(label="Job Score")
                            pg_jobs_detail = gr.Markdown(label="Job Detail")
                            pg_jobs_raw = gr.JSON(label="Raw Jobs JSON")

            pg_jobs_state = gr.State({})

            def pg_cv_summary_html(cv_json):
                cv = safe_dict(cv_json)
                if not cv or "error" in cv:
                    return "<div style='color:#aaa'>Chưa có CV state. Hãy Parse CV ở tab 1 trước.</div>"
                name = cv.get("name") or "—"
                email = cv.get("email") or "—"
                phone = cv.get("phone") or "—"
                skills = cv.get("skills", []) or []
                exp = cv.get("experience", []) or []
                edu = cv.get("education", []) or []
                return f"""
                <div style="font-size:13px;line-height:1.5">
                  <div><b>Candidate:</b> {name}</div>
                  <div><b>Email:</b> {email}</div>
                  <div><b>Phone:</b> {phone}</div>
                  <div style="margin-top:6px;">
                    <b>Stats:</b>
                    {len(skills)} skills · {len(exp)} experiences · {len(edu)} education items
                  </div>
                </div>
                """

            cv_state.change(pg_cv_summary_html, inputs=[cv_state], outputs=[pg_cv_summary])

            btn_pg_copy_jd.click(lambda t: t or "", inputs=[jd_state], outputs=[pg_jd_text])

            def pg_run_pipeline(api, use_mock, cv_json, jd_txt):
                empty_match_df = pd.DataFrame(columns=["Overlap skills", "Missing skills"])
                empty_jobs_df = pd.DataFrame(columns=["ID", "Title", "Company", "Location", "Score", "Top skills"])

                cv = safe_dict(cv_json)
                if not cv or "error" in cv:
                    msg = "⚠️ Chưa có CV. Hãy Parse CV ở tab 1 trước."
                    return (
                        "<div style='color:#aaa'>Chưa có CV.</div>",
                        "<div style='color:#aaa'>Chưa có CV.</div>",
                        html_progress(0),
                        msg,
                        empty_match_df,
                        score_circle_html(0),
                        html_stars(0),
                        "⚠️ Chưa có CV, chưa thể chấm điểm.",
                        empty_jobs_df,
                        gr.update(choices=[], value=None),
                        {},
                        html_progress(0),
                        "Chưa có job recommendation.",
                    )

                # Resume view
                skills = cv.get("skills", []) or []
                raw = cv.get("raw_text", "") or ""
                resume_skills_html = pills(skills)
                resume_highlight_html = highlight_text(raw, skills)

                # Matching
                if not jd_txt or not jd_txt.strip():
                    match_progress_html = html_progress(0)
                    match_reason = "Chưa có JD text (Matching bị bỏ qua)."
                    match_df = empty_match_df
                else:
                    out_match = mock_match() if use_mock else api_match(api, cv, jd_txt)
                    score_m = out_match.get("score_0_100", 0)
                    overlap = out_match.get("skills_overlap", []) or []
                    missing = out_match.get("skills_missing", []) or []
                    n = max(len(overlap), len(missing), 1)
                    match_df = pd.DataFrame({
                        "Overlap skills": overlap + [""] * (n - len(overlap)),
                        "Missing skills": missing + [""] * (n - len(missing)),
                    })
                    match_progress_html = html_progress(score_m)
                    match_reason = out_match.get("explanation", "—")

                # Scoring
                try:
                    out_score = mock_cv_score(cv) if use_mock else api_score_cv(api, cv)
                except Exception as e:
                    out_score = {"error": str(e)}
                if "error" in out_score:
                    score_circle = score_circle_html(0)
                    score_stars_html = html_stars(0)
                    score_suggest_md = f"⚠️ {out_score['error']}"
                else:
                    s = int(out_score.get("score", 0))
                    r = out_score.get("rating_0_5", round(s / 20.0, 1))
                    score_circle = score_circle_html(s)
                    score_stars_html = html_stars(r)
                    sug = out_score.get("suggestions", []) or []
                    score_suggest_md = "\n".join([f"- {x}" for x in sug]) if sug else "- —"

                # Job Recommendation
                try:
                    rec = mock_job_recommendations(cv) if use_mock else api_recommend_jobs(api, cv)
                except Exception as e:
                    rec = {"error": str(e), "jobs": []}

                jobs = rec.get("jobs", []) or []
                rows, choices = [], []
                for idx, job in enumerate(jobs):
                    rows.append({
                        "ID": job.get("id", ""),
                        "Title": job.get("title", ""),
                        "Company": job.get("company", ""),
                        "Location": job.get("location", ""),
                        "Score": job.get("score", 0),
                        "Top skills": ", ".join(job.get("top_skills", [])[:6]),
                    })
                    choices.append(f"{idx} - {job.get('title', '')}")
                jobs_df = pd.DataFrame(rows) if rows else empty_jobs_df

                if jobs:
                    first_choice = choices[0]
                    dd_update = gr.update(choices=choices, value=first_choice)
                    job0 = jobs[0]
                    job_score_html = html_progress(job0.get("score", 0))
                    job_detail_md = f"""#### {job0.get('title','')}

**Company:** {job0.get('company','')}  
**Location:** {job0.get('location','')}  

**Matching reason:** {job0.get('match_reason','—')}

**JD snippet:**

> {job0.get('jd_snippet','—')}
"""
                else:
                    dd_update = gr.update(choices=[], value=None)
                    job_score_html = html_progress(0)
                    job_detail_md = "Chưa có job recommendation."

                return (
                    resume_skills_html,
                    resume_highlight_html,
                    match_progress_html,
                    match_reason,
                    match_df,
                    score_circle,
                    score_stars_html,
                    score_suggest_md,
                    jobs_df,
                    dd_update,
                    rec,
                    job_score_html,
                    job_detail_md,
                )

            btn_pg_run.click(
                pg_run_pipeline,
                inputs=[api_base, mock_mode, cv_state, pg_jd_text],
                outputs=[
                    pg_resume_skills,
                    pg_resume_highlight,
                    pg_match_progress,
                    pg_match_reason,
                    pg_match_table,
                    pg_score_circle,
                    pg_score_stars,
                    pg_score_suggest,
                    pg_jobs_table,
                    pg_jobs_select,
                    pg_jobs_state,
                    pg_jobs_score,
                    pg_jobs_detail,
                ],
            )

            def pg_view_job_detail(result, choice):
                jobs = (result or {}).get("jobs", []) or []
                if not jobs:
                    return html_progress(0), "Chưa có job recommendation."
                if not choice:
                    job = jobs[0]
                else:
                    try:
                        idx = int(str(choice).split(" - ", 1)[0])
                    except Exception:
                        idx = 0
                    if idx < 0 or idx >= len(jobs):
                        idx = 0
                    job = jobs[idx]
                score = job.get("score", 0)
                md = f"""#### {job.get('title','')}

**Company:** {job.get('company','')}  
**Location:** {job.get('location','')}  

**Matching reason:** {job.get('match_reason','—')}

**JD snippet:**

> {job.get('jd_snippet','—')}
"""
                return html_progress(score), md

            pg_jobs_select.change(
                pg_view_job_detail,
                inputs=[pg_jobs_state, pg_jobs_select],
                outputs=[pg_jobs_score, pg_jobs_detail],
            )

            pg_jobs_state.change(
                lambda r: pg_view_job_detail(r, None),
                inputs=[pg_jobs_state],
                outputs=[pg_jobs_score, pg_jobs_detail],
            )

            pg_jobs_state.change(lambda r: r or {}, inputs=[pg_jobs_state], outputs=[pg_jobs_raw])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
