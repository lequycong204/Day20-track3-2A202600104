# Tóm tắt công việc dự án

Repository này là một lab khởi tạo cho hệ thống nghiên cứu đa‑agent. Các phần code hiện có chứa nhiều `TODO(student)` chỉ ra những công việc sinh viên cần thực hiện.

## Các công việc cốt lõi cần triển khai
1. **LLM client** – Thay thế placeholder trong `src/multi_agent_research_lab/services/llm_client.py` bằng một lời gọi thực tế tới LLM (OpenAI, Anthropic, v.v.).
2. **Search client** – Cài đặt hoặc mô phỏng client tìm kiếm/web trong `src/multi_agent_research_lab/services/search_client.py`.
3. **Routing của Supervisor** – Thêm chính sách định tuyến trong `src/multi_agent_research_lab/agents/supervisor.py` (xem TODO trong `docs/lab_guide.md`).

Lần gọi đầu tiên (iteration=0)
  │
  ├── _decompose_query()  ← LLM phân tích query
  │     "Research GraphRAG AND compare with RAG AND summarize trends"
  │     → sub_tasks: ["What is GraphRAG?", "Compare GraphRAG vs RAG", "Summarize trends"]
  │     → lưu vào state.trace
  │
  └── _decide_route()  ← quyết định agent tiếp theo
        ↓ deterministic rules (nhanh, không tốn token)
        ├── chưa có sources → "researcher"
        ├── có research_notes, chưa có analysis_notes → "analyst"
        ├── có analysis_notes, chưa có final_answer → "writer"
        └── ambiguous → _llm_route() (LLM làm trọng tài)


4. **Các agent worker** – Cài đặt `Researcher`, `Analyst`, và `Writer` ở `src/multi_agent_research_lab/agents/*.py`.

state.final_answer  +  state.sources
           │
           ▼ (LLM, temperature=0.0 — không được sáng tạo khi fact-check)
     JSON review:
       ├── score: 7.5/10
       ├── hallucinations: ["claim X không có trong sources"]
       ├── missing_citations: ["đoạn Y chưa có [n]"]
       ├── revision_notes: "Cần thêm citation cho..."
       └── approved: true/false
           │
    ┌──────┴──────┐
    │ score ≥ 6   │ score < 6
    ▼             ▼
  giữ nguyên   xóa final_answer
  final_answer  + gắn revision hints
                  vào analysis_notes
                  → Supervisor sẽ route lại Writer


ResearcherAgent → state.research_notes (tổng hợp thô từ web)
                            │
                            ▼
                    AnalystAgent
                    (temperature=0.1 — cẩn thận, ít sáng tạo)
                            │
                            ▼
              state.analysis_notes (5 sections)



5. **Workflow LangGraph** – Kết nối các agent lại trong một đồ thị LangGraph để Supervisor có thể giao state.
6. **Observability** – Kết nối một provider tracing thực tế (LangSmith, Langfuse, OpenTelemetry) trong `src/multi_agent_research_lab/observability/`.
7. **Báo cáo benchmark** – Tạo `reports/benchmark_report.md` so sánh chạy single‑agent vs multi‑agent (độ trễ, chi phí, chất lượng, tỷ lệ lỗi).

## Các công việc trong mẫu thiết kế (`docs/design_template.md`)
- Định nghĩa chi tiết **Supervisor**: trách nhiệm, input, output, failure mode.
- Hoàn thiện bảng **Researcher**, **Analyst**, **Writer** với các trường dữ liệu cụ thể.
- Liệt kê các trường trong **shared state** và giải thích lý do cần chúng.
- Vẽ hoặc mô tả **routing graph** cho Supervisor quyết định gọi agent nào.
- Xác định **guardrails**: số lần lặp tối đa, timeout, chính sách retry, fallback, và validation.
- Phác thảo **benchmark plan**: các query mẫu, metric đánh giá, và kết quả mong đợi.

## Các mốc thời gian (theo `README.md`)
| Thời lượng | Mốc | File cần chỉnh sửa |
|---|---|---|
| 0‑15' | Thiết lập, chạy baseline | `cli.py`, `services/llm_client.py` |
| 15‑45' | Xây dựng Supervisor / router | `agents/supervisor.py`, `graph/workflow.py` |
| 45‑75' | Thêm Researcher, Analyst, Writer | `agents/*.py`, `core/state.py` |
| 75‑95' | Trace & benchmark | `observability/tracing.py`, `evaluation/benchmark.py` |
| 95‑115' | Peer‑review (rubric) | `docs/peer_review_rubric.md` |
| 115‑120' | Exit ticket | `docs/lab_guide.md` |

## Nơi tìm các `TODO`
- `README.md` (phần **TODO chính cho học viên**)
- `docs/lab_guide.md` – LLM client, routing policy, triển khai worker.
- `docs/design_template.md` – Bảng vai trò agent, shared state, guardrails, benchmark plan.
- Các file nguồn có `TODO(student)` rõ ràng (ví dụ `src/multi_agent_research_lab/cli.py`, `agents/analyst.py`, `evaluation/benchmark.py`).

Sử dụng danh sách này để hướng dẫn công việc và xác nhận hoàn thành qua các unit test đi kèm.


