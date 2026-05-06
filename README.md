# Lab 20: Khởi đầu Hệ thống Nghiên cứu Multi‑Agent

Repo starter cho bài lab **Multi‑Agent Systems**: xây dựng một hệ thống nghiên cứu gồm **Supervisor, Researcher, Analyst, Writer** và so sánh với baseline single‑agent.

> Mục tiêu của repo này là cung cấp một **skeleton production‑grade** để học viên phát triển code cá nhân. Các phần logic quan trọng được để lại dưới dạng `TODO` để học viên tự triển khai.

## Kết quả học tập

Sau 2 giờ lab, học viên cần có thể:

1. Thiết kế vai trò (role) rõ ràng cho nhiều agent.
2. Xây dựng trạng thái chia sẻ (shared state) đủ thông tin cho việc handoff.
3. Thêm các guardrail tối thiểu: số vòng lặp tối đa, timeout, retry/fallback, validation.
4. Trace được luồng chạy và giải thích agent nào thực hiện gì.
5. Benchmark single‑agent vs multi‑agent về chất lượng, độ trễ, chi phí.

## Kiến trúc mục tiêu

```text
User Query
   |
   v
Supervisor / Router
   |------> Researcher Agent  -> research_notes
   |------> Analyst Agent     -> analysis_notes
   |------> Writer Agent      -> final_answer
   |
   v
Trace + Benchmark Report
```

## Cấu trúc repo

```text
.
├── src/multi_agent_research_lab/
│   ├── agents/            # Giao diện và skeleton cho các agent
│   ├── core/              # Cấu hình, state, schema, lỗi
│   ├── graph/             # Skeleton workflow LangGraph
│   ├── services/          # Khách hàng LLM, search, lưu trữ
│   ├── evaluation/        # Skeleton benchmark/evaluation
│   ├── observability/     # Hook logging/tracing
│   └── cli.py             # Điểm vào CLI
├── configs/               # Các file YAML cấu hình cho các biến thể lab
├── docs/                  # Hướng dẫn lab, rubric, notes thiết kế
├── tests/                 # Kiểm thử unit cho skeleton
├── notebooks/             # Notebook tùy chọn
├── scripts/               # Script hỗ trợ
├── .env.example           # Mẫu file môi trường
├── pyproject.toml         # Cấu hình dự án Python
├── Dockerfile             # Container hoá môi trường dev/runtime
└── Makefile               # Các lệnh thường dùng
```

## Hướng dẫn nhanh (Quickstart)

### 1. Tạo môi trường ảo

```bash
python -m venv .venv
# Kích hoạt môi trường
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
# Cài đặt các phụ thuộc được khai báo trong pyproject.toml (kèm dev extras)
pip install -e .[dev]
cp .env.example .env
```

### 2. Sử dụng `pyproject.toml`

Dự án sử dụng **hệ thống build PEP 517** với file `pyproject.toml` để khai báo phụ thuộc và cấu hình build. Sau khi tạo và kích hoạt môi trường ảo:

- Cài đặt package ở chế độ editable cùng các phụ thuộc phát triển:
  ```bash
  pip install -e .[dev]
  ```
- Lệnh trên sẽ đọc các mục `[project]` và `[tool.poetry.dev-dependencies]` (hoặc tương đương) trong `pyproject.toml` và cài đặt toàn bộ thư viện cần thiết.
- Để xem danh sách phụ thuộc đã được giải quyết:
  ```bash
  pip list
  ```
- Để nâng cấp các phụ thuộc được định nghĩa trong `pyproject.toml`:
  ```bash
  pip install -U -e .[dev]
  ```

Sau khi cài đặt xong, bạn có thể tiếp tục các bước tiếp theo.

### 3. Cấu hình API keys

Mở file `.env` và điền các khóa cần thiết.

```bash
OPENAI_API_KEY=...
# tuỳ chọn
LANGSMITH_API_KEY=...
TAVILY_API_KEY=...
```

### 4. Chạy smoke test

```bash
make test
python -m multi_agent_research_lab.cli --help
```

### 5. Chạy baseline skeleton

```bash
python -m multi_agent_research_lab.cli baseline \
  --query "Research GraphRAG state-of-the-art and write a 500-word summary"
```

Lệnh này chỉ chạy khung baseline tối giản. Học viên cần tự triển khai logic LLM thực tế trong `src/multi_agent_research_lab/services/llm_client.py`.

### 6. Chạy multi‑agent skeleton

```bash
python -m multi_agent_research_lab.cli multi-agent \
  --query "Research GraphRAG state-of-the-art and write a 500-word summary"
```

Mặc định lệnh sẽ báo các `TODO` cần thực hiện – đây là mục đích của starter repo.

## Các mốc thời gian (Milestones) trong 2 giờ lab

| Thời lượng | Milestone | File gợi ý |
|---:|---|---|
| 0‑15' | Setup, chạy baseline skeleton | `cli.py`, `services/llm_client.py` |
| 15‑45' | Xây dựng Supervisor / router | `agents/supervisor.py`, `graph/workflow.py` |
| 45‑75' | Thêm Researcher, Analyst, Writer | `agents/*.py`, `core/state.py` |
| 75‑95' | Trace + benchmark single vs multi | `observability/tracing.py`, `evaluation/benchmark.py` |
| 95‑115' | Peer review theo rubric | `docs/peer_review_rubric.md` |
| 115‑120' | Exit ticket | `docs/lab_guide.md` |

## Quy ước production trong repo

- Tách rõ các thư mục `agents`, `services`, `core`, `graph`, `evaluation`, `observability`.
- Không hard‑code API key trong code.
- Tất cả input/output chính sử dụng Pydantic schema.
- Có type hints, linting, formatting, và ít nhất một unit test.
- Có hook logging/tracing ngay từ đầu.
- Không để agent chạy vô hạn: dùng `max_iterations`, `timeout_seconds`.
- Có báo cáo benchmark thay vì chỉ demo output đẹp.

## TODO chính cho học viên

Tìm trong code các marker:

```bash
grep -R "TODO(student)" -n src tests docs
```

Các phần học viên cần tự triển khai:

1. Implement LLM client.
2. Implement client web/search hoặc mock source.
3. Implement quyết định routing trong Supervisor.
4. Implement từng worker agent.
5. Xây dựng workflow LangGraph.
6. Thêm tracing provider thực tế: LangSmith, Langfuse hoặc OpenTelemetry.
7. Viết benchmark report.

## Kết quả (Deliverables)

1. Repo GitHub cá nhân.
2. Ảnh chụp màn hình trace hoặc link trace.
3. `reports/benchmark_report.md` so sánh single vs multi‑agent.
4. Một đoạn mô tả failure mode và cách khắc phục.

## Tham khảo

- Anthropic: Building effective agents — https://www.anthropic.com/engineering/building-effective-agents
- OpenAI Agents SDK orchestration/handoffs — https://developers.openai.com/api/docs/guides/agents/orchestration
- LangGraph concepts — https://langchain-ai.github.io/langgraph/concepts/
- LangSmith tracing — https://docs.smith.langchain.com/
- Langfuse tracing — https://langfuse.com/docs