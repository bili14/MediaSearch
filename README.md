# MediaSearch（画面搜索）

在你**自己电脑的图/视频素材**上按**画面内容**做模糊搜索（不是搜文件名）。
后端是 WeMM-Embedding-2B 多模态嵌入模型（2048 维，文/图/视频同空间），
配一个单进程的"网页 + REST API + MCP + OpenAI 风格向量接口"，任何支持 MCP 的
harness（DSH / AutoClaw / Claude Desktop / Codex / OpenClaw …）都能直接挂上。

- 代码许可：**MIT**（见 [LICENSE](LICENSE)）
- 模型权重：`tencent/WeMM-Embedding-2B`（Hugging Face，**Apache-2.0**），
  首次运行自动下载，**本仓库不发布/不附带权重**。

## 仓库结构

```
repo/
├── src/medisearch/    # 全部代码（11 模块）：cfg 三层配置 · model_layer（WeMM 加载+自动下载+GPU 共存检测）
│                     #   libstore（库/范围/增量建库）· server（一进程四张脸）· mcp_proto+backend
│                     #   （MCP 协议/执行层）· mcp_server（stdio）· openai_compat（/v1/embeddings）
│                     #   · cli+__main__（console 双命令：medisearch serve / medisearch mcp）
├── src/medisearch/static/  # 网页 UI（index.html 单页，随包分发）
├── config.json.example  # 配置模板（env > data/config.json > 内置缺省）
├── start-service.bat / stop-service.bat  # Windows 双击起停（纯 ASCII）
├── samples/           # 演示素材 8 件（4 图+4 视频：猫/汽车/牛肉面/海鸥）+ 许可说明
└── data/（运行期自建） # 四件套：embeddings.json 向量库 · scopes.json · settings.json · config.json（均不入 git）
```

## 一、装（三步走 · 第 1 步）

```cmd
rem Windows cmd（PowerShell 报「禁止运行脚本」就用 cmd 全程）
python -m venv .venv
call .venv\Scripts\activate
pip install -e .
```

> **Windows 激活坑（必读）**：Windows 自带 PowerShell 5.1 默认执行策略 Restricted，
> `.venv\Scripts\Activate.ps1` 一律被拦（「在此系统上禁止运行脚本」）。中招就二选一：
> ① 切到 cmd 用 `call .venv\Scripts\activate`；② 干脆不激活，直接跑
> `.venv\Scripts\python.exe`（此时 `medisearch` 不在 PATH，一律写成
> `.venv\Scripts\python.exe -m medisearch serve`）。

macOS / Linux：

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -e .
```

没有 venv 的话也可以用 uv：`uv venv .venv && uv pip install -e .`。

依赖版本取自验收机实测 venv（Windows + CUDA 12.6 + torch 2.10.0+cu126）：
transformers 5.2.0 / accelerate 1.15.0 / qwen-vl-utils[decord] 0.0.14 /
fastapi 0.162.0 / uvicorn 0.54.0 等（完整清单见 [pyproject.toml](pyproject.toml)）。

> **torch 说明**：验证基线是 `torch==2.10.0+cu126`。`pip install torch==2.10.0`
> 装的是 PyPI 默认构建；需要完全复现 CUDA 12.6 构建时用
> `pip install "torch==2.10.0+cu126" --index-url https://download.pytorch.org/whl/cu126`。
> 无 GPU 也能跑（`device_map=auto` 会全落 CPU，只是慢）。

**国内镜像（可选）**：模型自动下载走 Hugging Face。直连慢/被墙时设
`HF_ENDPOINT=https://hf-mirror.com`（或你惯用的镜像）再首次运行即可，
`huggingface_hub` 原生读这个变量。完全断网时：把 `config.json` 的
`model_path` 指向你已有的本地权重目录（含 model.safetensors 的夹子），
代码检测到本地权重存在就不会下载。

## 二、起服务与建库（三步走 · 第 2 步）

```
medisearch serve
```

（或 Windows 双击 `start-service.bat`；停服 `stop-service.bat`。）
默认监听 **127.0.0.1:8686**，打开 http://127.0.0.1:8686/ ：

1. 左侧「＋ 加文件夹」粘素材文件夹路径（只索引其中的图/视频，可含子夹）；
2. 点「建库（增量）」——只嵌"没嵌过或 mtime 变了"的文件，
   图 ≈1.2s/件、视频 4–15s/件（视频为估算，以你机实测为准），进度条实时可见；
3. 搜索框输模糊短语（例：坐在窗台上的女孩 / 海鸥），回车。

仓库自带演示样例 `samples/`（4 图 + 4 视频：猫 / 汽车 / 红烧牛肉面 / 海鸥，
AI 生成原创、免费可用于演示与研究——见 [samples/README.md](samples/README.md)），
默认已登记为范围 `samples`：clone 下来直接建库即可试搜（「猫」「汽车」
「一碗红烧牛肉面」「海鸥」）。

数据都在 `<data>/`（仓库 checkout 下即 `./data/`，pip 安装下即
`~/.medisearch/data/`）：`embeddings.json`（向量库，全量一个库、带范围标签）、
`scopes.json`（登记的文件夹）、`settings.json`（参数，即改即生效）、
`config.json`（路径/模型/端口，模板 [config.json.example](config.json.example)）。

## 三、接 MCP（三步走 · 第 3 步）

### 3.1 单进程四张脸（`medisearch serve`，一个端口 8686）

| 脸 | 路径 | 给谁用 |
|---|---|---|
| 网页 UI | `GET /` | 人 |
| REST API | `POST /search`、`GET /status`、`POST /embed-scope`、`/add-scope`、`/remove-scope`、`GET/PUT /settings`、`/scopes`、`GET /file` | 老脚本/自己的前端 |
| MCP streamable-HTTP | `POST /mcp`（JSON-RPC；通知回 202；GET 回 405）、`GET /healthz` | 支持 MCP 的 harness（远程/多 client） |
| OpenAI 风格兜底 | `POST /v1/embeddings`（文本/图→向量，换 base_url 即用） | 不会 MCP 的老脚本 |

不需要第二个进程。旧版"8686 网页 + 8687 MCP"双进程已合并。

### 3.2 stdio（`medisearch mcp`，harness 派生子进程用）

通用三要素：**command = venv 里的 medisearch**（Windows：`.venv\Scripts\medisearch.exe`；
Linux/macOS：`.venv/bin/medisearch`）＋ **args = `["mcp"]`** ＋ env 可选
`WEMM_BACKEND=auto`（缺省即 auto：探测 8686 有服务=HTTP 透传，没有=本地向量库直读＋本地模型；
`=api` 强制透传、`=local` 强制本地）。

工具（stdio 与 HTTP 两传输相同）。其中 `scope` = 素材文件夹登记后的名称/编号（第 2 步网页里「＋ 加文件夹」时生成），不填 = 全库搜索：

```
media_search(q, scope?, topn?)  -> topN [{path, score, modality, scope}]
media_status()                  -> {backend, model: loaded|idle, items, lib, building, ...}
media_build(scope?)             -> 增量建库（direct/api 模式；轮询 media_status 看进度）
media_remove_scope(id, delete_vectors?) -> 移除登记范围（绝不删硬盘源文件）
```

**DSH 宿主**（`mcp_connector_configure`，stdio 自定义连接）：

```json
{
  "name": "medisearch",
  "transport": "stdio",
  "command": "C:\path\to\repo\.venv\Scripts\medisearch.exe",
  "args": ["mcp"],
  "env": { "WEMM_BACKEND": "auto" }
}
```

**AutoClaw / 任意 MCP 宿主**（通用模板，路径换成你的）：

```json
{
  "mcp": {
    "servers": {
      "medisearch": {
        "command": "C:\path\to\repo\.venv\Scripts\medisearch.exe",
        "args": ["mcp"],
        "env": { "WEMM_BACKEND": "auto" }
      }
    }
  }
}
```

Claude Desktop / Claude Code 同款（stdio）：command 指 venv 的
python 也可（args 写 `-m medisearch mcp`）；Codex CLI `~/.codex/config.toml`：

```toml
[mcp_servers.medisearch]
command = "C:\path\to\repo\.venv\Scripts\medisearch.exe"
args = ["mcp"]
env = { WEMM_BACKEND = "auto" }
enabled = true
```

MCP HTTP 版（多 client/远程）：服务起着时填 `url = "http://127.0.0.1:8686/mcp"`
（OpenClaw 等要写 `transport = "streamable-http"`，缺省被当 SSE 会连不上）。

## 四、坑清单（都踩过，别重踩）

1. **Windows 路径带空格**：harness/工具链个别环节对带空格路径处理不稳。
   两招：
   - 8.3 短名（无空格）方案：cmd 里
     `for %I in ("C:\My Projects\medisearch") do @echo %~sI`
     得到形如 `C:\MYPRO~1\MEDISEA~1` 的短路径，宿主配置里用短路径；
   - 或者把仓库放进**不带空格**的目录（推荐，一劳永逸）。
2. **只读开关 `WEMM_READ_ONLY=1`**（或 CLI `--read-only`）：拒绝一切写操作
   （加/改/删范围、建库、改 settings，全部 403 / 工具明确报错），
   搜索与状态口照常可用。**默认是写开启**（推荐默认行为），接入方想保险
   就显式开只读。
3. **模型许可**：代码 MIT；权重 Apache-2.0（Tencent 发布，随 Hugging Face
   拉取、遵循其自身许可）。**仓库不发布权重**，5.2GB 的 model.safetensors
   只会在你机器上第一次运行时下载/加载。
4. **实测基线**（验收机，GPU 空闲）：
   - 首查含 JIT 热身：权重已在本地约 **10s**（验收机干净窗实测 6.6s，仅参考）；
     首次运行需自动下载 5.2GB 权重，时间另计；不同机器有波动属正常；
   - 建库速度 **图 ≈1.2s/件**、视频 4–15s/件（估算，以你机实测为准）；
   - **闲置 30 分钟自动释放模型**（`settings.idle_release_min`，缺省 30；
     网页参数面板可改，0=不自动释放）——不跟你在同机跑的 LM Studio 抢床。
   - 参考命中（101 件媒体库）：「坐在窗台上的女孩」→ top1 14.png（0.4355）；
     「海鸥」→ top1 50.mp4（0.4929）/ top2 49.mp4（0.4834）。
5. **GPU 床规矩**：加载前探本机 LM Studio API（`WEMM_LM_BED_API`，缺省
   `http://127.0.0.1:1234/api/v0/models`）——探不通=没跑=GPU 归我们；
   探到已加载模型 → 拒绝加载（503 lm_bed_busy），不双开不抢床。
6. **4GiB 显存帽是命门**：模型 5.2GB 主体上卡、溢出层自动挪 CPU
   （`max_memory={0:"4GiB","cpu":"64GiB"}`，键必须整数 0）。改大反而 OOM。
   `model_max_memory` 是 {GPU 键, cpu} **成对项**：只写 GPU 帽、不写 "cpu" 键，
   溢出层无处可去、整模型挤上 GPU → OOM（config.json.example 两个键都留着别删；
   config.json 里键写成字符串 "0" 无妨，载入时会自动 int 化）。
7. **模型缺失才下载**：`model_path`（缺省 `~/.medisearch/models/WeMM-Embedding-2B`）
   里已有权重就直接用（断网可用）；支持 `HF_ENDPOINT` 走国内镜像。

## 五、配置总表（env > data/config.json > 缺省）

| 键（config.json） | 环境变量 | 缺省 |
|---|---|---|
| `data_dir` | `WEMM_DATA_DIR` | 仓库 `./data/`；pip 安装 `~/.medisearch/data` |
| `model_path` | `WEMM_MODEL_PATH` | `~/.medisearch/models/WeMM-Embedding-2B`（缺失自动下载） |
| `model_repo` | `WEMM_MODEL_REPO` | `tencent/WeMM-Embedding-2B` |
| `hf_home` / `hf_modules_cache` | `WEMM_HF_HOME` / `WEMM_HF_MODULES_CACHE` | `~/.medisearch/hf[/modules]` |
| `fallback_library` | `WEMM_FALLBACK_LIB` | 无（主库为空时的兜底库文件） |
| `host` / `port` | `WEMM_HOST` / `WEMM_PORT` | `127.0.0.1` / `8686` |
| `api_base` | `WEMM_API_BASE` | `http://127.0.0.1:8686` |
| `lm_bed_api` | `WEMM_LM_BED_API` | `http://127.0.0.1:1234/api/v0/models` |
| — | `WEMM_BACKEND` | `auto`（stdio 模式：探测服务→api，不通→local） |
| — | `WEMM_READ_ONLY` | `0`（开=只读） |
| — | `HF_ENDPOINT` | 无（镜像直传 huggingface_hub） |
| — | `WEMM_HOME` | `~/.medisearch`（用户目录根：未另设时 data/模型/HF 缓存都从它派生） |
| — | `WEMM_CONFIG` | `<data_dir>/config.json`（直指配置文件，盖过缺省位置） |
| — | `WEMM_DATA` | 无（stdio local 模式：直指向量库文件，优先级最高） |
| — | `WEMM_MODELS_DIR` | `~/.medisearch/models`（模型夹根；设了 `model_path` 时它让位） |
| — | `WEMM_API_TIMEOUT` | `300`（秒，stdio api 模式搜索透传超时；首搜 JIT 慢） |
| — | `WEMM_PROBE_TIMEOUT` | `2.0`（秒，auto 模式服务探测；冷 /status 偏慢可放宽） |

## 六、安全契约

`media_remove_scope` / `POST /remove-scope` **只删范围登记＋库内向量档案，
绝不删硬盘上的任何源文件**（返回 `files_deleted` 恒为 0）。
`media_build` / `POST /embed-scope` 只写 `data/` 下数据文件（原子替换＋
短锁），不碰素材文件夹。
`GET /file` 只服务库成员的图（缩略图），路径不在库内一律 404。
