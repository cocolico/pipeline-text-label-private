# Automatic Text Annotation

一个基于AI模型的多阶段视频自动标注管道，专门用于处理第一人称视角（egocentric）视频的自动标注任务。

## 项目概述

本项目实现了一个自动化视频标注系统，通过多阶段管道对第一人称视角视频进行深度分析，生成结构化的动作步骤、关键帧、手部动作和物体交互标注。该系统使用火山引擎的AI模型（豆包模型）进行视频理解和分析。

## 主要特性

- **多阶段管道架构**：6个独立阶段，循序渐进地完成视频标注
- **第一人称视角专注**：专门针对egocentric视频优化，识别主要操作者的手部动作和物体交互
- **AI驱动**：使用火山引擎的Doubao-seed模型进行视频理解
- **Docker支持**：提供Dockerfile便于环境部署
- **资源监控**：内置进程资源监控功能

> ⚠️ **重要提示**：
> - **Stage 4** 已启用；默认执行 `--stages 1,2,3,4`，无需 GPU。仅需步骤标注时可指定 `--stages 1,2,3`。
> - **Stage 5** 依赖于SAM3模型权重文件（`sam3.pt`），项目未提供此文件，目前无法正常运行

## 管道阶段详解

### Stage 1: 步骤计划生成 (Draft Plan Generation)

根据视频帧生成高层次的动作步骤计划。

**输入**：视频的均匀采样帧
**输出**：
- `high_level_goal`：整体目标描述
- `steps`：步骤列表（包含step_id和step_goal）

**主要功能**：
- 分析整个视频时间线
- 生成3-8个步骤的中等粒度计划
- 确保覆盖从第一帧到最后一帧的完整时间线

### Stage 2: 视频定位与切割 (Video Localization & Cutting)

将每个步骤定位到视频中的具体时间区间，并切割出对应的视频片段。

**输入**：Stage 1的步骤计划 + 视频帧边界点
**输出**：
- 每个步骤的时间边界（start_frame, end_frame）
- 对应的视频片段文件

**主要功能**：
- 步骤时间边界定位
- 边界验证与调整
- 视频片段自动切割

### Stage 2h: 手部深度质量控制 (Hand Depth Quality Control)

验证每个步骤片段中的手部检测和深度数据质量。

**输入**：Stage 2的视频片段 + 手部检测数据 + 深度数据
**输出**：
- `qc_summary.json`：手部质量评估报告（包含hand_status、depth_status等）

**主要功能**：
- 手部检测可用性检查
- 深度数据质量验证
- 识别手部可见性问题

### Stage 3: 关键帧细化 (Refine & Keyframes)

细化每个步骤的目标描述，并提取关键帧。

**输入**：Stage 2的视频片段
**输出**：
- 细化的step_goal
- hand_object_relations（手-物关系）
- 关键帧图像

**主要功能**：
- 步骤目标细化
- 手-物关系标注
- 关键帧提取

### Stage 4: 原子动作分解 (Atomic Actions)

> **运行状态**：Stage 4 默认启用。已有 Stage 1–3 结果时，可使用 `--stages 4` 单独补跑原子动作分解。

将每个步骤分解为更细粒度的原子动作。

**输入**：Stage 3的细化步骤和关键帧
**输出**：原子动作序列
- atomic_action_id
- atomic_action_goal
- 参与的手部（left_hand/right_hand）
- 涉及的物体

**主要功能**：
- 步骤到原子动作的分解
- 物体清单构建
- 手-物交互动作识别

### Stage 5: 边界框跟踪 (BBox Tracking)

> ⚠️ **注意**：Stage 5 依赖于SAM3模型权重文件（`sam3.pt`），目前无渠道获取，因此Stage 5无法正常运行。

使用SAM3和WiLoR进行精确的手部和物体边界框跟踪。

**输入**：Stage 4的原子动作 + 原始视频
**输出**：
- `frame_bboxes.jsonl`：每帧的边界框数据
- `tracks.json`：跟踪轨迹
- `qa_summary.json`：质量评估摘要
- `bbox_overlay.mp4`：可视化叠加视频

**主要功能**：
- 手部检测与跟踪
- 物体检测与跟踪
- 遮挡处理
- API辅助定位（当SAM3结果不理想时）

## 项目结构

```
Automatic-text-annotation/
├── api/
│   ├── api_seed_openai_easy_no_short_step/    # 主API实现
│   │   ├── pipeline.py                        # 主管道入口
│   │   ├── common.py                          # 公共工具函数
│   │   ├── prompts.py                         # AI提示词模板
│   │   ├── api_client_qwen_v2.py              # API客户端
│   │   ├── stage1_generate_draft.py           # Stage 1实现
│   │   ├── stage2_localize_and_cut.py         # Stage 2实现
│   │   ├── stage2h_qc_hand_depth.py           # Stage 2h实现
│   │   ├── stage3_refine_and_keyframes.py     # Stage 3实现
│   │   ├── stage4_atomic_actions.py           # Stage 4实现
│   │   ├── stage5_bbox_tracking.py             # Stage 5实现
│   │   ├── stage5_sam3_backend.py             # SAM3后端
│   │   └── tests/                             # 测试文件
│   ├── tools/                                  # 辅助工具
│   │   ├── ark_image_understanding_current.py
│   │   ├── ark_video_understanding_current.py
│   │   └── test_ark_*.py                      # Grounding测试脚本
│   └── litchi_718_aglin_6rgb_latest_full5_run_20260720_152921/
│       └── scripts/                            # 远程运行脚本
├── docker/
│   ├── Dockerfile                              # 基础Docker镜像
│   ├── Dockerfile.full                        # 完整Docker镜像
│   └── requirements.txt                        # Python依赖
├── proc_monitor.py                             # 资源监控脚本
└── run_pipeline_monitor.sh                     # 监控启动脚本
```

## 环境要求

- Python 3.12+
- FFmpeg
- OpenCV
- 支持CUDA的GPU（用于SAM3和物体检测）

### Python依赖

```
openai>=2.44.0
opencv-python>=4.13.0
numpy>=2.5.1
pillow>=12.3.0
requests>=2.34.2
httpx>=0.28.1
pydantic>=2.13.4
```

### Stage 5额外依赖

- SAM3模型
- WiLoR手部检测器
- Grounding DINO（可选）

## 安装配置

### 1. 克隆项目

```bash
git clone <repository-url>
cd Automatic-text-annotation
```

### 2. 创建Python虚拟环境

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
pip install -r docker/requirements.txt
```

### 4. 配置环境变量

本项目的所有配置均通过**环境变量**或**命令行参数**传递，**没有任何硬编码的密钥或路径**。

#### 必须配置的环境变量

| 环境变量 | 说明 | 默认值 | 必需 |
|---------|------|--------|------|
| `ARK_API_KEY` | 火山引擎API密钥 | 无 | ✅ 是 |
| `API_KEY` | ARK_API_KEY的备用名称 | 无 | ✅ 是 |

#### 可选的API配置环境变量

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| `API_BASE_URL` | API端点地址 | `https://ark.cn-beijing.volces.com/api/v3` |
| `MODEL_NAME` | 使用的模型名称 | `doubao-seed-2-0-lite-260428` |
| `MODEL_PROVIDER_ID` | 模型提供者ID | `volcengine_maas` |
| `ARK_REASONING_EFFORT` | 推理努力程度 | `medium` |
| `REASONING_EFFORT` | ARK_REASONING_EFFORT的备用名称 | `medium` |
| `MAX_TOKENS` | 最大token数 | `32000` |
| `TEMPERATURE` | 采样温度 | `0.8` |
| `VIDEO_UNDERSTANDING_FPS` | 视频理解采样帧率 | `5.0` |
| `VERBOSE_LOGGING` | 详细日志输出 (1/0) | `0` |
| `EMBED_INDEX_ON_API_images` | API图像嵌入索引 (1/0) | `1` |
| `API_CALL_RETRIES` | API调用重试次数 | `3` |
| `API_CALL_TIMEOUT_SEC` | API调用超时秒数 | `420` |

#### Stage 5 专用环境变量（需要SAM3权重文件）

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| `HAND_DETECTOR_PATH` | WiLoR手部检测器模型路径 | 无 |
| `SAM3_REPO` | SAM3仓库目录路径 | 无 |
| `SAM3_CHECKPOINT` | SAM3模型权重文件路径 | 无 |
| `BBOX_PYTHON` | BBox追踪环境Python路径 | 无 |
| `SAM3_PYTHON` | SAM3环境Python路径 | 无 |

> ⚠️ **注意**：Stage 5 的环境变量需要配合 `setup_stage5_bbox_env.sh` 脚本设置，且需要自行获取SAM3模型权重文件。

#### 配置示例

```bash
# 火山引擎API配置（必需）
export ARK_API_KEY="your-api-key-here"
export API_KEY="your-api-key-here"  # 备用

# 可选API配置
export API_BASE_URL="https://ark.cn-beijing.volces.com/api/v3"
export MODEL_NAME="doubao-seed-2-0-lite-260428"
export ARK_REASONING_EFFORT="medium"

# 可选参数调整
export MAX_TOKENS="32000"
export TEMPERATURE="0.8"
export VIDEO_UNDERSTANDING_FPS="5.0"
export VERBOSE_LOGGING="0"
```

除了环境变量，也可以通过命令行参数传递配置，详见"使用方法"章节。

### 5. Docker部署（可选）

```bash
# 构建镜像
docker build -f docker/Dockerfile -t video-annotation:latest .

# 运行容器
docker run -e ARK_API_KEY="your-api-key" \
           -e API_BASE_URL="https://ark.cn-beijing.volces.com/api/v3" \
           -v /path/to/videos:/data/videos \
           -v /path/to/output:/data/output \
           video-annotation:latest
```

## 使用方法

### API配置方式

除了环境变量，还可以通过命令行参数传递API配置：

| 命令行参数 | 对应环境变量 | 说明 |
|-----------|-------------|------|
| `--api-key` | `ARK_API_KEY` | API密钥 |
| `--api-base` | `API_BASE_URL` | API端点地址 |
| `--model` | `MODEL_NAME` | 模型名称 |
| `--reasoning-effort` | `ARK_REASONING_EFFORT` | 推理努力程度 (low/medium/high) |
| `--max-tokens` | `MAX_TOKENS` | 最大token数 |
| `--temperature` | `TEMPERATURE` | 采样温度 |

**优先级**：命令行参数 > 环境变量 > 默认值

### 基本用法

```bash
# 运行完整管道
python api/api_seed_openai_easy_no_short_step/pipeline.py \
    --source-video /path/to/video.mp4 \
    --output-root /path/to/output \
    --stages 1,2,3,4,5
```

### 运行单个阶段

```bash
# 仅运行Stage 1
python api/api_seed_openai_easy_no_short_step/pipeline.py \
    --source-video /path/to/video.mp4 \
    --output-root /path/to/output \
    --stages 1

# 仅运行Stage 5
python api/api_seed_openai_easy_no_short_step/pipeline.py \
    --source-video /path/to/video.mp4 \
    --output-root /path/to/output \
    --stages 5
```

### 常用参数

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--source-video` | 输入视频路径 | 必需 |
| `--output-root` | 输出目录 | 必需 |
| `--stages` | 要运行的阶段（逗号分隔） | "1,2,3,4" |
| `--overwrite` | 覆盖已有输出 | false |
| `--resume` | 从中断处恢复 | false |

### Stage 5 专用参数

```bash
# 使用SAM3后端（默认）
python pipeline.py --source-video video.mp4 --stages 5 --bbox-backend sam3

# 设置SAM3阈值
python pipeline.py --source-video video.mp4 --stages 5 --sam3-threshold 0.5

# 启用API辅助手部定位
python pipeline.py --source-video video.mp4 --stages 5 --api-hand-assist

# 启用API辅助物体定位
python pipeline.py --source-video video.mp4 --stages 5 --api-grounding-thinking enabled
```

### 批量处理多个视频

```bash
# 处理目录下所有视频
python api/api_seed_openai_easy_no_short_step/pipeline.py \
    --source-video /path/to/videos/ \
    --output-root /path/to/output \
    --stages 1,2,3,4,5
```

## 输出格式

### Stage 1 输出 (stage1/)

- `draft_plan.json`：步骤计划（JSON格式，结构如下）
- `stage1_raw_response.txt`：模型的原始响应
- `stage1_system_prompt.txt`：使用的系统提示词
- `stage1_user_prompt.txt`：使用的用户提示词
- `frame_manifest.json`：帧索引清单

**draft_plan.json 格式**：
```json
{
  "high_level_goal": "描述整体目标",
  "steps": [
    {
      "step_id": 1,
      "step_goal": "步骤1的目标描述"
    }
  ]
}
```

### Stage 2 输出 (stage2/)

- `localization_raw.json`：步骤时间边界（原始定位数据）
- `segments/`：切割后的视频片段

### Stage 3 输出 (stage3/)

- `final_plan.json`：细化后的步骤计划（包含step_goal和hand_object_relations）
- `instruction_subtask.json`：简化的任务指令格式（包含instruction和sub_task数组）
- `keyframes/`：关键帧图像
- - `object_inventory.json`：从hand_object_relations提取的物体清单

### Stage 4 输出 (stage4/)

- `atomic_plan_with_clips.json`：包含原子动作、原视频帧边界和片段路径的完整计划。
- `object_inventory.json`：详细的物体清单（包含 mentions 映射）。
- 每个步骤目录下的 `atomic_actions.json` 和 `atomic_clips/`：动作标注缓存与视频片段。
- 视频输出根目录的 `final_plan.json` 会合并原子动作结果。上游没有有效步骤时，输出空动作计划，不调用模型。

### Stage 5 输出 (stage5/)

- `frame_bboxes.jsonl`：每帧边界框（JSONL格式）
- `tracks.json`：跟踪轨迹
- `qa_summary.json`：质量评估
- `bbox_overlay.mp4`：可视化视频

## 主要操作者策略

系统遵循严格的主要操作者识别策略：

1. **唯一性**：在多人场景下，只标注一个主要操作者
2. **判定标准**：选择手部持续操作空间最接近画面中心的操作者
3. **身份一致性**：跨短暂遮挡、交叉或短暂运动时保持同一身份
4. **忽略边缘动作**：次要人物的手部动作不产生步骤、边界或关系

## 质量控制

### 自动质量门控

- Stage 2：边界验证与调整
- Stage 2h：手部深度数据质量检查
- Stage 5：跟踪质量评估（coverage、confidence、center jump等）

### 质量评估标签

- `passed`：通过质量门控，可自动使用
- `warning`：需要人工审核
- `failed`：未通过质量门控

## 资源监控

项目内置进程资源监控功能：

```bash
# 运行带监控的管道
bash run_pipeline_monitor.sh
```

监控日志保存在 `resource_monitor_logs/` 目录。

## 测试

运行测试：

```bash
# 进入API目录
cd api/api_seed_openai_easy_no_short_step

# 运行所有测试
python -m pytest tests/

# 运行特定测试
python -m pytest tests/test_stage3_terminal_outputs.py
python -m pytest tests/test_stage4_object_inventory.py
python -m pytest tests/test_stage5_quality.py
```

## 故障排查

### 配置问题

1. **项目没有硬编码任何API密钥或路径**：所有配置均通过环境变量或命令行参数传递
2. 如果未设置 `ARK_API_KEY`，运行时会报错并提示您设置该环境变量
3. 可以使用 `--api-key` 参数临时指定API密钥

### API调用失败

1. 检查环境变量 `ARK_API_KEY` 是否正确设置
2. 验证 `API_BASE_URL` 可访问
3. 检查 `MODEL_NAME` 是否有效

### Stage 5 SAM3问题

1. 确保SAM3模型文件存在于正确路径
2. 检查CUDA是否可用
3. 考虑使用 `--sam3-legacy-fallback` 降级到Grounding DINO

### 内存问题

1. 减少批处理大小
2. 使用 `--sam3-chunk-size` 调整SAM3分块大小
3. 监控资源使用并调整系统配置

## 许可证

本项目仅供研究使用。

## 联系方式

如有问题，请提交Issue或联系项目维护者。