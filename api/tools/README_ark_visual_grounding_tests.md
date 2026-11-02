# 火山方舟官方 Grounding 协议测试脚本（OpenAI SDK）

这组脚本使用 OpenAI SDK 调用火山方舟的 OpenAI 兼容接口，并严格采用官方《视觉定位 Grounding》的模型输出协议：

- 客户端 SDK：`openai.OpenAI`
- 调用方法：`client.chat.completions.create`
- 当前模型：`doubao-seed-2-1-pro-260628`
- 图片输入：本地图片 Base64 编码后的 `image_url`
- Point 模型输出：`<point>x y</point>`
- Bounding box 模型输出：`<bbox>x1 y1 x2 y2</bbox>`
- 坐标范围：`[0,999]`
- 归一化网格：`1000×1000`
- 像素换算：`int(coord * image_size / 1000)`

官方文档：<https://www.volcengine.com/docs/82379/1616136?lang=zh>

模型返回后，脚本才在本地生成 JSON、计算误差并绘制可视化。单目标 Point/Bbox 请求不会使用 Responses API，也不会要求模型返回 JSON Schema。

官方文档示例使用 Ark SDK，但 SDK 只是客户端封装。本项目按要求使用 OpenAI SDK 的 Chat Completions，并保持相同的 Ark endpoint、图片消息结构、模型、Grounding 标签和坐标换算。

## 环境

项目虚拟环境已经安装 OpenAI SDK。重新安装时运行：

```bash
.venv_litchi_pipeline/bin/pip install -r tools/requirements_ark_visual_grounding.txt
```

默认 API key、endpoint 和模型名称从 `ark_image_understanding_current.py` 继承，也可以通过 `ARK_API_KEY`、`API_BASE_URL`、`MODEL_NAME` 覆盖。

## 1. 左右手识别

文件：`test_ark_left_right_hands.py`

左右手是多目标分类加 Grounding 任务。官方多目标示例允许使用 JSON 数组，因此模型输出形式为：

```json
[
  {
    "category": "left_hand",
    "bbox": "<bbox>x1 y1 x2 y2</bbox>",
    "point": "<point>x y</point>"
  },
  {
    "category": "right_hand",
    "bbox": "<bbox>x1 y1 x2 y2</bbox>",
    "point": "<point>x y</point>"
  }
]
```

所有坐标仍使用官方标签和 `[0,999]` 范围。脚本在本地检查 point 是否位于对应 bbox 内。

```bash
.venv_litchi_pipeline/bin/python tools/test_ark_left_right_hands.py IMAGE.jpg \
  --expected-order left,right \
  --require-pass
```

`--expected-order` 按画面从左到右填写人物的解剖学左右手真值，而不是画面左右区域名称。

## 2. Point Grounding

文件：`test_ark_point_grounding.py`

模型只返回：

```xml
<point>x y</point>
```

运行：

```bash
.venv_litchi_pipeline/bin/python tools/test_ark_point_grounding.py IMAGE.jpg \
  --target "红柄剪刀的转轴"
```

提供人工像素真值后，脚本在本地计算欧氏距离：

```bash
.venv_litchi_pipeline/bin/python tools/test_ark_point_grounding.py IMAGE.jpg \
  --target "红柄剪刀的转轴" \
  --gt-point 1110,750 \
  --max-error-px 25 \
  --require-pass
```

## 3. Bounding Box Grounding

文件：`test_ark_bbox_grounding.py`

模型只返回：

```xml
<bbox>x_min y_min x_max y_max</bbox>
```

运行：

```bash
.venv_litchi_pipeline/bin/python tools/test_ark_bbox_grounding.py IMAGE.jpg \
  --target "红柄剪刀"
```

提供人工像素框后，脚本在本地计算 IoU：

```bash
.venv_litchi_pipeline/bin/python tools/test_ark_bbox_grounding.py IMAGE.jpg \
  --target "红柄剪刀" \
  --gt-box 790,685,1520,1070 \
  --min-iou 0.5 \
  --require-pass
```

## 官方坐标定义

```text
左上角：接近 (0, 0)
右下角：接近 (999, 999)
x 向右增大
y 向下增大
坐标范围：[0,999]
归一化网格：1000×1000
```

官方像素换算：

```python
x_pixel = int(x_normalized * image_width / 1000)
y_pixel = int(y_normalized * image_height / 1000)
```

脚本还会把结果限制在原图合法像素范围内。

## 输出文件

默认输出到：

```text
ark_visual_grounding_results/<图片名>/<测试类型>_official/
```

每次测试包含：

- `request_prompt.txt`：发送给模型的官方标签 prompt
- `raw_response.txt`：模型原始 `<point>`、`<bbox>` 或多目标标签响应
- `result.json`：本地解析后的 `[0,999]` 坐标、像素坐标、usage 和评测结果
- `overlay.svg`：自包含可视化
- `overlay.png`：普通图片查看器可直接打开的可视化

模型原始响应和本地结果 JSON 被明确分开，不会把本地 JSON 结构误认为官方模型输出格式。

## Dry Run

```bash
.venv_litchi_pipeline/bin/python tools/test_ark_point_grounding.py IMAGE.jpg \
  --target "红柄剪刀的转轴" \
  --dry-run
```

Dry run 会展示 OpenAI SDK、Chat Completions、官方坐标范围和实际 prompt，但不会调用 API。
