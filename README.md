# Audio Extractor

基于 [faster-whisper](https://github.com/SYSTRAN/faster-whisper) 的音频文字提取工具。

## 安装

```bash
uv sync
```

### CUDA（Windows）

GPU 加速需要额外安装 CUDA 12 运行库：

```bash
uv sync --extra cuda
```

## 使用

```bash
# 提取纯文本（输出到 stdout）
uv run audio-extractor audio.mp3

# 指定语言 + 输出 SRT 字幕
uv run audio-extractor audio.mp3 -l zh -f srt -o output.srt

# 输出 JSON（含时间戳）
uv run audio-extractor audio.mp3 -f json -o output.json

# 使用 GPU 加速
uv run audio-extractor audio.mp3 -d cuda -c float16

# 批量处理目录下所有音频
uv run audio-extractor-batch C:/audio --limit 10
```

### 配置文件

两个命令默认读取当前工作目录下的 `audio-extractor.toml`，也可通过 `--config` 指定其他文件。
配置优先级为：命令行参数 > 配置文件 > 内置默认值。

- `[extractor]`：`model`、`language`、`device`、`compute_type`、`beam_size`、`format`。
- `[batch]`：继承 `[extractor]` 中显式设置的公共转录参数，再应用本节的覆盖值；另支持 `limit`，输出固定为 SRT。

输入、输出路径通过命令行传入。默认配置文件不存在时使用内置默认值；显式指定的文件不存在、配置项未知或值无效时会报错。

```powershell
uv run audio-extractor audio.mp3 --config C:/config/settings.toml -f srt
uv run audio-extractor-batch C:/audio --config C:/config/settings.toml --limit 10
```

### 参数

以下默认值均为内置默认值，可由配置文件覆盖。

| 参数 | 说明 |
|------|------|
| `audio` | 输入音频文件路径 |
| `-m, --model` | 模型大小：`tiny`, `base`, `small`（默认）, `medium`, `large-v3` |
| `-l, --language` | 语言代码，如 `zh`, `en`；省略则自动检测 |
| `-o, --output` | 输出文件路径 |
| `-f, --format` | 输出格式：`text`, `srt`, `vtt`, `json`（默认 `text`） |
| `-d, --device` | 运行设备：`cpu`（默认）, `cuda` |
| `-c, --compute-type` | 计算类型：`auto`, `float16`, `float32`, `int8_float16`, `int8` |
| `-b, --beam-size` | beam search 宽度（默认 `5`） |

### 批量参数

| 参数 | 说明 |
|------|------|
| `root` | 扫描目录路径 |
| `--model` | 模型大小（默认 `small`） |
| `--language` | 语言代码（默认 `zh`） |
| `--device` | 运行设备（默认 `cuda`） |
| `--compute-type` | 计算类型（默认 `auto`） |
| `--beam-size` | beam search 宽度（默认 `5`） |
| `--limit` | 限制处理文件数（默认 `0`=全部） |
