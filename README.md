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

运行前将 DLL 目录加入 PATH：

```powershell
$venv = ".venv\Lib\site-packages"
$env:PATH = "$venv\nvidia\cublas\bin;$venv\nvidia\cuda_runtime\bin;$venv\nvidia\cuda_nvrtc\bin;$venv\nvidia\cudnn\bin;$env:PATH"
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
```



### 参数

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