# robocup_hri

人机交互模块，提供 TTS 接口供其他模块调用。

## 当前功能

- **TTS 话题** (`/tts/say`, `std_msgs/String`): 接收文本并播报
  - `ExecutePlace` 在目的地为洗碗机时发布通知 referee 开门的语音（rulebook2026 Rule #4）

## 计划功能

- 语音识别（ASR）
- 手势识别
- 任务状态 UI 界面

## 接口

```
订阅：
  /tts/say  [std_msgs/String]  — 文本转语音
```

## 使用

```bash
# 手动测试 TTS
rostopic pub /tts/say std_msgs/String "data: 'Please open the dishwasher door.'" --once
```
