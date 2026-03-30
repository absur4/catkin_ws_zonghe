# Open Vocabulary Object Detection and Segmentation

本目录包含基于 Grounded-SAM-2 的开放词汇物体检测和分割功能。

模型权重在链接: https://pan.baidu.com/s/1gBeEduYcSHdajqCQIQTOwg?pwd=qcxs 提取码: qcxs 
--来自百度网盘超级会员v6的分享
其中一个checkpoint在
SEU_robocup2026/perception/objects/open_vocabulary/Grounded-SAM-2/grounding_dino/groundingdino/config/bert-base-uncased/model.safetensors
## 📁 目录结构

```
open_vocabulary/
├── README.md                    # 本文件
├── Grounded-SAM-2/             # Grounded SAM 2 项目
│   ├── checkpoints/            # SAM 2 模型权重
│   ├── gdino_checkpoints/      # Grounding DINO 模型权重
│   ├── grounding_dino/         # Grounding DINO 源代码
│   ├── sam2/                   # SAM 2 源代码
│   ├── outputs/                # 默认输出目录
│   ├── bag.jpg                 # 测试图像
│   ├── grounded_sam2_local_demo.py  # 原始 demo 脚本
│   └── ...
└── API/                        # API 接口
    ├── README.md               # API 使用文档
    ├── __init__.py             # Python 包初始化
    ├── grounding_dino_api.py   # Grounding DINO API
    ├── grounding_sam_api.py    # Grounding SAM API
    ├── test_api.py             # API 测试脚本
    └── example_usage.py        # 使用示例
```

## 🚀 快速开始

### 1. 环境配置

确保已经安装所有依赖（详见 [Grounded-SAM-2/INSTALL.md](Grounded-SAM-2/INSTALL.md)）：

```bash
# 基础依赖
pip install torch torchvision
pip install opencv-python supervision pycocotools
pip install hydra-core iopath timm transformers

# 安装 SAM 2
cd Grounded-SAM-2
pip install -e .

# 安装 Grounding DINO
pip install -e grounding_dino --no-build-isolation
```

### 2. 使用 API

#### Python 代码

```python
# 方式 1: 仅检测（快速）
from API.grounding_dino_api import GroundingDINOAPI

api = GroundingDINOAPI()
results = api.detect(
    image_path="image.jpg",
    text_prompt="cat. dog.",
    output_path="output.jpg"
)

# 方式 2: 检测 + 分割（精确）
from API.grounding_sam_api import GroundingSAMAPI

api = GroundingSAMAPI()
results = api.segment(
    image_path="image.jpg",
    text_prompt="cat. dog.",
    output_dir="output/"
)
```

#### 命令行

```bash
# Grounding DINO
cd API
python grounding_dino_api.py --image ../Grounded-SAM-2/bag.jpg --prompt "bag." --output result.jpg

# Grounding SAM
python grounding_sam_api.py --image ../Grounded-SAM-2/bag.jpg --prompt "bag." --output output/
```

### 3. 运行示例

```bash
cd API
python example_usage.py
```

## 📚 详细文档

- **API 使用文档**: [API/README.md](API/README.md)
- **安装指南**: [Grounded-SAM-2/INSTALL.md](Grounded-SAM-2/INSTALL.md)
- **项目说明**: [Grounded-SAM-2/README.md](Grounded-SAM-2/README.md)

## 🎯 功能特性

### 1. Grounding DINO API
- ✅ 基于文本提示的物体检测
- ✅ 返回边界框和置信度
- ✅ 速度快，适合实时应用
- ✅ 支持开放词汇（任意物体类别）

### 2. Grounding SAM API
- ✅ 结合检测和分割
- ✅ 生成精确的分割掩码
- ✅ 输出标注图像和 JSON 结果
- ✅ 支持多物体同时分割

## ⚙️ 配置说明

### 设备选择

API 自动选择最优计算设备：
1. **CUDA** (NVIDIA GPU) - 最快
2. **MPS** (Mac GPU) - 较快
3. **CPU** - 最慢但兼容性最好

### 参数调整

```python
# 调整检测阈值
api = GroundingDINOAPI(
    box_threshold=0.35,   # 边界框置信度阈值（0-1）
    text_threshold=0.25   # 文本匹配阈值（0-1）
)
```

**建议值：**
- 严格检测（减少误检）：`box_threshold=0.5`
- 宽松检测（检测更多）：`box_threshold=0.25`

## 📝 使用注意事项

1. **文本提示格式**
   - 使用英文
   - 小写字母
   - 以 `.` 结尾
   - 多个物体用 `. ` 分隔
   - 示例：`"cat. dog. person."`

2. **模型权重**
   - SAM 2.1 Large: `checkpoints/sam2.1_hiera_large.pt` (856MB)
   - Grounding DINO: `gdino_checkpoints/groundingdino_swint_ogc.pth` (662MB)
   - 首次使用前请确保权重已下载

3. **性能优化**
   - 重用 API 实例，避免重复加载模型
   - 对于批量处理，建议使用 Grounding DINO API
   - 仅在需要精确分割时使用 Grounding SAM API

## 🔧 故障排除

### 问题 1: 找不到模型权重
```bash
# 检查权重文件是否存在
ls Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt
ls Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth
```

### 问题 2: CUDA/MPS 不可用
```python
# 强制使用 CPU
api = GroundingDINOAPI(device="cpu")
```

### 问题 3: 导入错误
```bash
# 确保在正确的目录
cd /path/to/perception/objects/open_vocabulary/API
python test_api.py
```

## 📊 性能基准

在 MacBook Pro (M1 Pro) 上的测试结果：

| API | 图像大小 | 检测时间 | 分割时间 | 总时间 |
|-----|---------|---------|---------|--------|
| Grounding DINO | 1000x1000 | ~2秒 | - | ~2秒 |
| Grounding SAM | 1000x1000 | ~2秒 | ~3秒 | ~5秒 |

*注：首次运行需要额外的模型加载时间*

## 🤝 贡献

如有问题或建议，请提交 issue 或 pull request。

## 📄 许可证

本项目基于 Grounded-SAM-2，遵循其原始许可证。详见：
- SAM 2: Apache 2.0
- Grounding DINO: Apache 2.0

## 🔗 相关链接

- [Grounded-SAM-2 GitHub](https://github.com/IDEA-Research/Grounded-SAM-2)
- [SAM 2 Paper](https://ai.meta.com/sam2/)
- [Grounding DINO Paper](https://arxiv.org/abs/2303.05499)