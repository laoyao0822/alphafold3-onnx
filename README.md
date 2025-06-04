
# AlphaFold3 OpenVINO/ONNX 推理优化

本项目基于AlphaFold3模型，实现了ONNX模型导出和OpenVINO推理加速功能，支持CPU/GPU高效推理与分布式计算。

## 环境要求

- Python 3.11
- PyTorch 2.7(如果要导出为onnx，推荐临时切换2.8)
- OpenVINO 2024.1
- Intel® Extension for PyTorch (IPEX)

```bash
pip install openvino-dev[pytorch]==2023.0.0 intel-extension-for-pytorch==2.0.0 rdkit
```

## 模型文件准备
1. 将原始AlphaFold3模型置于默认目录：
   ```bash
   mkdir -p ~/models/model_103275239_1
   cp /path/to/original_model/* ~/models/model_103275239_1/
   ```
2. 公共数据库路径：
   ```bash
   mkdir -p ~/public_databases
   ```

## 运行方式

### 基础推理命令
```bash
python main.py \
  --json_path=input.json \
  --output_dir=results/ \
  --cpu_inference=True \
  --use_ipex=True \
  --num_cpu_threads=16
```

### 关键参数说明
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `json_path` | None | 输入JSON文件路径 |
| `output_dir` | None | 结果输出目录 |
| `model_dir` | `~/models/model_103275239_1` | 模型文件目录 |
| `cpu_inference` | True | 启用CPU推理模式 |
| `use_ipex` | True | 启用Intel® PyTorch扩展加速 |
| `num_cpu_threads` | 16 | CPU线程数 |
| **OpenVINO加速模块** | | |
| `use_diffusion_vino` | False | 在扩散模块使用OpenVINO |
| `use_confidence_vino` | False | 在置信度模块使用OpenVINO |
| `use_evo_vino` | False | 在evoformer模块使用OpenVINO |
| **高级功能** | | |
| `save_onnx` | False | 导出ONNX模型 |
| `save_onnx_path` | `/root/asc25` | ONNX模型保存路径 |
| `world_size` | 1 | 分布式节点数量 |
| `rank` | 0 | 当前节点排名 |

### 示例场景

1. **导出ONNX模型**：
   ```bash
   python main.py \
     --json_path=input.json \
     --save_onnx=True \
     --save_onnx_path=/path/to/onnx_models
   ```
2 **导出为OpenVINO**加速：
  ```bash
   python convert_to_openvino.py
   ```


3. **全OpenVINO加速**：
   ```bash
   python main.py \
     --use_diffusion_vino=True \
     --use_confidence_vino=True \
     --use_evo_vino=True \
     --num_cpu_threads=32
   ```



## 输出文件
- `output_dir


## 性能优化建议
1. **CPU设置**：
   - 推荐使用`--num_cpu_threads`设置为物理核心数
   - 启用`--use_ipex=True`
