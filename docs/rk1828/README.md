# RK1828 LLM 部署指南（Qwen3）

在 **RK3576/RK3588 host + RK1828 PCIe 协处理器** 上端到端跑通 LLM 推理的完整流程，以 Qwen3-0.6B 为例。按顺序阅读：

| 文档 | 内容 | 在哪执行 | 需要板子 |
|---|---|---|---|
| [01-模型转换.md](01-模型转换.md) | HF 模型 → ONNX → `.rknn`（`--platform rk1828`），含可选 PC 模拟器验证 | PC (x86_64) | 否 |
| [02-目标设备环境修复.md](02-目标设备环境修复.md) | 运行库路径、协处理器启动（驱动+固件+代理）、板端 Python 环境 | 板 (dev14) | 是 |
| [03-部署与运行.md](03-部署与运行.md) | 传模型、跑 lite Python 推理 / C++ demo、重启恢复、排障 | PC + 板 | 是 |

## 一句话流程

```
[PC] export_llm.py → export_rknn.py --platform rk1828      （01）
        ↓ scp .rknn/.weight/.embed.bin + tokenizer
[板] 软链运行库 + rknn3_startup start + 建 venv 装 lite      （02）
        ↓
[板] python test_board.py --target rk1828 --prompt "..."   （03）
```

## 验证结果

Qwen3-0.6B 已在 RK1828 上跑通：prefill ~743 tok/s，decode ~144 tok/s，生成连贯中文。

## 关键事实

- 目标板 dev14 实测为 **RK3576 + RK1828 协处理器**（PCI `[1d87:182a]`），非 RK3588。
- 转换用 **full toolkit**（x86）；板上推理用 **lite wheel**（aarch64，不需要 torch）。
- 协处理器需 `rknn3_startup` 启动（加载 `pcie-rkep.ko` + 上传 `rknn3_rk1820.img`），**板子重启后要重做**（或配 systemd 自启）。
