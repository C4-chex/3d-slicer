# CBCT 椎骨分割：服务器试验包

当前阶段：用 SPINEPS 2.1.3 的 CT 预训练权重，对用户现有 CBCT 做分割与逐椎骨编号的可行性测试。尚未训练、微调或验证准确率。该项目不是已完成的 CBCT 专用模型。

## 已固定的方案

- 语义结构分割：CT_semantic v2.0.0，工作分辨率 0.8 mm。
- 椎骨实例分割：CT_instance v2.0.0。
- 解剖编号：VERIDAH CT_labeling v1.4.0。
- 官方权重来源和 SHA256 固定在 `model_manifest.json`，下载时验证长度、SHA256 和压缩包 CRC。
- 输入为一份三维 NIfTI；输出逐椎骨掩膜、合并标签图、中心点、叠加预览和运行记录，可选生成 RAS 毫米坐标的 PLY 网格。

这批扫描约为 20 cm 的局部视野，因此试验跳过上游额外的 CT 脊柱定位裁剪器，直接处理整个输入体积。上游 `needs_corp` 仅在运行内存配置中关闭，权重文件不改动。关闭镜像测试增强以缩短首次测试，可通过 `--tta` 开启做正式对照。不会自动下载其他定位模型。

`--offset` 默认 0，即不平移输入灰度。非零值仅供明确记录的对照实验，不代表 HU 校准，也不能直接推广到其他病例。

## 现在先准备什么

服务器试验起点可考虑单张 NVIDIA 24GB 显存卡、64GB 内存、12–16 vCPU、100–200GB SSD，按量使用。具体峰值显存尚未经租用 GPU 实测；下方脚本先做 CUDA 自检。若显存不足，先保持 `--batch-size 1`，再依据实际报错调整方案。

选择 Python 3.11 或 3.12、PyTorch 2.8 或更新且 CUDA 已能使用的 Linux 镜像。MONAI 1.6.1 要求 PyTorch >=2.8。不要先选择旧 PyTorch 镜像再依赖脚本自动更换 CUDA。安装脚本保留镜像中的 torch/torchvision/torchaudio 版本，不把它们替换成 CPU 包。

代码压缩包不包含任何医学影像、权重、登录信息或本地虚拟环境。将代码上传后，模型可在服务器从官方 GitHub 下载；医学数据需另行放入服务器的数据目录。当前脚本不会把本地影像发送到任何在线推理服务。

## Linux 服务器运行

假设项目上传到 `/root/autodl-tmp/cbct_vertebra_pipeline`，测试影像上传到 `/root/autodl-tmp/cbct_data/`。

```bash
cd /root/autodl-tmp/cbct_vertebra_pipeline
bash bootstrap.sh
source .venv/bin/activate
python download_models.py
python preflight.py --device cuda --require-models
python audit_dataset.py --input /root/autodl-tmp/cbct_data --output manifests/server_inventory.json

python -u run_case.py \
  --input /root/autodl-tmp/cbct_data/0008_0000.nii.gz \
  --output runs/0008_raw_gpu \
  --device cuda --batch-size 1 --mesh > runs_0008.log 2>&1

python verify_output.py --run runs/0008_raw_gpu
```

如访问 PyPI 较慢，可将安装命令改为：

```bash
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple bash bootstrap.sh
```

首先只跑一例，下载结果目视检查。不能因为生成了文件就开始批量处理或训练。每个新配置用新的输出目录，避免混淆结果；脚本会拒绝覆盖已有运行记录。

CPU 功能测试：在已安装依赖的环境中，将 `--device cuda` 改为 `--device cpu`；CPU 不用于正式训练。若只调试实例分割，可加 `--no-labeling`，此时 `instance_01` 等只是实例 ID，不是解剖编号。

## 结果怎么看

- `overlay.png`：原图与模型预测在三个方向的叠加。显示的是存储轴切片，不能单靠预览给出临床方向判断。
- `semantic_predicted_unverified.nii.gz`：模型的结构分区结果。
- `instances_raw_unverified.nii.gz`：上游原始实例/编号结果，保留供排错。
- `vertebrae_predicted_unverified.nii.gz`：仅保留椎骨类别及骨性分区的合并标签图。
- `per_vertebra/*.nii.gz`：每节椎骨独立二值掩膜，名称为模型预测，未经人工确认。
- `instances_ordered_unverified.nii.gz`、`per_instance/*.nii.gz`：按原始 NIfTI 的上方向坐标排序的临时实例 ID，1、2、3 等不表示 C/T/L 解剖编号，且依赖输入方向信息正确。
- `predicted_vertebrae.csv`：模型标签、体积、中心点及是否接触图像边界。
- `centroids_unverified.json`：上游产生的中心点。
- `meshes_RAS_mm/*.ply`：可选网格，顶点处于原始 NIfTI 的 RAS 毫米坐标；在使用 LPS 的系统中需明确转换，不能直接混用。
- `run_report.json`：软件版本、硬件、参数、耗时、成功/失败状态与限制。
- `geometry_checks.json`：形状、仿射矩阵、逐节掩膜与合并图一致性检查；通过不代表边界和编号正确。

保存结果时排除骶骨、骨盆、椎间盘和金属类别，但模型可能误识别，因此不能保证所有实际骨盆体素均正确排除。

## 怎样进入训练阶段

现阶段不提供伪装成训练的空跑脚本。训练或监督微调前，需要可靠的 CBCT 参考分割和解剖编号，并按独立患者划分训练与评估病例。失败预测不能直接当真值；患者重复扫描不能随机分入训练与测试两侧。若选择无配对域适配，则需核实对应实现，并准备有标注 CT 与 CBCT 数据；仍需参考标注检验结果。

已有 27 份局部扫描，不等于 27 位独立患者，也不能据此验证全颈胸腰能力。先用少量代表性病例完成目视审核和必要的参考标注，再确定是否微调及其预算。

## 来源

- [SPINEPS 官方代码](https://github.com/Hendrik-code/spineps)
- [SPINEPS 文档](https://spineps.readthedocs.io/)
- [SPINEPS 原论文](https://doi.org/10.1007/s00330-024-11155-y)（原论文针对 T2w MRI，不能当作 CT/CBCT 的验证证据）

官方代码以 Apache-2.0 发布，相关权重版本和下载来源已记录。生成的预测均需检查，不代表已达到可用分割精度。


## 源码包与 GitHub

本仓库包含推理封装、模型下载校验、结果导出和几何检查代码；SPINEPS 等第三方实现通过依赖安装，不在本仓库重复分发。模型权重按 `model_manifest.json` 单独下载。

源码打包命令为 `python package_project.py --output /path/to/cbct_github_source.zip`。上传 GitHub 的步骤见 `GITHUB_UPLOAD.md`。压缩包不包含本地专用启动器、医学影像、模型权重、运行环境或运行日志。
