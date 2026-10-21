# 上传 GitHub

解压源码压缩包，进入 `cbct_vertebra_pipeline` 文件夹。此文件夹就是仓库根目录，README.md 和 run_case.py 位于根层。

## 网页上传

1. 在 GitHub 新建一个空仓库；首次可选择 Private。
2. 在仓库中选择上传文件，将解压后的源码文件上传，不要直接上传 zip。
3. 确认包含 `.gitignore`，检查文件列表后提交。

## 使用 Git

在解压后的源码文件夹打开终端，执行：

```bash
git init
git add .
git diff --cached --stat
```

确认待提交项只有源码和文档，再执行：

```bash
git commit -m "Add exploratory CBCT vertebra segmentation pipeline"
git branch -M main
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```

将 `YOUR_REPOSITORY_URL` 换成你创建的仓库地址。

`.gitignore` 已排除常见医学影像、模型、数据清单、日志和环境目录。它不会移除之前已被 Git 跟踪的文件；使用新空仓库并核对待提交列表。

## 项目范围

这是基于 SPINEPS CT 预训练模型的 CBCT 探索性推理封装，尚未完成 CBCT 专门训练及可靠精度验证，当前解剖编号存在错误。没有包含 CT–CBCT 配准实现。

第三方依赖及模型的许可分别适用。本源码包未替你选择自有代码的开源许可证；需要对外授予复用许可时，再由仓库所有者选择并添加 LICENSE。
