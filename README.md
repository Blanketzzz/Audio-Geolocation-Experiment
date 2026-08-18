# GeoSound Lab 实验日志

零依赖静态实验日志网站。数据与页面分离，历史记录位于 `data/logs.json`。

## 启动

```bash
cd "/workspace/Visual-audio Geo-localization/experiment_log_web"
python -m http.server 8765 --bind 0.0.0.0
```

浏览器访问 `http://服务器地址:8765`。如果只在当前机器访问，使用 `http://127.0.0.1:8765`。

## 新增日志

复制 `data/logs.json` 中的一条 `entries` 记录，修改日期、分类、实验协议、表格、图表、结论、限制和原始文件链接。页面会自动生成时间线和筛选器。

建议每次实验至少记录：研究问题、代码版本、训练/验证/测试样本、随机种子、输入表征、训练目标、核心指标、失败现象、可以说明什么、不能说明什么。
# Visual–Audio Geolocation Experiment Log

Static research log for controlled experiments on visual–audio geolocation,
including the research map, protocols, plots, quantitative results, negative
findings, and current hypothesis boundaries.

## View locally

```bash
python3 -m http.server 8765
```

Then open `http://127.0.0.1:8765`.

## GitHub Pages

This directory is a self-contained static site. After pushing it to GitHub,
enable Pages for the `main` branch and `/ (root)` directory.
