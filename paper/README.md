# ST-SACA 论文源文件

主稿入口：`ST-SACA.tex`，参考文献：`ref.bib`，图片：`fig/`。投稿附信源文件位于 `letters/cover/` 和 `letters/difference/`，分别在各自目录编译。

2026-10-15 从 SACA_TMC_TKDE 根目录导入主稿，保持文件字节不变。`Main/` 是路径扁平化的重复稿，未作为第二个编辑入口导入。仅收录 TeX 实际引用的图片；保留各附信自己的参考文献和图片。未导入缓存、重复 PDF/zip、旧版、门户记录及与附信无关的 Figure 目录。完整源文件对应关系和初始 SHA-256 见 `IMPORT_MANIFEST.json`；该清单仅记录导入快照，不要求后续编辑继续匹配初始哈希。

## 编译

安装 TeX Live（包括 IEEEtran、latex-extra、science、推荐字体及 latexmk）。

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build ST-SACA.tex
cd letters/cover
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
cd ../difference
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build difference_letter.tex
```

生成 PDF 不提交 Git。GitHub Actions 会编译并提供 PDF 供审阅；历史内容中的结果、引用及期刊适配尚未通过本次导入重新核验。

学生协作见 [CONTRIBUTING.md](../CONTRIBUTING.md)。请保留现有段落布局，避免整篇重排，便于逐行及逐词审阅。

导入检查：主稿、cover letter、difference letter 均已在本地 TeX Live 2026 编译成功。源文件原有空白及排版警告保留，未作学术内容修改。
