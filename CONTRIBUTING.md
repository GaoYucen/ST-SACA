# 协作与论文审阅

`main` 是已审阅基线。学生请使用 `paper/<姓名>/<主题>` 或 `code/<姓名>/<主题>` 分支，通过 Pull Request 合并。禁止直接推送 main、强推、删除 main 和绕过审阅。

```bash
git switch main
git pull --ff-only origin main
git switch -c paper/pengfei/introduction
# 编辑并检查差异
git diff --word-diff -- paper/
git add paper/ST-SACA.tex
git commit -m "paper: clarify introduction motivation"
git push -u origin paper/pengfei/introduction
```

在 GitHub 创建目标为 main 的 Draft PR，完成后转为 Ready for review，请求 @GaoYucen 审阅。老师在 Files changed 中查看差异、逐行批注；学生在同一分支追加提交并回复批注，不另建重复 PR。每次更新后须重新获得批准，全部讨论解决、检查通过后由老师合并。建议 squash merge；不混入无关格式化、换行或文件重命名。

论文入口、编译方式及导入边界见 [paper/README.md](paper/README.md)。只提交源文件及必要图像；编译输出通过 PR 附件或 Actions artifacts 提供。替换二进制图时说明生成代码/数据及变化，保留可编辑源文件（若有）。论文中的数字和论断不得脱离证据修改；改变协议、预算或基线必须明确写入 PR。

仓库公开，请勿上传投稿门户页面、账号信息、审稿人身份、原始受限数据或其他未授权材料。
