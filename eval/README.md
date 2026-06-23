# Eval 使用说明

## 1. 仅跑检索评测
```bash
cd /Users/xujiajin/Desktop/RAG-Agent/代码/智扫通Agent
python3 eval/run_eval.py
```

输出：
- `eval/metrics_report.md`
- `eval/metrics_details.json`

## 2. 跑检索 + 回答质量评测
```bash
cd /Users/xujiajin/Desktop/RAG-Agent/代码/智扫通Agent
python3 eval/run_eval.py --with-answer-eval
```

新增回答指标：
- 回答成功率
- 回答错误率
- 参考行合规率（是否包含`参考：`）
- 平均关键词覆盖率
- 回答平均延迟

## 3. 注意事项
- 回答质量评测依赖在线大模型服务；如果网络或API不可用，回答成功率会下降。
- 检索评测在向量接口不可用时会自动降级为关键词检索，保证评测流程可完成。
