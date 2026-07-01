# GeoChef 对 Synoptic-Bench 的数据参考总结

> 论文：**GeoChef: A Data-Centric Guide to Tailoring Vision-Language Models for Remote Sensing**  
> 期刊/状态：IEEE Geoscience and Remote Sensing Magazine；项目仓库记录其于 2026-05-21 被 GRSM 接收。  
> DOI：https://doi.org/10.36227/techrxiv.176978652.29736845/v1  
> 作者：Yue Zhou, Shujun Zhao, Ruigang Li, Xue Yang, Mengcheng Lan, Chaofeng Chen, Tianwen Zhang, Lingfei Ma, Hongjie He, Jonathan Li。  
> 配套数据清单：https://github.com/VisionXLab/Awesome-RS-VL-Data

说明：本笔记结合 GeoChef 论文公开摘要、DOI 元数据、配套 Awesome-RS-VL-Data 数据清单，以及当前目录中的 Synoptic-Bench 代码、数据构建报告和训练日志整理。由于 TechRxiv PDF 镜像在本机网络环境下解析失败，本文不引用无法本地逐页核验的实验数值；重点提取可核验的数据方法论和对当前项目可操作的启发。

## 1. GeoChef 的核心信息

GeoChef 的中心观点是：遥感 VLM 的创新路径正在从“改模型结构”转向“用高质量遥感垂直领域数据适配通用 VLM”。大模型参数规模和训练成本很高，直接重训或频繁改架构不现实，因此更实际的路线是围绕数据做系统化设计：收集、清洗、标注、配比、任务拆解、评测和持续迭代。

论文把遥感视觉语言数据看作“配方”问题：不同任务需要不同数据原料、标注粒度和训练阶段。配套仓库将数据按用途整理为几大类：

- **Comprehensive Data**：综合指令微调数据，如 GeoChat-Instruct、SkyEye-968k、MMRS-1M、VRSBench-Train、SARChat-Bench-2M、REOBench、ThinkGeo、A2Seek、GeoReason-Bench 等。
- **Comprehensive Benchmarks**：综合评测集，如 GeoChat-Bench、LHRS-Bench、VRSBench、VLEO-Bench、GEOBench-VLM、XLRS-Bench、RSIEval、AirSpatial、SARChat-Bench-2M 等。
- **Image Captioning / Retrieval**：图文匹配和描述数据，如 UCM-Captions、Sydney-Captions、RSICD、RSITMD 等。
- **VQA**：遥感视觉问答数据，如 RSVQA-LR/HR、RSVQAxBEN、FloodNet、RSIVQA、CDVQA、VQA-TextRS、CRSVQA、RemoteCount、EarthVQA、GeoLLaVA、QAG-360K 等。
- **Meta Data**：底层遥感任务数据，覆盖分类、检测、分割、变化检测、地理定位、事件识别等，如 UCM、AID、NWPU-RESISC45、BigEarthNet、RSOD、iSAID、LoveDA、SEN1-2、SpaceNet6、SatlasPretrain、GeoPile、SSL4EO-S12、QuakeSet 等。
- **Models / Papers**：遥感 MLLM、视觉语言预训练模型和 Agent 相关工作，如 GeoChat、SkyEyeGPT、EarthGPT、LHRS-Bot、RS-LLaVA、SkySenseGPT、GeoGround、RS5M/GeoRSCLIP、RemoteCLIP、SkyScript、TEOChat、GeoPix、GeoPixel、ImageRAG 等。

对你的项目最重要的不是“直接照搬某个遥感数据集”，而是 GeoChef 强调的三点：

1. **数据比模型结构更关键**：你当前已经在用 Qwen3-VL 做 LoRA，这符合“用垂直领域数据适配通用 VLM”的路线。
2. **任务要拆细**：不能只训练“看图写预报讨论”，还可以拆成高低压识别、槽脊定位、温度距平判断、风场影响、区域归因、时间趋势等子任务。
3. **评测要领域化**：传统 ROUGE/BERTScore 只能看文本相似度，必须加入现象、极性和位置一致性的评测。你项目里的 SPACE 正是在做这件事。

## 2. 当前目录项目概况

当前仓库是 **Synoptic-Bench**，目标是让 VLM 根据天气图生成类似 National Weather Service Area Forecast Discussions 的大尺度天气预报讨论。

README 中定义的完整数据集规模为：

- **1,367,041 条 AFD 文本样本**。
- 图像来自 GFS 预报变量的可视化：500 mb geopotential height、2 m temperature、850 mb wind velocity。
- 评估框架为 **SPACE**，用于衡量生成文本是否描述了正确极性和位置的天气尺度现象。
- 基准模型包括 LLaVA、Qwen、LLaMA-3.2 等 VLM/MLLM，以及 Gemini、Nearest Neighbor、Climatology、Blind LLM 等基线。

当前本地数据构建状态：

- `data/images/American/generation_report.json` 显示 JSON 侧样本量为：
  - train: 457,230
  - val: 50,727
  - test: 73,351
- `data/synoptic_lf/build_report.json` 显示转换为 LLaMA-Factory 格式并过滤缺失图像后：
  - train: 327,222 / 457,230
  - val: 26,944 / 50,727
  - test: 57,423 / 73,351
- 缺失图像数量：
  - train: 130,008
  - val: 23,783
  - test: 15,928
- `data/synoptic_lf/dataset_info.json` 已把数据注册成 ShareGPT 风格的多模态数据集：
  - `synoptic_train`
  - `synoptic_val`
  - `synoptic_test`

当前训练状态：

- 日志 `outputs/logs/synoptic_qwen3vl_8b_lora_fa2_img64_full_train.log` 显示正在/曾经使用 **Qwen3-VL-8B-Instruct** 做 LoRA 微调。
- 训练加载了 `synoptic_train.json` 和 `synoptic_val.json`。
- 已生成 tokenized dataset：`data/tokenized/synoptic_qwen3vl_8b_fa2_img64_full`。
- 训练样例输入是天气图 + 地区提示，例如 Las Vegas, Nevada，并要求生成大尺度天气讨论。
- 日志中训练进度到约 73%-74% 时，loss 在约 1.82-1.93 区间波动；早期 loss 从约 3.59 降到 2.x，再到 1.8x，说明模型已经学到相当多的领域文本格式和天气讨论模式。

当前预处理逻辑：

- `preprocessing/prepare_dataset_in_parallel_synoptic.py` 从 HDF5 中读取 AFD 与 GFS 预报变量。
- 使用 3h 到 48h lead time，计算 0-48h 平均：
  - `GH500`
  - `U850`
  - `V850`
  - `t2m`
- 使用月平均气候态计算 `t2m_anomaly`。
- 输出图像包含：
  - 2m 温度距平填色
  - 500mb 位势高度等值线
  - 850mb 风羽
  - 黄色框标注本地预报区域
- 文本过滤聚焦 synoptic 关键词，如 trough、ridge、high/low pressure、front、warmer/cooler/freezing 等。

当前评估逻辑：

- `SPACE/SPACE_aggregate.py` 和 `SPACE/SPACE_local.py` 抽取压力系统/槽脊等现象和位置。
- 将现象归为 `HIGH` / `LOW` 极性。
- 用位置层级图计算地理距离和聚类。
- SPACE 分数 = 位置/极性匹配分数 * 覆盖率。
- 这与 GeoChef 的“领域数据 + 领域评测”思想高度一致。

## 3. GeoChef 对当前项目最有价值的启发

### 3.1 不要只追求样本量，要做数据分层

你现在已有几十万可训练样本，本地 train 可用 327,222 条。下一步最有价值的不是单纯继续堆数据，而是把样本按天气现象、区域、季节、难度和文本质量分层。

建议增加以下数据标签：

- 现象类型：ridge、trough、closed low、upper high、front、blocking、temperature anomaly、wind event。
- 极性：HIGH / LOW / warm / cold / neutral。
- 位置粒度：local CWA、state、region、CONUS、ocean-side upstream region。
- 时间范围：0-24h、24-48h、48h+。
- 图像复杂度：单一主导系统、多系统、弱信号、强梯度、边界区域。
- 文本质量：是否包含明显无关短期地面现象、是否只有一句话、是否带过多 NWS 缩写、是否与图像变量不匹配。

这些标签可以支撑更清晰的数据配方，例如：

- 先用高质量、主现象明显的样本训练基础生成能力。
- 再加入多系统、边界区域、弱信号样本提升鲁棒性。
- 对高/低压、槽/脊、温度距平分布不均的数据做重采样，避免模型只学到常见天气叙述。

### 3.2 把“看图写一段话”拆成可控中间任务

GeoChef 的遥感数据清单强调 caption、VQA、grounding、change/temporal、reasoning 等任务并行建设。对应到 Synoptic-Bench，可以把你的训练数据扩展为多任务指令：

- Caption：描述整张天气图的大尺度格局。
- VQA：回答“预报区域附近是槽还是脊？”“温度距平偏暖还是偏冷？”“850mb 风向是否支持暖平流？”。
- Grounding：输出“low pressure over the northern Great Basin”这种现象-位置对。
- Reasoning：解释“为什么该区域将降温/升温/转风？”。
- Temporal：从 0-48h 平均图升级到多时次图像或视频，生成趋势讨论。
- Verification：给定生成文本，让模型判断是否与图像现象一致。

这会让模型少一些模板化 AFD 复述，多一些可验证的天气图理解能力。

### 3.3 用遥感 VLM 数据做“辅助预适配”，但最终能力必须靠气象数据

遥感数据集能帮助模型理解俯视图、地理区域、空间关系、多尺度视觉对象、图文对齐和指令格式；但它们通常不能直接教会模型读 500mb 高度场、温度距平、850mb 风场和 NWS 预报语体。

因此推荐路线是：

1. 通用 VLM 基座。
2. 可选：遥感 VLM 数据做轻量预适配，让模型更熟悉地理/遥感视角。
3. Synoptic-Bench 做主训练。
4. 人工/规则/LLM 构造的气象子任务数据做二阶段强化。
5. SPACE + 传统指标 + 人工抽样做综合评测。

## 4. 能用什么数据达成什么效果

| 数据来源/数据类型 | 在当前项目中的用途 | 预期效果 | 风险/注意点 |
|---|---|---|---|
| 当前 Synoptic-Bench 核心数据：GFS 图像 + AFD 文本 | 主训练数据；训练“天气图 -> forecast discussion” | 生成 NWS 风格的大尺度天气讨论；学习槽脊、高低压、温度趋势、区域表达 | 文本可能包含非图像变量能解释的现象，如雾、海风、局地对流；需要过滤或标注 |
| 当前本地可用 327k train + 26.9k val + 57.4k test | Qwen3-VL LoRA 完整训练和验证 | 已足以形成稳定领域风格；loss 已明显下降 | 缺失图像较多，train 缺 130k，需要补图或确认过滤是否引入时间/区域偏差 |
| 高质量子集：主现象明确、文本和图像一致、长度适中 | 第一阶段 SFT 或 curriculum learning | 更快学会现象-位置-影响的稳定映射 | 需要额外 QC；过度过滤会减少真实复杂度 |
| Hard subset：多系统、弱梯度、边界区域、复杂上游形势 | 第二阶段训练/验证 | 提升复杂天气形势下的判断能力，减少泛化失败 | 需要 SPACE 或规则标签辅助筛选 |
| VQA 式气象问答数据：由现有图像和 AFD/规则生成 | 训练图像理解的中间能力 | 提升对“哪里有 trough/ridge/high/low、偏暖/偏冷、风向如何”的可控理解 | 自动生成问答需抽样审查，防止标签噪声 |
| Grounding 数据：现象-位置对，如 `LOW: Pacific Northwest` | 对齐 SPACE 评估目标 | 提升生成文本中的位置准确性和极性一致性 | 位置层级词典要覆盖气象常用地名和海区 |
| GeoChef 清单中的 caption/retrieval 数据：RSICD、RSITMD、SkyScript、RS5M/GeoRSCLIP 等 | 可选预适配；训练图文对齐和遥感视角描述 | 改善模型对俯视图、地理空间和图文匹配的基础能力 | 对天气图变量的帮助有限，不能替代 Synoptic-Bench |
| GeoChef 清单中的综合指令数据：GeoChat-Instruct、SkyEye-968k、MMRS-1M、VRSBench-Train 等 | 可选混合少量数据，增强遥感指令跟随和空间描述 | 更稳地按提示输出、描述区域关系、处理多任务格式 | 混入比例要小；过多会冲淡气象语域 |
| GeoChef 清单中的 VQA 数据：RSVQA、EarthVQA、FloodNet、CDVQA、RemoteCount 等 | 借鉴格式，或做少量辅助训练 | 增强问答、计数、变化、灾害语义的泛化 | 数据语义与天气图不同，最好只用于格式/能力预热 |
| Temporal/change 数据：CDVQA、LEVIR-MCI、GeoLLaVA、UniRS、TEOChat 等 | 借鉴多时相组织方式，用于未来多 lead-time 图像输入 | 从单张 0-48h 平均图升级为时序天气演变讨论 | 当前 Qwen3-VL 支持 video processor，但需重新设计输入和显存预算 |
| SAR/多光谱数据：SARChat-Bench-2M、SEN1-2、Beyond the Visible 等 | 仅在扩展到卫星云图、雷达、红外/水汽图时使用 | 支持多传感器天气/遥感综合理解 | 对当前 GFS 派生图不是直接必要 |
| XLRS-Bench、ImageRAG、UHR-CoZ 类超高分辨率数据/方法 | 借鉴“粗到细检索/缩放”的评测和推理方式 | 未来可用于高分辨率天气图局部细读 | 当前图像是固定绘图，短期收益不如数据 QC |

## 5. 推荐实验路线

### 实验 A：补齐和审计当前数据

目标：确认当前过滤后的 327k/26.9k/57.4k 是否存在明显偏差。

做法：

- 按年份、月份、NWS station、区域统计缺失图像比例。
- 补生成缺失图像，或在报告中明确只使用完整图像样本。
- 抽样检查图像与文本是否对应同一地区、日期和 lead time。

预期效果：

- 减少训练/验证分布偏差。
- 避免某些年份、区域或季节被系统性过滤。

### 实验 B：构建 Synoptic-VQA 子任务集

目标：让模型显式学会天气图理解，而不只是模仿 AFD 文风。

可以自动生成的问题：

- `Is the forecast region under a ridge or trough?`
- `Is the 2m temperature anomaly mostly above or below climatology near the yellow box?`
- `Which large-scale feature is upstream of the forecast region?`
- `Does the 850mb wind suggest warm advection, cold advection, or weak advection?`
- `Name the main pressure-pattern feature and its approximate location.`

可用数据：

- 现有 GFS 数值场。
- `t2m_anomaly`。
- 500mb 高度场的局部梯度/曲率或规则检测。
- AFD 文本中的关键词和位置。

预期效果：

- 提升 SPACE 分数中的 coverage 和 polarity/location match。
- 减少生成文本中“看起来像预报但位置错了”的问题。

### 实验 C：高质量样本 curriculum

目标：先让模型吃“干净样本”，再吃复杂样本。

第一阶段筛选：

- 文本包含 ridge/trough/high/low/front 等核心 synoptic 词。
- 文本长度适中，例如 30-150 words。
- 排除主要讨论雾、局地对流、海风、火险、航空、海洋但缺少大尺度解释的样本。
- 图像存在明显温度距平或高度场结构。

第二阶段加入：

- 多现象样本。
- 区域边界样本。
- 弱信号样本。
- NWS 缩写较多的真实 AFD 样本。

预期效果：

- 训练更稳定。
- 降低模板化输出。
- 对复杂形势泛化更好。

### 实验 D：少量遥感指令数据预适配

目标：验证 GeoChef 清单中遥感数据是否能提升图文/空间能力。

建议数据：

- Caption/retrieval：RSICD、RSITMD、SkyScript 或 RS5M 类数据。
- Instruction：GeoChat-Instruct、SkyEye-968k、MMRS-1M。
- Benchmark 仅用于参考，不建议混入训练，避免评测污染。

训练方式：

- 用很小比例做预适配，例如总训练 token 的 5%-15%。
- 然后只用 Synoptic-Bench 做主训练。
- 对比不预适配版本的 SPACE、ROUGE/BERTScore、人工样例。

预期效果：

- 可能提升空间描述和指令遵循。
- 不一定提升天气物理理解；若主指标下降，应取消混入。

### 实验 E：多时次天气图输入

目标：从“0-48h 平均图”扩展到“天气演变”。

可用数据：

- 原 HDF5 中 `f_3` 到 `f_48` 的逐 3 小时预报变量。
- Qwen3-VL 的 video/multi-image processor。

做法：

- 方案 1：多图输入，选择 0h/12h/24h/36h/48h。
- 方案 2：生成小视频或图像序列。
- 方案 3：仍输出单图，但增加趋势标签作为辅助任务。

预期效果：

- 更好生成 “front will move through”、“ridge shifts east”、“low deepens” 这类时间演变句子。
- 对长期讨论和趋势判断更有帮助。

风险：

- 显存和 token 数增加。
- 需要重新控制图像分辨率、batch size 和 LoRA 配置。

## 6. 优先级建议

最高优先级：

1. **补齐/审计缺失图像与样本分布**：这是当前数据闭环里最直接的风险。
2. **构建 Synoptic-VQA 和现象-位置 grounding 数据**：最贴近 SPACE，也最能提升可验证的气象理解。
3. **按天气现象和质量分层训练**：比盲目扩大训练规模更符合 GeoChef 的数据中心路线。

中等优先级：

4. **尝试少量遥感 instruction/caption 数据预适配**：可以做 ablation，但不要让遥感数据喧宾夺主。
5. **构建 hard validation split**：专门评估复杂槽脊、多系统、强冷暖平流和边界区域。

长期优先级：

6. **多时次/视频输入**：最可能突破单张平均图无法表达天气演变的问题。
7. **融合卫星/雷达/水汽图像**：如果项目从 GFS 图生成扩展到真实观测解释，可以借 GeoChef 清单中的多传感器遥感数据路线。

## 7. 一句话结论

GeoChef 对当前 Synoptic-Bench 最直接的帮助是提供一种“数据配方”思路：你的核心数据仍应是 GFS 天气图 + AFD 文本，但应把它拆成高质量生成、VQA、现象-位置 grounding、时序趋势和 hard-case 评测等多类数据；遥感 VLM 数据可以作为少量预适配和格式参考，用来提升空间/图文能力，但真正决定效果的仍是气象领域数据的质量、分层和 SPACE 对齐程度。

## 参考链接

- GeoChef DOI：https://doi.org/10.36227/techrxiv.176978652.29736845/v1
- GeoChef / Awesome-RS-VL-Data：https://github.com/VisionXLab/Awesome-RS-VL-Data
- ResearchGate 元数据页：https://www.researchgate.net/publication/400275995_GeoChef_A_Data-Centric_Guide_to_Tailoring_Vision-Language_Models_for_Remote_Sensing
- ORCID/Crossref 记录：https://orcid.org/0000-0002-3080-6721
- Synoptic-Bench 数据集：https://huggingface.co/datasets/Aikyam-Lab/Synoptic-Bench
- Synoptic-Bench 论文：https://arxiv.org/abs/2604.16451
