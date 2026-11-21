<!--
PURSUE replication: pre-registration (original Chinese text).

Source: section 14.1-14.8 of the study's internal engineering guide (docs/engineering_guide.md),
copied verbatim from commit d2a8f71 of the study repository, tagged prereg-pursue-v1 on
2026-10-15 15:27 (UTC-4). The PURSUE EEG data were downloaded after this commit. The internal
repository and its history are not public; an English translation is in
pursue_preregistration_en.md. Section numbers refer to the internal guide (see the note in the
translation). Deviations from this plan are reported in the Methods of the paper.
-->

## 14. PURSUE 预先登记的复制（第 7 版，2026-10-15）

### 14.1 目的与定位

在一个独立的大样本上，对三任务联合检验中两条最稳健的结论做**预先登记的直接复制**。本复制是补充结果，不进入第 13.5 节的联合检验，也不改变已有的结论。研究者此前（2026-10-12）曾排除 PURSUE；在三个数据集的研究完成后改为做这一小规模复制（2026-10-15）。只检验两条结论，原因见 14.5 节。

### 14.2 数据

- **来源**：PURSUE N400 Word Processing（Couperus, Bukach & Reed），NEMAR `on007052` 版本 v1.0.0，即 OpenNeuro `ds007052`（DOI 10.82901/nemar.on007052），CC0。2017–2018 年在 3 所以本科教学为主的院校记录（Mt. Holyoke College、University of Richmond、Claremont McKenna College），被试编号分三组：`sub-1xxx` 104 人、`sub-2xxx` 78 人、`sub-3xxx` 106 人，共 288 人，与 ERP CORE、Draschkow 的被试无关。
- **任务**：沿用 ERP CORE 的 N400 词对范式，启动词 → 目标词 → 按键判断（相关 / 不相关）。事件码与 ERP CORE 相同：启动词 111/112/121/122，目标词相关 211/212、不相关 221/222，反应 201/202；另有练习试次 110/120/210/220（事件说明为 "Unknown"），不纳入。每名被试预期 120 个目标词（每类 60 个）。
- **记录**：BrainVision actiCHamp，500 Hz，29 个 EEG 电极（FP1、Fz、F3、F7、FC3、C3、C5、TP9、CPz、Pz、P3、P7、PO3、PO7、O1、Oz、FP2、F4、F8、FC4、Cz、C4、C6、TP10、P4、P8、PO4、PO8、O2）与 3 个眼电通道（HEOG R、HEOG L、VEOG Lower），接地 AFz；在线参考未在元数据中写明（"common"）。没有 FCz 与 P9、P10。BIDS 格式，每名被试一个 EEGLAB `.set`（数据内嵌）。
- **获取与校验**：在 FIR 上按 NEMAR 的文件清单（`manifest.json`）逐个下载，`.set` 文件用清单中的 sha256 校验，其余文件核对大小。
- **登记之前已查看的内容**（不含任何 EEG 信号值）：数据集说明、README、`dataset_description.json`、`task-WordPR_events.json`、`participants.tsv`、文件清单（含各文件大小）、作者的 `code/` 文件（双侧乳突参考与眼电导联的通道运算；ERPLAB 分箱文件），以及 sub-1001 的通道表、事件表与 `.set` 文件头。

### 14.3 预处理

采用第 4、5 节 ERP CORE 的固定路径（Kessler 的 Luck 参照路径，含第 5 版的两处偏离），不使用第 13.8 节对 Draschkow 任务的修改。只做以下必要的适配：

| 第 4 节的步骤 | ERP CORE | PURSUE |
|---|---|---|
| 1. 读入与事件 | EEGLAB；目标词 + 26 毫秒 | EEGLAB（BIDS）；目标词按相同事件码，**不加显示延迟**（数据集未报告显示器延迟；ERP CORE 的 26 毫秒是其显示器的实测值） |
| 2. 降采样 | 1024 → 256 Hz | 500 → 256 Hz |
| 3. 眼电通道 | HEOG = 左 − 右；VEOG = 下 − FP2 | HEOG = HEOG L − HEOG R；VEOG = VEOG Lower − FP2（与作者的通道运算一致） |
| 4. 初始参考 | Cz | Cz |
| 9. 最终参考 | P9、P10 平均 | TP9、TP10 平均（作者的离线参考；两者都位于耳后下方） |

- **通道**：保存与特征使用 ERP CORE 的 28 个记录电极中存在的 27 个（缺 FCz；不重建 FCz，因为在线参考电极无法核实）。TP9、TP10 与 ERP CORE 的 P9、P10 一样不进入任何通道集合。本复制只用 C1。
- **试次**：全部目标词试次进入解码，不按反应正确性或反应时筛选（与 ERP CORE 相同）。
- 包络按第 5 节计算（B2、B8 需要）。

### 14.4 解码

- 3 个配置：`B1-W1-N2-C1`（基线）、`B8-W1-N2-C1`、`B2-W1-N2-C1`。
- 主模型：L2 逻辑回归，C 在预先规定的网格 {10⁻³, …, 100} 中由内层 5 折交叉验证选出；外层 10 折 × 5 次重复（种子为 20261011 加被试序号），平衡准确率；与第 6 节相同。
- 次要：收缩 LDA 在同样 3 个配置上（N2 以等价的 N1 代替），只作报告。

### 14.5 假设、检验与复制标准

对每名被试计算 50 个外层折的平均平衡准确率：

| 编号 | 预测（单侧） | 对应的联合检验结论 | 已有数据中的条件效应（逻辑回归，ERP CORE / 场景 / 句子） |
|---|---|---|---|
| **R1** | `B8-W1-N2-C1` < `B1-W1-N2-C1`：在波形上加频段包络降低准确率 | P1（普遍成立） | −0.037 / −0.035 / −0.028（dz −0.64 / −0.64 / −0.53） |
| **R2** | `B1-W1-N2-C1` > `B2-W1-N2-C1`：波形优于 θ 包络 | P3（方向一致） | +0.135 / +0.057 / +0.066（dz 1.50 / 0.85 / 0.95） |

- 检验：被试水平差值的单侧配对 t 检验，R1、R2 一起做 Holm 校正，α = 0.05；Wilcoxon 符号秩检验作复核。报告差值、95% 置信区间与 dz。
- **复制成功**：R1 与 R2 在 Holm 校正后都显著且方向符合预测。任何一条不显著或方向相反，即报告为该条未复制。
- **操作检查**：基线配置的平衡准确率高于 0.5（单侧 t 检验），只作报告。
- 检验力：已有数据中最小的条件效应为 dz = 0.53；按约 280 名被试计算，单侧 α = 0.05 下检验力 > 0.999。
- 不检验其他比较。P7（逐试次 z-score）在边际均值上普遍成立，但在基线水平的条件效应很小且方向不一致（ERP CORE −0.005、场景 +0.001、句子 −0.004，都不显著），预测其方向不可靠，因此不登记。复制检验的是"其余维度取基线水平时的效应"，与联合检验使用的边际均值不完全相同，论文中作为补充复制报告。

### 14.6 排除标准

- **事先排除**：`.set` 文件不含 EEG 数据的记录（文件只有 19 KB：sub-2002、sub-2025、sub-3042）。
- **处理后排除**：无法读取的记录；预处理后任一类别少于 30 个试次的被试。
- 不按解码准确率或其他结果排除被试。报告每所院校纳入与排除的人数，并报告不做"处理后排除"时 R1、R2 的结果。

### 14.7 次要与探索性结果

- 每所院校分别的 R1、R2 结果（描述性）；LDA 的结果。
- 被试编号：按 BIDS 编号排序映射为 sub-001 至 sub-288，保存对应表。
- 与本节不一致的任何实施细节，都在实施后记录为偏离，并说明原因。

### 14.8 固定方式

本节合并到 main 并打标签 `prereg-pursue-v1` 之后，才在 FIR 上下载 PURSUE 的 EEG 数据；此后 14.2–14.6 不再修改。实现代码在标签之后提交，实现不得改变本节规定的预处理、配置、检验与排除标准。
