# 第三章 BETR：评估、消融与注意力可视化

依据 `~/main.v2.3.3.mask.pdf` 第三章，重点核对印刷页 52–60、表 3.1–3.5 和图 3.6。
以下命令均从仓库根目录、`conda activate detr` 后运行。新实验不会改写 `train_coco.sh`。

## 本次验证记录

已用当前 checkpoint 完成 5000 张 COCO val2017 评估，加载时 epoch=23（第 24 轮），
AP/AP50/AP75/APs/APm/APl 为 38.00/55.73/41.27/23.10/41.25/49.56。
完整结果在 `workdir/betr_eval_full/metrics.json`。训练仍会更新 checkpoint.pth，
该文件之后可能不再对应本次加载的 epoch；比较结果请核对各次 metrics.json 中的 epoch。
这不是论文 50 轮终值，也不代表三组消融已经训练完成。

`workdir/attention_smoke/139/A2F_figure3_6.png` 是已实际生成并检查的布局示例。
四项测试覆盖损失关闭、全局匹配配置、零半径/空目标、padding 排除，以及
none/gt/dam 三组模型的 CUDA 前向与反向；测试命令：

```bash
python -m unittest discover -s tests -p test_betr_reproduction.py
```

## 1. COCO 评估

```bash
python -m tools.eval_betr \
  --checkpoint workdir/betr_coco/checkpoint.pth \
  --coco-path ../det_data/coco \
  --output-dir workdir/evaluation/current
```

脚本只加载 val2017，采用仓库验证变换（短边 800，长边不超过 1333），不做测试增强。
使用原始 COCO category ID、原始图像尺寸和仓库后处理，官方 COCOeval 默认 maxDets=100。
输出 `predictions.json`、`metrics.json`，其中 AP/AR 均为百分数；COCO 的不可用指标 -1 会显示为 -100。
权重严格加载，模型配置取自 checkpoint.args，不会额外加载 ImageNet 权重，也不会重新创建优化器。
没有 args 的旧 checkpoint 必须提供 `--enc-layers 0 --backbone swin_nano`，其余采用当前默认配置；
若严格加载不通过，需要确认该 checkpoint 的实际结构，不能忽略 missing/unexpected keys。

可用 `--limit 8` 做小样本检查，但这不是论文 AP。默认不限制，评估全部 5000 张。
`--batch-size 1` 是默认值；模型计时跳过前 10 个 batch，CUDA 同步后统计，排除数据读取和后处理。
它是当前机器的模型吞吐率，包含当前代码实际执行的辅助分支，不能直接当作论文 V100 FPS。
论文表 3.1 的 Nano 0enc/2dec、Nano 1enc/2dec、Tiny、Small、Base AP 分别为
40.6、43.4、47.4、49.8、50.2；这是论文报告值，不是脚本运行结果。

## 2. 公平训练协议与配置差异

论文明确给出：COCO train2017 / val2017；ImageNet 预训练 Swin；50 epoch；AdamW；
检测器学习率 1e-4；余弦退火；全局 batch 16；weight decay 1e-4；clip norm 0.1；300 queries；
分类/L1/GIoU 权重 2/5/2；训练短边 480–800、最长边 1333。
论文原始硬件为 8 张 V100；当前为 2 张 5090，默认每卡 batch 8，以保持全局 batch 16。
如果显存不足，不应悄悄改小全局 batch 后宣称严格复现；应调整硬件或另行实现并验证梯度累积。

论文没有完整指定 warmup、主干学习率、投影层倍率、随机种子和额外分类 KD。
新入口明确采用 warmup=0、主干 lr=1e-4、投影倍率=1、seed=42、额外 KD 开启（系数 2；旧版本入口默认关闭），
可用 `--warmup-epochs`、`--lr_backbone`、`--lr_linear_proj_mult`、`--seed`、`--no_kd_from_dec` 覆盖。
当前训练 checkpoint 可能启用 5 epoch warmup 和 KD，且 batch/lr 不同，不能作为表 3.5 的公平对照。
保存的 `experiment.json` 和 checkpoint.args 是判断实验配置的依据。

## 3. 表 3.5：真值辅助损失是重点

三个主实验均使用 Swin-nano、0 编码器、2 解码器、局部匹配 r=1.5，其余超参数完全相同。
每组独立从同一个 ImageNet 权重开始训练，不能从另一个组的检测 checkpoint 接着微调。

| 实验 | 命令中的 experiment | 辅助监督 | 权重 | 论文 AP |
|---|---|---|---|---|
| 无辅助损失 | baseline | 只有前置一对一检测损失和解码器损失 | 0 | 40.0 |
| 真值辅助分类 | gt | 一对多标签分配后，对共享前置分类 logits 施加 focal loss | 2 | 32.2 |
| A2F | a2f | 解码器累计注意力 top-20% 二值目标，监督独立前置预测头 | 2 | 40.6 |

```bash
# 先检查参数，不启动训练
python -m tools.train_betr_ablation --experiment gt --dry-run

# 三个命令依次运行，不要同时占用同一组 GPU 和端口
python -m torch.distributed.run --nproc_per_node=2 --master_port=29541 \
  -m tools.train_betr_ablation --experiment baseline --seed 42
python -m torch.distributed.run --nproc_per_node=2 --master_port=29541 \
  -m tools.train_betr_ablation --experiment gt --seed 42
python -m torch.distributed.run --nproc_per_node=2 --master_port=29541 \
  -m tools.train_betr_ablation --experiment a2f --seed 42
```

默认目录为 `workdir/ablation/<experiment>`；后续种子建议 43、44，并指定不同 `--output_dir`。
报告三个种子的 mean/std、单种子 AP/AP50/AP75/APs/APm/APl、最终 epoch、所有超参数。
以最后一个 epoch 为主结果，若报告 best checkpoint，必须对每组使用相同选择规则。

**真值损失实现边界：** `gt` 复用当前 `DenseAuxMatcherV0`：每层最多选 9 个最低
分类＋L1＋GIoU 代价的候选，以候选代价 mean+std 筛选，要求点在 GT 框内，重叠按最低代价分配。
这是原仓库的 ATSS 风格改写，不是标准 ATSS 的最近中心候选＋IoU mean+std；论文未给足细节，
因此不能保证这份实现必然复现 32.2 AP。分类 focal loss 使用 alpha=.25、gamma=2，
按分布式平均正样本数归一化，其他位置仍是背景。保留共享分类头，正是为了测量一对一/一对多冲突。
`--with_gt_mask` 是另外的机制，不应用来代表此实验；将 GT 框涂成二值 mask 做 BCE 也不是表 3.5。

原 `--dense_aux_loss o2m` 还计算 `loss_o2miou_enc_aux`，会引入额外变量。
用 `--experiment legacy-o2m` 单列该对照，不能和 `gt` 混称。
推荐再对 gt 权重 {0.5,1,2,4} 做敏感性分析：`--dense_aux_loss_coef 1`，
以检验性能下降是否只是损失尺度问题，而不是直接用 AP 下降证明梯度冲突。
如需进一步证明梯度冲突，应在相同 minibatch 上分别求主损失/辅助损失对共享主干参数的梯度，
统计余弦相似度和范数；注意力图本身不能证明梯度方向相反。

## 4. 表 3.2–3.4 的实验矩阵

| 表 | 配置 | 入口/覆盖参数 | 论文 AP |
|---|---|---|---|
| 3.2 | 稠密前置 0enc/6dec，无局部匹配/辅助损失 | dense6 | 41.5 |
| 3.2 | 稠密前置 0enc/2dec，无局部匹配/辅助损失 | dense2 | 39.8 |
| 3.2 | 加局部匹配 | baseline | 40.0 |
| 3.2 | 加 A2F | a2f | 40.6 |
| 3.2 | 再加 1 层编码器 | full | 43.4 |
| 3.3 | r={inf,0,1.5,3,5}，无辅助损失 | baseline + --spatial_prior_radius R | 39.8/38.5/40.0/39.9/39.8 |
| 3.4 | lambda={0,1,2,4}，gamma=.2 | a2f + --dense_aux_loss_coef W | 40.0/40.4/40.6/33.2 |
| 3.4 | gamma={.1,.2,.6}，lambda=2 | a2f + --a2f_ratio G | 40.1/40.6/40.2 |

稀疏 ViDT 0enc/6dec 是单独架构，表 3.2 报告 AP 40.4；不能用 BETR dense6 冒充它。
这里未新增 ViDT 重训练入口，也未实现 CityPersons/CrowdHuman 的 MR/JI 专用协议。

原代码 r 写死为 1.5；现在 r=inf 使用全局 Hungarian，r=0 按论文文字保留每层最近中心格点。
旧代码中 radius=0 实际意味着框内所有点，本入口不沿用那个相反语义。
零半径下不同目标可能竞争同一个点；这是诊断性极端配置，不能保证每个目标都有独立可用候选。
A2F top-k 保留原仓库 floor(valid_tokens*gamma)+1 的舍入方式，上限截到有效 token 数。
不同实验使用唯一输出目录，例如：

```bash
python -m torch.distributed.run --nproc_per_node=2 --master_port=29541 \
  -m tools.train_betr_ablation --experiment a2f --a2f_ratio .1 \
  --output_dir workdir/ablation/a2f_gamma10_seed42
```

## 5. 图 3.6 与 A2F/真值对比

```bash
python -m tools.visualize_betr_attention \
  --model A2F=workdir/ablation/a2f/checkpoint.pth \
  --model GT=workdir/ablation/gt/checkpoint.pth \
  --image-ids 139 285 \
  --coco-path ../det_data/coco --output-dir workdir/attention_comparison
```

图像 ID 是示例，不是论文原图的已确认 ID。图 3.6 是滑雪者图片，PDF 未注明 COCO ID；
选到原图后直接替换 `--image-ids`。可只传一个 `--model`，先查看已有 checkpoint。

每张图输出：

- `comparison.png/pdf`：每个模型一行，GT 框、连续交叉注意力、top-k 二值教师、前置分类响应、A2F 预测头。
- `<LABEL>_figure3_6.png/pdf`：左侧图像与 GT，右上二值交叉注意力、右下前置 A2F 图；采用最细特征层。
- `<LABEL>_raw.npz`：原始未归一化、未上采样的每层数据，可重绘/统计。
- 各种热图及 `level0..3` 原分辨率图，和 checkpoint/epoch/config 等元信息。

累计注意力复用训练 `util.dam.attn_map_to_flat_grid`，将所有 query/layer/head 的稀疏采样注意力
双线性散射回特征网格，与现有 A2F 目标一致；不使用主干自注意力或 Grad-CAM 替代。
该历史实现的网格坐标约定与 CUDA grid_sample 的半像素约定不同，此处为保持训练一致性不更改它。
多层汇总图各层先上采样再平均；平均后的二值图会有灰度，原生每层二值图和图3.6面板保留二值。
每张图片不同模型的连续交叉注意力采用同一个最大值；预测概率固定显示 [0,1]，不逐图拉伸反差。

**GT 模型没有 A2F 预测头。** 脚本明确显示缺失，绝不会画一个随机初始化的 A2F 头冒充结果。
真正可直接比较的是两种模型共有的解码器注意力，以及前置类别概率最大值（有 filter 则相乘）。
后者是分类响应代理，不是与 A2F 独立预测头同定义的注意力。
为了公平，比较同一图片、同一层、同一 epoch、同一架构和同一训练协议，并预先固定可视化样本集；
不要只挑选支持“中心/边缘”结论的图片。A2F、GT 都需要各自训练后的 checkpoint，单个模型无法产生真实组间比较。

## 6. 评委意见补充：实例级空间偏好与量化统计

新增 `tools/analyze_betr_attention.py`。与前面的整图脚本不同，此脚本对每个 GT
用 Hungarian 匹配找到最终解码器 query，并只累积该 query 的注意力（默认所有解码器层和头）。
模型间以 **COCO annotation ID** 对齐，query ID 可以不同。不会按预测置信度筛选好看的实例。
三个模型必须分别训练；以下路径是各组训练完成后的实际 checkpoint：

```bash
conda activate detr
python -m tools.analyze_betr_attention \
  --model BASE=workdir/ablation/baseline/checkpoint.pth \
  --model GT=workdir/ablation/gt/checkpoint.pth \
  --model A2F=workdir/ablation/a2f/checkpoint.pth \
  --sample-size 100 --seed 42 \
  --output-dir workdir/mechanism_comparison
```

- 默认在 val2017 固定随机抽 100 张，所有模型使用同一份 ID；`image_ids.json` 留档。
- `--sample-size 0` 分析全部 val2017；`--image-ids 139 285` 优先于抽样参数，适合检查流程。
- `--decoder-layer 0` 只分析第一个解码器层，`1` 分析第二个；默认 `-1` 累积所有层。
- `--plot-images 3 --plot-instances 2` 仅限制生成面板的数量，不限制统计样本。
- `--min-points 4` 要求中心和边缘各有至少 4 个有效网格点。不满足的记录写入 `excluded.csv`，
  不强行上采样造出独立样本。统计因此可能偏向较大实例，须连同排除量一并报告。
- 使用原生四层网格，按实际 stride 和变换后图像尺寸定位点，排除 padding；不跨尺度平均后再做统计。

**对比面板 `<image_id>_<annotation_id>_comparison.png/pdf`：**
每模型一行：原图与该实例 GT 框 → 匹配 query 的连续注意力 → 该 query 的二值 top-k →
该实例 GT 类别的前置分类响应 → 该实例实际一对多分配点 → A2F 专有预测头。
连续注意力在同一个面板各模型间共用最大值，其余概率/二值图固定 [0,1]。
注意力面板为原生最细网格；青色轮廓是框中心区，橙色轮廓是框内边缘带。
不同图片之间连续注意力不共享最大值，不可仅通过灰度深浅作跨图片强度比较。
另有同名 `_zoom.png/pdf`：GT 框外扩 25% 的实例放大图，沿用原图颜色范围，便于观察小目标。
A2F/BASE 行的一对多分配图是**用相同分配器计算的假设监督**，不表示它们训练时使用了真值辅助。
GT 行没有 A2F 头，明确显示 Head absent，不添加随机头。

**两种区域定义（分别报告，不能混称）：**

1. `region=box`：以 GT 框归一化坐标
   `r=max(|x-cx|/(w/2), |y-cy|/(h/2))`，中心 `r<=0.5`，框内边缘 `0.8<r<=1`。
   用 `--center .5 --edge .8` 调整，推荐补充 `.4/.75` 等敏感性检查。
2. `region=contour`：COCO 实例 mask 的**内部轮廓带**与剩余内部区域。
   在原图 mask 上计算欧氏距离变换，带宽 `max(1px, 0.1*sqrt(mask面积))`，
   可通过 `--contour-width .05` 调整。超过带宽的内部称为 interior（CSV 中沿用 center 字段），
   它不等价于框中心。分割标注仅用于事后统计，不用于训练。`--no-contours` 可关闭。

**输出与指标：**

| 文件 | 内容 |
|---|---|
| `instances.csv` | 每实例、模型、尺度、区域、图类型的统计；small/medium/large 按 COCO 原始实例面积分组 |
| `radial.csv` | 框内 r 从 0 到 1 的 10 个区间，各区间每格点平均响应与格点数 |
| `radial_level*.png/pdf` | 中心到边缘分布曲线：实例 query 注意力、GT 类别响应、一对多正样本密度 |
| `summary.json` | 按模型、尺度、区域、图类型、尺寸分组的均值和 95% 图像 bootstrap CI |
| `paired_differences.json` | 相同 image/annotation/尺度/区域的模型配对差值；明确 right-minus-left |
| `alignment.csv` | 全 query A2F 教师与预测 top-k 的逐图、逐层 IoU/precision/recall，仅 A2F 头存在时输出 |
| `excluded.csv` | 区域格点不足或未匹配 GT 的排除记录 |
| `metadata.json` | checkpoint 路径、epoch、完整配置、环境和分析参数 |

`center_mean/edge_mean` 是区域内**每格点平均值**，本身已经面积归一化，不能再除一次面积。
`edge_minus_center>0` 表示边缘平均响应较高；`edge_center_ratio>1` 表示相同方向，
中心响应为零时 ratio 记为缺失而非无穷大。`center_coverage/edge_coverage` 是区域内
top-k 正样本比例，`coverage_difference=edge-center`。
对于 `gt_assignment`，这些值就是实际分配的正样本密度；对于分类/A2F 预测图，coverage
是它们各自 top-k 集合的覆盖率；所有原始连续响应仍单独保留。

图类型中 `query_attention` 是实例 query；`a2f_teacher` 是训练使用的全 query 累计教师；
`a2f_prediction` 是全局前置预测头，不能声称它是某个 query 的专有预测。
实例 query 二值图与全 query 教师都按**全图所有有效层 token**排序，保留仓库
`floor(N*gamma)+1`（上限 N）的规则，不是在每个 GT 框内重新选 20%。
稀疏 query 可能不足 20% 非零点，此时阈值落在零值并有并列值；因此必须同时检查连续响应，
不能只用二值覆盖率推断机制。`--ratio .1/.2/.3` 可做敏感性分析。

统计先在同图内平均实例，再以图像为单位 bootstrap 1000 次（seed=42），避免同图实例相关性
造成伪精确；配对统计先限定相同 annotation，再计算图像平均差值。不同种子训练的模型应分别
运行再报告种子间波动，当前 CI 不包括训练随机性。单图 CI 只是退化区间，不具有总体推断意义。
径向曲线按模型自身的可用实例统计，若两模型未匹配实例不同，严谨差值结论以配对结果为准。

## 7. 梯度冲突验证：不更新模型参数

新增 `tools/analyze_betr_gradients.py`，将空间偏好与梯度干扰的证据分开。

```bash
python -m tools.analyze_betr_gradients \
  --model BASE=workdir/ablation/baseline/checkpoint.pth \
  --model GT=workdir/ablation/gt/checkpoint.pth \
  --model A2F=workdir/ablation/a2f/checkpoint.pth \
  --sample-size 100 --seed 42 --batch-size 1 --aux-weight 2 \
  --output-dir workdir/gradient_comparison
```

`--image-ids` 和随机抽样规则与空间脚本相同。建议固定图集，并分别对训练早、中、后期
保存的 checkpoint 执行；输出到不同目录。显存不足时保持 `--batch-size 1`，所有对照一致。

分析模式固定为 `model.eval()`，启用 autograd，使用确定性 val 变换，不执行 optimizer.step，
不改权重。它控制 dropout 等随机性，但**不是训练模式下实际梯度轨迹的在线记录**。
每个 checkpoint 上对同一 minibatch 分别求：

- `decoder`：最终及中间 decoder 层分类/L1/GIoU 损失，使用 checkpoint 中的损失系数。
- `early_o2o`：前置一对一分类/L1/GIoU 主损失。
- `gt_focal`：当前一对多分配器产生的分类 focal loss × `--aux-weight`，所有模型都可计算；
  在 BASE/A2F 上属于**反事实诊断损失**。
- `a2f`：实际 A2F 头对教师目标的损失 × `--aux-weight`，仅存在该头的模型输出。

比较 decoder↔gt_focal、early_o2o↔gt_focal、decoder↔a2f、early_o2o↔a2f。
额外分类 KD、旧 o2m 的 IoU 分支不计入这些隔离目标。
统计全部可训练 backbone 参数及各 Swin stage；无梯度参数按零向量处理，零范数时 cosine 缺失。
输出：

- `gradients.csv`：每批次/模型/stage 的损失值、两侧梯度范数、余弦相似度、右/左范数比。
- `summary.json`：余弦及范数比的均值和批次 bootstrap CI、负余弦比例。
- `*_conflict.png/pdf`：各 stage 的余弦区间图和负余弦比例图。
- `metadata.json`、`image_ids.json`：诊断协议、checkpoint 与样本清单。

解释：负余弦表示共享参数上的局部优化方向相反；辅助范数很大且余弦很负时可能干扰主任务。
推荐 `--aux-weight 0.5/1/2/4` 检查损失尺度；正的标量权重不会改变余弦，只会改变范数比。
空间图、梯度统计和检测 AP 应相互印证，不能由某一张图或两张图片的梯度直接声称因果。

## 8. 验证范围与推荐论文呈现

本次以现有 A2F checkpoint、COCO 图 139 跑通空间统计（含真实轮廓）和实例图导出；
梯度诊断在图 139、285 跑通。位置：`workdir/mechanism_smoke`、`workdir/gradient_smoke`。
这仅证明脚本流程可运行，不代表已完成三模型机制实验，也不预设结果必然支持论文解释。
数值测试检查 top-k 排除 padding、不同区域面积下的密度、负向/零梯度、按图像配对统计：

```bash
python -m unittest discover -s tests -p test_betr_mechanism.py
```

建议论文补充：三模型同实例可视化（含实际分配点）；分尺度径向曲线；按目标尺寸的
边缘/中心均值与覆盖率表（含 CI 和排除数）；轮廓带验证；若继续保留“梯度冲突”明确论断，
再附按 stage/训练阶段的负余弦比例和范数比。图例应写清“框边缘”或“分割轮廓”，
且根据实际结果使用“观察到/支持/未支持”，不要把假设写成脚本必然生成的结论。


## FPS、参数量与 FLOPs 测试

`tools/benchmark_betr.py` 的默认 FPS 计时协议对齐 ViDT `fps_calculator.py`：
FP32、eval、torch.no_grad，每次前向前后 CUDA 同步，使用 perf_counter；
总共 300 次，丢弃前 5 次，FPS = batch size / 剩余 295 次的平均耗时。
不包含读取、预处理、CPU→GPU 传输或检测后处理。

固定输入高 800、宽 1300，batch size 1：

```bash
conda activate detr
python -m tools.benchmark_betr \
  --checkpoint workdir/betr_coco/checkpoint.pth \
  --height 800 --width 1300 --batch-size 1 \
  --num-iters 300 --warm-iters 5 \
  --output workdir/benchmark/fps_800x1300_bs1.json
```

若要像 ViDT 一样使用 COCO 验证集第一张图片，增加 `--coco-path /path/to/coco`。
此时采用仓库验证集预处理，覆盖 height/width，实际输入尺寸见 JSON 的 input_shape；
固定尺寸模式仍使用随机输入。公平比较时两个模型必须使用相同输入尺寸、硬件、
精度和软件环境，并在 GPU 空闲时分别测试。不得将本机结果直接与论文 V100 FPS 比较。

`--iterations` 是 `--num-iters` 的别名，**现在表示包含预热的总次数**；
`--warmup` 是 `--warm-iters` 的别名。不要再把 iterations 设置为 295 来表示有效次数。
`--postprocess` 会额外计入 bbox 后处理，不属于 ViDT 默认协议。
`--skip-flops` 仅跳过 FLOPs 统计。FLOPs 在 FPS 计时结束后统计，不影响计时区间。
参数量同时记录总参数与可训练参数（ViDT 输出的是可训练参数）；
FLOPs 使用 fvcore 与可变形注意力近似计数，partial=true 表示存在未支持算子。


## DeFCN-style 真值辅助损失（gt-defcn）

新增 preset `gt-defcn`：0 encoder / 2 decoder，在原有检测损失上增加权重 2 的一对多分类 focal；不增加 IoU BCE 或辅助框回归。
分配遵循 DeFCN 官方 `poto.res50.fpn.coco.800size.3x_ms.3dmf.aux/fcos.py` 的 `get_aux_ground_truth`：
原始类别概率 p（不乘 filter）与预测框 IoU 构成质量 p^0.2 × IoU^0.8；
每层选质量最高的 9 个候选，保留质量 ≥ 候选均值 + 样本标准差且位置严格位于 GT 框内的点；多 GT 冲突取最大质量。
小特征层候选不足 9 时取全部有效点，单候选标准差设为 0；padding 不参与候选，空 GT 返回空匹配。
分类 focal 的 alpha=0.25、gamma=2，按分布式平均正样本数归一化（下限 1）。
旧 `gt`/`legacy-o2m` 的 cost-based 分配保持不变。新实现不等于原始距离型 ATSS。

```bash
conda activate detr
python -m torch.distributed.run --nproc_per_node=2 -m tools.train_betr_ablation \
  --experiment gt-defcn --lr 2e-4 --lr_backbone 1e-4 \
  --kd_from_dec --dense_kd_loss_coef 2 --dense_aux_loss_coef 2 \
  --output_dir workdir/ablation/gt_defcn_kd
```

所有 preset 现在默认开启 KD，baseline/gt-defcn/a2f 应保持相同 KD 配置。
复现旧 KD=False 对照时显式传 `--no_kd_from_dec`。不要混用新旧默认设置的结果；核对 experiment.json。
注意 `a2f` 是 0 encoder，而 `full` 是 1 encoder，KD 一致不代表结构一致。
注意力/梯度分析工具通过 checkpoint 的 dense_aux_loss 自动选择对应 matcher；对旧 checkpoint 仍采用旧分配。
此模式是依据官方实现补充的实验，不保证复现论文 32.2 AP。


### 原图尺寸的注意力对比面板

`tools.visualize_betr_attention` 原命令无需增加参数。各 `*_levelN.png` 现在用最近邻放大至原图尺寸，二值目标保持 0/1；原生数值保存在 `*_raw.npz`。
`comparison.png/pdf` 保留多尺度平均视图，其 teacher 是平均投影，不是二值图。
`comparison_level0.png/pdf` 至 `comparison_level3.png/pdf` 分别比较各模型对应特征层（按实际层数自动生成），各层均放大到原图尺寸，教师是该层实际二值目标；该层正样本比例不保证为 20%，20% 定义在全部有效多尺度 token 上。
`<模型>_panel_A_level0.png/pdf` 为原图、二值教师、A2F 预测，仅有 A2F 头的模型生成。
`panel_B_class<类别ID>_level0.png/pdf` 按 GT 类别生成共有类别响应对照：原图、原始概率 [0,1]、同图同类跨模型共享最大值显示、全部入选 proposal 中心与前 30 个框。
面板 B 分类响应不乘 filter；proposal 使用实际排名。归一化只用于显示，不能跨图片比较归一化亮度。这些图不单独证明梯度冲突或重复 proposal 导致 AP 下降。
