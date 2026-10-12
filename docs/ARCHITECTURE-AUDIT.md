# 只读架构审计 — Minecraft 建筑生成项目

日期: 2026-10-09。方法: 只读代码、数据文件与已有 checkpoint/日志;所有结论标注
证据出处;无法从代码确认的写"未知/无法确认"。本审计不修改代码、不训练。

---

## 一、项目总体数据流

项目实际存在**两条互不相交的数据流**——生产装配线(出图)与实验模型栈(研究)。

### 1A. 生产装配线(当前出地图用的)

```
rom1504 schematics 库(TFRecord→schematic/schem 字节)
  → src/schemlib.py 解码 → out_lib/buildings/*.npz (legacy ID 体素, [x,y,z] uint8)
  → src/features.py(类特征) → out_lib/features.npz (N×d float, 具体 d 见文件)
  → src/assemble2.py: MMR+风格锚点检索选楼(仅用 embedding 选,不用生成)
  → src/worldgen.py 地形/道路/街区
  → 体素拼装 → 渲染/导出 npz
```

| 层 | 输入 | 输出 | shape/dtype | 离散? | 条件? | 训练? | 推理? |
|---|---|---|---|---|---|---|---|
| schemlib 解码 | schematic 字节 | 体素网格 | (x,y,z) uint8 | 是(ID) | 否 | — | 是 |
| features | 体素网格 | 嵌入向量 | (N,d) float32 | 否 | 否 | 无训练 | 是 |
| assemble2 检索 | 嵌入+风格锚点 | 楼+坐标 | 结构化 | — | 风格锚点 | — | 是 |

**生产线里没有任何生成模型**;建筑全部来自检索库原样搬运。

### 1B. 实验模型栈(VQ-VAE + prior)

```
buildings/*.npz (legacy ID, 任意尺寸)
  → src/prep_vq_data.py: crop bbox → 每轴独立缩放到 30³ → 居中 pad 到 32³
    → legacy ID 折叠为 6 类(air/mineral/wood/glass/organic/colourful)
  → out_lib/vq_data.npz  X uint8 (3262, 32, 32, 32)
  → one-hot (B, 6, 32, 32, 32) float32
  → src/train_vq.py VQVAE.enc: Conv3d(6→64,k4,s2)+Res → Conv3d(64→64,k4,s2)+Res
  → z (B, 64, 8, 8, 8) → reshape (B, 512, 64)
  → VectorQuantize(dim=64, codebook=1024, EMA decay=0.99, commitment=0.25)
  → codes (B, 512) int64 + q (B, 512, 64) float
  → reshape (B,8,8,8,64).permute → dec: Res → ConvT(64→64,s2) → Res
    → ConvT(64→128,s2) → ReLU → Conv3d(128→6, k3)
  → logits (B, 6, 32, 32, 32) → argmax → 6 类体素
[生成路径]
  codes 序列(512, 光栅序) → src/train_prior.py GPT(dim=192,4层,8头,causal)
  → 逐位置预测下一 code → 采样 512 个 code → 8³ 网格
  → 同一 VQ decoder → 体素
```

| 层 | 输入 | 输出 | shape/dtype | 离散? | 条件? | 训练? | 推理? |
|---|---|---|---|---|---|---|---|
| prep | (x,y,z) ID 网格 | 6 类张量 | (32,32,32) uint8 | 是 | 否 | — | 是 |
| encoder | one-hot | 连续特征 | (64,8,8,8) float32 | 否 | 否 | 是 | 是 |
| quantizer | (512,64) | code 序列 | (512,) int64 | 是 | 否 | 是 | 是 |
| decoder | (64,8,8,8) | 类 logits | (6,32,32,32) float32 | 输出离散 | 否 | 是 | 是 |
| GPT prior(v2) | (≤513,) int64 | 下一 code logits | (VOCAB=1032,) | 是 | style token(v2) | 是 | 是 |

注意: **v1 prior 与 v2 的区别**。v1 = 无条件, VOCAB=1024;v2 = 序首加一个
style token(7 风格+unknown), VOCAB=1032(证据: train_prior.py:28-32, 188)。
v2 正在 mcbox 训练;v1 的 checkpoint 已被 v2 覆盖(run_prior.sh 先
`docker rm -f prior`,v2 因词表不匹配 fresh start 后会覆盖写 prior_ckpt.pt,
train_prior.py:168-176, 200-202)。v1 的成绩只存在于 docs/prior_train.log。

---

## 二、数据集

- **样本数**: 3,262(meta.jsonl 行数;vq_data.npz X.shape=(3262,32,32,32))。
- **来源**: rom1504 schematics 归档(TFRecord 打包 planetminecraft 项目,
  meta.jsonl 每条含 url;schemlib.py:1-12 记录三种封装格式)。
- **原始格式**: .schematic(MCEdit NBT)与 .schem(Sponge v2),均为建筑存档。
- **原始尺寸分布**(meta.jsonl dims 字段, x,y,z):
  min (15,1,1), median (50,27,32), mean (56.9,34.2,38.5), max (128,128,128);
  209 个样本有一维 ≤1(地图画/平面物)。nblocks: median 7,240, max 883,395。
- **resize/crop/pad**(prep_vq_data.py:61-70): crop 到非零 bbox → **每轴独立**
  `zoom(f=30/该轴尺寸, order=0)` → 居中 pad 到 32³。
  **后果: 长宽高比例被完全抹掉**——128×1×114 的地图画和 46×19×46 的房子都被
  拉伸成 ~30³ 立方体。绝对尺寸信息同样丢失。
- **模型最终输入空间尺寸**: 32³。
- **block vocabulary**: 原始库全量扫描 251 种 legacy ID(max id 255);
  模型只见 **6 类**(air/mineral/wood/glass/organic/colourful,
  prep_vq_data.py:21-47;未匹配 ID 的兜底: `i<35→mineral 否则 colourful`,行 46)。
- **block state**: **不保留**。schemlib.schematic_to_grid 只读 `Blocks` 字节数组,
  .schematic 的 `Data` 元数据数组(朝向/朝向/含水等)直接丢弃
  (schemlib.py:186-191);Sponge 路径用 20 项小 LUT 映射名字→legacy,
  未收录名字全部变石头(schemlib.py:181-184, 195-203)。
  **orientation/facing/waterlogged 等全部丢失。**
- **air**: 区分(类 0),占 75.8%。
- **类别/风格/功能标签**: 有,vision_types.json 3,181/3,262 条
  (MiMo flash 视觉标注):
  style: modern 1248 / other 861 / medieval 503 / scifi 221 / nautical 196 /
  asian 132 / western 20;
  type: building 1690 / other 823 / vehicle 341 / statue 205 / sprite 74 /
  vegetation 48。
  **这些标签 v1 模型完全没用;v2 只用 style;type 从未进入任何模型。**
  **库中只有 52% 是 building,车/雕/像素画/植被混训。**
- **biome/terrain/位置**: 无。
- **数据增强**: 无(prep/train 代码中没有任何增强)。
- **train/val/test 划分**: **没有**。train_vq.py:81 与 train_prior.py:185
  都从全量 N 随机抽批;不存在任何 split 文件。
- **泄漏风险**: ① 3,262 张量中 **44 张是完全重复**(逐字节 hash 统计);
  ② 同一 planetminecraft 项目的变体建筑可能近似重复,未做去重,规模未知;
  ③ 无 split,谈不上切片泄漏(样本=整栋,不是切片)。
- **模型实际看到的信息(明确字段表)**:

| 字段 | 看到? | 说明 |
|---|---|---|
| geometry | 是 | 但**比例已被拉伸归一化**(见上) |
| block type | 部分 | 仅 6 材质类 |
| block state | 否 | 解析期即丢 |
| style | 否(v1)/单 token(v2) | 7 离散类,非连续嵌入 |
| function | 否 | 标签有,未入模 |
| floor plan | 否 | |
| room topology | 否 | |
| terrain | 否 | |
| position | 否(世界坐标) | 序内位置用学习的 1D pos emb |
| orientation | 否 | |
| metadata(描述/作者/链接) | 否 | |

---

## 三、Tokenizer / VQ-VAE

证据: src/train_vq.py:25-57。

- **输入**: (B, 6, 32, 32, 32) float32 one-hot。
- **encoder**: Conv3d(6→64, k4, s2, p1) → ResBlock(64) →
  Conv3d(64→64, k4, s2, p1) → ResBlock(64)。两层 ResBlock=
  ReLU+Conv3d×2 残差。
- **下采样**: 2×2 = **4×**(32³→8³)。
- **latent**: (B, 64, 8, 8, 8) → 序列化 (B, 512, 64)。
- **latent dim**: 64。**codebook size**: 1024(lucidrains VectorQuantize,
  EMA decay=0.99, commitment_weight=0.25)。
- **每个 code 覆盖空间**: 8³ 体素中的一个 4×4×4 原始格…准确说每个 code 对应
  输入 32³ 里的 **4×4×4 voxel 区域**(8³ 个 code 格,每格 4³)。
- **decoder**: ResBlock(64) → ConvTranspose3d(64→64, k4, s2) → ResBlock(64) →
  ConvTranspose3d(64→128, k4, s2) → ReLU → Conv3d(128→6, k3)。
- **reconstruction loss**: cross_entropy(logits, 6类)(train_vq.py:88),
  权重 1。
- **codebook/commitment**: VectorQuantize 返回的 vq_loss(EMA 码本 + 0.25
  commitment),权重 1 相加(train_vq.py:89)。无 β 调度。
- **perceptual loss**: 无。**adversarial loss**: 无。
- **一个建筑经过 encoder 后变成什么**(准确表述):
  `(32,32,32) uint8 六类` → one-hot `(6,32,32,32)` → encoder →
  `(64,8,8,8)` → 每个 8³ 格量化为 **1 个 code(0..1023)** →
  **512 个离散 code 的序列**(光栅序, x 慢 z 快, train_prior.py:110 的
  permute+reshape 次序)。
- **当前重建质量**: voxel acc **0.963**(out_lib/vq_eval,mcbox;含 air,
  air 占 75.8%,acc 有虚高);目检 6/10——大体块保留,窗/叶/细节糊
  (对比图 out_lib/vq_eval/)。

## 四、Codebook 审计

- **size**: 1024。
- **实际使用**: 全语料 token 统计(vq_tokens.npz, 3262×512):
  **1024/1024 全部出现过**;频次严重倾斜——top code(164)占 9.5%,
  top10 合计 33.6%;**perplexity ≈ 249**。
  (此前报告的 "349/1024" 是 vq_eval 子集口径;全语料口径为 1024。两个数字
  都真实,口径不同,以全语料为准。)
- **collapse**: 不存在硬塌缩(全码在用),但**头部集中明显**(有效词表 ~249)。
- **一个 code 的典型局部结构**: 码本图谱工程正在生成
  (src/codebook_atlas.py, 逐码单独解码渲染);**结果未出,本审计不脑补**。
- **"一个 code 有没有稳定空间/语义意义?"**: 目前证据偏否定/未定——
  latent 扰动实验(src/latent_audit.py): 换 1 个 token 后重建只有
  0.04–0.2% 体素变化(约 13–70/32768),目检两张渲染**肉眼无差别**。
  两种解释无法区分: ① 局部性好(变化被限制在 4×4×4 区域);
  ② 解码器对单 code 不敏感(过度平滑,code 语义弱)。
  **需要的实验**: 多 code 连续扰动 + 码本图谱 + 按位置分组扰动(槽位 vs 墙体),
  才能拆开这两种解释。

## 五、Prior / 生成模型

证据: src/train_prior.py:35-73, 139-160, 183-193。

- **模型类型**: 手写 GPT(自回归 Transformer decoder)。
- **参数量**: **约 2.3M**(按代码结构推算: tok emb 1032×192≈0.20M,
  pos emb 513×192≈0.10M, 4×Block≈1.77M, head 192×1032≈0.20M;
  未从 checkpoint 精确统计)。
- **输入**: code 序列(v1: 512 code;v2: 1 style token + 512 code)。
  **输出**: 每位置对 VOCAB 的 logits(v1 1024 / v2 1032)。
- **sequence length**: 512(v2 训练时 513)。
- **token vocabulary**: 1024 codes(+8 style token, v2)。
- **positional encoding**: **学习的 1D embedding 相加**(train_prior.py:63,70)。
- **3D 空间信息**: **只存在于 token 的光栅排序里**;无 3D 坐标嵌入,
  无相对位置偏置。
- **attention**: 标准 1D causal self-attention(F.scaled_dot_product_attention,
  is_causal=True, train_prior.py:53),8 头。
- **预测什么**: 下一个 code(交叉熵)。
- **training objective**: next-token CE(train_prior.py:190)。
- **sampling**: 逐位置 softmax **temperature 0.9** 多项式采样,
  **无 top-k / top-p / beam**(train_prior.py:142-144)。
- **当前生成质量**: v1(loss 1.11 @12,620 步)样本科幻→工业骨架渐变可见,
  但目检 2–4/10: **无建筑框架、无功能逻辑**(用户判词"稀碎",承认成立)。
  v2 训练中(写本报告时 step ~500/12000, loss 3.3–3.5,
  tok emb shape (1032,192) 已验)。
- **模型在学什么(特别判断)**:
  **B 为主(code token 统计规律),A 为辅(经 decoder 间接学到局部材质
  占用统计),C 基本没有**。理由: 输入里不存在任何高层语义(无功能/风格/平面);
  光栅 next-token 目标鼓励的是局部共现;decoder 平滑性(扰动实验)说明 code
  空间的信息被摊平成"材质纹理统计"而非部件。顶层语义无从习得。

## 六、条件信息

**进入模型的条件只有一个: style token(仅 v2,训练中)。**

| 条件 | 进入模型? | 哪层 | 维度 | 融合方式 | 训练用? |
|---|---|---|---|---|---|
| style | v2 是 / v1 否 | GPT 输入序列首位 | 192(tok emb) | 拼接为第 0 个 token | v2 |
| building type | **否** | — | — | — | 否 |
| function/floor count/尺寸/材质/biome/terrain/text/image/floor plan/room graph | 否 | — | — | — | 否 |

代码里有但未进 forward 的字段: vision_types.json 的 `type` 字段
(building/vehicle/...)——**读取代码只取 style**(train_prior.py:91-95)。
skeleton.json 的 enclosure 指标只被生产线 assemble2 用作检索过滤,
**不进任何模型**。

## 七、3D 表示问题(只描述现状与后果)

- **是,3D 被 flatten 成 1D**: 8³ code 格按 x 慢、y 中、z 快的光栅序排 512
  token(train_prior.py:110)。
- **模型不知道两个 token 3D 相邻**: 唯一的邻近信号是光栅序里前后 + 1D pos emb;
  (x,y,z) 与 (x,y+1,z) 在序列里隔 8 个位置,与 (x+1,y,z) 隔 64——
  **序列距离与空间距离不一致**,且无相对偏置补偿。
- **8³ token 是否固定空间区域**: 是,规则 4× 下采样,每 code 恒对应 4×4×4。
- **token 边界破坏结构**: 未定量;扰动实验显示 decoder 跨格平滑,边界效应
  疑似存在但未测。
- **32³ 是否太小**: 是,且被 prep 的比例归一化放大了这个问题——
  一堵带门窗的墙 32³ 里只有 2-4 格厚,细节层级(窗框/屋檐)低于格子分辨率。
- **长程依赖**: 512 长度 + 4 层 192 维,GPT 理论可见全序列,但光栅序下
  对角结构跨度 256+,学习难度大;无层次化先验。
- **roof/wall/opening 被压缩**: 是——6 类材质把"墙 vs 屋顶"的语义差完全抹掉
  (都是 mineral),开口(air)与背景 air 无差别。

## 八、Decoder / reconstruction audit

证据: out_lib/vq_eval/(mcbox, 重建对比渲染)+ acc 0.963。

| 项目 | 保持? | 证据 |
|---|---|---|
| 墙体 | 部分 | 大体块在,薄墙可糊成块 |
| 房间 | 部分 | 大房间在,小房间可并墙 |
| 门窗 | **差** | 窗格细节丢失(目检 6/10 报告"窗/叶细节糊") |
| 屋顶 | 大致保持 | 陡坡/细节简化 |
| 对称性 | 未知 | 无专项指标 |
| block 类型 | 6 类层面基本保持 | acc 0.963 含 air 虚高;细类不存在 |
| 结构连通性 | 未知 | 无专项指标 |

**目前没有**对称性/连通性指标——需要新写,本审计不执行。

## 九、Generation audit(四阶段分开)

1. **reconstruction**(真实→encode→decode): acc 0.963 / 目检 6/10。
   **问题已经存在**: 窗叶细节糊、比例被 prep 归一化——tokenizer 就是瓶颈之一。
2. **latent reconstruction**(真实 token→decode): 同上(同一路径)。
3. **prior-generated latent**(GPT 采样 token→decode): 目检 2–4/10,
   结构碎、无框架。
4. **final generated building**: 生产线不用 prior,实际出图=检索搬运,
   质量=库内建筑原样。

**哪个阶段最先出问题**: **encoder/decoder 一侧先出问题**(重建就丢细节),
prior 再叠加(统计规律≠建筑语法)。按反馈建议的判读规则:
重建不好时**无资格单独评价 prior**——两者都欠账,tokenizer 优先修。

## 十、信息瓶颈表

| Information | Input? | Preserved? | Where lost? | Evidence |
|---|---|---|---|---|
| Geometry | 是 | 部分 | prep 每轴拉伸归一化 | prep_vq_data.py:64-65 |
| Topology(房间连通) | 否 | — | 从未输入 | — |
| Structure(承重/骨架) | 否 | — | 从未输入;6 类不含结构语义 | prep 分类表 |
| Function | 否 | — | 标签存在未使用 | vision_types.json |
| Style | v2 输入 | 未知 | 7 离散类,无连续嵌入 | train_prior.py:29-32 |
| Material | 部分(6 类) | 部分 | 251 种 ID→6 类 | prep_vq_data.py:21-47 |
| Scale | 否 | **否** | bbox 缩放抹平 | prep_vq_data.py:64-65 |
| Orientation | 否 | **否** | schematic Data 字段丢弃 | schemlib.py:189-191 |
| Terrain | 否 | — | 从未输入 | — |
| Context(环境/邻接) | 否 | — | 从未输入 | — |
| Decoration(窗/檐/灯) | 部分 | **差** | 分辨率+6类+重建糊 | vq_eval 目检 |

## 十一、方向可行性判断(当前架构)

| 方向 | 判断 | 原因 |
|---|---|---|
| A. skeleton→building | **不可以** | 模型栈无任何骨架表示/条件;skeleton 指标只用于检索过滤 |
| B. style-conditioned | **部分可以** | v2 style token 在训;单离散 token,无 CFG,效果未验 |
| C. function-conditioned | **不可以** | 标签在库但未入模(架构可平移 v2 做法,当前没有) |
| D. style interpolation | **不可以** | 离散 token 无插值空间 |
| E. latent interpolation | **部分可以** | code 空间可平滑过渡(解码器平滑),但语义性未证;扰动实验显示 code 语义弱 |
| F. retrieval by embedding | **可以** | 生产线在用(features.npz+MMR),已验证可用 |
| G. structure→decoration | **不可以** | 单阶段,无结构/装饰分解 |
| H. village→town→city | **部分可以(系统级)** | worldgen+assemble2 规则装配已能出图;模型栈不参与,无法学"城市逻辑" |
| I. mixed-style generation | **不可以** | 单标签;无多标签/连续风格向量 |
| J. MoE | **不可以** | 2.3M 参数单体,无专家结构;数据/任务也未到分工条件 |

## 十二、真实架构总结(给另一个研究者)

**500-1000 字版本:**

这个项目有一条**生产线**和一套**实验模型栈**,两者目前没有交汇。生产线是
检索式: 从 3,262 栋 planetminecraft 存档库里按嵌入相似度+风格锚点选楼,
原样搬进程序化地形。实验模型栈是一个 3D VQ-VAE 加小型自回归 GPT,
是"向量→建筑"方向的第一版可行性验证。

**系统真正学习的对象**: 一团"被压成 30×30×30 立方体、把 251 种方块折成
6 种材质、丢掉全部朝向状态"的混合类型物体(只有 52% 是建筑,还有车、雕像、
像素画、植被)在 8³ 网格光栅序下的 code 共现统计。GPT 预测的不是建筑语法,
而是"这句话的下一个词"。

**输入信息**: 6 类材质占用体素,32³,仅此而已。风格(v2)是第一个也是唯一
一个条件。功能、平面、地形、尺寸、朝向全部缺席。

**latent 表示**: 32³ → 4× 下采样 → 8³ 格 × 64 维 → 1024 码本量化 →
512 个离散 code。全码在用(有效词表 perplexity≈249),单 code 扰动只改变
0.04-0.2% 体素——code 更像"局部材质纹理的软统计",不是可解释部件。

**生成模型预测**: 下一个 code 的分布(交叉熵,causal GPT,~2.3M 参数,
temperature 0.9 无截断采样)。

**三大瓶颈**:
1. **预处理毁掉几何**: 每轴独立缩放把所有建筑拉成正方体,比例/尺寸信息归零;
   block state 与朝向在解析期就丢了。
2. **数据**: 3,262 样本太少,混入 48% 非建筑,44 完全重复,无 train/val/test。
3. **表示太粗且无条件**: 32³+6 类,门窗檐口低于分辨率;无平面/功能/结构条件,
   模型无从学"建筑逻辑"。

**最不该做的三件事**:
1. 不要堆参数/堆 GPU——瓶颈在表示和数据,放大会得到"更贵的看不懂"。
2. 不要现在上 MoE/多专家——条件分解还没做,先分信息流。
3. 不要拿 prior 的采样质量单独评判生成方向——重建本身就有损,两笔账分开算。

**只允许三个最小实验**:
1. **保比例重预处理**: 保留长宽高比例(等比缩放+pad,不逐轴拉伸)+ 只留
   type=building → 重训 VQ → 对比重建质量(直接检验"tokenizer 瓶颈"假设)。
2. **补完表示审计**: 码本图谱 + 多 code 连续扰动 + 共现聚类可视化,
   判定 code 是否有稳定语义。
3. **skeleton DSL 最小原型**: 手工定义墙/楼板/屋顶/开口/房间的 JSON 表示,
   选 10 栋房子做 ground truth,测"骨架→体素壳"纯规则转换的可行性。

---

## 附: 关键文件与符号索引

| 文件 | 关键类/函数 | 作用 |
|---|---|---|
| src/prep_vq_data.py | `classify`, `main` | 6 类折叠、bbox 裁剪、逐轴缩放、pad 32³ |
| src/train_vq.py | `ResBlock`, `VQVAE`(enc/vq/dec), `main` | VQ-VAE 训练 |
| src/train_prior.py | `Block`, `GPT`, `get_vq`, `extract_tokens`, `main`(train/sample) | code 提取、GPT 训练、采样 |
| src/schemlib.py | `schematic_to_grid`(156-191), `_NAME_TO_LEGACY` | schematic/schem→体素;Data 丢弃点 |
| src/latent_audit.py | `main` | 共现/位置/扰动审计 |
| src/codebook_atlas.py | `main` | 码本图谱 |
| src/skeleton_metrics.py | enclosure v3 | 骨架指标(仅生产线用) |
| src/assemble2.py | 检索/MMR/风格锚点 | 生产线装配 |
| src/features.py | 嵌入提取 | 生产线嵌入 |
| out_lib/meta.jsonl | — | 3262 条: idx/file/desc/dims/nblocks/url |
| out_lib/vq_data.npz | X, idx | (3262,32,32,32) uint8 |
| out_lib/vq_tokens.npz | tokens, idx, styles | (3262,512) code 序列 |
| out_lib/vision_types.json | — | 风格/类型标签 3181 条 |
| out_lib/vq_ckpt.pt / prior_ckpt.pt | state dict | VQ(prior_ckpt 已被 v2 覆盖) |
| docs/prior_train.log | — | v1 训练记录(loss 5.45→1.11) |
| out_lib/vq_eval/ | — | 重建/插值/采样对比渲染 |

——审计完。未修改任何文件(本报告除外),未启动任何训练。
