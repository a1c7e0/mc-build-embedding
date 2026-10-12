# Plan-Layer Survey: HouseDiffusion & Graph2Plan (A3 integration memo)

调研来源: delegated research agent, 2026-10-09. 中文存档(给项目组阅读)。

## Repos

- **HouseDiffusion 官方** https://github.com/aminshabani/house_diffusion — CVPR 2023,
  基于 openai/guided-diffusion;可运行;依赖钉死旧版 nightly torch,但
  floorplan_eval 证明 torch 2.4.1 免改可用(去掉 tensorflow/pytorch_fid,
  Shapely 留 1.8.x)。
- **Graph2Plan 官方** https://github.com/HanHan55/Graph2Plan — SIGGRAPH 2020;
  Network/ 训练 + PostProcess/ 推理 + Django UI;align 后处理依赖 Matlab Engine,
  但纯推理 App.forward 可脱离 Matlab。
- **floorplan_eval** https://github.com/birsakm/floorplan_eval — 第三方评测框架,
  HouseDiffusion 采样驱动 + HG++ JSON 转换器 + 规则校验,集成模板直接抄。
- **Housegan-data-reader** https://github.com/sepidsh/Housegan-data-reader — RPLAN→HG++ 读取器。
- **RPLAN-Toolbox** / **PuzzleFussion** (MagicPlan 替代数据,含 reader)。

## 输入/输出

- HouseDiffusion: 输入 = bubble diagram(House-GAN++ JSON:room_type 25 类 one-hot、
  boxes、edges、ed_rm;门是节点:15 大门/17 内门;邻接+门边转 attention mask;
  画布 256px,序列≤100 corners)。**不需要 boundary**。
  输出 = vector floorplan(每房间有序 corner 多边形 [-1,1]→256px),无结构化导出
  (自己从 sample tensor 序列化,floorplan_eval 的 raw npz + common JSON 是范式)。
  多边形可能自交/重叠,官方不 repair。
- Graph2Plan: 输入 = layout graph + boundary;输出 = 256² raster + 每房间
  axis-aligned bbox;decorate 补门窗;与我们的 rooms(rect)+doors 几乎 1:1,房间仅矩形。
- 两者输入都能由我们的 rooms(type,rect)+doors+walls 直接推导。

## 权重

- HouseDiffusion: README "temporary model" Google Drive
  (id 16zKmtxwY5lF6JE-CJGkRf3-OFoD1TrdR),可能失效。
- Graph2Plan: 仓内 Interface/model/model.pth (30.6MB) 实为完整 state_dict
  (与某些说法矛盾,需新版 torch 下自验 load_state_dict)。

## 推理成本

- HouseDiffusion: TransformerModel(4 层,d_model=512,4 heads)× 1000 步 DDPM。
  GPU 实测 0.5-0.9 s/plan;CPU 推算 1-5 分钟/plan;timestep_respacing 到
  100-250 步可降到几十秒。
- Graph2Plan: GNN+CNN 单前向,CPU 秒级/plan;Matlab 只卡 align 后处理。

## 集成方案(双路)

主路 HouseDiffusion(纯气泡图条件,契合"先平面骨架后体素"):
1. 适配层 A: 我们 JSON → HG++ JSON(rect→boxes,共享墙→ed_rm,doors→门节点,
   type→RPLAN 25 类映射;改造 floorplan_eval 的 g2p_to_hgjson.py)。
2. cndist: RPLAN 统计拿不到 → 用我们 3262 库转平面后池化角点分布。
3. 采样: p_sample_loop(可 respacing 提速);256px → 32-64 block 平面网格
   (1 block≈0.5m,比例定为管线常量)。
4. 适配层 B: 多边形→轴对齐 snap→rooms[].rect;并集边界→walls(扣门洞);门落共享墙。
   non-Manhattan 必须 snap(A3 先 snap,或把格式扩成 polygon rooms)。
5. 交 voxel builder: 平面拉伸房间壳、墙体带洞、doors 开口;VQ/prior 不动。
6. 质量闸: floorplan_eval 规则(无重叠/孔洞/连通/最小面积)不过则重采样。
备路 Graph2Plan: 有 boundary 时走 refineBox(跳过 Matlab align),天然矩形 rooms。

## 风险清单

- **许可**: HouseDiffusion 禁商用/GPL v3(产品集成有传染风险);Graph2Plan 无
  license 文件(默认保留所有权利)。
- RPLAN 数据 Google Form 审批制,多人拿不到;官方权重是"temporary"。
- 输出多边形可自交/重叠;non-Manhattan vs rect-only 冲突。
- 条件分布敏感(门是图内节点;--target_set 8 训练,≥8 房质量未知)。
- px→block 尺度、门窗尺寸、层高都是体素层的设计决策。
- RPLAN 25 类 ↔ 我们 type 词汇要建映射表。
