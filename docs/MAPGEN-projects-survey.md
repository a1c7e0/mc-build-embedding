# Similar Projects Survey: AI/agent Minecraft map & world generation

Compiled 2026-10-07 at the user's request ("所有类似的项目你都可以拿过来看看").
Focus: what each project does and what we can borrow.

## A. Agent-driven (LLM in the loop)

| project | approach | borrow |
|---|---|---|
| **JEV + Astra Minecraft speedrun** (rmalde/minecraft-agent) | Two-tier: Astra plans (35 calls), JEV selects bounded actions from structured game state (131 decisions), Mineflayer executes on vanilla server. Read-only sensors, 17 run checks. 8:43 dragon kill | planner(贵) + selector(便宜) 双层分工;structured state + bounded action space(= msg.txt 的 validator 思路);每次运行带验证清单 |
| **Codex-Minecraft-Gameplay** (wz1119) | Computer-use toolkit: screenshots + OS input events; observe → act → verify loop in the live game | 闭环范式与我们的"生成→渲染→视觉验证→修复"同构;可把我们的管线包成同样的 toolkit 给 agent 驱动 |
| **Astra vanilla Minecraft map** (r/codex demo) | One/few prompts, Codex writes map-gen code and iterates; reviewers highlight "procedural map generation" as the impressive part | 证实 procedural + 代码生成路线(我们已有);agent 自迭代代码而非直接输出体素 |
| **GPT-6 Astra demos in general** (awesome-gpt-6-astra, 181 cases) | Multi-day, asset-by-asset generation (a city's every building generated one by one over 5 days), then compose; known failure mode "AI design smell": flat design + same green palette unless explicitly steered | 逐资产生成+组装 = 我们的库+装配;显式风格约束是必须项(我们已加调色板/风格锚) |
| **Claude Code + 30 MCP tools 3D world builder** (r/ClaudeAI) | Model drives terrain/model/weather tools in real time through MCP | 把管线工具化(worldgen/assemble/render)供 agent 调用的架构 |
| **Planner/executor split blog** (Claude plan → Gemini implement) | One model writes a 3000-word plan, another implements | 规划/实现分模;与 JEV 双层同思路 |

## B. Terrain & world generation engines (no LLM needed)

| project | approach | borrow |
|---|---|---|
| **DreamLab-AI/terracraft** (uses **arnis** Rust engine) | Real place → world: OSM (buildings/roads/water/rails/landuse) + elevation (AWS Terrarium 30m / LIDAR 1-2m) → vanilla MC world, 290+ parametric building templates (walls/doors/windows/roofs/interiors). Optional LLM "AI enrichment" that only assigns building levels/materials/roof types per region | 强烈对口:LLM 只做元数据增强、模板做体素(= 我们的 motif/DSL 路线);"真实地点"可成为第二种 prompt 模式;arnis 引擎可直接复用做真实地形模式 |
| **CartoVoxel** | Browser: OSM + elevation → terrain/roads/buildings, Java/Bedrock export; positions itself as the automated alternative to WorldPainter | Web 产品形态;数据源清单 |
| **Neural-Terrain** | Web app: text prompt → "AI agent world generation" → hosted playable server + community sharing | 产品闭环:prompt → 可玩服务器 + 分享 |
| **BeamNG.WorldForge** | Local LLM (Ollama/Qwen) drives terrain/roads/buildings for BeamNG maps | 本地小模型驱动分区/道路/建筑的配方 |
| **WorldPainter** (classic) | Interactive terrain painting + stamps (tree/building brushes) on heightmaps | "笔刷/stamp"心智模型 = 我们的 motif 库;人机混合修补界面 |
| **Minecraft-Terrain-GAN** / research (World-GAN etc.) | Learned terrain generators | 已记入 RELATED-WORK.md |

## Conclusions for our pipeline

1. **架构验证**:工业界/社区在做的正是我们这套——procedural 骨架 + 参数化建筑模板 + LLM 只碰语义/元数据 + 视觉闭环迭代。terracraft 的"LLM enrichment 只填 levels/materials/roof"与我们的"LLM 语义层、编译器执行"完全一致。
2. **可直接借的件**:
   - arnis(Rust,真实世界→MC)做"真实地点"模式;
   - 双层模型分工(贵 planner + 便宜 selector/critic)——flash 视觉就是我们的 cheap critic;
   - 运行验证清单(run checks)制度化进管线(我们有 probe*.py,继续加);
   - 逐资产生成+组装的长跑模式(motif/VQ 实验就按这个跑)。
3. **该避免的坑**:"AI design smell"(扁平+绿色泛滥)在多个 demo 复现——显式调色板/风格锚不是可选项;资产质量不一致 = 需要逐资产 QA(双视角渲染 + flash 核型)。
