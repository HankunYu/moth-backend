# 蛾子后端系统 (Moth Backend)

一个基于遗传算法的蛾子繁殖和生命周期管理系统，支持智能贴图混合、动态寿命控制和 AI 驱动的提示词整合。

## 🚀 功能特性

### 核心功能
- **自动蛾子生成**: 监控目录，自动从贴图文件创建蛾子
- **智能繁殖系统**: 使用遗传算法混合父代贴图生成子代
- **动态生命周期**: 根据种群数量智能调整蛾子寿命
- **AI 提示词整合**: 使用 Ollama 智能合并父代提示词
- **UDP 通信**: 实时发送蛾子命令到外部系统

### 高级特性
- **种群控制**: 自动维持蛾子数量在指定范围内 (5-200只)
- **遗传算法优化**: 使用 SSIM 和边缘保护评估贴图混合质量
- **实时监控**: 交互式命令行界面，实时查看蛾子状态
- **完整数据追踪**: 记录每只蛾子的完整生命周期数据

## 📋 系统要求

- Python 3.8+
- Ollama 服务 (可选，用于 AI 提示词整合)
- 足够的磁盘空间存储贴图文件

## 🔧 安装步骤

### 1. 环境准备
```bash
# 克隆项目
git clone <repository-url>
cd moth-backend

# 创建虚拟环境 (推荐)
python3 -m venv moth-env
source moth-env/bin/activate  # macOS/Linux
# 或 moth-env\Scripts\activate  # Windows
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 创建必要目录
```bash
mkdir -p example
mkdir -p offspring_textures
```

### 4. Ollama 设置 (可选)
```bash
# 安装 Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 拉取模型
ollama pull deepseek-r1:1.5b
```

## ⚙️ 配置说明

编辑 `config.json` 文件进行个性化配置：

```json
{
  "monitoring": {
    "watch_directory": "./example",          // 监控目录
    "texture_path_pattern": "./mothtexture/generated_textures_000_2000.png"
  },
  "moth_settings": {
    "min_moths": 5,                         // 最少蛾子数量
    "max_moths": 200,                       // 最多蛾子数量
    "base_lifespan_seconds": 300,           // 基础寿命(秒)
    "lifespan_adjustment_factor": 1.2       // 寿命调整倍数
  },
  "ollama": {
    "enabled": true,                        // 是否启用 AI 整合
    "model": "deepseek-r1:1.5b",           // 使用的模型
    "host": "http://localhost:11434"        // Ollama 服务地址
  }
}
```

## 🎯 使用指南

### 1. 启动系统
```bash
python moth_controller.py
```

### 2. 创建蛾子
在 `./example` 目录下创建新文件夹，包含：
```
example/
  └── 蛾子名称/
      ├── mothtexture/
      │   └── generated_textures_000_2000.png  # 贴图文件
      └── prompt.txt                           # 提示词文件
```

### 3. 交互式命令

启动后可使用以下命令：

#### 📊 查看状态
```bash
> status
```
显示种群统计信息：
- 总蛾子数量
- 存活/死亡数量  
- 已交配数量
- 种群状态 (正常/过少/过多)

#### 📋 列出所有蛾子
```bash
> list
```
显示每只蛾子的详细信息：
- UUID 和贴图路径
- 提示词内容
- 年龄和剩余寿命
- 交配状态和族谱信息

#### 💕 蛾子交配
```bash
> mate <蛾子ID1> <蛾子ID2>
```
让两只蛾子交配，系统会：
1. 使用遗传算法混合父代贴图
2. 用 Ollama AI 整合父代提示词
3. 自动创建子代蛾子

#### ☠️ 手动杀死蛾子
```bash
> kill <蛾子ID>
```

#### 🚪 退出系统
```bash
> quit
```

## 🧬 繁殖机制详解

### 贴图混合
- 使用遗传算法优化混合掩码
- 基于 SSIM (结构相似性) 评估混合质量
- 自动调整掩码分辨率和进化参数

### 提示词整合
当两只蛾子交配时，AI 会智能整合父代提示词：

```
父代1: "红色的美丽蛾子带着金色纹路"
父代2: "蓝色的夜行蛾喜欢月光"
      ↓ AI 整合
子代: "金色纹路的夜行蛾在月光下展现红蓝交融的美丽"
```

### 生命周期管理
- **种群过少**: 自动延长蛾子寿命
- **种群过多**: 缩短老蛾子寿命让其自然死亡
- **智能平衡**: 始终维持在配置的数量范围内

## 🔌 UDP 通信协议

系统通过 UDP 发送以下命令：

```bash
# 生成新蛾子
generatemoth|<蛾子ID>|<贴图路径>|<提示词>

# 子代蛾子诞生  
birthmoth|<蛾子ID>|<贴图路径>|<整合后提示词>

# 蛾子交配
matingmoth|<蛾子ID1>|<蛾子ID2>

# 蛾子死亡
killmoth|<蛾子ID>
```

## 📁 文件结构

```
moth-backend/
├── moth_controller.py      # 主控制器
├── udp_bridge_client.py    # UDP 通信客户端
├── mask.py                 # 遗传算法贴图混合
├── config.json             # 配置文件
├── requirements.txt        # Python 依赖
├── example/                # 蛾子输入目录
├── offspring_textures/     # 子代贴图输出
└── moth_controller.log     # 系统日志
```

## 🐛 故障排除

### 常见问题

1. **Ollama 连接失败**
   ```bash
   # 检查 Ollama 服务状态
   ollama list
   
   # 重启 Ollama 服务
   ollama serve
   ```

2. **贴图文件未找到**
   - 确保贴图文件路径正确：`./mothtexture/generated_textures_000_2000.png`
   - 检查文件权限

3. **种群数量异常**
   - 调整 `config.json` 中的 `min_moths` 和 `max_moths`
   - 检查寿命相关参数设置

4. **UDP 通信失败**
   - 确认端口未被占用
   - 检查防火墙设置

### 日志调试
查看详细日志：
```bash
tail -f moth_controller.log
```