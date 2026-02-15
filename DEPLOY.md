# QuantX 云服务器部署方案

> 最后更新：2026-02-13

---

## 一、项目现状分析

### 需要部署的3个服务

| 服务 | 文件 | 运行方式 | 说明 |
|:---|:---|:---|:---|
| Web 界面 | `app.py` | 常驻运行 | Streamlit Web UI，端口 8501 |
| 每日收盘任务 | `daily_job.py` | 定时触发 | 周一~周五 16:15，全市场扫描+AI+邮件推送 |
| 盘中实时监控 | `intraday_monitor.py` | 定时触发 | 周一~周五 09:25 启动，15:00 自动退出 |

### 需要迁移的数据

| 数据 | 路径 | 大小估计 | 说明 |
|:---|:---|:---|:---|
| K线缓存 | `data/stock_cache.db` | 500MB~2GB | SQLite，可在服务器重新生成 |
| 交易数据库 | `data/trading.db` | 很小 | 需迁移（含持仓/交易记录） |
| AI模型 | `data/*.pkl, *.pt, *.json` | ~100MB | XGBoost/形态/Transformer 模型 |
| 日志 | `data/logs/` | 可选 | 历史日志，可不迁移 |

### 需要适配的 Windows 专有代码

| 模块 | 问题 | 解决方案 |
|:---|:---|:---|
| `winotify` | Windows 桌面通知，Linux 无法用 | 条件导入，Linux 跳过 |
| `run_daily.bat` | Windows 批处理 | 改写为 Shell 脚本 |
| `setup_scheduler.ps1` | Windows 任务计划 | 改用 Linux crontab |

---

## 二、云服务器选型建议

### 推荐配置

| 配置项 | 最低配置 | 推荐配置 | 说明 |
|:---|:---|:---|:---|
| **CPU** | 2 核 | 4 核 | AI扫描全市场时需要算力 |
| **内存** | 4 GB | 8 GB | XGBoost+Transformer 内存消耗大 |
| **硬盘** | 40 GB SSD | 80 GB SSD | K线缓存+模型文件 |
| **系统** | Ubuntu 22.04 | Ubuntu 22.04 | 稳定、社区资源丰富 |
| **带宽** | 3 Mbps | 5 Mbps | AkShare数据拉取+Web访问 |
| **GPU** | 不需要 | 不需要 | PyTorch推理用CPU够了 |

### 云厂商推荐（国内）

| 厂商 | 推荐机型 | 预估月费 | 优势 |
|:---|:---|:---|:---|
| **阿里云 ECS** | ecs.c7.xlarge (4C8G) | 150~300元 | 百炼API同网络延迟低 |
| **腾讯云 CVM** | S5.LARGE8 (4C8G) | 150~300元 | QQ邮箱SMTP同生态 |
| **华为云 ECS** | c7.xlarge.2 (4C8G) | 150~300元 | 性价比高 |

> **提示**：首购新用户一般有大额折扣（1~3折），可关注各平台活动
>
> **提示**：不需要 GPU，不需要买 GPU 实例
>
> **提示**：选国内服务器（上海/北京/深圳），AkShare 拉东财数据延迟最低

---

## 三、部署流程（10步分步指南）

### 第0步：购买服务器后的初始设置

```bash
# SSH 连接服务器
ssh root@你的服务器IP

# 更新系统
apt update && apt upgrade -y

# 安装基础工具
apt install -y git curl wget vim htop unzip

# 配置时区（重要！定时任务依赖正确时区）
timedatectl set-timezone Asia/Shanghai
```

### 第1步：安装 Python 3.12

```bash
# 安装编译依赖
apt install -y build-essential libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev libffi-dev \
  liblzma-dev libncurses-dev software-properties-common

# 用 deadsnakes PPA
add-apt-repository ppa:deadsnakes/ppa -y
apt update
apt install -y python3.12 python3.12-venv python3.12-dev

# 验证
python3.12 --version
```

### 第2步：上传项目代码

```bash
# 方案A：Git 克隆（推荐）
cd /opt
git clone https://github.com/你的用户名/my-finance.git quantx
cd quantx

# 方案B：SCP 直接上传（本地 PowerShell 执行）
# scp -r "F:\project\my finance" root@服务器IP:/opt/quantx
```

### 第3步：配置 Python 环境

```bash
cd /opt/quantx
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# 注意：先注释掉 requirements.txt 中的 winotify 行
pip install -r requirements.txt \
  -i https://mirrors.aliyun.com/pypi/simple/ \
  --trusted-host mirrors.aliyun.com

# PyTorch CPU版（无GPU服务器用这个）
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 第4步：迁移数据文件

```bash
# 服务器上创建目录
mkdir -p /opt/quantx/data/logs

# 本地 PowerShell 上传关键数据：
# scp data\trading.db root@服务器IP:/opt/quantx/data/
# scp data\ai_daily_scores.json root@服务器IP:/opt/quantx/data/
# scp data\xgb_v2_model.json root@服务器IP:/opt/quantx/data/
# scp data\pattern_engine.pkl root@服务器IP:/opt/quantx/data/
# scp data\transformer_model.pt root@服务器IP:/opt/quantx/data/
#
# stock_cache.db 太大，建议在服务器上重新生成（daily_job会自动更新）
```

### 第5步：处理 Windows 专有代码

修改 `src/utils/notifier.py`，让它在 Linux 上不报错：

```python
import platform
if platform.system() == 'Windows':
    import winotify
else:
    winotify = None  # Linux 上不支持桌面通知，用邮件替代
```

### 第6步：配置 Streamlit Web 服务（systemd 常驻）

```bash
# 先测试启动
source venv/bin/activate
streamlit run app.py --server.port 8501 --server.headless true --server.address 0.0.0.0

# 创建 systemd 服务（开机自启 + 崩溃自动重启）
sudo tee /etc/systemd/system/quantx-web.service << 'EOF'
[Unit]
Description=QuantX Streamlit Web UI
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/quantx
ExecStart=/opt/quantx/venv/bin/streamlit run app.py --server.port 8501 --server.headless true --server.address 0.0.0.0
Restart=always
RestartSec=10
Environment="LANG=en_US.UTF-8"

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable quantx-web
sudo systemctl start quantx-web
sudo systemctl status quantx-web
```

### 第7步：配置定时任务（crontab 替代 Windows 任务计划）

```bash
crontab -e

# 添加以下内容：

# 收盘任务：周一~五 16:15
15 16 * * 1-5 cd /opt/quantx && /opt/quantx/venv/bin/python daily_job.py >> /opt/quantx/data/logs/cron_daily.log 2>&1

# 盘中监控：周一~五 09:25（脚本自带循环，15:00自动退出）
25 9 * * 1-5 cd /opt/quantx && /opt/quantx/venv/bin/python intraday_monitor.py >> /opt/quantx/data/logs/cron_intraday.log 2>&1

# 日志清理：每月1号
0 3 1 * * find /opt/quantx/data/logs -name "*.log" -mtime +30 -delete
```

### 第8步：配置防火墙和安全组

```bash
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 8501/tcp  # Streamlit Web
sudo ufw enable

# 还需要在云控制台的「安全组」中开放端口：
# - 22/TCP   (SSH)
# - 8501/TCP (Streamlit Web UI)
```

### 第9步：验证部署

```bash
# 检查 Web 服务
curl http://localhost:8501

# 手动运行一次每日任务
cd /opt/quantx && source venv/bin/activate
python daily_job.py

# 浏览器访问: http://服务器公网IP:8501
# 检查邮件是否收到推送
```

---

## 四、进阶优化（可选）

### 4.1 Nginx 反向代理 + HTTPS（需要域名）

```bash
apt install -y nginx

sudo tee /etc/nginx/sites-available/quantx << 'EOF'
server {
    listen 80;
    server_name 你的域名.com;
    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }
}
EOF

sudo ln -s /etc/nginx/sites-available/quantx /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# 免费 HTTPS 证书
apt install -y certbot python3-certbot-nginx
certbot --nginx -d 你的域名.com
```

### 4.2 访问密码保护

```bash
apt install -y apache2-utils
htpasswd -c /etc/nginx/.htpasswd admin
# 在 Nginx location 内添加:
#   auth_basic "QuantX Login";
#   auth_basic_user_file /etc/nginx/.htpasswd;
```

### 4.3 数据自动备份

```bash
# 创建备份脚本 /opt/quantx/backup.sh
#!/bin/bash
BACKUP_DIR="/opt/quantx/backups"
mkdir -p $BACKUP_DIR
DATE=$(date +%Y%m%d)
cp /opt/quantx/data/trading.db "$BACKUP_DIR/trading_$DATE.db"
find $BACKUP_DIR -name "*.db" -mtime +7 -delete

# crontab添加: 0 23 * * * /opt/quantx/backup.sh
```

### 4.4 服务监控（自动重启）

```bash
# crontab添加: 每5分钟检查
# */5 * * * * systemctl is-active quantx-web || systemctl restart quantx-web
```

---

## 五、完整 crontab 汇总

```cron
# QuantX 量化系统 - 全部定时任务

# 收盘任务：周一~五 16:15
15 16 * * 1-5 cd /opt/quantx && /opt/quantx/venv/bin/python daily_job.py >> /opt/quantx/data/logs/cron_daily.log 2>&1

# 盘中监控：周一~五 09:25
25 9 * * 1-5 cd /opt/quantx && /opt/quantx/venv/bin/python intraday_monitor.py >> /opt/quantx/data/logs/cron_intraday.log 2>&1

# 数据备份：每天 23:00
0 23 * * * /opt/quantx/backup.sh >> /opt/quantx/data/logs/backup.log 2>&1

# 日志清理：每月1号
0 3 1 * * find /opt/quantx/data/logs -name "*.log" -mtime +30 -delete

# 服务监控：每5分钟
*/5 * * * * systemctl is-active quantx-web || systemctl restart quantx-web
```

---

## 六、部署检查清单

- [ ] 服务器已购买，SSH 可连接
- [ ] 时区设为 Asia/Shanghai
- [ ] Python 3.12 已安装
- [ ] 项目代码已上传到 /opt/quantx
- [ ] 虚拟环境+依赖已安装
- [ ] winotify 兼容处理完成
- [ ] config.py 配置已检查（邮件/API Key）
- [ ] trading.db + AI模型已迁移
- [ ] Streamlit systemd 服务运行正常
- [ ] 安全组已开放 8501 端口
- [ ] 可通过公网 IP 访问 Web 界面
- [ ] crontab 定时任务已配置
- [ ] 手动运行 daily_job.py 测试通过
- [ ] 邮件推送测试成功
- [ ] （可选）Nginx + HTTPS
- [ ] （可选）访问密码
- [ ] （可选）数据备份
- [ ] （可选）服务监控

---

## 七、预计费用

| 项目 | 月费用 | 备注 |
|:---|:---|:---|
| 云服务器 4C8G | 100~300元 | 新用户首购更低 |
| 域名（可选） | ~5元/月 | .com 约60元/年 |
| SSL证书 | 免费 | Let's Encrypt |
| **合计** | **100~300元/月** | |

---

## 八、需要改的代码清单

购买服务器后需要适配的代码：

| 序号 | 文件 | 改动内容 |
|:---|:---|:---|
| 1 | `requirements.txt` | 注释掉 `winotify` |
| 2 | `src/utils/notifier.py` | 条件导入 winotify，Linux 下跳过 |
| 3 | `config.py` | 敏感信息可改为环境变量（可选） |
| 4 | 新建 `run_daily.sh` | 替代 `run_daily.bat` |
| 5 | 新建 `run_intraday.sh` | 替代 `run_intraday.bat` |
| 6 | 新建 `deploy.sh` | 一键部署脚本 |

> **购买服务器后告诉我服务器信息，我会帮你自动完成所有代码适配并生成一键部署脚本。**
