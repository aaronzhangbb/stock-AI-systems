# 本地与远程仓库合并指南

> 当网络可用时，按以下步骤合并远程最新代码，同时保留本地所有修复与功能。

---

## 一、本地独有修改（需保留）

本机已完成的修复，**远程仓库可能没有**，合并时必须保留：

### app.py
| 修改 | 说明 |
|------|------|
| `width='stretch'` → `use_container_width=True` | 修复 Streamlit `TypeError: ButtonMixin.button() got an unexpected keyword argument 'width'` |
| `st.dataframe(..., width='stretch')` → `use_container_width=True` | 同上，dataframe 组件兼容 |
| AI 扫描前模型存在性检查 | 模型不存在时显示友好提示，而非 `FileNotFoundError` |

### retrain_all.py
| 修改 | 说明 |
|------|------|
| `XGBOOST_DEVICE` 环境变量 | 支持 `XGBOOST_DEVICE=cpu` 强制 CPU 训练 |
| GPU 内存不足自动回退 CPU | 捕获 "Unable to allocate" 等错误，自动改用 CPU 训练 |

---

## 二、推荐合并流程

### 步骤 1：确保 VPN/代理已开启（访问 GitHub 需外网）

若使用代理（如端口 17890）：
```powershell
git config http.proxy "http://127.0.0.1:17890"
git config https.proxy "http://127.0.0.1:17890"
```

### 步骤 2：暂存本地修改
```powershell
cd "c:\xunqing\project\finace dock system\stock-AI-systems"
git stash push -m "local_fixes" -- app.py retrain_all.py
```

### 步骤 3：拉取远程最新
```powershell
git fetch origin
git pull origin master
```

### 步骤 4：合并并解决冲突（若有）
```powershell
git stash pop
```

若出现冲突，手动编辑冲突文件：
- **app.py**：保留 `use_container_width=True`、模型检查逻辑；采纳远程新增的其他功能
- **retrain_all.py**：保留 `XGBOOST_DEVICE`、GPU→CPU 回退逻辑；采纳远程其他改动

### 步骤 5：验证
```powershell
python -m py_compile app.py retrain_all.py
streamlit run app.py
```

---

## 三、合并冲突时的取舍原则

| 情况 | 建议 |
|------|------|
| 同一处既有本地修改又有远程修改 | 以「功能最全」为准：本地修复 + 远程新功能都要保留 |
| 参数名 `width` vs `use_container_width` | **必须**用 `use_container_width`（Streamlit 新版不支持 `width`） |
| XGBoost device / 内存回退 | **必须**保留本地逻辑 |
| 远程新增模块/功能 | **全部**采纳 |

---

## 四、合并后建议提交

```powershell
git add app.py retrain_all.py
git commit -m "fix: Streamlit 兼容 + XGBoost 内存回退 + 模型检查提示"
git push origin master
```

这样远程仓库也会拥有这些修复，其他电脑拉取后即可直接使用。
