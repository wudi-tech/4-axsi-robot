# Git 提交流程指南

本文档说明如何将本地代码同步推送到 GitHub 仓库 `wudi-tech/4-axsi-robot`。

---

## 一、日常提交推送（最常用）

每次修改代码后，按以下步骤操作：

```bash
# 第一步：进入项目目录
cd E:\PYproject\vision_system

# 第二步：查看当前改动（可选，建议每次都看一眼）
git status

# 第三步：将所有改动加入暂存区
git add .

# 第四步：提交，-m 后面填写本次提交说明
git commit -m "你的提交说明，例如：修复舵机角度计算bug"

# 第五步：推送到 GitHub
git push origin main
```

---

## 二、只提交部分文件

如果不想提交所有改动，只提交指定文件：

```bash
# 只添加某个文件
git add robot/servo.py

# 或添加某个目录
git add robot/

# 然后提交和推送
git commit -m "更新舵机控制逻辑"
git push origin main
```

---

## 三、从 GitHub 拉取最新代码

在开始工作前，建议先拉取远程最新代码，避免冲突：

```bash
cd E:\PYproject\vision_system
git pull origin main
```

---

## 四、查看提交历史

```bash
# 查看最近 10 条提交记录
git log --oneline -10
```

---

## 五、常见问题处理

### 推送被拒绝（远程有更新）
```bash
# 先拉取远程代码，再推送
git pull origin main
git push origin main
```

### 撤销最后一次提交（还未推送）
```bash
git reset --soft HEAD~1
```

### 放弃本地所有未提交的修改（谨慎！不可恢复）
```bash
git checkout -- .
```

---

## 六、提交信息规范建议

好的提交信息让历史记录一目了然，建议格式：

| 场景 | 示例 |
|------|------|
| 新增功能 | `新增语音控制模块` |
| 修复问题 | `修复PCA9685初始化失败问题` |
| 修改代码 | `优化舵机角度映射算法` |
| 日常同步 | `3.31 同步更新，修改ch347驱动` |

---

## 七、远程仓库信息

- **GitHub 地址**：https://github.com/wudi-tech/4-axsi-robot
- **默认分支**：main
- **本地路径**：`E:\PYproject\vision_system`
