"""
语音控制模块
============
基于 Vosk 的离线中文语音识别，独立后台线程持续监听麦克风，
匹配关键词后将动作名称写入队列，主循环按需取用。
支持 pyttsx3 语音反馈（TTS），抓取成功后可调用 speak("真棒") 给出奖励。

安装依赖
--------
    pip install vosk pyaudio pyttsx3

下载中文小模型（约 50 MB，够用于固定指令识别）
    https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip
    解压后重命名为 vosk-model-small-cn，放到：
        vision_system/data/models/vosk-model-small-cn/

典型用法
--------
    vc = VoiceController()
    vc.start()

    while True:
        cmd = vc.get_command()          # 非阻塞，无指令返回 None
        if cmd == "home":
            arm.home()
            vc.speak("已归零")
        elif cmd == "stop":
            arm.emergency_stop()
            vc.speak("已停止")
        elif cmd == "pick_carrot":
            success = pipeline.pick("carrot")
            vc.speak("真棒" if success else "抓取失败")

    vc.stop()

内置指令表（可通过 set_commands() 替换）
----------------------------------------
    "归零" / "回原点"          → "home"
    "停止" / "急停"            → "stop"
    "开始" / "启动"            → "start"
    "暂停"                     → "pause"
    "萝卜"                     → "pick_carrot"
    "纸巾"                     → "pick_tissue"
    "米老鼠"                   → "pick_mickey"
    "状态"                     → "status"
"""
from __future__ import annotations
import json
import os
import queue
import sys
import threading
import time

# ─── 默认内置指令表 ────────────────────────────────────────────────────── #
# key: 识别到的关键词（子串匹配）  value: 动作名称（传给主循环）
# 同一动作可对应多个关键词；关键词越长越具体，建议放前面（匹配时按顺序）
DEFAULT_COMMANDS: dict[str, str] = {
    # 控制（长词优先）
    "急停":   "stop",
    "停止":   "stop",
    "回原点": "home",
    "归零":   "home",
    "启动":   "start",
    "开始":   "start",
    "暂停":   "pause",
    "状态":   "status",
    # 抓取目标
    "米老鼠": "pick_mickey",   # 最长，放最前
    "萝卜":   "pick_carrot",
    "纸巾":   "pick_tissue",
}

# 默认模型路径（相对项目根）
_DEFAULT_MODEL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "models", "vosk-model-small-cn"
)


class VoiceController:
    """
    持续监听麦克风，后台线程识别语音，主线程通过 get_command() 取指令。

    线程安全：start()/stop() 和 get_command() 可在不同线程调用。
    """

    def __init__(self,
                 model_path: str = _DEFAULT_MODEL,
                 sample_rate: int = 16000,
                 device_index: int = None,
                 debounce_s: float = 1.5):
        """
        Args:
            model_path  : Vosk 中文模型目录路径
            sample_rate : 采样率，Vosk 推荐 16000
            device_index: PyAudio 麦克风设备索引，None 使用系统默认
            debounce_s  : 同一指令的最短触发间隔（秒），防止重复识别
        """
        self.model_path   = model_path
        self.sample_rate  = sample_rate
        self.device_index = device_index
        self.debounce_s   = debounce_s

        self._commands: dict[str, str] = dict(DEFAULT_COMMANDS)
        self._cmd_queue: queue.Queue   = queue.Queue()
        self._stop_event: threading.Event = threading.Event()
        self._thread: threading.Thread    = None

        # 防抖：记录每个 action 上次触发时间
        self._last_trigger: dict[str, float] = {}

        self._model  = None   # vosk.Model
        self._ready  = False  # 模型加载完成标志

        # TTS 引擎（pyttsx3），独立线程播放，不阻塞主线程
        self._tts_engine  = None
        self._tts_queue: queue.Queue  = queue.Queue()
        self._tts_thread: threading.Thread = None
        self._init_tts()

    # ── 指令管理 ──────────────────────────────────────────────────────── #

    def set_commands(self, commands: dict[str, str]):
        """
        替换全部指令表。
        commands: {关键词: 动作名称}，例如 {"停止": "stop"}
        """
        self._commands = dict(commands)

    def add_command(self, keyword: str, action: str):
        """追加单条指令映射"""
        self._commands[keyword] = action

    def remove_command(self, keyword: str):
        """删除单条指令"""
        self._commands.pop(keyword, None)

    def list_commands(self):
        """打印当前全部指令表"""
        print("\n[语音] 当前指令表：")
        grouped: dict[str, list[str]] = {}
        for kw, act in self._commands.items():
            grouped.setdefault(act, []).append(kw)
        for act, kws in grouped.items():
            print(f"  {act:16s} ← {' / '.join(kws)}")
        print()

    # ── TTS 语音反馈 ──────────────────────────────────────────────────── #

    def _init_tts(self):
        """初始化 pyttsx3，失败时静默降级（不影响识别功能）"""
        try:
            import pyttsx3
            self._tts_engine = pyttsx3.init()
            # 设置中文语速和音量
            self._tts_engine.setProperty("rate",   180)   # 语速（默认200）
            self._tts_engine.setProperty("volume", 1.0)   # 音量 0~1

            # 优先选择中文女声（Microsoft Huihui / Yaoyao 等）
            voices = self._tts_engine.getProperty("voices")
            for v in voices:
                if any(k in v.name for k in ("Huihui", "Yaoyao", "Kangkang",
                                              "Chinese", "中文")):
                    self._tts_engine.setProperty("voice", v.id)
                    break

            # 独立 TTS 播放线程（串行队列，避免引擎并发冲突）
            self._tts_thread = threading.Thread(
                target=self._tts_loop, name="TTSThread", daemon=True
            )
            self._tts_thread.start()
            print("[语音] TTS 初始化成功")
        except Exception as e:
            print(f"[语音] TTS 不可用（{e}），语音反馈已禁用")
            print("       安装方式：pip install pyttsx3")
            self._tts_engine = None

    def _tts_loop(self):
        """TTS 播放线程：从队列取文本逐条播放"""
        while True:
            text = self._tts_queue.get()
            if text is None:          # 退出哨兵
                break
            try:
                self._tts_engine.say(text)
                self._tts_engine.runAndWait()
            except Exception as e:
                print(f"[语音] TTS 播放失败: {e}")

    def speak(self, text: str):
        """
        异步播放语音（不阻塞调用方）。
        抓取成功后调用 vc.speak("真棒") 即可触发奖励音。

        Args:
            text: 要播放的中文文本，例如 "真棒" / "已归零" / "抓取失败"
        """
        if self._tts_engine is None:
            print(f"[语音] TTS 未就绪，跳过播放: {text}")
            return
        self._tts_queue.put(text)

    # ── 线程控制 ──────────────────────────────────────────────────────── #

    def start(self) -> bool:
        """
        启动后台监听线程。
        Returns:
            True  启动成功
            False 依赖缺失或模型不存在
        """
        if self._thread and self._thread.is_alive():
            print("[语音] 已在运行，无需重复启动")
            return True

        # 检查依赖
        if not self._check_deps():
            return False

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._listen_loop, name="VoiceThread", daemon=True
        )
        self._thread.start()
        return True

    def stop(self):
        """停止监听线程和 TTS 线程（最多等待 3 秒）"""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._thread = None
        self._ready  = False
        # 停止 TTS 线程
        if self._tts_thread and self._tts_thread.is_alive():
            self._tts_queue.put(None)   # 发送退出哨兵
            self._tts_thread.join(timeout=3.0)
        print("[语音] 已停止")

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def is_ready(self) -> bool:
        """模型是否已加载完成（start() 后需等待约 1-2 秒）"""
        return self._ready

    # ── 主循环取指令 ──────────────────────────────────────────────────── #

    def get_command(self, block: bool = False,
                    timeout: float = None) -> str | None:
        """
        从识别队列取出一条指令动作名称。
        Args:
            block  : True 则阻塞等待；False（默认）立即返回
            timeout: block=True 时最长等待秒数，None 表示永久等待
        Returns:
            动作名称字符串（如 "stop"/"home"/"pick_red"），
            队列为空时返回 None
        """
        try:
            return self._cmd_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def has_command(self) -> bool:
        return not self._cmd_queue.empty()

    # ── 内部：加载模型 ────────────────────────────────────────────────── #

    def _check_deps(self) -> bool:
        try:
            import vosk       # noqa: F401
            import pyaudio    # noqa: F401
        except ImportError as e:
            print(f"[语音] 缺少依赖：{e}")
            print("       请运行：pip install vosk pyaudio")
            return False

        if not os.path.isdir(self.model_path):
            print(f"[语音] 模型目录不存在：{self.model_path}")
            print("       请下载中文模型并解压到该路径：")
            print("       https://alphacephei.com/vosk/models/"
                  "vosk-model-small-cn-0.22.zip")
            return False
        return True

    def _load_model(self):
        import vosk
        vosk.SetLogLevel(-1)          # 屏蔽 Vosk 内部日志
        print(f"[语音] 加载模型: {self.model_path} ...")
        self._model = vosk.Model(self.model_path)
        print("[语音] 模型加载完成，开始监听")

    # ── 内部：主监听循环 ──────────────────────────────────────────────── #

    def _listen_loop(self):
        import vosk
        import pyaudio

        self._load_model()

        pa     = pyaudio.PyAudio()
        rec    = vosk.KaldiRecognizer(self._model, self.sample_rate)
        stream = pa.open(
            format            = pyaudio.paInt16,
            channels          = 1,
            rate              = self.sample_rate,
            input             = True,
            input_device_index= self.device_index,
            frames_per_buffer = 4000,
        )
        stream.start_stream()
        self._ready = True

        try:
            while not self._stop_event.is_set():
                data = stream.read(4000, exception_on_overflow=False)
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    text   = result.get("text", "").strip()
                    if text:
                        print(f"[语音] 识别: 「{text}」")
                        self._dispatch(text)
        except Exception as e:
            print(f"[语音] 监听异常: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()
            self._ready = False

    # ── 内部：关键词匹配 + 防抖 ───────────────────────────────────────── #

    def _dispatch(self, text: str):
        """在识别文本中查找关键词，命中则入队（带防抖）"""
        matched_action = None
        matched_kw     = None

        # 按关键词长度从长到短匹配，避免短词误匹配
        for kw in sorted(self._commands, key=len, reverse=True):
            if kw in text:
                matched_action = self._commands[kw]
                matched_kw     = kw
                break

        if matched_action is None:
            return

        # 防抖检查
        now  = time.time()
        last = self._last_trigger.get(matched_action, 0.0)
        if now - last < self.debounce_s:
            print(f"[语音] 防抖跳过: {matched_action}")
            return

        self._last_trigger[matched_action] = now
        self._cmd_queue.put(matched_action)
        print(f"[语音] 指令入队: 「{matched_kw}」→ {matched_action}")


# ─── 便捷函数：列出所有麦克风设备 ─────────────────────────────────────── #

def list_microphones():
    """打印系统可用麦克风设备列表"""
    try:
        import pyaudio
    except ImportError:
        print("请先安装 pyaudio：pip install pyaudio")
        return
    pa = pyaudio.PyAudio()
    print("\n[语音] 可用麦克风设备：")
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            print(f"  [{i}] {info['name']}")
    pa.terminate()
    print()


# ─── 独立运行：功能测试 ────────────────────────────────────────────────── #

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="语音控制模块测试")
    parser.add_argument("--list-mics", action="store_true",
                        help="列出可用麦克风")
    parser.add_argument("--mic", type=int, default=None,
                        help="指定麦克风设备索引")
    parser.add_argument("--model", default=_DEFAULT_MODEL,
                        help="Vosk 模型路径")
    args = parser.parse_args()

    if args.list_mics:
        list_microphones()
        sys.exit(0)

    vc = VoiceController(model_path=args.model, device_index=args.mic)
    vc.list_commands()

    if not vc.start():
        sys.exit(1)

    print("=" * 50)
    print("语音控制测试模式 - 对麦克风说指令")
    print("按 Ctrl+C 退出")
    print("=" * 50)

    try:
        while True:
            cmd = vc.get_command()
            if cmd:
                print(f"\n>>> 触发动作: [{cmd}]")
                # 模拟抓取成功反馈
                if cmd.startswith("pick_"):
                    target = {"pick_carrot": "萝卜",
                               "pick_tissue": "纸巾",
                               "pick_mickey": "米老鼠"}.get(cmd, "目标")
                    vc.speak(f"正在抓取{target}")
                    time.sleep(1.5)   # 模拟抓取耗时
                    vc.speak("真棒")  # 奖励反馈
                elif cmd == "home":
                    vc.speak("已归零")
                elif cmd == "stop":
                    vc.speak("已停止")
                print()
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n退出...")
    finally:
        vc.stop()
