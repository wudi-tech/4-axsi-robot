"""
CH347F USB → I2C 驱动
WCH CH347 芯片 Python 封装（Windows 平台，调用官方 DLL）

CH347 官方 DLL 下载：
    https://www.wch.cn/products/CH347.html  →  资料下载 → CH347DLL

64位系统放置：CH347DLLA64.dll
32位系统放置：CH347DLL.dll
两者放在项目根目录或系统 PATH 中均可。

I2C 硬件连接：
    CH347F SCL → PCA9685 SCL
    CH347F SDA → PCA9685 SDA
    CH347F GND → PCA9685 GND
    PCA9685 VCC → 3.3V/5V（CH347 可提供 3.3V）
"""
from __future__ import annotations
import ctypes
import sys
import os
import platform
from ctypes import c_ubyte, c_ulong, POINTER, byref


# ──────────────────────────────────────────────────────────────────────── #
#  DLL 加载                                                                 #
# ──────────────────────────────────────────────────────────────────────── #
def _load_dll() -> ctypes.WinDLL:
    """自动选择并加载 CH347 DLL"""
    if platform.system() != "Windows":
        raise OSError("CH347 DLL 仅支持 Windows 平台")

    # 优先搜索顺序：项目根目录 → 系统 PATH
    search_names = (
        ["CH347DLLA64.dll", "CH347DLL.dll"]
        if platform.machine().endswith("64")
        else ["CH347DLL.dll", "CH347DLLA64.dll"]
    )

    # 搜索路径：项目根 / 当前目录 / robot目录
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    search_dirs = [project_root, script_dir, os.getcwd()]

    for dll_name in search_names:
        for d in search_dirs:
            full_path = os.path.join(d, dll_name)
            if os.path.exists(full_path):
                try:
                    return ctypes.WinDLL(full_path)
                except OSError:
                    continue
        # 最后尝试系统 PATH
        try:
            return ctypes.WinDLL(dll_name)
        except OSError:
            continue

    raise FileNotFoundError(
        "未找到 CH347 DLL 文件。\n"
        "请从 https://www.wch.cn/products/CH347.html 下载，\n"
        "并将 CH347DLLA64.dll (64位) 放入项目根目录。"
    )


# ──────────────────────────────────────────────────────────────────────── #
#  CH347 I2C 接口                                                           #
# ──────────────────────────────────────────────────────────────────────── #
class CH347I2C:
    """
    CH347F USB 转 I2C 接口

    使用示例:
        with CH347I2C(device_index=0, speed=CH347I2C.SPEED_FAST) as i2c:
            i2c.write(0x40, bytes([0x00, 0x00]))   # 写 PCA9685 寄存器
            data = i2c.read(0x40, 1)               # 读 1 字节
    """

    # I2C 速度档位（对应 CH347 SetStream 参数）
    SPEED_LOW    = 0   # 20  kHz
    SPEED_STD    = 1   # 100 kHz（标准）
    SPEED_FAST   = 2   # 400 kHz（快速，PCA9685 支持）
    SPEED_HIGH   = 3   # 750 kHz

    # 最大单次传输字节数
    MAX_WRITE_LEN = 4096
    MAX_READ_LEN  = 4096

    def __init__(self, device_index: int = 0,
                 speed: int = SPEED_FAST):
        self._dev_idx = device_index
        self._speed   = speed
        self._dll     = None
        self._opened  = False

    # ------------------------------------------------------------------ #
    #  打开 / 关闭                                                          #
    # ------------------------------------------------------------------ #
    def open(self) -> bool:
        """加载 DLL 并打开 CH347 设备"""
        self._dll = _load_dll()

        # CH347OpenDevice(ULONG DevI) -> HANDLE (非0成功)
        ret = self._dll.CH347OpenDevice(c_ulong(self._dev_idx))
        if ret == 0 or ret == ctypes.c_void_p(-1).value:
            raise RuntimeError(
                f"CH347 设备 {self._dev_idx} 打开失败，"
                "请检查 USB 连接或驱动安装"
            )

        # 设置 I2C 速度
        # CH347SetStream(DevI, Mode): Mode bit[1:0] = I2C速度
        self._dll.CH347SetStream(c_ulong(self._dev_idx), c_ulong(self._speed))

        self._opened = True
        speed_labels = {0: "20kHz", 1: "100kHz", 2: "400kHz", 3: "750kHz"}
        print(f"[CH347] 设备 {self._dev_idx} 已打开，"
              f"I2C速度={speed_labels.get(self._speed, '?')}")
        return True

    def close(self):
        """关闭 CH347 设备"""
        if self._dll and self._opened:
            self._dll.CH347CloseDevice(c_ulong(self._dev_idx))
            self._opened = False
            print(f"[CH347] 设备 {self._dev_idx} 已关闭")

    def is_opened(self) -> bool:
        return self._opened

    # ------------------------------------------------------------------ #
    #  I2C 读写                                                            #
    # ------------------------------------------------------------------ #
    def write(self, i2c_addr: int, data: bytes) -> bool:
        """
        I2C 写操作
        Args:
            i2c_addr: 7位 I2C 设备地址（如 PCA9685 默认 0x40）
            data: 要写入的字节序列（含寄存器地址）
        Returns:
            True 表示成功
        """
        if not self._opened:
            raise RuntimeError("设备未打开")

        # 构造写缓冲：[地址<<1 | 0(写)] + data
        addr_byte   = (i2c_addr << 1) & 0xFE
        write_buf   = bytes([addr_byte]) + bytes(data)
        buf_type    = c_ubyte * len(write_buf)
        c_write_buf = buf_type(*write_buf)

        # CH347StreamI2C(DevI, WriteLen, WriteBuf, ReadLen, ReadBuf)
        ret = self._dll.CH347StreamI2C(
            c_ulong(self._dev_idx),
            c_ulong(len(write_buf)),
            c_write_buf,
            c_ulong(0),
            None
        )
        return bool(ret)

    def read(self, i2c_addr: int, length: int) -> bytes:
        """
        I2C 读操作
        Args:
            i2c_addr: 7位 I2C 设备地址
            length: 要读取的字节数
        Returns:
            读取到的字节序列
        """
        if not self._opened:
            raise RuntimeError("设备未打开")

        addr_byte   = (i2c_addr << 1) | 0x01  # 读标志位
        write_buf   = bytes([addr_byte])
        wbuf_type   = c_ubyte * 1
        c_write_buf = wbuf_type(*write_buf)

        rbuf_type   = c_ubyte * length
        c_read_buf  = rbuf_type()

        ret = self._dll.CH347StreamI2C(
            c_ulong(self._dev_idx),
            c_ulong(1),
            c_write_buf,
            c_ulong(length),
            c_read_buf
        )
        if not ret:
            return b""
        return bytes(c_read_buf)

    def write_reg(self, i2c_addr: int, reg: int, data: bytes) -> bool:
        """
        写寄存器（最常用封装）：先发寄存器地址，再发数据
        Args:
            i2c_addr: I2C 设备地址
            reg: 目标寄存器地址
            data: 要写入的数据字节
        """
        return self.write(i2c_addr, bytes([reg]) + bytes(data))

    def read_reg(self, i2c_addr: int, reg: int, length: int = 1) -> bytes:
        """
        读寄存器：先写寄存器地址，再读数据
        """
        self.write(i2c_addr, bytes([reg]))
        return self.read(i2c_addr, length)

    def scan(self, start: int = 0x08, end: int = 0x77) -> list[int]:
        """
        扫描 I2C 总线上的设备，返回响应的地址列表
        用于硬件调试：确认 PCA9685 是否正确连接
        """
        found = []
        for addr in range(start, end + 1):
            # 发送仅含地址的写请求，有 ACK 则设备存在
            addr_byte   = (addr << 1) & 0xFE
            buf_type    = c_ubyte * 1
            c_write_buf = buf_type(addr_byte)
            ret = self._dll.CH347StreamI2C(
                c_ulong(self._dev_idx),
                c_ulong(1), c_write_buf,
                c_ulong(0), None
            )
            if ret:
                found.append(addr)
                print(f"[CH347] 发现 I2C 设备: 0x{addr:02X}")
        if not found:
            print("[CH347] 未发现任何 I2C 设备")
        return found

    # ------------------------------------------------------------------ #
    #  上下文管理器                                                          #
    # ------------------------------------------------------------------ #
    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()
        return False


# ──────────────────────────────────────────────────────────────────────── #
#  快速测试                                                                 #
# ──────────────────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    print("CH347 I2C 总线扫描测试")
    with CH347I2C(device_index=0, speed=CH347I2C.SPEED_FAST) as i2c:
        devices = i2c.scan()
        print(f"共发现 {len(devices)} 个设备: {[hex(a) for a in devices]}")
