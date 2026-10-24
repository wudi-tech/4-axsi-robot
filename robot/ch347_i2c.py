"""
CH347F USB → I2C 驱动
WCH CH347 芯片 Python 封装（Windows 平台，调用官方 DLL）

DLL 文件位置（已随项目内置，无需另行下载）：
    robot/libs/CH347DLLA64.DLL   ← 64位系统（主用）
    robot/libs/CH347DLL.DLL      ← 32位系统

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
    """自动选择并加载 CH347 DLL（优先使用项目内置 robot/libs/ 中的版本）"""
    if platform.system() != "Windows":
        raise OSError("CH347 DLL 仅支持 Windows 平台")

    is_64 = platform.machine().endswith("64")
    search_names = (
        ["CH347DLLA64.DLL", "CH347DLL.DLL"]
        if is_64
        else ["CH347DLL.DLL", "CH347DLLA64.DLL"]
    )

    # 搜索路径：robot/libs（首选，项目内置）→ robot目录 → 项目根 → 当前目录
    script_dir   = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    libs_dir     = os.path.join(script_dir, "libs")
    search_dirs  = [libs_dir, script_dir, project_root, os.getcwd()]

    for dll_name in search_names:
        for d in search_dirs:
            full_path = os.path.join(d, dll_name)
            if os.path.exists(full_path):
                try:
                    dll = ctypes.WinDLL(full_path)
                    print(f"[CH347] 加载 DLL: {full_path}")
                    return dll
                except OSError:
                    continue
        # 最后尝试系统 PATH
        try:
            return ctypes.WinDLL(dll_name)
        except OSError:
            continue

    raise FileNotFoundError(
        "未找到 CH347 DLL 文件。\n"
        f"预期路径：{os.path.join(script_dir, 'libs', 'CH347DLLA64.DLL')}\n"
        "请确认 robot/libs/ 目录下存在对应 DLL 文件。"
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

    速度档位（CH347I2C_Set iMode 低3位）：
        SPEED_20K   = 0   20  kHz
        SPEED_STD   = 1   100 kHz（标准，上电默认）
        SPEED_FAST  = 2   400 kHz（快速，PCA9685 最高支持 400kHz）
        SPEED_750K  = 3   750 kHz
        SPEED_50K   = 4   50  kHz
        SPEED_200K  = 5   200 kHz
        SPEED_1M    = 6   1   MHz（仅 CH347F 支持）
    """

    # I2C 速度档位（对应 CH347DLL.H CH347I2C_Set 的 iMode bit[2:0]）
    SPEED_20K    = 0   # 20  kHz
    SPEED_STD    = 1   # 100 kHz（标准，上电默认）
    SPEED_FAST   = 2   # 400 kHz（快速，PCA9685 支持）
    SPEED_750K   = 3   # 750 kHz
    SPEED_50K    = 4   # 50  kHz
    SPEED_200K   = 5   # 200 kHz
    SPEED_1M     = 6   # 1   MHz（仅 CH347F）

    # 向后兼容别名
    SPEED_LOW    = SPEED_20K
    SPEED_HIGH   = SPEED_750K

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

        # 设置 I2C 速度（CH347DLL.H: CH347I2C_Set(iIndex, iMode)）
        # iMode bit[2:0] = I2C速度档位，见 SPEED_* 常量
        self._dll.CH347I2C_Set(c_ulong(self._dev_idx), c_ulong(self._speed))

        self._opened = True
        speed_labels = {
            0: "20kHz", 1: "100kHz", 2: "400kHz", 3: "750kHz",
            4: "50kHz", 5: "200kHz", 6: "1MHz",
        }
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

    def get_chip_type(self) -> str:
        """
        查询芯片类型（CH347DLL.H: CH347GetChipType）
        Returns:
            "CH341" / "CH347" / "CH347F" / "Unknown"
        """
        if not self._opened:
            raise RuntimeError("设备未打开")
        chip_id = self._dll.CH347GetChipType(c_ulong(self._dev_idx))
        return {0: "CH341", 1: "CH347", 2: "CH347F"}.get(chip_id, f"Unknown({chip_id})")

    def get_device_info(self) -> dict:
        """
        获取设备信息（CH347DLL.H: CH347GetDeviceInfor）
        Returns:
            dict with keys: iIndex, DevicePath, UsbClass, FuncType,
                            DeviceID, ChipMode, BulkOutEndpMaxSize,
                            BulkInEndpMaxSize, UsbSpeedType, CH347IfNum,
                            DataUpEndp, DataDnEndp, ProductString, ManufacturerString
        """
        if not self._opened:
            raise RuntimeError("设备未打开")

        # mDeviceInforS 结构体（对应 CH347DLL.H 定义，简化版只取关键字段）
        class _DevInfo(ctypes.Structure):
            _fields_ = [
                ("iIndex",              c_ulong),
                ("DevicePath",          ctypes.c_char * 260),
                ("UsbClass",            ctypes.c_ubyte),
                ("FuncType",            ctypes.c_ubyte),
                ("DeviceID",            ctypes.c_char * 64),
                ("ChipMode",            ctypes.c_ubyte),
                ("BulkOutEndpMaxSize",  ctypes.c_ushort),
                ("BulkInEndpMaxSize",   ctypes.c_ushort),
                ("UsbSpeedType",        ctypes.c_ubyte),
                ("CH347IfNum",          ctypes.c_ubyte),
                ("DataUpEndp",          ctypes.c_ubyte),
                ("DataDnEndp",          ctypes.c_ubyte),
                ("ProductString",       ctypes.c_char * 64),
                ("ManufacturerString",  ctypes.c_char * 64),
            ]

        info = _DevInfo()
        ret  = self._dll.CH347GetDeviceInfor(
            c_ulong(self._dev_idx), ctypes.byref(info)
        )
        if not ret:
            return {}
        return {
            "iIndex":             info.iIndex,
            "DeviceID":           info.DeviceID.decode(errors="ignore"),
            "ChipMode":           info.ChipMode,
            "UsbSpeedType":       {0: "FullSpeed(12Mbps)",
                                   1: "HighSpeed(480Mbps)"}.get(info.UsbSpeedType, "?"),
            "BulkOutEndpMaxSize": info.BulkOutEndpMaxSize,
            "BulkInEndpMaxSize":  info.BulkInEndpMaxSize,
            "ProductString":      info.ProductString.decode(errors="ignore"),
            "ManufacturerString": info.ManufacturerString.decode(errors="ignore"),
        }

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
        chip = i2c.get_chip_type()
        print(f"[CH347] 芯片型号: {chip}")
        info = i2c.get_device_info()
        if info:
            print(f"[CH347] 产品名称: {info['ProductString']}")
            print(f"[CH347] USB速度:  {info['UsbSpeedType']}")
        devices = i2c.scan()
        print(f"共发现 {len(devices)} 个设备: {[hex(a) for a in devices]}")
