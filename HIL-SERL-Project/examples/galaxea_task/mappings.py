# 这个文件的作用是一个注册表 / 路由表。

# 它把：

# 命令行里的实验名字符串
# 映射到
# 具体任务的配置类

# 然后不需要知道你现在跑的是 RAM、USB、handover 还是 egg。
# 也就是说，它解决的是“脚本怎么根据实验名找到对应任务配置”。
#它只是把实验名映射到 usb_pick_insertion等任务/config.py 里的训练配置类。

# 实验名
# 映射到具体任务配置类
# from .usb_pick_insertion.config import GalaxeaUSBTrainConfig
from .usb_pick_insertion_single.config import GalaxeaUSBTrainConfig

CONFIG_MAPPING = {
    "galaxea_usb_insertion_single": GalaxeaUSBTrainConfig,
}

