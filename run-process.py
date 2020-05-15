#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- version: v0.1 -*-
# 脚本用途：重启jar，需要传入jar名, 如nziot_api
# 注意事项：
#   jar名不要传入版本号，也不要加入通配符*。
#
# 运行命令： python restart-jar.py nziot_api

import os
import subprocess
import sys


def run_cmd(cmd):
    """
    运行单条命令
    :return:
    """
    print("cmd: " + cmd)
    return os.popen(cmd).readlines()


if __name__ == "__main__":
    if len(sys.argv) <= 2:
        print('请输入程序名，以及输出文件!')
    run_file = sys.argv[1]
    out_file = sys.argv[2]
    start_cmd = "nohup python -u %s > runconfig/%s 2>&1 &" % (run_file, out_file)
    print("启动命令：" + start_cmd)
    try:
        res = subprocess.check_call([start_cmd], shell=True)
    except BaseException:
        print("启动process失败")
    else:
        print("启动结果：" + str(res))
