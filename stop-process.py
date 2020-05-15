#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 脚本用途：关闭进程，需要传入进程名
#
# 运行命令： python stop-process.py nziot_api

import os
import sys

def run_cmd(cmd):
    """
    运行单条命令
    :return:
    """
    print("cmd: " + cmd)
    return os.popen(cmd).readlines()

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print('请输入程序名!')
    else:
        """
        sys.argv[1]: proc_name
        """
        proc_name = sys.argv[1]
        print("kill进程：" + proc_name) 
        grep_pid_cmd = "ps -ef | grep "  + proc_name + " | grep -vE '(grep|stop-process)' | awk '{print $2}'"
        pids = run_cmd(grep_pid_cmd)

        if pids:
            for pid in pids:
                print("正在kill进程，进程id：" + pid)
                kill_pid_cmd = "kill " + pid
                run_cmd(kill_pid_cmd)
        else:
            print("没有进程在运行。")
