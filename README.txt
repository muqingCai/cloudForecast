1、直接跑asl.py即可，可以设置对应的参数
2、run-process.py 和 stop-process.py是运行/停止的脚本
3、data目录存放数据，models存放模型类，runs存放跑出来的模型，utils存放工具函数
4、数据是阿里巴巴的数据集，sortM1_5000是抽取了machine_usage表里面全部M1机器的数据，
并且按照时间戳排列好，取了其中5000个数据来跑
