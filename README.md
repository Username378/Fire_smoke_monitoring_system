# Fire_smoke_monitoring_system
基于yolov8的烟火监测系统  
=====
1.演示视频  
-----

  
2.安装  
-----
（1）安装anaconda  
  1）访问Anaconda官网：https://www.anaconda.com/products/individual  
  2）选择相应的操作系统版本并下载对应的安装包（推荐下载64位版本）  
  3）打开下载的安装包，按照提示进行安装即可  
  4）创建一个虚拟环境：  
    conda create --name 自命名 python  
    
（2）安装需求包  
  激活环境并安装相应的库： activate 自命名-> pip install -r requirements.txt  
  这一步会安装cpu版本的torch与torchvision，建议安装cuda版本，  
  安装cuda版本很简单，首先要有英伟达显卡，其次nvdia-smi查看cuda driver驱动版本号，  
  上英伟达官网选择对应cuda版本号的cuda套件安装，最后去torch官网选择自己安装的cuda套件版本使用conda或者pip安装即可。  
  
3.运行  
-----
配置好环境后在含有main.py的工作目录下运行main.py即可  
