Fire_smoke_monitoring_system
基于yolov8的烟火监测系统  
=====
作者：孔德睿
=====
1.系统介绍  
-----
>（1）基于pyside6的ui设计  
>>系统包括本地图片/视频检测模块、屏幕实时检测模块、本地摄像头检测模块、检测结果查看模块  、参数调节模块  
>>![UI设计](https://github.com/Username378/Fire_smoke_monitoring_system/assets/112858821/f4c79763-2887-430b-b427-2447f9771261)  
>（2）本地图片/视频检测模块  
>>用户可以选择从本地导入图像或视频文件，也可以实时抓取本地摄像头的图像或视频数据。系统能够自动识别图像或视频的格式、分辨率、帧率等参数，并进行适当的预处理，用户可以在系统内部同时查看原文件以及检测结果。  
>>![本地文件检测](https://github.com/Username378/Fire_smoke_monitoring_system/assets/112858821/e52841c4-8c0a-454a-8357-06d39c910334)  
>（3）屏幕实时检测模块  
>>本系统可以对指定屏幕的图像或视频进行实时烟火检测，并显示当前的检测帧率、当前时间、目标数量、警报信息等。如果检测到烟雾火焰，系统会在图像或视频上用红色框标出其位置，并显示其置信度。  
>>![实时监测图片](https://github.com/Username378/Fire_smoke_monitoring_system/assets/112858821/de77048b-0bae-43f3-a1e7-c680d49483cc)  
>（4）检测结果查看模块  
>>当检测到烟雾火焰时，除了在图形界面上显示结果外，还会自动记录报警信息，包括报警时间、日期、现场图片等，用户可自行选择是否保存图像或文本检测结果。  
>>查看检测结果  
>>![图片信息记录](https://github.com/Username378/Fire_smoke_monitoring_system/assets/112858821/df80307a-d633-46ea-876d-5850a942857e)  
>>查看信息记录  
>>![文本信息记录](https://github.com/Username378/Fire_smoke_monitoring_system/assets/112858821/8493e0cf-81b7-439b-a030-0d39d7bb2982)  
>（5）参数调节模块  
>>支持用户更换不同的烟火检测模型，以适应不同的场景和需求。用户可以从提供的模型库中选择合适的模型，也可以从网络上下载或自己训练新的模型，并导入到系统中。此外，用户还可以根据实际情况，调节检测模型的参数，如交并比、置信度等。  
>>![参数调节](https://github.com/Username378/Fire_smoke_monitoring_system/assets/112858821/d8749954-e11e-4e39-b360-adf657f594b6)  

2.安装  
-----
>（1）安装anaconda  
>>1）访问Anaconda官网：https://www.anaconda.com/products/individual  
>>2）选择相应的操作系统版本并下载对应的安装包（推荐下载64位版本）  
>>3）打开下载的安装包，按照提示进行安装即可  
>>4）创建一个虚拟环境：  
    conda create --name 自命名 python  
    
>（2）安装需求包  
>>激活环境并安装相应的库： activate 自命名-> pip install -r requirements.txt。这一步会安装cpu版本的torch与torchvision，建议安装cuda版本，安装cuda版本很简单，首先要有英伟达显卡，其次nvdia-smi查看cuda driver驱动版本号，上英伟达官网选择对应cuda版本号的cuda套件安装，最后去torch官网选择自己安装的cuda套件版本使用conda或者pip安装即可。  
  
3.运行  
-----
配置好环境后在含有main.py的工作目录下运行main.py即可  
