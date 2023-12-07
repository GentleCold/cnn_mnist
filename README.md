## 项目文件

```shell
.
├── analyse.ipynb       // 实验分析文件
├── data
│   ├── data_handler.py // 数据处理文件
│   ├── MNIST           // MNIST数据集
├── imgs                // 文档图片
├── main.py             // 主运行文件
├── model               // 模型文件
│   ├── alexnet.py
│   ├── googlenet.py
│   ├── lenet.py
│   ├── mobilenet.py
│   └── resnet.py
├── README.md           // 项目用法介绍
├── REPORT.md           // 项目实验报告
├── REPORT.pdf          // 项目实验报告
├── requirements.txt    // 项目依赖
└── train.py            // 模型训练文件
```

- 各模型训练结果见analyse.ipynb

- 实验报告见REPORT.md/REPORT.pdf

## 结果复现

通过requirements.txt安装对应包后，再运行如下命令

```shell
python main.py -m lenet

python main.py -m alexnet

python main.py -m googlenet -d 0.2

python main.py -m resnet

python main.py -m mobilenet
```

可得到analyse.ipynb中的结果

其余各参数说明如下，其中

- `-m`参数的可选值为: `lenet、alexnet、alexnetlarge、googlenet、resnet、mobilenet`

- `-o`参数的可选值为: `adam、adamw`

```shell
usage: main.py [-h] [-b B] [-e E] [-l L] [-d D] [-m M] [-o O]

options:
  -h, --help  show this help message and exit
  -b B        batch size, defaul=128
  -e E        number of epochs, default=5
  -l L        learning rate, default=0.001
  -d D        dropout rate, default=0.5
  -m M        model type, default=lenet
  -o O        optimizer type, default=adamw
```
