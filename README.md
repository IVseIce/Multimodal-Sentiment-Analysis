# 多模态情感分析

当代人工智能课程实验。

## Setup

This implemetation is based on Python3. To run the code, you need the following dependencies:

- torch==2.1.2+cu118

- numpy==1.21.5

- pandas==1.4.2

- Pillow==9.0.1

- scikit_learn==1.0.2

- tqdm==4.64.0

- transformers==4.30.2


You can simply run 

```python
pip install -r requirements.txt
```

## Repository structure
We select some important files for detailed description.

```python
|-- data/ # 训练数据
|-- pretrained_models # 下载到本地的预训练模型
    |-- bert-base-uncased/ # bert-base-uncased
    |-- resnet-50/  # resnet-50
|-- my_model # 训练之后保存的pth文件
    |-- text&image_5epochs_best # 多模态最佳效果
    |-- image_only_model_15epochs.pth # 仅图像
    |-- text_only_model_15epochs.pth # 仅文本
    |-- text&image_model_15epochs.pth # 多模态
|-- main.py # 整体代码
|-- requirements.txt # 依赖
|-- README.md # readme文档
|-- train.txt # 训练数据
|-- test_without_label.txt # 测试数据
|-- test_with_label.txt # 预测结果

```

## Run 
1. 使用下面的命令来运行代码实现多模态模型的训练与预测，训练完成后生成的pth文件默认名为`my_model.pth`，预测文件保存为`test_with_label.txt`。
```python
python main.py
```

2. 对于仅图片或仅文本的消融实验，使用以下参数：
```python
python main.py --image_only
```
```python
python main.py --text_only
```
3. 你也可以自行调整以下参数：
```python
python main.py --epochs 5 --learning_rate 5e-6 --batch_size 64 --weight_decay 0.01
```




## Attribution

Parts of this code are based on the following repositories:

- [guitld/Transfer-Learning-with-Joint-Fine-Tuning-for-Multimodal-Sentiment-Analysis](https://github.com/guitld/Transfer-Learning-with-Joint-Fine-Tuning-for-Multimodal-Sentiment-Analysis)

- [YeexiaoZheng/Multimodal-Sentiment-Analysis](https://github.com/YeexiaoZheng/Multimodal-Sentiment-Analysis)

- [Linshou99/Multimodal-sentiment-analysis](https://github.com/Linshou99/Multimodal-sentiment-analysis)


