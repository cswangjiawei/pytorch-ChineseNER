# ChineseNER
  这是一个基于BiLSTM-CRF的字级别的中文命名实体识别库，可以从文本中抽取出人名、地名、组织名三大类实体。  
## installation
  pip install ChineseNER
## 使用
```
from ChineseNER import model  
text = '国务院总理李克强在中南海紫光阁会见来华访问的德国总理默克尔'  
ner = model.load()  
dict1 = ner.get_entity_from_sent(text)
```
你将得到字典：{'location': ['中南海', '紫光阁', '华', '德国'], 'orgnization': ['国务院'], 'person': ['李克强', '默克尔']}  
  
也可以从一个文本文件中抽取实体，得到一个实体文件：
```
from ChineseNER import model  
ner = model.load()  
ner.get_entity_from_file('1.txt', '2.txt')  
```
输入文件'1.txt'，将得到文件'2.txt'。1.txt和2.txt如下图所示：  
![1.txt](https://github.com/cswangjiawei/ChineseNER/blob/master/iamge/1.png '1.txt')  
![2.txt](https://github.com/cswangjiawei/ChineseNER/blob/master/iamge/2.png '2.txt')
