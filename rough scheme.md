# 楚辞生成

## 初步计划
### 输入
~~根据不同输入方式，为了能应用上数据集，我们应采用不同的输出过程。~~
~~1. **输入关键词**。如此我们则要先根据关键词生成现代文本，再把文本翻译成楚辞~~
2. **输入开头几个字**。这样的话我们可以直接生成楚辞。

### 输出
逐句输出一段楚辞，句数random

### 基本思路
使用transformers预训练的**tokenizer**，或者手动构建词表也行~~（若选择第一种输入，我们也可以用别人的文本生成模型）~~
然后采用**seq2seq**的**attention**模型，先使用古诗古文进行预训练。再采用楚辞进行fine-tuning。

### 问题与初步应对
1. **韵律与平仄：** 或许能自己学会,训练时应该生成到句号而不是逗号为止。
2. **字数：** 训练时的应对见下，生成时或可在567间随机截断
3. **楚地方言与生僻字**：在训练时将生僻字先替换为同义乃至同长的翻译，在生成时将翻译再换回生僻字
4. **训练集少：** 先用古诗古文预训练

## 具体分析
数据标注、存储、映射 
seq2seq模型，attention打分、合成 
预训练、训练、评估函数
生成、注释 
### 数据预处理
0. 把楚辞的字变成简体字
1. 生僻字定义：在古诗和楚辞混合集中出现次数少于等于三次的字
2. 生僻字处理方式：若该生僻字不在楚辞中，则将其映射成生僻字符的独特id；若该生僻字位于楚辞中，则**手动将其替换为一两个字长的常见字**
3. dataloader的要求：
   古诗集：每个batch中的每首诗的句子数要相同，且数字化后末尾（保留标点？）加sep符，并pad字符padding至max_len
   楚辞集：去除特别不规则的句子，替换完生僻字后同古诗，每4/6句分割？

### 模型训练
以batch为单位
与藏头诗区别：
1. 一个batch中生成到终止符的字数不同———while循环，当所有句子都生成完则标志完成，再手动将每句结束符后padding成规定字数
2. Attention忽略padding：手动修改打分矩阵？
3. 损失函数接受不同长度：将target集padding，交叉熵采用ignore_index以忽略padding

##### remark 加padding产生影响的地方（暂时已知的，欢迎补充）
1. `dataloader` 中需要加padding至max_len ,希望一个batch中的数据是根据原句长降序排序的
2. `embedding`过程中，直接加`padding_idx = PAD`即可
3. 计算损失函数loss时,应忽略padding
4. Attention机制中的source。给padding的位置全部加上$-\infty$($e^{-\infty} = 0$)

### 输出
同藏头诗？外加随机指定句数。

## 汇报模块
### 为什么选楚辞
一页
1. 《楚辞》是最早的浪漫主义诗歌总集及浪漫主义文学源头
2. 当下人们已经有许多古诗词生成的尝试，但是楚辞生成仍然冷门。这也给我们探索这一领域的兴趣。
### 楚辞带来的困难与应对
1. 数据集少 两页
   存世65篇，只有两千余行，与诗词库相比十分稀少（图）
   在训练集中加入不同长度的诗对模型进行预训练，使其掌握基本“语感”。
   接着再用楚辞对模型进行fine-tuning
   同时，由于流传的楚辞风格皆以屈原为宗，对于楚辞数据集的过拟合并非不可接受。
2. 句式句长丰富 一页 
   楚辞较为灵活，因而不能限定其句长与句数
   应对方案：将不同长度的诗混合投入训练。采用seq2seq模型进行不定长输出。
3. 生僻字众多（文质兼美的困难）两页
   
   出现少，难学会（数据）
   文质兼美解释：文字古雅，意象美

### 数据预处理
统计生僻字，爬虫，取字，构建字典 一页 图
替换、整理格式、tokenize 一页 图
### 模型
attention mask seq2seq  一两页 图
### 交互效果
两页 界面图，模块标注
### 分工与收获
一页
