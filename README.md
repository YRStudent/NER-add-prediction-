1. 在models里面加入了一个predict_tags；然后写了一个predict文件，输入一句话可以预测句子中的实体信息。
2. train函数里面为了数据训练的均衡性 我加了一个stratified k-fold的写法 但是注释起来了 有需要的可以解开注释 并验证在k-fold下的数据效果是否比之前较好，k值一般选取5或者10 可根据具体需要进行选择数值。
3. 这里就没有放数据处理的过程了，我用的是BIO格式的数据；具体数据及处理过程可以看下面链接。
4. 原博主链接：https://github.com/XavierWww/Chinese-Medical-Entity-Recognition
