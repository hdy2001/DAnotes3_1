import os

# path表示路径
path = "./新冠辅助诊断数据集/NonCOVID"
# 返回path下所有文件构成的一个list列表
filelist = os.listdir(path)
# 遍历输出每一个文件的名字和类型
file = open('./新冠辅助诊断数据集/train.txt', 'w+')
for item in filelist:
    # 输出指定后缀类型的文件
    # if(item.endswith('.jpg')):
    file.write('./新冠辅助诊断数据集/NonCOVID/' + item + ' 0' + '\n')
file.close()
