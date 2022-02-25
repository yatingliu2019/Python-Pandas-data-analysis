# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 13:20:27 2022

@author: Dell
"""
import pandas as pd
import numpy as np

courses = ["语文","数学","英语"]#list
data = pd.Series(data=courses)
print(data)#list->series
print("----------------------")


grades = {"语文":80,"数学":90,"英语":85}#dict
data = pd.Series(data=grades)
print(data)#dict->series
print("----------------------")


numbers = data.tolist()
print(numbers)#series->list
print("----------------------")


df = pd.DataFrame(data,columns=['grade'])
print(df)#series->dataframe
print("----------------------")


s = pd.Series(
    np.arange(10,100,10),#数值：10~90，间隔10
    index=np.arange(101,110),#索引：101~109，间隔1
    dtype='float'#类型：float64
    )
print(s)#numpy创建series
print("----------------------")


s = pd.Series(
    data  = ["001","002","003","004"],
    index = list("abcd"))
s = s.astype(int)
s = s.map(int)
print(s)#string series -> int series 
print("----------------------")


data = data.append(pd.Series({
    "物理":80,
    "化学":89}))
print(data)#add series
print("----------------------")


df = data.reset_index()
df.columns = ["course","grade"]
print(df)#series -> dataframe
print("----------------------")


df = pd.DataFrame({
    "姓名":["小张","小王","小李","小赵"],
    "性别":["男","女","男","女"],
    "年龄":[18,19,20,21]})
print(df)#创建 dataframe 
print("----------------------")


df.set_index("姓名",inplace=True)
print(df)#改变 df 的索引列
print("----------------------")


date_range = pd.date_range(start='2022-01-01',end='2022-01-31')
date_range = pd.date_range(start='2022-01-01',periods=31)
print(date_range)#生成一个月份所有天
print("----------------------")


date_range = pd.date_range(start='2022-01-01',end='2022-12-31',freq='W-MON')
date_range = pd.date_range(start='2022-01-01',periods=52,freq='W-MON')
print(date_range)#生成一年所有周一
print("----------------------")


date_range = pd.date_range(start='2022-01-01',periods=24,freq='H')
date_range = pd.date_range(start='2022-01-01',end='2022-01-02',freq='H',closed='left')
print(date_range)#生成一天所有小时
print("----------------------")


date_range = pd.date_range(start='2022-01-01',periods=31)
df = pd.DataFrame(data=date_range,columns=['day'])
df['dayofyear'] = df['day'].dt.dayofyear
print(df)#生成日期
print("----------------------")


date_range = pd.date_range(start='2022-01-01',periods=1000)
data = {"norm":np.random.normal(loc=0,scale=1,size=1000),
        "uniform":np.random.uniform(low=0,high=1,size=1000),
        "binomial":np.random.binomial(n=1,p=0.2,size=1000)}
df = pd.DataFrame(data=data,index=date_range)
print(df)#生成日期和随机分布的 dataframe
print("----------------------")


print(df.head(10))#打印 dataframe 前 10行
print()#打印空行
print(df.tail(5))#打印 dataframe 后 5行
print("----------------------")


print(df.info())#打印 dataframe 的 基本信息
print()
print(df.describe())#打印 dataframe 的 统计信息
print("----------------------")


print(df["binomial"].value_counts())#统计 series 数据列 出现的 次数
print("----------------------")


date_range = pd.DataFrame(
    data = {"norm":np.random.normal(loc=0,scale=1,size=1000),
            "uniform":np.random.uniform(low=0,high=1,size=1000),
            "binomial":np.random.binomial(n=1,p=0.2,size=1000)},
            index = pd.date_range(start='2022-01-01',periods=1000))
df.head(100).to_csv('./分布数据前100.csv')#df前N行存入csv文件
print("----------------------")


df = pd.read_csv("分布数据前100.csv",index_col=0)#恢复索引列
print(df.info())#打印 dataframe 的 基本信息
print(df.head())#打印 dataframe 的 head
print("----------------------")


'''
股票 实例一
import pandas as pd

df = pd.read_csv("./00700.HK.csv",index_col=0)
df = pd.read_csv("./00700.HK.csv")#加载 csv 到 df

print(df.info())#df 基本信息
print()
print(df.describe())#df 描述性信息

df.reset_index(inplace=True)#将 df 的索引列改为普通数据列

df["Date"] = pd.to_datetime(df["Date"])#给 df 数据添加月份和年份
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month


print(df.groupby("Year")["Close"].mean())#计算 平均 收盘价

print(df["Close"].min())
print(df["Close"].argmin())
print(df.loc[[df["Close"].argmin()]])#找出收盘价 最低的 数据行


df_new = df[["Date","Open","Close","Volume"]]#刷选出 部分 数据列


df.set_index("Date",inplace=True)#设置日期为 索引列


df = pd.read_csv("./00700.HK.csv",index_col=0)
df.drop(labels=['High','Low'], axis=1,inplace=True)#删除 不需要的 数据列


df = pd.read_csv("./00700.HK.csv")#对 数据列 重命名
df.columns = ["D","O","H","L","C","V"]#法一
df.rename(columns={"Date":"D","Open":"O","High":"H","Low":"L","Close":"C","Volume":"V"},inplace=True)#法二

'''
'''
电信用户
import pandas as pd

df=pd.read_csv("Telco-Customer-Churn.csv")

print(df.isnull().sum())#判断是否有空值


#print(df["TotalCharges"].value_counts())#每个数值出现多少次

#中位数填充空值
median = df["TotalCharges"][df["TotalChrges"] != " "].median()
df.loc[df["TotalCharges"] == " ","TotalCharges"] = median
df["TotalCharges"] = df["TotalCharges"].astype(float)


#将分类列转换成Categorical类型
number_columns = ["tenure","MonthlyCharges","TotalCharges"]
for column in number_columns:
    df[column] = df[column].astype(float)
for column in set(df.columns) - set(number_columns):
    df[column] = pd.Categorical(df[column])


print(df.describe(include=["category"]))#对 cat 类型字段数据统计

print(df["Churn"].value_counts())#churn 字段的数据分布

print(df.groupby(["Churn","PaymentMethod"])["MonthlyCharges"].mean())#建立一个统计表

#数字映射 ：yes->1 no->0
df["Churn"] = df["Churn"].map({"Yes":1,"No":0}) 
print(df["Churn"].value_counts())

print(df.corr())#查看字段相关性矩阵

df.sample(10).to_csv("sample10.csv")#采样数据 并保存到 csv
'''





#合并两个 series 生成一个 df
np.random.seed(66)
s1 = pd.Series(np.random.rand(20))
s2 = pd.Series(np.random.randn(20))
df = pd.concat([s1,s2],axis=1)
df.columns = ["col1","col2"]
print(df)
print("----------------------")

#筛选 df 的数值 
df1=df[(df["col2"] >= 0) & (df["col2"] <= 1)]
print(df1)
print("----------------------")

#给 df 增加一个新的数据列
df["col3"] = df["col2"].map(lambda x: 1 if x>=0 else -1)
print(df)
print("----------------------")

#截断 df 的数值数据列
df["col4"] = df["col2"].clip(-1.0,1.0)
print(df)
print("----------------------")

#计算 df 的 最大和最小 的数字
print(df["col2"].nlargest(5))
print()
print(df["col2"].nsmallest(5))
print("----------------------")

#df 的数字 累计加和值
print(df.cumsum())
print("----------------------")

#计算 df 的 中位数
print(df["col2"].median())#法一
print()
print(df["col2"].quantile())#法二
print("----------------------")

#按条件筛选 df 过滤后的结果
print(df[df["col2"]>0])
print()
print(df.query("col2 > 0"))
print("----------------------")

#df 变成 dict
print(df.head(5).to_dict())
print("----------------------")

#df 变成 html
print(df.head(5).to_html())
print("----------------------")

#按列名筛选 df 使用 .loc方法进行
np.random.seed(66)
df = pd.DataFrame(np.random.rand(10,4),columns=list("ABCD"))
print(df)
print(df.loc[df['C']>0.8])
print("----------------------")

#使用两列进行组合筛选
print(df[(df['C']>0.3) & (df['D']<0.7)])
print("----------------------")

#用 for 遍历 df
for index,row in df.iterrows():
    print(row)
print("----------------------")

#设置 df 精确位置的值
df.iloc[3,1] = np.nan
df.loc[8,"D"] = np.nan
print(df)
print("----------------------")

#移除包含空行的值
df2 = df.dropna()
print(df2)
print("----------------------")

#移除空行的 df 重新设置索引
df2 = df2.reset_index(drop=True)
print(df2)
print("----------------------")

#统计 df 的缺失值
print(df.isnull().sum())
print("----------------------")

#使用数字填充 df 的空值
df = df.fillna(0)
print(df)
print("----------------------")

#修改 df 列 的前后顺序
df=df[["D","A","B","C"]]
print(df)
print("----------------------")

#删除 df 指定的数据列
print(df.drop(["C","D"],axis=1))
print(df)
print("----------------------")

'''
二手车 实例三
#数据加载与介绍
import pandas as pd
df = pd.read_csv("used_cars.csv",index_col=0)
print(head(5))

#输出 df 的列名称列表
print(list(df.columns))
'''

'''
两只股票 实例四
#读取股票数据 csv
import pandas as pd
bidu = pd.read_csv("./stock.BIDU.csv",index_col=0)
iq = pd.read_csv("./stock.IQ.csv",index_col=0)

#给列名加前缀 列名小写
bidu.columns = ["bidu"+col.lower() for col in bidu.columns]
iq.columns = ["iq"+col.lower() for col in iq.columns]

#将两个 df ，按照日期进行 concat
df = pd.concat([bidu,iq],axis=1)


#提取开盘价和收盘价
quotations = result[["bidu_open","bidu_close","iq_open","iq_close"]].copy()


#计算每只股票的当日涨跌幅 %
quotations["bidu_change"] = (quotations["bidu_close"] / quotations["bidu_open"]-1)*100
quotations["iq_change"] = (quotations["iq_close"] / quotations["iq_open"]-1)*100


#计算同涨同跌信号
quotations["similarity_change"] = (quotations["bidu_change"] * quotations["iq_change"] > 0 )*1


#计算同时涨跌的比例
result= quotations["similarity_flag"].sum() / len(quotations)*100

'''












































