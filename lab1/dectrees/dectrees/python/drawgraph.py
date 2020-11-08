
import monkdata as mdata
import dtree 
import lab1 as l 
import matplotlib.pyplot as plt
import numpy as np

x=[0.3,0.4,0.5,0.6,0.7,0.8]
avgscores=[]
std=[]
#monk1
for i in x:
    temp=[]
    for _ in range(10):
        _,ptree,_=l.get_val_score(mdata.monk1,i,mdata.attributes)
        score1=dtree.check(ptree, mdata.monk1test)
        temp.append(score1)
    avg1=np.mean(temp)
    std1=np.std(temp)
    avgscores.append(avg1)
    std.append(std1)

avgscoresmonk3=[]
stdmonk3=[]
#monk3
for i in x:
    temp=[]
    for _ in range(10):
        _,ptree,_=l.get_val_score(mdata.monk3,i,mdata.attributes)
        score3=dtree.check(ptree, mdata.monk3test)
        temp.append(score3)
    avg3=np.mean(temp)
    std3=np.std(temp)
    avgscoresmonk3.append(avg3)
    stdmonk3.append(std3)

plt.xlabel('fraction')
plt.ylabel('testscore')
plt.title('Testscore of MONK under different fraction')
l1=plt.errorbar(x,avgscores,yerr=std,fmt='x',ecolor='r',color='b',elinewidth=2,capsize=4)
l2=plt.errorbar(x,avgscoresmonk3,yerr=stdmonk3,fmt='o',ecolor='g',color='y',elinewidth=2,capsize=4)
#fmt :   'o' ',' '.' 'x' '+' 'v' '^' '<' '>' 's' 'd' 'p'
plt.legend(handles=[l1,l2],labels=['MONK1','MONK3'],loc='best')
plt.show()