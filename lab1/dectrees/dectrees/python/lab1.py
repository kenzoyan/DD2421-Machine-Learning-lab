
import monkdata as mdata
import dtree 
import drawtree_qt5 as draw
#Assignment 1 
# r1=dtree.entropy(mdata.monk1)
# r2=dtree.entropy(mdata.monk2)
# r3=dtree.entropy(mdata.monk3)
# print(f"r1={r1} || r2={r2} || r3={r3}")


#**********************************
# Assignment 3 [0.07527255560831925, 0.005838429962909286, 0.00470756661729721, 0.02631169650768228, 0.28703074971578435, 0.0007578557158638421]
#[0.0037561773775118823, 0.0024584986660830532, 0.0010561477158920196, 0.015664247292643818, 0.01727717693791797, 0.006247622236881467]
#[0.007120868396071844, 0.29373617350838865, 0.0008311140445336207, 0.002891817288654397, 0.25591172461972755, 0.007077026074097326]

# ag1=[]
# ag2=[]
# ag3=[]

# for index in range(6):
#     gain1=dtree.averageGain(mdata.monk1 , mdata.attributes[index])
#     gain2=dtree.averageGain(mdata.monk2 , mdata.attributes[index])
#     gain3=dtree.averageGain(mdata.monk3 , mdata.attributes[index])

#     ag1.append(gain1)
#     ag2.append(gain2)
#     ag3.append(gain3)

# print(ag1) #a5
# print(ag2) #a5
# print(ag3) #a2 a5


#**********************************
# Assignment 5



a=dtree.bestAttribute(mdata.monk1,mdata.attributes)
attributesLeft = [x for x in mdata.attributes if x != a]
#print(a,attributesLeft) #a5

subsets=[]
for v in a.values:
    temp=dtree.select(mdata.monk1, a, v)
    subsets.append(temp)

ag_in2level=[]
subsets_ag=[]
#print(len(a.values))
for subset in subsets:
    for i in range(len(attributesLeft)):
        gain1=dtree.averageGain(subset, attributesLeft[i])
        ag_in2level.append(gain1)   
    subsets_ag.append(ag_in2level)
    ag_in2level=[]
#print(subsets_ag)

def Tree(dataset, attributes, maxdepth=3):

    def Branch(dataset, default, attributes):
        if not dataset:
            return dtree.TreeLeaf(default)
        if dtree.allPositive(dataset):
            return dtree.TreeLeaf(True)
        if dtree.allNegative(dataset):
            return dtree.TreeLeaf(False)
        return Tree(dataset, attributes, maxdepth-1)

    default = dtree.mostCommon(dataset)
    if maxdepth < 1:
        return dtree.TreeLeaf(default)
    a = dtree.bestAttribute(dataset, attributes)
    attributesLeft = [x for x in attributes if x != a]
    branches = [(v, Branch(dtree.select(dataset, a, v), default, attributesLeft))
                for v in a.values]
    return dtree.TreeNode(a, dict(branches), default)

# result=Tree(mdata.monk1,mdata.attributes)
# print(result)
# draw.drawTree(result)



# t=dtree.buildTree(mdata.monk1, mdata.attributes)
# print(f'M1_train={1-dtree.check(t, mdata.monk1)},M1_Test={1-dtree.check(t, mdata.monk1test)}')
# #print(t)

# t=dtree.buildTree(mdata.monk2, mdata.attributes)
# print(f'M2_train={1-dtree.check(t, mdata.monk2)},M2_Test={1-dtree.check(t, mdata.monk2test)}')
# #print(t)
# t=dtree.buildTree(mdata.monk3, mdata.attributes)
# print(f'M3_train={1-dtree.check(t, mdata.monk3)},M3_Test={1-dtree.check(t, mdata.monk3test)}')
#print(t)
#draw.drawTree(t)


# Assignment 6
import random

def partition(data,fraction):
    ldata=list(data)
    random.shuffle(ldata)
    breakpoint=int(len(ldata)* fraction)
    return ldata[:breakpoint], ldata[breakpoint:]



def selectBestTree(tree,best_score,dataset):

    P_trees=dtree.allPruned(tree)
    for subtree in P_trees[1:]:
        new_score=dtree.check(subtree,dataset)
        #print(new_score,subtree)
        if new_score>=best_score:
            #print('into backtracking now score=',(new_score))
            tree,best_score=selectBestTree(subtree,new_score,dataset)
            #return selectBestTree(subtree,new_score,dataset)
                
    #print('out score=',(best_score))
    return  tree,best_score
    
def get_val_score(dataset,fraction,attributes):
    dtrain,dval=partition(dataset,fraction)
    origin_tree=dtree.buildTree(dtrain,attributes,3)
    re,score=selectBestTree(origin_tree,0,dval)
    return origin_tree,re,score
    #draw.drawTree(m1t)
    #draw.drawTree(re)

print('------Validation-----------')
otree1,ptree1,score1=get_val_score(mdata.monk1,0.6,mdata.attributes)
print(f'Original tree={otree1} ||Pruned tree={ptree1},score={score1}')

otree2,ptree2,score2=get_val_score(mdata.monk2,0.6,mdata.attributes)
print(f'Original tree={otree2} ||Pruned tree={ptree2},score={score2}')

otree3,ptree3,score3=get_val_score(mdata.monk3,0.6,mdata.attributes)
print(f'Original tree={otree3} ||Pruned tree={ptree3},score={score3}')

#draw.drawTree(ptree1)






