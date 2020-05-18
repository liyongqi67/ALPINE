def computeEdge(history):
    history=visual[history] #T H

    history2=np.expand_dims(history,1) #T 1 H

    a=history2-history
    a=np.sum(np.square(a),2)



    for i in range(len(a)):
        a[i][i]=sys.maxsize
    b=[]
    for i in range(len(a)):
        for j in range(i-1):
            b.append(1)
        for j in range(min(len(a)-i+1,len(a))):
            b.append(sys.maxsize)
    b=np.array(b).reshape([-1,len(a)])
    b[0][0]=1
    b[1][0]=1


    a=a*b

    a=np.argmin(a, axis=1)

    return a.tolist()