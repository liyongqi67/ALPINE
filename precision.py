user_click={}
for i in range(len(ground_truth)):
    user_id=ground_truth[i][0]
    click=ground_truth[i][1]
    probability=result[i]

    if(user_id not in user_click):
        user_click[user_id]=[]
    user_click[user_id].append((click,probability))





for K in range(1,11):
    recall=0
    precision=0
    f1=0
    for key in user_click:
        user_click[key].sort(key=lambda x:-x[1])
        

        click_all=0
        predicate_correct=0
        for i in range(len(user_click[key])):
            if(user_click[key][i][0]==1):
                click_all=click_all+1
        for i in range(min(K*10,len(user_click[key]))):
            if(user_click[key][i][0]==1):
                predicate_correct=predicate_correct+1
        if(predicate_correct==0):
            recall_temp=0
            precision_temp=0
            f1_temp=0
        else:
            recall_temp=float(predicate_correct)/click_all
            precision_temp=float(predicate_correct)/min(K*10,len(user_click[key]))
            f1_temp=2*(recall_temp*precision_temp)/(recall_temp+precision_temp)
        recall=recall+recall_temp
        precision=precision+precision_temp
        f1=f1+f1_temp

    recall=recall/(len(user_click))
    precision=precision/(len(user_click))
    f1=f1/(len(user_click))

    print("K",K*10)
    print('recall',recall)
    print('precision',precision)
    print('f1',f1)
