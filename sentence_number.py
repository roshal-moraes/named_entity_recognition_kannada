word = []
pos=[]
num=[]
sentence_number = 1
with open('viterbi_output.txt', mode='r',encoding='utf-8') as filey:
    for line in filey:
        l = line.split('\t')
        word.append(l[0])
        pos.append(l[1])
        num.append(sentence_number)
        if(l[0]=='.'):
            sentence_number+=1
f = open('processed_input.txt', mode='w',encoding='utf-8')
f.write('Sentence#\tWord\tPOS\n')
for i in range(len(word)):
    f.write(str(num[i])+"\t"+str(word[i])+"\t"+str(pos[i]))
f.close()        
