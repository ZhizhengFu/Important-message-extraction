import re
import json
import random
list = []
with open('/home/charles/code/content_all.txt', "r", encoding="utf-8") as file:
    chat_record = file.readlines()
    for line in chat_record:
        list.append({'text': line, 'label': 1})
f = open('/home/charles/code/newo.txt','r',encoding='utf-8')
lines = f.readlines()
for line in lines:
    s = re.findall(r'(.*?)\t(\d)\n',line)
    for s1,s2 in s:
        if s2 == '1':
            list.append({'text': s1, 'label': 1})
        else:
            list.append({'text': s1, 'label': 0})

tlist = []
Rand = random.sample(range(0,4131),827)
Rand.sort(reverse=True)
for i in Rand:
    tlist.append(list[i])
    del list[i]
# print(len(list)) 4132 827
b = json.dumps(list)
c = json.dumps(tlist)
f = open('/home/charles/code/my_json.json','w')
f.write(b)
f.close()
f = open('/home/charles/code/my_test.json','w')
f.write(c)
f.close()
