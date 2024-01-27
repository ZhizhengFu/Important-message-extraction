import re

def process_notifications(file_path):
    notifications = []
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    notifications_raw = re.findall(r'(\d{4}-\d{2}-\d{2} \d{1,2}:\d{2}:\d{2}) (.*?)\n(.*?)\n', content, re.DOTALL)
    for timestamp, author, message in notifications_raw:
        notification = {}
        notification['timestamp'] = timestamp.strip()
        notification['author'] = author.strip()
        notification['message'] = message.strip()
        notification['message'] = re.sub(r'[^\w\s@:/?=&.-]+(?![^<>]*>)', ' ',notification['message'])
        notification['message'] = notification['message'].replace('\n',' ')
        notification['message'] = re.sub(r'\s+', ' ', notification['message'])
        notification['message'] = notification['message'].strip()
        notifications.append(notification)
    return notifications
import os
root = "/home/charles/code/data"
for dirpath, dirnames, filenames in os.walk(root):
    for filepath in filenames:
        si = dirpath+'/'+filepath
        sj = dirpath+filepath
        # print(si,sj)
        notifications = process_notifications(si)
        with open(sj,'w') as file:
            for notification in notifications:
                if notification['message']=='' or notification['message']=='图片':
                    continue
        # file.write('时间:'+notification['timestamp']+'\n')
        # file.write('作者:'+notification['author']+'\n')
                file.write(notification['message']+'\t'+'0'+'\n')
        # file.write('---'+'\n')
        file.close()
        
