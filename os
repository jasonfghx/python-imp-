os.mkdir("C:/Users/123/Desktop/test")#產生資料夾 file name is "test"
os.rmkdir("C:/Users/123/Desktop/test")#delete file

os.listdir("C:/Users/123/Desktop/")

#檔案管理系統
file=os.listdir("C:/Users/123/Desktop")
    i=0
    for d in file:
        print(i,d)
        i=i+1
    
    a=input("enter")    
    os.remove(file[int(a)])    
    

import os

def fileList():
	print("=====================")
	fileList=os.listdir(os.getcwd())
	i=0
	for d in fileList:
		print("("+str(i)+")"+d)
		i+=1
	print("=====================")

os.system("cls")
while True:
	print("(1) 列出當前資料夾內的檔案")
	print("(2) 刪除檔案")
	print("(3) 建立資料夾")
	command=input("你要做甚麼：")
	if command=="1":
		os.system("cls")
		fileList()
	elif command=="2":
		fileList()
		fileindex=input("請選擇您要刪除的檔案：")
		fn=fileList[int(fileindex)]
		if os.path.isdir(fn):
			os.rmdir(fn)
		else:
			os.remove(fn)
		os.system("cls")
	elif command=="3":
		fileiName=input("請輸入你要建立的資料夾名稱：")
		os.mkdir(fileiName)
		os.system("cls")
