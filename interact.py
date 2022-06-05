from tkinter import *
from PIL import Image,ImageTk
from tkinter import scrolledtext    
from following_process import * 
from dataloader import Vocab
sent_num = 10

root=Tk()
root.geometry("400x700+600+30")#宽乘以高加上水平偏移量加上垂直偏移量
root.title("233")
def get_image(image_name , width , height):
    im = Image.open(image_name).resize((width , height))
    return ImageTk.PhotoImage(im)
can = Canvas(root , width = 400 , height = 800 , bg = "white")
im = get_image('chuci.gif' , 400 , 800)
can.create_image(200,300,image = im)
can.pack()
label=Label(root,text="楚辞生成器",font=("华文行楷",20),fg="green",bg=None)
label.pack()#调用pack方法将label标签显示在主界面
data = StringVar()
entry =Entry(root ,font=("华文行楷",15),textvariable=data)#创建labal组件并将其与data关联

def callback():
    import torch
    import config
    inputs = torch.tensor([[Vocab.Pad]*config.max_len]*sent_num).unsqueeze(0)
    for i,word in enumerate(data.get()):
        inputs[0,0,i] = Vocab.vocab[word] if word in Vocab.vocab else 0
    inputs[0,0,len(data.get())] = Vocab.vocab['，']
    sents,annotations = generate(data.get(),inputs)      #outputs[:,sent_id,i,:] batch_size*num_sents*maxlen*voc_size
    # sents = ["帝高阳之苗裔兮","朕皇考曰伯约","摄提贞于孟陬兮","唯庚寅吾以降"]
    text.delete('1.0','end')   
    for sent in sents:
        text.insert(INSERT,sent)        
        text.insert(INSERT,'\n') 
    text.insert(INSERT,'\n')
    for k,v in annotations.items():
        text.insert(INSERT,k)
        text.insert(INSERT,'：')
        text.insert(INSERT,v)
        text.insert(INSERT,'\n') 
    text.update()
    
button=Button(root,text='开始生成',font=("华文行楷",15),command=callback)
text =scrolledtext.ScrolledText(root,width=25,height=20,font=("",16))#,bg='azure') 
can.create_window(200, 30, width = 150, height=40,window=label)
can.create_window(200, 70, width = 250, height=35,window=entry)
can.create_window(200, 115, width = 100, height=40,window=button)
can.create_window(200, 415,width = 300, height=550,window=text)

root.mainloop()



