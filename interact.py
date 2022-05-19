from tkinter import *
from PIL import Image,ImageTk
from tkinter import scrolledtext     
sent_num = 20

def generate(inputs):
    # inputs: batch_size*num_sents*max_len    
    #outputs[:,sent_id,i,:] batch_size*num_sents*maxlen*voc_size
    import torch
    import config
    from webbrowser import get
    from model import S2SModel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = f'checkpoints/{args.dataset}_{args.model}_best_model.pt'
    ckpt = torch.load(model_path)
    vocab = ckpt['vocab']
    inversed_vocab = ckpt['inversed_vcaob']
    # 建立模型
    input_size = 300
    hidden_size = 512
    n_layers = 2
    model = S2SModel(
        voc_size=len(vocab) + 1,
        input_size=input_size,
        hidden_size=hidden_size,
        n_layers=n_layers,
    )
    # 加载保存的参数到模型当中
    model.load_state_dict(ckpt['model'])
    model = model.to(device) 
    outputs,hidden = model(inputs, teacher_force_ratio=0)
    sents=[]
    sent =''
    for i in range(sent_num):
        for j in range(0,config.max_len): #从第二个字到标点
            possiblity = outputs[0,i,j,:]            
            value,index = torch.topk(possiblity,5)
            word = inversed_vocab[index[0].item()]
            sent = sent + word
        sents.append(sent)
        sent = ''
    return sents
            


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
    vocab = {}
    inputs = torch.zeros(1,sent_num,config.max_len)
    for i,word in enumerate(data.get()):
        inputs[0,0,i] = vocab[word]
    for i in range(len(data.get()),config.max_len):
        inputs[0,0,i] = config.Pad 
    sents = generate(inputs)      #outputs[:,sent_id,i,:] batch_size*num_sents*maxlen*voc_size
    # sents = ["帝高阳之苗裔兮","朕皇考曰伯约","摄提贞于孟陬兮","唯庚寅吾以降"]
    # text.delete('1.0','end')
    for sent in sents:
        text.insert(INSERT,sent)        
        text.insert(INSERT,'\n')  
    text.update()
    
button=Button(root,text='开始生成',font=("华文行楷",15),command=callback)
text =scrolledtext.ScrolledText(root,width=25,height=20,font=("华文行楷",20))#,bg='azure') 
can.create_window(200, 30, width = 150, height=40,window=label)
can.create_window(200, 70, width = 250, height=35,window=entry)
can.create_window(200, 115, width = 100, height=40,window=button)
can.create_window(200, 415,width = 300, height=550,window=text)

root.mainloop()



