from email.mime import image
from tkinter import *
from PIL import Image,ImageTk
from tkinter import scrolledtext        # 导入滚动文本框的模块
root=Tk()
root.geometry("400x700+600+30")#对应的格式为宽乘以高加上水平偏移量加上垂直偏移量

# def get_image(image_name , width , height):
#     im = Image.open(image_name).resize((width , height))
#     return ImageTk.PhotoImage(im)
# can = Canvas(root , width = 400 , height = 800 , bg = "pink")
# im = get_image('chuci.gif' , 400 , 800)
# can.create_image(200,300,image = im)
# can.pack()

# 定义lable对象用Lable方法，顺序分别为窗口对象，显示文本python程序设计,字体内型为华文行楷，大小为20
# 字体颜色为绿色，背景颜色为粉色
label=Label(root,text="楚辞生成器",font=("华文行楷",20),fg="green",bg=None)
label.pack()#调用pack方法将label标签显示在主界面

data = StringVar()
entry =Entry(root ,font=("华文行楷",15),textvariable=data)#创建labal组件并将其与data关联
entry.pack()

def callback():
    print(data.get())
    text.delete('1.0','end')
    text.insert(INSERT,data.get())
    text.update()

button=Button(root,text='开始生成',font=("华文行楷",15),command=callback)
button.pack()

text =scrolledtext.ScrolledText(root,width=25,height=20,font=("华文行楷",20)) #创建labal组件并将其与data关联

text.pack()

#can.create_window(200, 30, width = 400, height=800,window=text)

root.mainloop()



