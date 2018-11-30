from Tkinter import *
import Tkinter, Tkconstants, tkFileDialog
from PIL import Image, ImageTk
from ttk import Frame, Style, Button, Radiobutton
import os, time, cv2
import numpy as np
import tkMessageBox as mbox
from our_func_cvpr18 import our_func
import tensorflow as tf

class Example(Frame):

    cnt = 0
    imIdx = 0
    usrId = -1
    flag = 0
    filename = ""
    x_bd = 0
    y_bd = 0

    def __init__(self, parent):
        Frame.__init__(self, parent)

        self.parent = parent
        self.initUI()

    def onNextImg(self):
        mbox.showinfo("Information", "One task completed! Thank you!")

    def onNextMethod(self):
        mbox.showinfo("Information", "One task completed! Thank you!" )

    def callback_left(self, event):
        if self.flag == 1:
            return
        self.flag = 1
        self.focus_set()
        T.insert('1.0', "Click %d: Positive click at [%d, %d].\n\n"%(self.cnt, event.x, event.y))
        # print "left clicked at", event.x, event.y, self.cnt
        if event.y >= self.y_bd or event.x >= self.x_bd:
            T.insert('1.0', "Click is outside the image! Ignored!\n")
            self.flag = 0
            return
        target0 = open("res/%d/Ours/time_log.txt" % self.usrId, 'a+')
        st = time.time()
        our_iou = our_func(self.usrId, self.imIdx, self.filename, self.cnt, 1, event)
        target0.write("%f\n" % (time.time()-st))
        target0.close()
        self.update_show()
        self.cnt = self.cnt + 1
        self.flag = 0

    def callback_right(self, event):
        if self.flag == 1:
            return
        self.flag = 1
        self.focus_set()
        T.insert('1.0', "Click %d: Negative click at [%d, %d].\n\n" % (self.cnt, event.x, event.y))
        # print "right clicked at", event.x, event.y, self.cnt
        if event.y >= self.y_bd or event.x >= self.x_bd:
            T.insert('1.0', "Click is outside the image! Ignored!\n")
            self.flag = 0
            return
        target0 = open("res/%d/Ours/time_log.txt" % self.usrId, 'a+')
        st = time.time()
        our_iou = our_func(self.usrId, self.imIdx, self.filename, self.cnt, 2, event)
        target0.write("%f\n" % (time.time()-st))
        target0.close()
        self.update_show()
        self.cnt = self.cnt + 1
        self.flag = 0

    def initUI(self):
        # os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
        # os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax([int(x.split()[2]) for x in open('tmp', 'r').readlines()]))
        # os.system('rm tmp')
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        sess=tf.Session()
        self.usrId = time.time()
        self.usrId = int(time.time()-1495000000)

        if not os.path.isdir("res/%d" % self.usrId):
            os.makedirs("res/%d" % self.usrId)
            os.makedirs("res/%d/Ours" % self.usrId)

        self.parent.title("Interactive Image Segmentation")
        self.pack(fill=BOTH, expand=1)

        global T
        T = Text(self, height=100, width=20)
        T.pack()
        T.insert(END, "Welcome!\n")
        T.place(x=1760, y=20)
        Style().configure("TFrame", background="#333")

        w = 1920
        h = 1080
        # w = 1280
        # h = 900

        sw = self.parent.winfo_screenwidth()
        sh = self.parent.winfo_screenheight()

        x = (sw - w) / 2
        y = (sh - h) / 2
        self.parent.geometry('%dx%d+%d+%d' % (w, h, x, y))

        self.filename = tkFileDialog.askopenfilename(initialdir="./", title="Select file",
                                                     filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))

        self.update_all()

        self.bind_all("<Button-1>", self.callback_left)
        self.bind_all("<Button-3>", self.callback_right)

    def update_all(self):
        time.sleep(2)
        im_path = self.filename
        bard = Image.open(im_path)
        self.x_bd = bard.width
        self.y_bd = bard.height
        bardejov = ImageTk.PhotoImage(bard)
        label1 = Label(self, image=bardejov)
        label1.image = bardejov
        label1.place(x=10, y=10)

        imgray_path = "helper/graycolor.png"
        rot = Image.open(imgray_path)
        rotunda = ImageTk.PhotoImage(rot)
        label2 = Label(self, image=rotunda)
        label2.image = rotunda
        label2.place(x=880, y=500)

        bard = Image.open(imgray_path)
        bardejov = ImageTk.PhotoImage(bard)
        label1 = Label(self, image=bardejov)
        label1.image = bardejov
        label1.place(x=10, y=500)

        rot = Image.open(imgray_path)
        rotunda = ImageTk.PhotoImage(rot)
        label2 = Label(self, image=rotunda)
        label2.image = rotunda
        label2.place(x=880, y=10)

        self.update()

    def update_show(self):

        res_path = "res/%d/Ours/%05d/segs/%03d.png" % (self.usrId, self.imIdx, self.cnt)
        tmp_clk_path = 'res/%d/Ours/%05d/tmps/clk_%03d.png' % (self.usrId, self.imIdx, self.cnt)
        tmp_ol_path = 'res/%d/Ours/%05d/tmps/ol_%03d.png' % (self.usrId, self.imIdx, self.cnt)

        bard = Image.open(tmp_clk_path)
        bardejov = ImageTk.PhotoImage(bard)
        label1 = Label(self, image=bardejov)
        label1.image = bardejov
        label1.place(x=10, y=10)
        bard = Image.open(tmp_ol_path)
        bardejov = ImageTk.PhotoImage(bard)
        label1 = Label(self, image=bardejov)
        label1.image = bardejov
        label1.place(x=880, y=10)
        minc = Image.open(res_path)
        mincol = ImageTk.PhotoImage(minc)
        label3 = Label(self, image=mincol)
        label3.image = mincol
        label3.place(x=10, y=500)
        self.update()

def main():
    root = Tk()
    app = Example(root)
    root.mainloop()


if __name__ == '__main__':
    main()