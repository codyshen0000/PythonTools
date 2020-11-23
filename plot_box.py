from  PIL import Image
import numpy as np
import os


def plot_box(loadpath,savepath, box, point, red_line=True):
    point_x, point_y= point[0],point[1]
    box_h, box_w = box[0], box[1]
    filename, ext = os.path.splitext(os.path.basename(loadpath))
    savepath = os.path.join(savepath,str(point_x)+'_'+str(point_y))
    os.makedirs(savepath,exist_ok=True)

    img = Image.open(loadpath)
    img = np.array(img)
    bx = img[point_x:point_x+box_w,point_y:point_y+box_h]
    if red_line:
        img[point_x:point_x+box_w,point_y,0]= 255
        img[point_x:point_x+box_w,point_y,1:]=0
        img[point_x,point_y:point_y+box_h,0]= 255
        img[point_x,point_y:point_y+box_h,1:]=0

        img[point_x+box_w,point_y:point_y+box_h,0]= 255
        img[point_x+box_w,point_y:point_y+box_h,1:]=0

        img[point_x:point_x+box_w,point_y+box_h,0]= 255
        img[point_x:point_x+box_w,point_y+box_h,1:]=0

    bx = Image.fromarray(bx)
    img = Image.fromarray(img)
    img.save(os.path.join(savepath,filename+ext))
    bx.save(os.path.join(savepath,filename+'box'+ext))

if __name__ == "__main__":
    loadpath = r'C:\Users\ShenSheng\Desktop\B100\edsr\00166_x4_SR.png'
    savepath = r'C:\Users\ShenSheng\Desktop\B100\edsr\box'
    box = [381, 78]
    point = [285, 854]
    plot_box(loadpath,savepath,box,point,red_line=True)