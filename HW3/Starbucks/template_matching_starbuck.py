import cv2
import numpy as np
from matplotlib import pyplot as plt
import PIL
from PIL import Image

def change_image_type(infile):
    try:
        im = Image.open(infile)
    except IOError:
        print "Cant load", infile
        sys.exit(1)

    mypalette = im.getpalette()

    try:
        while 1:
            im.putpalette(mypalette)
            new_im = Image.new("RGBA", im.size)
            new_im.paste(im)
            new_im.save("change_template" +".jpg")
            im.seek(im.tell() + 1)

    except EOFError:
        pass

def rescale_image(filename,rescale_width):
    img = Image.open(filename)
    wpercent = (rescale_width/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((rescale_width,hsize), PIL.Image.ANTIALIAS)
    img.save('rescale_template.jpg')


#change the file format
change_image_type("template.gif")

rescale_size = [[250],[65],[140,120,150],[110,135,170],[120],[70],[200],[75,80,100],[120],[120],[75],[62],[240],[300],[1000],[600]]
methods = ['cv2.TM_SQDIFF_NORMED','cv2.TM_CCOEFF','cv2.TM_CCOEFF_NORMED','cv2.TM_CCOEFF_NORMED','cv2.TM_CCOEFF','cv2.TM_CCORR','cv2.TM_CCOEFF','cv2.TM_CCOEFF','cv2.TM_CCOEFF','cv2.TM_SQDIFF_NORMED','cv2.TM_CCOEFF','cv2.TM_CCOEFF','cv2.TM_CCORR','cv2.TM_CCOEFF','cv2.TM_CCORR','cv2.TM_CCORR']


for i in range(16):
    top_left = []
    bottom_right = []

    for j in range(len(rescale_size[i])):
        rescale_image("change_template.jpg",rescale_size[i][j])
        target_image_name = "image" + str(i+1) + ".jpg"
        img = cv2.imread(target_image_name)
        img2 = img
        template = cv2.imread("rescale_template.jpg")
        w, h, xxx = np.array(template).shape

        img = img2.copy()
        method = eval(methods[i])
    
    # Apply template Matching
        res = cv2.matchTemplate(img,template,method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            temp_1 = min_loc
        else:
            temp_1 = max_loc

        top_left.append(temp_1)
        temp_2 = (temp_1[0] + h, temp_1[1] + w)
        bottom_right.append(temp_2)
    
    for j in range(len(top_left)):    
        cv2.rectangle(img,top_left[j], bottom_right[j], 255, 10)

    plt.plot(122),plt.imshow(img,cmap = 'gray')
    plt.title(methods[i]), plt.xticks([]), plt.yticks([])

    plt.show()    