import cv2
import numpy as np
from matplotlib import pyplot as plt
import PIL
from PIL import Image

#rescale the template to the same size of the template part in original picture
def rescale_image(filename,rescale_width):
	img = Image.open(filename)
	wpercent = (rescale_width/float(img.size[0]))
	hsize = int((float(img.size[1])*float(wpercent)))
	img = img.resize((rescale_width,hsize), PIL.Image.ANTIALIAS)
	img.save('rescale_template.jpg')

def template_matching(template_file,matching_file, match_method):
    img = cv2.imread(matching_file)
    img2 = img
    template = cv2.imread(template_file)
    w, h, j = np.array(template).shape
   
    matching_method = match_method

    img = img2.copy()
    method = eval(matching_method)
    
    # Apply template Matching
    res = cv2.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
#As for we used 'cv2.TM_CCOEFF' as our matching method, we get the biggest value among the pixals and get the (x,y) of that pixal.
    top_left = max_loc
    bottom_right = (top_left[0] + h, top_left[1] + w)
    cv2.rectangle(img,top_left, bottom_right, 255, 5)

    # draw the detected part in the orinigal picture
    plt.plot(),plt.imshow(img,cmap = 'gray')
    plt.title(matching_method), plt.xticks([]), plt.yticks([])

    plt.show()   
'''
After experiments a lot of times, we got the exact window size of the template part in the orinigal picture,
so we rescale the template into that size
'''
rescale_size = [200,110,300,215,240,65,310,130,120]
methods = ['cv2.TM_CCOEFF','cv2.TM_CCOEFF','cv2.TM_CCOEFF','cv2.TM_CCORR','cv2.TM_CCOEFF','cv2.TM_CCOEFF','cv2.TM_CCOEFF','cv2.TM_CCOEFF','cv2.TM_CCORR_NORMED']

for i in range(9):
    rescale_image("template.png",rescale_size[i])
    target_image_name = "image" + str(i+1) + ".jpg"
    template_matching("rescale_template.jpg", target_image_name, methods[i])
