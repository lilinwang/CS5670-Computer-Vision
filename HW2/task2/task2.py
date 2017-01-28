#task 2 Epipolar Geometry and Camera Calibration

import numpy as np

def readpoints(filename):
	data = []
	with open(filename, "r") as ins:
		for line in ins:
			data += line.strip(' ').strip('\n').split('  ')

	return data

def imagepoints_homogeneous_coordinate():
	result_image = readpoints("HW2_image.txt")
	result_image_new = []
	image = []

	for item in result_image:
		item = float(item)
		result_image_new.append(item)
	
	image.append(result_image_new[:10])
	image.append(result_image_new[10:])
	image.append([1]*10)
	return image

def worldpoints_homogeneous_coordinate():
	result_world = readpoints("HW2_world.txt")
	result_world_new = []
	world = []

	for item in result_world:
		item = float(item)
		result_world_new.append(item)

	world.append(result_world_new[:10])
	world.append(result_world_new[10:20])
	world.append(result_world_new[20:])
	world.append([1]*10)
	return world

# Use linear equation to construct A
def compute_a(img_points,world_points):
	A = []
	
	for i in range (10):
		a1 = []
		a2 = []
		a1 += [0,0,0,0]
		
		for j in range (4):
			a1.append(-img_points[2][i] * world_points[j][i])
			a2.append(img_points[2][i] * world_points[j][i]) 

		a2 += [0,0,0,0]
		for j in range (4):
			a1.append(img_points[1][i] * world_points[j][i])
			a2.append(-img_points[0][i] * world_points[j][i])
			
		A.append(a1)
		A.append(a2)
	A= np.array(A)

	return A

def compute_svd(a):
	U, s, V = np.linalg.svd(a, full_matrices=True)
	i_min = np.argmin(s)
	min_eigenvector = V[i_min,:]

	return min_eigenvector

#computed the difference of the original points and the projected points
def verify(img_points,world_points,p):
	diff = []
	p_new = p.tolist()
	for k in range(10):
		df = 0.0
		array = []
		for i in range(3):
			temp = 0.0
			for j in range(4):
				temp += p_new[i][j] * world_points[j][k]
			array.append(temp)
		for q in range(3):
			df += (img_points[q][k]-array[q]/array[2])**2
		df = df **(0.5)
		diff.append(df)
	print
	print "the difference between the original points and the projected points"
	print diff


if __name__ == '__main__':

	#convert the world points and the image points to homogenous points by adding a row of 1 
	result1 = imagepoints_homogeneous_coordinate()
	result2 = worldpoints_homogeneous_coordinate()


	#Construct A
	task_A = compute_a(result1,result2)
	print "matrix A:"
	print task_A
	
	#Compute SVD for A
	p = compute_svd(task_A)
	p = p.reshape((3,4))
	print
	print "the 3x4 camera matrix P is:"
	print p
	
	# print the difference of the original points and the projected points
	verify(result1,result2,p)
    
    # compute the world coordinates of the projection center of the camera C
	c = compute_svd(p)
	c = c/c[c.size-1]
	camera_coordinate = c[:3]
	print
	print "the world coordinates of the projection center of the camera:"
	print camera_coordinate	
