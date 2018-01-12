import os, sys
import numpy as np
from glob import glob
from skimage import io 

def average(images, output_path):
    	print("Calculating images' average ...")
    	sum = np.array(io.imread(images[0]), dtype=np.float32)
    	num = len(images)
    
    	for i in range(num-1):
    		im = np.array(io.imread(images[i+1]), dtype=np.float32)
    		sum += im
            
    	avg_float = sum / float(num)
    	avg = np.array(avg_float, dtype=np.uint8)
    
    #	print('Saving average image ...')
    	#io.imsave(output_path,avg)
    	return avg_float

def imAdjust(M, output_path):
    	sqrt_dim = np.sqrt(len(M)/3).astype(np.uint16) # supported to be 600, uint8 will overflow
    	
    	M -= np.min(M)
    	M /= np.max(M)
    	M = (M * 255).astype(np.uint8)
    	im = M.reshape(sqrt_dim,sqrt_dim,3)
    
    	print('Making output named %s ...' % output_path)	
    	io.imsave(output_path,im)

def pca(name_image, avgImage):
    	
    avgImage = avgImage.reshape(1,-1)
    
    print('Reading images ...')
    num_image = len(name_image)
    images = []
    for i in range(num_image):
        im = io.imread(name_image[i])
        im = im.reshape(1,-1)
        images.append(im)
        if (i+1) % 10 == 0:
            print('%d / %d images readed ...' % (i+1, num_image))
    print('Image reading is done ...')    
    images = np.array(images, dtype=np.float32)
    images = images.reshape(num_image, -1)   
    
    print("Calculating eigenvalues and eigenvectors ...")	  
    images -= avgImage	   
    U,s,V = np.linalg.svd(images.T, full_matrices = False)
    
    '''
    	print('pick the first %d eigenface ...' % num_eig)
    	eigFace = U[:,:num_eig]
    	eigValue = s[:num_eig]
    	for i in range(num_eig):
    		output_path = 'eigenface' + str(i) + '.jpg'
    		imAdjust(eigFace[:,i], output_path)
    '''
    return U,s

def reconstruct(image_path, avgImage, eigFace, eig_num, output_path):
    im = np.array(io.imread(image_path), dtype=np.float32)
    im = im.reshape(1,-1)
    avgImage = avgImage.reshape(1,-1)
	
    im -= avgImage
    w = np.dot(im, eigFace[:,:eig_num])
    imProject = np.dot(eigFace[:,:eig_num], w.T)
    imProject += avgImage.T
    imAdjust(imProject, output_path)

def eigRatio(eigValue, eig_num):
    eigSum = np.sum(eigValue)
    for i in range(eig_num):
        print("the eigenface %d 's ratio = %f / %f = %f" %(i, eigValue[i], eigSum, eigValue[i]/eigSum))


def main():
    folder = sys.argv[1]      #folder = 'Aberdeen'
    pattern = os.path.join(folder,'*.jpg')
    name_image = glob(pattern)
    avgImage = average(name_image, 'myAverage.jpg')  #avgImage = average(name_image, 'myAverage.jpg')
    eigFace, eigValue = pca(name_image, avgImage)
    #eigRatio(eigValue, 4)
    target = os.path.join(folder, sys.argv[2])
    reconstruct(target, avgImage, eigFace, 4, 'reconstruction.jpg')
  	
if __name__ == '__main__':
    main()
