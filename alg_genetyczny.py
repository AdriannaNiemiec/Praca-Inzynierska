import cv2
import numpy as np
from matplotlib import pyplot as plt
import random
from numpy.random import randint
from numpy.random import rand
import math
import os   
from skimage import measure, color, io


def Hist(image):
    H = np.zeros(shape=(256,1))
    shape = image.shape
    for i in range(shape[0]):  
        for j in range(shape[1]):   
            k=image[i,j]   
            H[k,0]=H[k,0]+1
    return H

def inverse(image):
    shape = image.shape
    for i in range(shape[0]):  
        for j in range(shape[1]):  
            if image[i,j]==0:
                image[i,j]=255
            elif image[i,j]==255:
                image[i,j]=0
    return image


def splitImage(image, x, y):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    width = image.shape[1]
    width_cutoff = width // x
    verti_parts = []
    for i in range(x):
        if i==x-1:
            verti_parts.append(image[:,i*width_cutoff:])
            break
        verti_parts.append(image[:, i*width_cutoff:(i+1)*width_cutoff])
    fin_pcs = []
    for i in verti_parts:
        image = cv2.rotate(i, cv2.ROTATE_90_CLOCKWISE)
        width = image.shape[1]
        width_cutoff = width // y
        for j in range(y):
            if j == y-1:
                fin_pcs.append(image[:,j*width_cutoff:])
                fin_pcs[len(fin_pcs)-1] = cv2.rotate(fin_pcs[len(fin_pcs)-1], cv2.ROTATE_90_COUNTERCLOCKWISE)
                break
            fin_pcs.append(image[:,j*width_cutoff:(j+1)*width_cutoff])
            fin_pcs[len(fin_pcs)-1] = cv2.rotate(fin_pcs[len(fin_pcs)-1], cv2.ROTATE_90_COUNTERCLOCKWISE)

    return fin_pcs


def mergeImage(pcsList, x, y): 
    height = 0
    width = 0

    for i in range(0, len(pcsList), y):
        width += pcsList[i].shape[1]
    
    for i in range(y):
        height += pcsList[i].shape[0]

    base_widths = pcsList[0].shape[1]
    base_heights = pcsList[0].shape[0]
    big_height = pcsList[len(pcsList)-1].shape[0]
    new_image = np.zeros((height, width), np.uint8)
    for img in range(len(pcsList)):
        for i in range(pcsList[img].shape[0]):
            for j in range(pcsList[img].shape[1]):
                new_image[((y-1-(img%y))*base_heights+(((y-1-(img%y))>0)*big_height))+i-(((y-1-(img%y))>0)*base_heights), ((math.floor(img/y))*base_widths)+j] = pcsList[img][i,j]
    return new_image

#binarne na dziesietne
def binaryToDeci(tablica):  
    deci = []
    tmp_val = 0
    for j in range(int(len(tablica)/8)):
        for n in range(7, -1, -1):
            tmp_val += (tablica[n+(j*8)]*(2**(-n+7)))
        deci.append(tmp_val)
        tmp_val=0
    return deci

def binaryToDeciSingle(tablica):
    tmp_val=0
    for n in range(7, -1, -1):
        tmp_val += (tablica[n]*(2**(-n+7)))
    return tmp_val

def deciToBinary(deci):
    binary = []
    for i in deci:
        for j in range(7,-1,-1):
            if i > 2**j:
                binary.append(1)
                i -= 2**j
            else:
                binary.append(0)
    return binary


def calcVariance(threshs, imgqs):
    final_var = [] 
    for i in range(len(imgqs)):
        img = imgqs[i]
        hist = plt.hist(img.ravel(),256,[0,256])
        total = np.sum(hist[0])
        final = 0
        left, right = np.hsplit(hist[0],[threshs[i]])
        left_bins, right_bins = np.hsplit(hist[1],[threshs[i]])

        if np.sum(left) !=0 and np.sum(right) !=0:
            # weights
            w_0 = np.sum(left)/total
            w_1 = np.sum(right)/total
            # mean 
            mean_0 = np.dot(left,left_bins)/np.sum(left)
            mean_1 = np.dot(right,right_bins[:-1])/np.sum(right)  
            # variance 
            var_0 = np.dot(((left_bins-mean_0)**2),left)/np.sum(left)
            var_1 = np.dot(((right_bins[:-1]-mean_1)**2),right)/np.sum(right)
            # final within class variance
            final = w_0*var_0 + w_1*var_1
        final_var.append(final)
    return final_var


# f. przystosowania - zwraca srednia z within-class variance (Otsu)
def fitness(thresh, image, x, y):
    image_parts = splitImage(image, x ,y)
    ths = binaryToDeci(thresh)
    list = calcVariance(ths, image_parts)
    total = 0
    counter = 0
    for i in list:
        total += i
        counter += 1
    avg = total/counter
    print(avg)
    return avg

# selekcja jednego rodzica
def selection(pop, scores, k=3):
	# pierwsza losowa selekcja
	selection_ix = randint(len(pop))
	for ix in randint(0, len(pop), k-1):
		# sprawdz czy rozwiazanie jest lepsze
		if scores[ix] < scores[selection_ix]:
			selection_ix = ix
	return pop[selection_ix]

def compareThresh(maxs, mins, chrom):
    deci = binaryToDeci(chrom)
    for i in range(len(deci)):
        if deci[i] > maxs[i]:
            return False
        elif deci[i] < mins[i]:
            return False
    return True


def crossover(p1, p2, r_cross, maxs, mins):
    # dzieci - kopie rodzicow
    c1, c2 = p1.copy(), p2.copy()
	# rekombinacja
    if rand() < r_cross:
        pt = randint(1, len(p1)-2)
        # perform crossover
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
        if compareThresh(maxs, mins, c1):
            if compareThresh(maxs, mins, c2):
                return [c1,c2]
            else:
                return crossover(p1, p2, r_cross, maxs, mins)
        else:
            return crossover(p1, p2, r_cross, maxs, mins)
    return [c1, c2]

#mutacja - odwraca bity z niskim p-stwem kontrolowanym przez parametr r_mut
def mutation(bitstring, r_mut, maxs, mins):
    for i in range(len(bitstring)):
        # check for a mutation
        if rand() < r_mut:
            # flip the bit
            bitstring[i] = 1 - bitstring[i]
            if not compareThresh(maxs, mins, bitstring):
                bitstring[i] = 1 - bitstring[i]

#sprawdzenie limitu progow (najwyzszego i najnizszego)
def limiter(img, x ,y):
    image_parts = splitImage(img, x ,y)
    maxs = []
    mins = []
    for i in image_parts:
        tmp = Hist(i)
        for j in range(255,-1,-1):
            if tmp[j][0]>0:
                maxs.append(j)
                break
        for n in range(255):
            if tmp[n][0]>0:
                mins.append(n)
                break
    return maxs, mins

#progi adaptacyjne - losowany nowy prog
def adaptiveThreshold(maxs, mins, n_bits, n_pop):
    pop = [randint(0, 2, n_bits).tolist() for _ in range(n_pop)]
    for i in range(len(pop)):
        deci = binaryToDeci(pop[i])
        while True:
            counter=0
            for j in range(len(deci)):
                if deci[j] > maxs[j]:
                    deci[j] = binaryToDeciSingle(randint(0, 2, n_bits).tolist())
                    counter+=1
                elif deci[j] < mins[j]:
                    deci[j] = binaryToDeciSingle(randint(0, 2, n_bits).tolist())
                    counter+=1
            if counter==0:
                break
        pop[i] = deciToBinary(deci)
    return pop

# algorytm genetyczny
def genetic_algorithm(objective, n_bits, n_iter, n_pop, r_cross, r_mut, image, x, y):
    #losowanie pierwotnej populacji
	maxs, mins = limiter(image, x, y)
	pop = adaptiveThreshold(maxs, mins, n_bits, n_pop)
	#sledz najlepsze rozwiazanie
	best, best_eval = 0, objective(pop[0], image, x, y)
	for gen in range(n_iter):
		#ocen wszystkich kandydatow w populacji
		scores = [objective(c, image, x, y) for c in pop]
		#sprawdz nowe najlepsze rozwiazanie
		for i in range(n_pop):
			if scores[i] < best_eval:
				best, best_eval = pop[i], scores[i]
				print(">%d, new best f(%s) = %.3f" % (gen,  pop[i], scores[i]))
		#utworzenie listy rodzicow (p1,p2)
		selected = [selection(pop, scores) for _ in range(n_pop)]
		#stworz nowe pokolenie
		children = list()
		for i in range(0, n_pop, 2):
			#polacz wybranych rodzicow w pary
			p1, p2 = selected[i], selected[i+1]
			#krzyzowanie i mutacja
			for c in crossover(p1, p2, r_cross, maxs, mins):
				mutation(c, r_mut, maxs, mins)
				children.append(c)
		#zamien populacje
		pop = children
	return [best, best_eval]

#algorytm watershed
def separation(img, thresh):
    image_thresh = thresh
    #usun szum
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(image_thresh, cv2.MORPH_OPEN,kernel, iterations = 2)
    ### Segmentacja Watershed
    #sure background
    sure_bg = cv2.dilate(opening, kernel,iterations=3)
    #sure foreground
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)
    ret2, sure_fg = cv2.threshold(dist_transform,0.4*dist_transform.max(),255,0)
    #Unknown region
    sure_fg = np.uint8(sure_fg)  #Convert to uint8 from float
    unknown = cv2.subtract(sure_bg,sure_fg)
    #markers
    ret3, markers = cv2.connectedComponents(sure_fg)
    markers = markers+10
    markers[unknown==255] = 0
    markers = cv2.watershed(img,markers)
    img[markers == -1] = [0,255,255]
    image_result = [image_thresh]
    counter=0
    #petla przez unikalne etykiety zwrocone przez watershed
    for label in np.unique(markers):
	#jesli etykieta ma wartosc 0, sprawdzamy tlo => ignorowane
        if counter == 0 or counter ==1:
            counter+=1
            continue
        if label == 0:
            continue
	#jesli nie, wyrysuj region etykiety
        mask = np.zeros(img.shape, dtype="uint8")
        mask[markers == label] = 255
        image_result.append(mask)
    return image_result

#zapisz maski
#def saveMasks(img_result):
#    path1 = "/sciezkaDoFolderu/"
#    os.mkdir(path1) #tworzenie sciezki do pliku
#    for index in range(1, len(img_result)):  
#        path2 = '/sciezkaDoFolderu/' + '/mask' + str(index) + '.png'
#        io.imsave(path2, img_result[index])
#    return img_result


# liczba czesci obrazu w pionie
x=2
# liczba czesci obrazu w poziomie
y=2
# liczba iteracji
n_iter = 100
# rozmiar populacji
n_pop = x*y
# bits
n_bits = n_pop*8
# crossover rate
r_cross = 0.9
# mutation rate
r_mut = 1.0 / float(n_bits)

if (x*y)%2 == 0:
    img_name = 'dataset/0b2e702f90aee4fff2bc6e4326308d50cf04701082e718d4f831c8959fbcda93/images/0b2e702f90aee4fff2bc6e4326308d50cf04701082e718d4f831c8959fbcda93.png'
    # przeprowadz wyszukiwanie algorytmem genetycznym
    image_original = cv2.imread(img_name)
    image = cv2.GaussianBlur(image_original,(5,5),0)
    best, score = genetic_algorithm(fitness, n_bits, n_iter, n_pop, r_cross, r_mut, image, x, y)

    image_parts = splitImage(image, x, y)
    bestDeci = binaryToDeci(best)
    avg_thresh = 0
    for j in range(len(bestDeci)):
        avg_thresh += bestDeci[j]
    avg_thresh = avg_thresh/len(bestDeci)
    threshedList = []
    for i in range(len(image_parts)):
        (otsu_threshold, image_thresh) = cv2.threshold(image_parts[i], avg_thresh, 255, cv2.THRESH_BINARY)
        threshedList.append(image_thresh)
        cv2.imshow('test', image_thresh)
        cv2.waitKey()
    final_image = mergeImage(threshedList, x, y)
    histx = Hist(final_image)
    if histx[0]<histx[255]:
        image_thresh = inverse(final_image)

    cv2.imshow('final', final_image)
    cv2.waitKey()
    print('Done!')
    print('f(%s) = %f' % (best, score))
    
    best_val = binaryToDeci(best) 
    print(best_val)
    print('avg_thresh: ', avg_thresh)

    #Separacja jader
    kernel_list = separation(image, final_image)
    list=[]
    for img in kernel_list:
        cv2.imshow('', img)
        cv2.waitKey()
        list.append(img)
    
    #saveMasks(list)
else:
    print('Nalezy podac wymiary których iloczyn jest parzysty')