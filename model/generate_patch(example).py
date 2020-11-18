#This code used to generate the training data from the 3d dataset
#JUST FOR AN EXAMPLE AND NO DATA AUGMENTATION

def seispatch_2d(section,patch_size,os_x=0,os_y=0,show=False,colors="Dark2"):
    '''
    section :the 2-D seismic data
    patch_size :the size of patches
    os_x/_y: the length of overlap 
    show: 1:visualization
    colors：if show==True; label:"Dark2"; seismic:"seismic" or "Greys"
    '''
    
    m1,m2 = section.shape                            
    n1,n2 = patch_size,patch_size  #for square patch              
    c1 = int(np.round((m1+os_y)/(n1-os_y)+0.5))   
    c2 = int(np.round((m2+os_x)/(n2-os_x)+0.5))-2  
    p1 = (n1-os_y)*c1+os_y   #expand the edge
    p2 = (n2-os_x)*c2+os_x   
    img_pad   = np.zeros((p1,p2),dtype=np.single)
    img_pad[0:m1,0:m2]=section
    img_patch = np.zeros((n1,n2),dtype=np.single)
    i=0
    result=np.zeros((c1*c2,n1,n2),dtype=np.single)  
    for k1 in range(c1):
        for k2 in range(c2):
            b1 = k1*(n1-os_y) 
            e1 = b1+n1          
            b2 = k2*(n2-os_x)
            e2 = b2+n2                   
            img_patch=img_pad[b1:e1,b2:e2]
            result[i,]=img_patch
            i+=1
            if show:
                import matplotlib.pyplot as plt
                plt.subplot(c1,c2,i)
                imgplot1 = plt.imshow(img_patch,cmap=colors,interpolation='nearest')            
    return result

def extract_2d_from_3d(seismic_3d,axis=0,patch_size=128,os_x=0,os_y=0):
    '''
    seismic_3d :the 3-D seismic data
    axis :0:patch from inline; 1:xline
    '''
    for i in range(np.shape(seismic_3d)[axis]):
        line=seispatch_2d(np.transpose(seismic_3d[i]),patch_size,os_x,os_y)
        if i == 0:
            c = line
        else:
            c=np.vstack((c,line))
    return c
    
if __name__ == "__main__":
    sei_patch = extract_2d_from_3d(train_seismic,patch_size=256,os_x=128,os_y=0)
    lab_patch = extract_2d_from_3d(train_labels,patch_size=256,os_x=128,os_y=0)
