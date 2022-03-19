import numpy as np
import cv2
import os


def init_particles(state,n):
    particles = np.array([state,]*n)
    return particles
    

def get_view(image,x,y,x_size, y_size):
    """
    Get a smaller image, centered at (x,y) with size (sq_size x sq_size)
    """
    # print("get_view", image.shape, x, y ,sq_size)
    # with numpy arrays this is an O(1) operation
    view = image[int(x-x_size/2):int(x+x_size/2),
                 int(y-y_size/2):int(y+y_size/2),:]
    return view
    
def calc_hist(image, mask=None, histogram =["h"]):
    """
    Computes the color histogram of an image (or from a region of an image).
    
    image: 3D Numpy array (X,Y,RGB)

    return: One dimensional Numpy array
    """
    Nh,Ns,Nv=180,255,255 
    if len(histogram)==1:
        if histogram[0]=="h":
            Nbins = Nh
            if mask is None:
                mask = cv2.inRange(image, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
                
                hist = cv2.calcHist([image], [0], mask, [Nh], [0, 181]) # Value histogram
                cv2.normalize(hist, hist, 0, 1, norm_type=cv2.NORM_MINMAX)
                return hist
            else :
                hist = cv2.calcHist([image], [0], mask, [Nh], [0, 181]) # Value histogram
                cv2.normalize(hist, hist, 0, 1, norm_type=cv2.NORM_MINMAX)
                return hist
        elif histogram[0]=="s":
            Nbins = Ns
            if mask is None:
                mask = cv2.inRange(image, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
                
                hist = cv2.calcHist([image], [1], mask, [Ns], [0, 256]) # Value histogram
                cv2.normalize(hist, hist, 0, 1, norm_type=cv2.NORM_MINMAX)
                return hist
            else :
                hist = cv2.calcHist([image], [1], mask, [Ns], [0, 256]) # Value histogram
                cv2.normalize(hist, hist, 0, 1, norm_type=cv2.NORM_MINMAX)
                return hist
        elif histogram[0]=="v":
            Nbins = Nv
            if mask is None:
                mask = cv2.inRange(image, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
                
                hist = cv2.calcHist([image], [2], mask, [Nv], [0, 256]) # Value histogram
                cv2.normalize(hist, hist, 0, 1, norm_type=cv2.NORM_MINMAX)
                return hist
            else :
                hist = cv2.calcHist([image], [2], mask, [Nv], [0, 256]) # Value histogram
                cv2.normalize(hist, hist, 0, 1, norm_type=cv2.NORM_MINMAX)
                return hist
    elif len(histogram)==1:
        if histogram==["h","s"]:
            if mask is None:
                mask = cv2.inRange(image, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
                Nbins = Nh*Ns
                hist = cv2.calcHist([image], [0, 1], mask, [Nh, Ns], [0, 181, 0, 256]) # Hue/Saturation histogram
                cv2.normalize(hist, hist, 0, 1, norm_type=cv2.NORM_MINMAX)
                return hist
            else:
                Nbins = Nh*Ns
                hist = cv2.calcHist([image], [0, 1], mask, [Nh, Ns], [0, 181, 0, 256]) # Hue/Saturation histogram
                cv2.normalize(hist, hist, 0, 1, norm_type=cv2.NORM_MINMAX)
                return hist
        elif histogram==["h","v"]:
            Nbins = Nh*Nv
            if mask is None:
                mask = cv2.inRange(image, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
                hist = cv2.calcHist([image], [0, 2], mask, [Nh, Nv], [0, 181, 0, 256]) # Hue/Saturation histogram
                cv2.normalize(hist, hist, 0, 1, norm_type=cv2.NORM_MINMAX)
                return hist
            else:
                hist = cv2.calcHist([image], [0, 2], mask, [Nh, Nv], [0, 181, 0, 256]) # Hue/Saturation histogram
                cv2.normalize(hist, hist, 0, 1, norm_type=cv2.NORM_MINMAX)
                return hist
        elif histogram==["s","v"]:
            if mask is None:
                mask = cv2.inRange(image, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
                Nbins = Nh*Nv
                hist = cv2.calcHist([image], [1, 2], mask, [Ns, Nv], [0, 256, 0, 256]) # Hue/Saturation histogram
                cv2.normalize(hist, hist, 0, 1, norm_type=cv2.NORM_MINMAX)
                return hist
            else:
                Nbins = Nh*Nv
                hist = cv2.calcHist([image], [1, 2], mask, [Ns, Nv], [0, 256, 0, 256]) # Hue/Saturation histogram
                cv2.normalize(hist, hist, 0, 1, norm_type=cv2.NORM_MINMAX)   
                return hist
        elif histogram ==["hs","v"]:
            Nbins = Nh*Ns + Nv # Total number of bins
            if mask is None:
                mask = cv2.inRange(image, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
                hist_hs = cv2.calcHist([image], [0, 1], mask, [Nh, Ns], [0, 181, 0, 256]) # Hue/Saturation histogram
                hist_v = cv2.calcHist([image], [2], mask, [Nv], [0, 256]) # Value histogram
                
                # Normalize histograms
                cv2.normalize(hist_hs, hist_hs, 0, 1, norm_type=cv2.NORM_MINMAX)
                cv2.normalize(hist_v, hist_v, 0, 1, norm_type=cv2.NORM_MINMAX)
                    
                # Concatenate both histograms (weighted)
                hist = np.concatenate((hist_hs.flatten()*Nh*Ns/Nbins, hist_v.flatten()*Nv/Nbins))
                return hist
            else:
                Nh,Ns,Nv=180,255,255 
                Nbins = Nh*Ns + Nv # Total number of bins
                hist_hs = cv2.calcHist([image], [0, 1], mask, [Nh, Ns], [0, 181, 0, 256]) # Hue/Saturation histogram
                hist_v = cv2.calcHist([image], [2], mask, [Nv], [0, 256]) # Value histogram
                
                # Normalize histograms
                cv2.normalize(hist_hs, hist_hs, 0, 1, norm_type=cv2.NORM_MINMAX)
                cv2.normalize(hist_v, hist_v, 0, 1, norm_type=cv2.NORM_MINMAX)
                    
                # Concatenate both histograms (weighted)
                hist = np.concatenate((hist_hs.flatten()*Nh*Ns/Nbins, hist_v.flatten()*Nv/Nbins))       
                return hist
        elif histogram ==["hv","s"]:
            Nbins = Nh*Nv + Ns # Total number of bins
            if mask is None:
                mask = cv2.inRange(image, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
                hist_hs = cv2.calcHist([image], [0, 2], mask, [Nh, Nv], [0, 181, 0, 256]) # Hue/Saturation histogram
                hist_v = cv2.calcHist([image], [1], mask, [Ns], [0, 256]) # Value histogram
                
                # Normalize histograms
                cv2.normalize(hist_hs, hist_hs, 0, 1, norm_type=cv2.NORM_MINMAX)
                cv2.normalize(hist_v, hist_v, 0, 1, norm_type=cv2.NORM_MINMAX)
                    
                # Concatenate both histograms (weighted)
                hist = np.concatenate((hist_hs.flatten()*Nh*Nv/Nbins, hist_v.flatten()*Ns/Nbins))
                return hist
            else:
                
                hist_hs = cv2.calcHist([image], [0, 2], mask, [Nh, Nv], [0, 181, 0, 256]) # Hue/Saturation histogram
                hist_v = cv2.calcHist([image], [1], mask, [Ns], [0, 256]) # Value histogram
                
                # Normalize histograms
                cv2.normalize(hist_hs, hist_hs, 0, 1, norm_type=cv2.NORM_MINMAX)
                cv2.normalize(hist_v, hist_v, 0, 1, norm_type=cv2.NORM_MINMAX)
                    
                # Concatenate both histograms (weighted)
                hist = np.concatenate((hist_hs.flatten()*Nh*Nv/Nbins, hist_v.flatten()*Ns/Nbins))
                return hist
        elif histogram ==["sv","h"]:
            Nbins = Nh*Nv + Ns # Total number of bins
            if mask is None:
                mask = cv2.inRange(image, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
                hist_hs = cv2.calcHist([image], [1, 2], mask, [Ns, Nv], [0, 256, 0, 256]) # Hue/Saturation histogram
                hist_v = cv2.calcHist([image], [0], mask, [Ns], [0, 181]) # Value histogram
                
                # Normalize histograms
                cv2.normalize(hist_hs, hist_hs, 0, 1, norm_type=cv2.NORM_MINMAX)
                cv2.normalize(hist_v, hist_v, 0, 1, norm_type=cv2.NORM_MINMAX)
                    
                # Concatenate both histograms (weighted)
                hist = np.concatenate((hist_hs.flatten()*Ns*Nv/Nbins, hist_v.flatten()*Nh/Nbins))
                return hist
            else:
                
                hist_hs = cv2.calcHist([image], [1, 2], mask, [Ns, Nv], [0, 256, 0, 256]) # Hue/Saturation histogram
                hist_v = cv2.calcHist([image], [0], mask, [Ns], [0, 181]) # Value histogram
                
                # Normalize histograms
                cv2.normalize(hist_hs, hist_hs, 0, 1, norm_type=cv2.NORM_MINMAX)
                cv2.normalize(hist_v, hist_v, 0, 1, norm_type=cv2.NORM_MINMAX)
                    
                # Concatenate both histograms (weighted)
                hist = np.concatenate((hist_hs.flatten()*Ns*Nv/Nbins, hist_v.flatten()*Nh/Nbins))
                return hist
        
           
    # print(image.shape)
    if mask is None:
        mask = cv2.inRange(image, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
        
        Nh,Ns,Nv=180,255,255 
        Nbins = Nh*Ns + Nv # Total number of bins
        hist_hs = cv2.calcHist([image], [0, 1], mask, [Nh, Ns], [0, 181, 0, 256]) # Hue/Saturation histogram
        hist_v = cv2.calcHist([image], [2], mask, [Nv], [0, 256]) # Value histogram
        
        # Normalize histograms
        cv2.normalize(hist_hs, hist_hs, 0, 1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist_v, hist_v, 0, 1, norm_type=cv2.NORM_MINMAX)
            
        # Concatenate both histograms (weighted)
        hist = np.concatenate((hist_hs.flatten()*Nh*Ns/Nbins, hist_v.flatten()*Nv/Nbins))
    else:
        Nh,Ns,Nv=180,255,255 
        Nbins = Nh*Ns + Nv # Total number of bins
        hist_hs = cv2.calcHist([image], [0, 1], mask, [Nh, Ns], [0, 181, 0, 256]) # Hue/Saturation histogram
        hist_v = cv2.calcHist([image], [2], mask, [Nv], [0, 256]) # Value histogram
        
        # Normalize histograms
        cv2.normalize(hist_hs, hist_hs, 0, 1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist_v, hist_v, 0, 1, norm_type=cv2.NORM_MINMAX)
            
        # Concatenate both histograms (weighted)
        hist = np.concatenate((hist_hs.flatten()*Nh*Ns/Nbins, hist_v.flatten()*Nv/Nbins))

     
    return hist

def comp_hist(hist1,hist2):
    """
    Compares two histograms together using the article's metric

    hist1,hist2: One dimensional numpy arrays
    return: A number
    """
    lbd = 20
    return np.exp(lbd*np.sum(hist1*hist2))

def read_video(name,directory):
    path = directory+'/' +name 
    first_frame = cv2.imread(path+'-%0*d.bmp'%(3,1))
    files_names = os.listdir(directory)
    frames_names = [ file_name for file_name in files_names if file_name[-3:]=='bmp' and file_name.find(name)!=-1 ]
    # return frames_names

    video = np.zeros(first_frame.shape + (len(frames_names),))
    for index in range(1,len(frames_names)):
        frame = cv2.imread(path+'-%0*d.bmp'%(3,index))
        
        cols, rows,_ = frame.shape
        brightness = np.sum(frame) / (255 * cols * rows)
        minimum_brightness = 0.8

        new_frame = cv2.convertScaleAbs(frame, alpha=minimum_brightness/brightness, beta=0)
        # new_frame = cv2.cvtColor(new_frame.astype('uint'), cv2.COLOR)
        video[:,:,:,index-1] = new_frame
    return video

def read_masks(name,directory):
    path = directory+'/' +name 
    first_frame = cv2.imread(path+'-%0*d.png'%(3,1))
    files_names = os.listdir(directory)
    frames_names = [ file_name for file_name in files_names if file_name[-3:]=='png' and file_name.find(name)!=-1 ]
    masks = np.zeros(first_frame.shape + (len(frames_names),))
    for index in range(1,len(frames_names)):
        frame = cv2.imread(path+'-%0*d.png'%(3,index))
        masks[:,:,:,index-1] = frame
    return masks

def gt_centroid(mask):
    x_list, y_list, _ = (mask == 255).nonzero()
    x_gtcentroid, y_gtcentroid = x_list.mean(), y_list.mean()
    return x_gtcentroid,y_gtcentroid

def get_size(mask):
    x_list, y_list, _ = (mask == 255).nonzero()
    x_edge = (np.max(x_list)-np.min(x_list))/2
    y_edge = (np.max(y_list)-np.min(y_list))/2
    if abs(x_edge-y_edge) >10:
        return x_edge,y_edge
    else:
        return max(x_edge, y_edge),max(x_edge, y_edge)
    
class ParticleFilter(object):
    def __init__(self,x,y,first_frame, first_mask,n_particles=1000,dt=0.04,sizes=(20,40), histogram=["h"]):
        self.n_particles = n_particles
        self.n_iter = 0
        self.state = np.array([x,y,sizes[0], sizes[1]]) 
        self.histogram = histogram
        # state =[X[t],Y[t],S[t],X[t-1],Y[t-1],S[t-1]]
        self.std_state = np.array([15,15,0,0])
        self.window_size = first_frame.shape
        
        self.max_square = self.window_size[0]*0.5
        self.min_square = self.window_size[0]*0.1

        self.A = np.array([[1+dt,0,0,0],
                           [0,1+dt,0,0],
                           [0,0,1+dt/4,0],
                           [0,0,0,1+dt/4]])


        self.B = np.array([[-dt,0,0,0],
                           [0,-dt,0,0],
                           [0,0,-dt/4,0],
                           [0,0,0,-dt/4]])


        self.particles = init_particles(self.state,n_particles)
        self.last_particles = np.array(self.particles)             
        self.hist = calc_hist(first_frame, first_mask, histogram)
        self.start_hist = self.hist
        
        
     
    def next_state(self,frame):       
      
        control_prediction = self.transition()
        control_prediction = self.filter_borders(control_prediction)
       
        hists = self.candidate_histograms(control_prediction,frame)

        weights = self.compare_histograms(hists,self.start_hist)
        self.last_particles = np.copy(self.particles)
        self.particles = self.resample(control_prediction,weights)
        self.state = np.mean(self.particles,axis=0)
      
        self.last_frame = np.array(frame)
        # self.n_iter += 1
        self.hist = calc_hist(get_view(frame,self.state[0],self.state[1],self.state[2], self.state[3]), histogram = self.histogram)
        

        
        return int(self.state[0]),int(self.state[1]),int(self.state[2]),int(self.state[3]) ,self.particles,control_prediction
        
        
    def transition(self):

        n_state = self.state.shape[0]
        noises = self.std_state*np.random.randn(self.n_particles,n_state)
        particles = np.dot(self.particles,self.A) + np.dot(self.last_particles,self.B) + noises
        return particles

    def candidate_histograms(self,predictions,image):
        "Compute histograms for all candidates"
        hists = [] 

        for x in predictions:
            v = get_view(image,x[0],x[1],x[2], x[3])
            hists.append(calc_hist(v, histogram = self.histogram))
        return hists
        
    def compare_histograms(self,hists,last_hist):
        "Compare histogram of current (last) histogram and all candidates"
        weights = np.array(list(map(lambda x: comp_hist(x,last_hist),hists)))
        return weights/np.sum(weights)

    def resample(self,predictions,weights):
        "Scatter new particles according to the weights of the predictions"
        indexes = np.arange(weights.shape[0])
        inds = np.random.choice(indexes,self.n_particles,p=weights)
        return predictions[inds]
    def filter_borders(self,predictions):  
        "Remove candidates that will not have the correct square size."
        np.clip(predictions[:,0],self.state[2]+1,self.window_size[0]-(1+self.state[2]),predictions[:,0])        
        np.clip(predictions[:,1],self.state[3]+1,self.window_size[1]-(1+self.state[3]),predictions[:,1])
        np.clip(predictions[:,2],self.min_square,self.max_square,predictions[:,2])
        
        return predictions
    
    
# masks = read_masks('bag',r'C:\Users\achraf\Desktop\IMT Atlantique\3A\computer vision\sequences-train' )
# gt_centroid(masks[:,:,:,0])