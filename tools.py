from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import cv2
from skimage.feature import hog

def spatial_binning_color(img, features_size):
    return cv2.resize(img, (features_size, features_size)).ravel()

class ColorBinner(BaseEstimator, TransformerMixin):
    def __init__(self, features_size=1, color_space="RGB"):
        self.features_size = features_size
        self.color_space = color_space
        
    def fit(self, X, y):
        return self

    def feature_vector(self, img):
        if self.color_space == "RGB":
            pass
        elif self.color_space == "HSV":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif self.color_space == "LAB":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
        elif self.color_space == "YUV":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif self.color_space == "HLS":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        return spatial_binning_color(img, self.features_size)
    
    def transform(self, X):
        return [(img, np.append(vec, self.feature_vector(img))) for (img, vec) in X]

class ColorHistogram(BaseEstimator, TransformerMixin):
    def __init__(self, bins=2, color_space="RGB"):
        self.bins = bins
        self.color_space = color_space
        
    def fit(self, X, y):
        return self

    def feature_vector(self, img):
        if self.color_space == "RGB":
            pass
        elif self.color_space == "HSV":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif self.color_space == "LAB":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
        elif self.color_space == "YUV":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif self.color_space == "HLS":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        
        rhist = np.histogram(img[:,:,0], bins=self.bins, range=(0, 256))
        ghist = np.histogram(img[:,:,1], bins=self.bins, range=(0, 256))
        bhist = np.histogram(img[:,:,2], bins=self.bins, range=(0, 256))
        
        return np.concatenate((rhist[0], ghist[0], bhist[0]))
    
    def transform(self, X):
        return [(img, np.append(vec, self.feature_vector(img))) for (img, vec) in X]

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features    
    
class HOGFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, color_space="RGB", hog_channel='ALL', orient=9, pix_per_cell=8, cell_per_block=2):
        self.color_space = color_space
        self.hog_channel = hog_channel
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block

    def fit(self, X, y):
        return self
        
    def feature_vector(self, img):
        if self.color_space == "RGB":
            pass
        elif self.color_space == "HSV":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif self.color_space == "LAB":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
        elif self.color_space == "YUV":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif self.color_space == "HLS":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

        if self.hog_channel == 'ALL':
            hog_features = []
            for channel in range(img.shape[2]):
                hog_features.append(get_hog_features(img[:,:,channel], 
                                    self.orient, self.pix_per_cell, self.cell_per_block, 
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)        
        else:
            hog_features = get_hog_features(feature_image[:,:,self.hog_channel], self.orient, 
                        self.pix_per_cell, self.cell_per_block, vis=False, feature_vec=True)
        return hog_features

    def transform(self, X):
        return [(img, np.append(vec, self.feature_vector(img))) for (img, vec) in X]
    
    
class ImageRemover(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, X, y):
        return self
    
    def transform(self, X):
        return np.vstack([vec for (_, vec) in X])
    
# Define a function to draw windows
def draw_windows(img, windows, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the windows
    for window in windows:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, window[0], window[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def get_heatmap(img, bboxes):
    heatmap = np.zeros_like(img[:,:,0]).astype(np.float)
    for ((xstart, ystart), (xend, yend)) in bboxes:
        # Add some weight for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[ystart:yend, xstart:xend] += 1 / (100 + xend - xstart)

    return heatmap

def draw_labeled_bboxes(img, labels):
    imcopy = img.copy()
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(imcopy, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return imcopy

def find_bounding_boxes(img, clf, window_size, x_range=[None, None], y_range=[None, None], pix_per_cell=8, cell_per_block=3, orient=9):
    xstart = x_range[0] if x_range[0] is not None else 0
    xend = x_range[1] if x_range[1] is not None else img.shape[1]
    ystart = y_range[0] if y_range[0] is not None else 0
    yend = y_range[1] if y_range[1] is not None else img.shape[0]

    search_region_img = img[ystart:yend, xstart:xend]
    resized_img = cv2.resize(
        search_region_img, 
        (
            (search_region_img.shape[1] * 64 // window_size + 7) // 8 * 8, 
            (search_region_img.shape[0] * 64 // window_size + 7) // 8 * 8
        )
    )
        
    hog_trans = cv2.cvtColor(resized_img, cv2.COLOR_RGB2YUV)
    
    nxblocks = (hog_trans.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (hog_trans.shape[0] // pix_per_cell) - cell_per_block + 1 
    
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 1  # Instead of overlap, define how many cells to step
    
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    hog1 = get_hog_features(hog_trans[..., 0], orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(hog_trans[..., 1], orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(hog_trans[..., 2], orient, pix_per_cell, cell_per_block, feature_vec=False)

    result = []
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = resized_img[ytop:ytop+window, xleft:xleft+window]
            prediction = clf.predict_proba([(subimg, hog_features)])
            if(prediction[0][1] > 0.5):
                xleft_orig = xleft * window_size // 64 + xstart
                ytop_orig = ytop * window_size // 64 + ystart
                result.append(((xleft_orig, ytop_orig), (xleft_orig + window_size, ytop_orig + window_size)))
    return result

y_limits = {
    48: [384, 528],
    64: [384, 544],
    96: [384, 576],
    128: [320, 576],
    160: [320, 640]
}
