import json
import os
import numpy as np
import cv2
import color_uti as cu
import matplotlib.pyplot as plt
from scipy.stats import entropy

def gen_roi_mask(img, shape, location):
    ## convert location to numpy number array
    location = np.array(location.split(',')).astype(float)
    ## generate mask with zeros
    mask = np.zeros((img.shape[0], img.shape[1], 3), dtype = float)
    ## draw shape
    ## center points = x, y; size = width, height; 
    center_x = int(img.shape[1] * location[0])
    center_y = int(img.shape[0] * location[1])
    width = int(img.shape[1] * location[2])
    height = int(img.shape[0] * location[3])
    
    if 'rect' in shape:
        cv2.rectangle(mask, (center_x, center_y), (center_x + width, center_y + height), color = (1.0, 1.0, 1.0), thickness = -1)
        ## add rotation is needed
    if 'ellipse' in shape:
        cv2.ellipse(mask, (center_x, center_y), (width, height), 0, 0, 360, (1.0, 1.0, 1.0), -1)
        ## add rotation is needed
    return mask

def parse_img_name(img_name):
    ## parse img name
    img_name_list = img_name.split('_')
    project = img_name_list[0]
    case_index = img_name_list[1]
    timestamp = img_name_list[2]
    
    return project, case_index, timestamp


def crop_roi(img_name, index, shape, location):
    ## load image
    img = cv2.imread('imgs/' +img_name)
    ## gen roi binary mask
    mask = gen_roi_mask(img, shape, location)
    ## element wise mask
    img_masked = np.where(mask, img, 0)
    ##img_masked = np.multiply(mask, img)
    
    ## get pixel elements and mean std etc...
    ## convert roi mean to rgb
    roi_size_per_channel = mask[:, :, 0].sum()
    roi = np.zeros((int(roi_size_per_channel), 3))
    for c in range(3):
        ## use masked array to get valid roi pixels
        roi_ = np.ma.masked_array(img[:, :, c], mask[:, :, c] - 1.0)
        ## get only valid values
        roi[:,c] = roi_[~roi_.mask]
    
    plot_hist(roi, img_name, index)
    
    
    ## add hpf
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1,0, ksize = 3, scale=1)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0,1, ksize = 3, scale=1)
    absx= cv2.convertScaleAbs(dx)
    absy = cv2.convertScaleAbs(dy)
    edge = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)

    edge_masked = np.where(mask[:,:,0], edge, 0)
    roi_edge = np.ma.masked_array(edge, mask[:, :, 0] - 1.0)
    roi_edge = roi_edge[~roi_edge.mask]
    
    roi_edge_ave = np.mean(roi_edge)

    ## debug info
    if 1:
        cv2.imwrite('debug/' + img_name.replace('.jpg','') + '_debug_roi' + str(index + 1) + '.jpg', img_masked.astype(np.uint8))
        #cv2.imwrite('debug/' + img_name.replace('.jpg','') + '_debug_roi_hpf' + str(index + 1) + '.jpg', edge_masked.astype(np.uint8))
        
    #cv2.namedWindow('img ', cv2.WINDOW_NORMAL)
    #cv2.imshow('img ', img)
    #cv2.waitKey()
    #
    #
    #cv2.namedWindow('mask ', cv2.WINDOW_NORMAL)
    #cv2.imshow('mask ', mask)
    #cv2.waitKey()

    
    return roi, roi_edge_ave
    

def plot_hist(roi, img_name, index):

    ## plot histogram of ROIs
    n, bins, patches = plt.hist(roi[:, 0], 50, density = True, facecolor = 'b', alpha = 0.3, label = 'blue')
    n, bins, patches = plt.hist(roi[:, 1], 50, density = True, facecolor = 'g', alpha = 0.3, label = 'green')
    n, bins, patches = plt.hist(roi[:, 2], 50, density = True, facecolor = 'r', alpha = 0.3, label = 'red')
    plt.xlabel('pixel value')
    plt.ylabel('Probability')
    plt.title('Hist of ' + img_name + ' roi' + str(index))
    plt.text(60, .025, r' ')
    plt.xlim(0, 255)
    plt.ylim(0, 0.1)
    plt.legend(loc = 'upper right')
    #plt.grid(True)
    #plt.savefig('debug/' + img_name.replace('.jpg','') + '_debug_roi' + str(index + 1) + 'hist.jpg')
    plt.close()
   
def cal_stats(roi):    
    ## compute mean, std and etc.
    roi_sRGB_mean = np.array([roi[:, 2].mean(), \
                         roi[:, 1].mean(), \
                         roi[:, 0].mean()] )
    roi_Lab_mean = cu.sRGB2Lab(roi_sRGB_mean)
    roi_LCH_mean = cu.Lab2LCh(roi_Lab_mean)
    
    roi_sRGB_std = np.array([roi[:, 2].std(), \
                         roi[:, 1].std(), \
                         roi[:, 0].std()] )
    
    
    roi_sRGB_entropy = np.zeros((3,1))
    for c in range(3):
        value, counts = np.unique(roi[:, 2 - c].flatten(), return_counts = True)
        roi_sRGB_entropy[c] = entropy(counts, base = 2)
        
        
        
    
    #print('roi_sRGB_entropy ', roi_sRGB_entropy)
    #print('roi_red   mean ', "{:.2f}".format(roi[:, 2].mean()))
    #print('roi_green mean ', "{:.2f}".format(roi[:, 1].mean()))
    #print('roi_blue  mean ', "{:.2f}".format(roi[:, 0].mean()))
    
    
    return roi_sRGB_mean, roi_Lab_mean, roi_LCH_mean, roi_sRGB_std, roi_sRGB_entropy

    
def dump_roi_results(roi_results):
    
    roi_results_dict = {}
    roi_results_dict['img_name'] = roi_results.file
    #roi_results_dict['roi'] = roi_results.index
    #
    roi_results_dict['roi_results'] = {}
    roi_results_dict['roi_results'] = roi_results.roi_results
    #
    #roi_results_dict['roi_results']['basic_sRGB'] = roi_results.roi_sRGB.flatten().tolist()
    #roi_results_dict['roi_results']['basic_Lab'] = roi_results.roi_Lab.flatten().tolist()
    #roi_results_dict['roi_results']['basic_LCH'] = roi_results.roi_LCH.flatten().tolist()
    #roi_results_dict['roi_results']['basic_sRGB_std'] = roi_results.roi_sRGB_std.flatten().tolist()
    
    ##roi_results_dict['roi_results']['fre_hpf_sobel_abs_ave'] = self.roi_results_fre_hpf_sobel_abs_ave
    ##roi_results_dict['roi_results']['fre_dwt_haar_dig_5'] = self.roi_results_fre_dwt_haar_dig_5
    #### convert nd arrays to 1d
    ##roi_results_dict['roi_results']['fre_dft_psd'] = self.roi_results_fre_dft_psd.flatten().tolist()
    ##roi_results_dict['roi_results']['fre_dft_fre'] = self.roi_results_fre_dft_fre.flatten().tolist()
    
    with open('results/' + roi_results.file + '.json', "w") as outfile:
        #roi_results_dict = json.dumps(roi_results_dict)
        json.dump(roi_results_dict, outfile, indent = 4)