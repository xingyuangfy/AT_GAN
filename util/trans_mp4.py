"""
Video format conversion utility for AT-GAN
Copyright (c) Xingyuangfy 2025. All rights reserved.

This module provides functionality for video format conversion and resizing.
"""

import numpy as np
import cv2 as cv
import imageio as iio
from PIL import Image

def video_trans_size(input_mp4, output_h264):
    cap = cv.VideoCapture(input_mp4)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    
    print(width,height)
    
    # Define codec and create VideoWriter object
    out = iio.get_writer(output_h264, format='ffmpeg', mode='I', fps=25, codec='libx264', 
                        pixelformat='yuv420p', macro_block_size=None)
    while(True):
        ret, frame = cap.read()
        if ret is True:
            # Convert BGR to RGB
            image = frame[:, :, (2, 1, 0)]
            # Write the flipped frame
            out.append_data(image)
            
            if cv.waitKey(1) == ord('q'):
                break
        else:
            break
    # Release everything when job is finished
    cap.release()
    out.close()
    cv.destroyAllWindows()


if __name__ == '__main__':
    video_trans_size('tmp.mp4', 'tmpx.mp4')

