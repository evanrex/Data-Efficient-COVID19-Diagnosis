import pydicom as dicom
import os
from PIL import Image
from datetime import date, datetime
import numpy as np


# rootdir = os.getcwd()
SAVE_PATH = "/home/evan/NLST_jpg/" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
filedir = '/mnt/data1/nbia/nlst/manifest-NLST_allCT/NLST'


def renameNotDICOM(path):
    new_path = path.replace('dcm','dcmNONCONFORM')
    os.rename(path,new_path)


def handle_image(path,n):
    ds = dicom.dcmread(path, force=True)
    ds.file_meta.TransferSyntaxUID = dicom.uid.ImplicitVRLittleEndian

    try:

        # get numpy array of pixel values
        A = ds.pixel_array
    except Exception:
        print("Failed to get numpy data, marking file as dcmNONCONFORM  and moving on to next.")
        renameNotDICOM(path)
        return None
    
    # get numpy array of pixel values
    A = ds.pixel_array

    # scaling
    A = A-A.min()
    A = A/A.max() * 255.0

    # Convert to uint
    A = np.uint8(A)
    
    # convert to PIL dataype for saving
    im = Image.fromarray(A)

    # save
    save_path = SAVE_PATH+'_{}.jpg'.format(str(n))
    im.save(save_path)
    os.remove(path)


def main():
    print(os.getcwd())
    print(filedir)
    print(datetime.now())
    n=1
    chkpt = 1
    for subdir,dirs,files in os.walk(filedir):
        for file in files:
            if file.endswith('dcm'):

                path = os.path.join(subdir,file)  
                # print(path)
                handle_image(path,n)
                # os.remove(path)
                if n==chkpt:
                    print(datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+": {} converted!".format(n))
                    if chkpt >= 1000000:
                        chkpt += 1000000
                    else:
                        chkpt*=10
                n+=1


if __name__ == "__main__":
    main()