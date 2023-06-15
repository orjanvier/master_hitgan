import cv2
import numpy as np
import os
from PIL import Image
from scipy import ndimage
import config3D


class Preprocess(object):
    def __init__(self, Original_PATH, Output_Path, Patients_paths=None,resize=False, new_size=None):
        '''

        :param Patient_Path: String Path for Patient
        :param Output_Path: String Path for saving new images
        :param resize:  Boolean for resizing
        :param new_size: Int for new image shapes
        '''
        #self.Folder_Names = ['Original', 'Mirror', 'Full_Left', 'Full_Right']
        #self.Folder_Names = ['Mirrored']
        #self.Folder_Names = ['Original', 'Mirrored']
        self.Folder_Names=['Original']
        self.resize = resize

        if new_size is None:
            self.new_size=512
        else:
            self.new_size = new_size

        self.orig_path=Original_PATH
        self.outputPath = Output_Path
        self.CustomPatientsPaths=Patients_paths

    def get_Specific_patients(self,number):
        '''
        Function for get specific patient list
        :param number: Patient with fixed slice numbers
        :return:list of patients Id
        '''

        p_list=[]
        assert(self.CustomPatientsPaths is not None),'Patient path is none'
        for path in self.CustomPatientsPaths:
            with open(path, 'r') as file:
                text = file.read()
            all_content = text.replace(' ', '').replace('\n', '').split(',')
            p_number = int(all_content[0])
            if p_number == number:
                p_list = p_list + all_content[1:]
        return p_list


    def start1(self):
        if self.CustomPatientsPaths is not None:
            Patient_list = self.get_Specific_patients(config3D.SLICE_NUMBER)
        else:
            Patient_list = os.listdir(self.orig_path)
        self.outputPath = self.outputPath + '/' + f'Size_{self.new_size}'

        if not os.path.exists(self.outputPath):
            for Patient in Patient_list:
                print(f'Patient is {Patient}')
                Patient_Path=self.orig_path+'/'+Patient
                self.input_path=Patient_Path
                self.slices=os.listdir(Patient_Path)
                self.Patient_Id=Patient_Path.split('/')[-1]
                self.augment_folder()
        else:
            print(f'Preprocessing for {self.new_size} size was done before ')

    def start(self):
        Patient_list = os.listdir(self.orig_path)
        if self.resize:
            self.Folder_Names = ['Original', 'Mirrored']
            self.outputPath=self.outputPath+'/'+f'Resize_{self.new_size}'
        for Patient in Patient_list:
            print(f'Patient is {Patient}')
            Patient_Path=self.orig_path+'/'+Patient
            self.input_path=Patient_Path
            self.slices=os.listdir(Patient_Path)
            self.Patient_Id=Patient_Path.split('/')[-1]
            self.augment_folder()



    def resize_images(self, image):
        '''
        Resize images by using scipy ndimage
        :param image:  original image
        :return: Resized image
        '''

        ratio = self.new_size / image.shape[0]
        new_image = ndimage.interpolation.zoom(image, ratio)
        return new_image

    def augment_folder(self):
        '''
        Function for augment all the folders
        '''

        for slice in self.slices:
            Slice_path=self.input_path+'/'+slice

            augments=[]
            for i in os.listdir(Slice_path):
                image_path=Slice_path+'/'+i
                image=cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if self.resize:
                    image=self.resize_images(image)
                all_augment=self.augment_image(image)
                augments.append(all_augment)
            self.save_images(slice,augments)


    def augment_image(self, image):
        '''
        Function for create 4 different image from 1 image
        :param image:
        :return: Array of augmented images
        First index:  Original image
        Second index: Mirrored image
        Third index:  Image that created from left side of original image
        Fourth index: Image that created from right side of original image
        '''

        #points=self.find_points(image)
        #middle_col=self.input.shape[1]//2

        #middle_col=points[0][0]+(points[1]//2)
        #left_half=image[:,:middle_col]
        #right_half=image[:,middle_col:]


        mirror_image=np.fliplr(image)

        #full_left=np.concatenate([left_half, np.fliplr(left_half)], axis=1)
        #full_right=np.concatenate([np.fliplr(right_half), right_half], axis=1)

        #alls= [image, mirror_image, full_left, full_right]

        # if self.resize:
        #         #     alls=[image,mirror_image]
        #         # else:
        #         #     alls=[mirror_image]

        alls=[image]
        #alls=[image, mirror_image]

        final=self.Check_size(alls)

        final=np.stack(final)

        return final

    def find_points(self, image):

        '''
        Function for finding starting index of brain
        :param image: Original image
        :return: Tuple(top_left, width, height)
        '''
        rows=np.argwhere(image)[:,0]
        cols=np.argwhere(image)[:,1]

        bottom_border=np.max(rows)
        top_border=np.min(rows)

        left_border=np.min(cols)
        right_border=np.max(cols)

        top_left=(left_border, top_border)
        bottom_right=(right_border, bottom_border)

        width=right_border-left_border
        height=bottom_border-top_border

        return (top_left, width, height)

    def save_images(self, slice,augments):
        '''
        Function for saving the arrays as image
        :param slice: Number of slice
        :param augments: Images
        '''
        augments=np.array(augments)

        for i,name in enumerate(self.Folder_Names):
            #new_path=self.outputPath+'/'+self.Patient_Id+'/'+slice+'/'+name
            new_path = self.outputPath + '/' + name+'/' + self.Patient_Id + '/' + slice
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            for j,image in enumerate(augments[:,i]):
                name=f'{new_path}/{str(j+1).zfill(2)}.tiff'
                im = Image.fromarray(image)
                im.save(name)
                #scipy.misc.toimage(image, cmin=0.0, cmax=...).save(name)
                #cv2.imwrite(name, image, )

    def Check_size(self, images):
        '''
        Function for checking the size of new images
        :param images: Created Images
        :return: Created images with same size as original images
        '''

        orig_image_shape=images[0].shape
        for i in range(1,len(images)):
            aug_shape=images[i].shape
            if aug_shape!=orig_image_shape:
                images[i]=self.change_size(images[0], images[i])

        return images

    def change_size(self, orig_image, aug_image):
        '''
        Functiion for change the size of images
        :param orig_image: Original image
        :param aug_image: Created image that has different size
        :return: New image with the same size
        '''
        difference=aug_image.shape[1]-orig_image.shape[1]
        if difference>0:
            aug_image=self.cut_edge(aug_image, difference)
        else:
            aug_image=self.padding(aug_image, difference)

        return aug_image


    def cut_edge(self, input, difference):
        '''
        FUnction for cropping image size if the created image size is larger than the original one

        :param input: Created image
        :param difference: Int
        :return: new image with same size
        '''

        mid_col = input.shape[1] // 2
        if difference%2==0:
            half_dif=difference//2
            aug_image=np.concatenate([input[:,half_dif:mid_col], input[:,mid_col:(input.shape[1])-half_dif]], axis=1)
        else:
            first=int(difference/2)+1
            last=difference-first
            aug_image=np.concatenate([input[:,first:mid_col],input[:,mid_col:(input.shape[1])-last]], axis=1)
        return aug_image

    def padding(self, input, difference):
        '''
        FUnction for padding image size if the created image size is smaller than the original one

        :param input: Created image
        :param difference: Int
        :return: new image with same size
        '''
        new_shape=input.shape[1]+np.abs(difference)
        shape = input.shape
        try:
            row_padding=(new_shape-shape[0])//2
            col_padding=(new_shape-shape[1])//2

            final_img=np.pad(input,((row_padding, row_padding),(col_padding, col_padding)), 'constant')
        except:
            print("Error")
            return

        return final_img




