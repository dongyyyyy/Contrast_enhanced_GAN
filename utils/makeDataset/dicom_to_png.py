from include.header import *
from pydicom.pixel_data_handlers.util import apply_voi_lut
import pydicom
import multiprocessing
from multiprocessing import Process, Manager, Pool, Lock

def dicom_to_png(arg_list):
    save_path = arg_list[1]
    img_path = arg_list[0]
    no_of_picture = os.listdir(img_path)

    os.makedirs(save_path, exist_ok=True)

    for name in no_of_picture:
        current_path = img_path + name
        current_save_path = save_path + current_path.split('/')[-3]+'/'
        os.makedirs(current_save_path,exist_ok=True)
        window_center = -600
        window_width = 1600

        slice = pydicom.read_file(current_path)
        s = int(slice.RescaleSlope)
        b = int(slice.RescaleIntercept)
        image = s * slice.pixel_array + b

        slice.WindowCenter = window_center
        slice.WindowWidth = window_width
        image2 = apply_voi_lut(image, slice)

        plt.axis('off')
        plt.imshow(image2, cmap='gray')
        filename = current_path.split('/')[-1].split(' ')[-1].split('.dcm')[0][-4:] +'.png'
        plt.savefig(current_save_path+filename)
        plt.cla()
        plt.clf()


def make_dataloader_dataset():
    # print(file_list)
    cpu_num = multiprocessing.cpu_count()
    print('cpu_num : ', cpu_num)
    arg_list = []
    path = 'G:/Contrast-enhanced Problem/Nect CT/'
    enhance_save_path = 'G:/Contrast-enhanced Problem/Nect CT Png/enhanced_png/'
    none_enhance_save_path = 'G:/Contrast-enhanced Problem/Nect CT Png/none_enhanced_png/'

    patient_list = os.listdir(path)

    enhance_patient_list = [path + filename + '/' + '%s axi ce/' % filename for filename in patient_list]
    none_enhance_patient_list = [path + filename + '/' + '%s axi n/' % filename for filename in patient_list]

    for i in range(len(enhance_patient_list)):
        if enhance_patient_list[i].split('/')[-2][0] != '~':
            arg_list.append([enhance_patient_list[i], enhance_save_path])
    for i in range(len(none_enhance_patient_list)):
        if enhance_patient_list[i].split('/')[-2][0] != '~':
            arg_list.append([none_enhance_patient_list[i], none_enhance_save_path])
    pool = Pool(cpu_num)

    pool.map(dicom_to_png, arg_list)
    pool.close()
    pool.join()


# if __name__ == '__main__':
#     make_dataloader_dataset()
