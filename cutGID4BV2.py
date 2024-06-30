import os
import struct
import numpy as np
import torch
from osgeo import gdal
from PIL import Image
import shutil
import random
import json

def save_geotiff(output, H, V, B, filename, geotransform=None, projection=None):
    driver = gdal.GetDriverByName('GTiff')
    if driver is None:
        raise RuntimeError("El driver GTiff no está disponible.")
    
    dataset = driver.Create(filename, H, V, B, gdal.GDT_UInt16)
    if dataset is None:
        raise RuntimeError("No se pudo crear el archivo GeoTIFF.")

    if geotransform is not None:
        dataset.SetGeoTransform(geotransform)
    
    if projection is not None:
        dataset.SetProjection(projection)

    for i in range(B):
        band = dataset.GetRasterBand(i + 1)
        band.WriteArray(output[:, :, i])
    
    dataset.FlushCache()
    dataset = None
    print(f'* Saved GeoTIFF: {filename}')


def process_folders(train_images_folder, val_images_folder, train_annotations_folder, val_annotations_folder, base_path):
    all_patches = []
    all_patches_gt = []
    
    print("Processing training images...")
    all_patches += process_individual_folder(train_images_folder, train_annotations_folder, 'training', base_path)
    
    print("Processing validation images...")
    all_patches += process_individual_folder(val_images_folder, val_annotations_folder, 'validation', base_path)

    return all_patches

def process_individual_folder(images_folder, annotations_folder, folder_type, base_path):
    image_files = [f for f in os.listdir(images_folder) if f.endswith('.tiff')]
    gt_files = {f[:-12]: f for f in os.listdir(annotations_folder) if f.endswith('_24label.png')}
    all_patches = []
    all_patches_gt = []
    print(f"Images ({folder_type}): ", image_files)
    print(f"Annotations ({folder_type}): ", gt_files)
    for image_file in image_files:
        base_name = image_file[:-5]
        gt_file = gt_files.get(base_name)
        if gt_file:
            image_path = os.path.join(images_folder, image_file)
            gt_path = os.path.join(annotations_folder, gt_file)

            img_tiff, w, h, b = read_tiff(image_path)
            print(img_tiff.shape)
            img_png, w_gt, h_gt = read_png(gt_path)

            patches, patches_gt = cut_into_patches(img_tiff, img_png, max_size=500)
            all_patches.extend(patches)
            all_patches_gt.extend(patches_gt)
    save_patches(all_patches, all_patches_gt, base_path, 500, 500, folder_type)
    return all_patches

def write_odgt(entries, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        for entry in entries:
            json.dump(entry, f)
            f.write('\n')

def vaciar_carpeta(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Error al eliminar %s. Razón: %s' % (file_path, e))

def read_tiff(fichero):
    dataset = gdal.Open(fichero)
    if dataset is None:
        raise Exception("No se pudo abrir el archivo TIFF.")

    B = dataset.RasterCount
    H = dataset.RasterXSize
    V = dataset.RasterYSize

    datos = np.zeros((V, H, B), dtype=np.float32)
    for i in range(B):
        banda = dataset.GetRasterBand(i + 1)
        datos[:, :, i] = banda.ReadAsArray().astype(np.float32)
    
    dataset = None
    datos = 65535 * (datos - datos.min()) / (datos.max() - datos.min())
    datos = datos.astype(np.uint16)

    return datos, H, V, B

def save_raw(output, H, V, B, filename):
    try:
        f = open(filename, "wb")
    except IOError:
        print('No puedo abrir ', filename)
        exit(0)
    else:
        f.write(struct.pack('i', B))
        f.write(struct.pack('i', H))
        f.write(struct.pack('i', V))
        output = output.reshape(H * V * B)
        for i in range(H * V * B):
            f.write(struct.pack('H', np.uint16(output[i])))
        f.close()
        print('* Saved file:', filename)
    
def read_png(fichero):
    try:
        dataset = gdal.Open(fichero)
    except RuntimeError:
        print('No puedo abrir ', fichero)
    else:
        H = dataset.RasterXSize
        V = dataset.RasterYSize
        B = dataset.RasterCount

        datos = np.zeros((B, V, H), dtype=np.uint8)
        for i in range(B):
            banda = dataset.GetRasterBand(i + 1)
            datos[i, :, :] = banda.ReadAsArray().astype(np.uint8)

        if datos.ndim == 3 and datos.shape[0] == 1:
            datos = Image.fromarray(datos[0].astype(np.uint8))
        else:
            datos = Image.fromarray(datos.astype(np.uint8))
        print('* Read GT:', fichero)
        print('  H:', H, 'V:', V, 'B:', B)
        print('  Read:', datos.size)

        return datos, H, V

def cut_into_patches(datos, gt, max_size=600):
    V, H, B = datos.shape

    patches = []
    patches_gt = []

    for i in range(0, V, max_size):
        for j in range(0, H, max_size):
            patch = datos[i:i+max_size, j:j+max_size, :]
            patch_gt = gt.crop((j, i, j + max_size, i + max_size))
            if patch.shape[0] < max_size or patch.shape[1] < max_size:
                padded_patch = np.zeros((max_size, max_size, B), dtype=patch.dtype)
                padded_patch[:patch.shape[0], :patch.shape[1], :] = patch
                patch = padded_patch
            if patch_gt.size[1] < max_size or patch_gt.size[0] < max_size:
                padded_patch_gt = Image.new("L", (max_size, max_size))
                padded_patch_gt.paste(patch_gt, (0, 0))
                patch_gt = padded_patch_gt
            patches.append(patch)
            patches_gt.append(patch_gt)

    return patches, patches_gt

def save_patches(patches, patches_gt, base_path, patch_height, patch_width, folder_type, patch_bands=4):
    image_folder_path = os.path.join(base_path, 'images', folder_type)
    annotation_folder_path = os.path.join(base_path, 'annotations', folder_type)
    vaciar_carpeta(image_folder_path)
    vaciar_carpeta(annotation_folder_path)
    
    entries = []
    for i, (patch, patch_gt) in enumerate(zip(patches, patches_gt)):
        patch_path = os.path.join(image_folder_path, f'imagen_patch_{i}.tif')
        patch_gt_path = os.path.join(annotation_folder_path, f'imagen_patchgt_{i}.png')
        
        save_geotiff(patch, patch_height, patch_width, patch_bands, patch_path)
        patch_gt.save(patch_gt_path, 'PNG')

        entry = {
            "fpath_img": os.path.join("GIDpatches", os.path.relpath(patch_path, start=base_path)),
            "fpath_segm": os.path.join("GIDpatches", os.path.relpath(patch_gt_path, start=base_path)),
            "width": patch_width,
            "height": patch_height
        }
        entries.append(entry)

    write_odgt(entries, os.path.join('data/', f'{folder_type}.odgt'))

if __name__ == "__main__":
    base_path = 'data/GIDpatches'
    train_images_folder = '/home/jesus/Escritorio/TFG/GID/DatasetV5/Train'
    val_images_folder = '/home/jesus/Escritorio/TFG/GID/DatasetV5/Val'
    train_annotations_folder = '/home/jesus/Escritorio/TFG/GID/24ClassAnnotations'
    val_annotations_folder = '/home/jesus/Escritorio/TFG/GID/24ClassAnnotations'
    all_patches = process_folders(train_images_folder, val_images_folder, train_annotations_folder, val_annotations_folder, base_path)
