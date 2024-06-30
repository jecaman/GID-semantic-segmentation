import json
import os
import numpy as np
from osgeo import gdal
from PIL import Image
import subprocess
import tempfile
from mit_semseg.lib.utils import as_numpy
import torch
from mit_semseg.utils import AverageMeter, colorEncode, accuracy, intersectionAndUnion, parse_devices, setup_logger
from scipy.io import loadmat


colors = loadmat('data/color5.mat')['colors']

def load_full_ground_truth(info, base_path):
    img_name = os.path.basename(info)
    gt_name = img_name.replace('.tiff', '_24label.png')
    gt_path = os.path.join(base_path, gt_name)
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth not found at path: {gt_path}")
    ground_truth = Image.open(gt_path).convert('L')
    return np.array(ground_truth)

def read_gt(fichero):
    image = Image.open(fichero)
    assert(image.mode == "L")
    gray_image = np.array(image)
    gray_image = np.where(gray_image==0, -1, gray_image-1)
    gray_image = torch.from_numpy(np.array(gray_image)).long()
    seg=as_numpy(gray_image)
    
    return seg

def write_odgt(entries, file_path):
    """
    Escribe entradas en un archivo .odgt en formato JSON.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        for entry in entries:
            json.dump(entry, f)
            f.write('\n')

def save_geotiff(output, H, V, B, filename, geotransform=None, projection=None):
    """
    Guarda una imagen GeoTIFF a partir de una matriz numpy.

    :param output: Matriz numpy con los datos de la imagen.
    :param H: Ancho de la imagen.
    :param V: Altura de la imagen.
    :param B: Número de bandas.
    :param filename: Nombre del archivo de salida.
    :param geotransform: (opcional) Lista de 6 elementos para la geotransformación.
    :param projection: (opcional) Proyección WKT.
    """
    # Crear el driver para el archivo GeoTIFF
    driver = gdal.GetDriverByName('GTiff')
    if driver is None:
        raise RuntimeError("El driver GTiff no está disponible.")
    
    # Crear el dataset
    dataset = driver.Create(filename, H, V, B, gdal.GDT_UInt16)
    if dataset is None:
        raise RuntimeError("No se pudo crear el archivo GeoTIFF.")

    # Establecer la geotransformación, si se proporciona
    if geotransform is not None:
        dataset.SetGeoTransform(geotransform)
    
    # Establecer la proyección, si se proporciona
    if projection is not None:
        dataset.SetProjection(projection)

    # Escribir cada banda
    for i in range(B):
        band = dataset.GetRasterBand(i + 1)
        band.WriteArray(output[:, :, i])
    
    # Cerrar el dataset
    dataset.FlushCache()
    dataset = None

def read_png(fichero):
    try:
        dataset = gdal.Open(fichero)
    except RuntimeError:
        print('No puedo abrir ', fichero)
    else:
        # Obtener información sobre la imagen
        H = dataset.RasterXSize
        V = dataset.RasterYSize
        B = dataset.RasterCount

        # Leer datos de las bandas
        datos = np.zeros((B, V, H), dtype=np.uint8)
        for i in range(B):
            banda = dataset.GetRasterBand(i + 1)
            datos[i, :, :] = banda.ReadAsArray().astype(np.uint8)

        if datos.ndim == 3 and datos.shape[0] == 1:  # Caso de una sola banda
                datos = Image.fromarray(datos[0].astype(np.uint8))
        else:  # Caso multibanda, selecciona una banda o ajusta según sea necesario
                datos = Image.fromarray(datos.astype(np.uint8))

        return datos, H, V
    
def save_as_png(array, output_path):
    """
    Guarda un array de NumPy como una imagen PNG.
    """
    # Seleccionar las bandas R, G, B (índices 2, 1, 0)
    if array.shape[2] == 4:
        array = array[:, :, [2, 1, 0]]
    
    # Asegurarse de que el tipo de datos sea uint8
    if array.dtype != np.uint8:
        array = (array / array.max() * 255).astype(np.uint8)

    # Convertir a imagen PIL y guardar
    image = Image.fromarray(array)
    image.save(output_path, 'PNG')

def read_tiff(fichero):
    # Abrir el archivo TIFF con GDAL
    dataset = gdal.Open(fichero)
    if dataset is None:
      raise Exception("No se pudo abrir el archivo TIFF.")

    # Obtener información sobre la imagen
    B = dataset.RasterCount
    H = dataset.RasterXSize
    V = dataset.RasterYSize

    # Leer datos de las bandas
    datos = np.zeros(( V, H,B), dtype=np.float32)
    for i in range(B):
        banda = dataset.GetRasterBand(i + 1)
        datos[ :, :,i] = banda.ReadAsArray().astype(np.float32)
    # Cerrar el dataset GDAL
    dataset = None
    # Normalizar los datos a 16 bits
    datos = 65535 * (datos - datos.min()) / (datos.max() - datos.min())
    datos = datos.astype(np.uint16)

    return datos, H, V, B

def compare_patches(original_patch, loaded_patch):
    if np.array_equal(original_patch, loaded_patch):
        print("Parches son idénticos")
    else:
        print("Parches son diferentes")

def cut_into_patches(datos, gt, max_size=600):
    """
    Corta una imagen en parches de tamaño máximo max_size x max_size.
    """
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

def reconstruct_image(patches, original_shape, max_size=500):
    V, H, B = original_shape
    reconstructed_image = np.zeros((V, H, 3), dtype=np.uint8)  # Asegurarse de que la imagen final tenga 3 bandas
    patch_idx = 0
    for i in range(0, V, max_size):
        for j in range(0, H, max_size):
            patch = patches[patch_idx]
            patch_height, patch_width, _ = patch.shape
            
            # Asegurarse de que el parche no se salga del borde derecho o inferior
            if i + patch_height > V:
                patch_height = V - i
            if j + patch_width > H:
                patch_width = H - j

            
            try:
                reconstructed_image[i:i+patch_height, j:j+patch_width, :] = patch[:patch_height, :patch_width, :3]  # Usar solo las primeras 3 bandas
            except ValueError as e:
                print(f"Error while assigning patch {patch_idx} to reconstructed image: {e}")
                print(f"reconstructed_image shape: {reconstructed_image[i:i+patch_height, j:j+patch_width, :].shape}")
                print(f"Patch shape: {patch[:patch_height, :patch_width, :3].shape}")
                raise e
            
            patch_idx += 1
    
    return reconstructed_image

def save_reconstructed_image_as_png(image, filename):
    """
    Guarda una imagen reconstruida como PNG.

    :param image: Imagen reconstruida en formato numpy array.
    :param filename: Nombre del archivo de salida.
    """
    # Convertir el array numpy a una imagen PIL
    image_pil = Image.fromarray(image)
    # Guardar la imagen como PNG
    image_pil.save(filename, 'PNG')

def save_combined_image(original, ground_truth, inference, filename):
    combined_image = np.concatenate((np.array(original), (ground_truth), (inference)), axis=1).astype(np.uint8)
    combined_image_pil = Image.fromarray(combined_image)
    combined_image_pil.save(filename, 'PNG')

def apply_postprocessing_full(inference_img, ground_truth):
    postprocessed_img = np.array(inference_img)
    postprocessed_img[ground_truth == 0] = [0, 0, 0]
    return postprocessed_img

def process_and_infer_images(folder_path, test_script_path):
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.tiff')]
    odgt_entries = []
    for image_file in image_files:
        base_name = image_file[:-5]
        print("Procesando imagen:", base_name)
        with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as tmpdir2:
            # Rutas
            image_path = os.path.join(folder_path, image_file)
            gt_path = image_path.replace('.tiff', '_24label.png')

            # Cargar imagen y ground truth
            img_tiff, w, h, b = read_tiff(image_path)
            gt_image, w_gt, h_gt = read_png(gt_path)            
            img_patches, gt_patches = cut_into_patches(img_tiff, gt_image, max_size=500)

            for i, (img_patch, gt_patch) in enumerate(zip(img_patches, gt_patches)):
                img_patch_filename = os.path.join(tmpdir, f'patch_{i}.tif')
                gt_patch_filename = os.path.join(tmpdir2, f'patchgt_{i}.png')
                save_geotiff(img_patch, 500, 500, b, img_patch_filename)
                gt_patch.save(gt_patch_filename, 'PNG')  # Guardar como PNG
                entry = {
                    "fpath_img": img_patch_filename,
                    "fpath_segm": gt_patch_filename,
                    "width": 500,
                    "height": 500
                }
                odgt_entries.append(entry)
            
            write_odgt(odgt_entries, os.path.join('data/', 'test.odgt'))
            subprocess.run(['python3', test_script_path, '--gpu', '0', '--cfg', 'config/GID-resnet50dilated-ppm_deepsup.yaml', '--output_dir', tmpdir, '--base_path', tmpdir2, 'VAL.visualize', 'True'])


            # Leer y procesar la imagen ground truth
            ground_truth_image = read_gt(gt_path)
            color_image = colorEncode(ground_truth_image, colors).astype(np.uint8)
            image = Image.fromarray(color_image)
            ground_truth_image_path = os.path.join(os.getcwd(), f'gt_{base_name}.png')
            image.save(ground_truth_image_path, 'PNG')

            # Procesar los resultados
            result_patches = [np.array(Image.open(os.path.join(tmpdir, f'patch_{i}.png'))) for i in range(len(img_patches))]
            reconstructed_image = reconstruct_image(result_patches, (h, w, 3), max_size=500)
            output_filename = os.path.join(os.getcwd(), f'reconstructed_{base_name}.png')
            full_ground_truth = load_full_ground_truth(image_path, folder_path) 
            postprocessed_image = apply_postprocessing_full(reconstructed_image, full_ground_truth)
            save_reconstructed_image_as_png(postprocessed_image, output_filename)

            original_image_path = os.path.join(os.getcwd(), f'original_{base_name}.png')
            save_as_png(img_tiff, original_image_path)

            # Asegurarse de que original_image es un array de NumPy
            original_image = np.array(Image.open(original_image_path))
            combined_output_path = os.path.join(os.getcwd(), f'combined_{base_name}.png')
            save_combined_image(original_image, color_image, postprocessed_image, combined_output_path)

if __name__ == "__main__":
    folder_path = '/home/jesus/Escritorio/TFG/GID/test2'
    test_script_path = 'eval_multipro2.py'
    process_and_infer_images(folder_path, test_script_path)
    num_classes = 24  # Set this to the number of classes in your dataset