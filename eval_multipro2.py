# System libs
import os
import argparse
from distutils.version import LooseVersion
from multiprocessing import Queue, Process
# Numerical libs
import numpy as np
import math
import torch
import torch.nn as nn
from scipy.io import loadmat
# Our libs
from mit_semseg.config import cfg
from mit_semseg.dataset import ValDataset
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import AverageMeter, colorEncode, accuracy, intersectionAndUnion, parse_devices, setup_logger
from mit_semseg.lib.nn import user_scattered_collate, async_copy_to
from mit_semseg.lib.utils import as_numpy
from PIL import Image
from tqdm import tqdm

colors = loadmat('data/color5.mat')['colors']

def save_as_png(array):
    """
    Guarda un array de NumPy como una imagen PNG.
    """
    # Si es un tensor, convertirlo a array de NumPy
    if isinstance(array, torch.Tensor):
        array = array.cpu().numpy()

    # Seleccionar las bandas R, G, B (índices 2, 1, 0)
    array = array[:, :, [2, 1, 0]]
    # Asegurarse de que el tipo de datos sea uint8
    if array.dtype != np.uint8:
        array = (array / array.max() * 255).astype(np.uint8)

    # Convertir a imagen PIL
    return Image.fromarray(array, 'RGB')


def visualize_result(data, pred, dir_result, base_path):
    (img, seg, info) = data
    img = save_as_png(img)

    # Colorear la segmentación
    seg_color = colorEncode(seg, colors)

    # prediction
    pred_color = colorEncode(pred, colors)

    # Ajustar tamaño de pred_color para que coincida con img
    if pred_color.shape[0] != img.height or pred_color.shape[1] != img.width:
        pred_color = np.array(Image.fromarray(pred_color).resize((img.width, img.height)))

    # Guardar solo la imagen de predicción
    img_name = info.split('/')[-1]
    Image.fromarray(pred_color).save(os.path.join(dir_result, img_name.replace('.tif', '.png')))

def evaluate(segmentation_module, loader, cfg, gpu_id, result_queue, dir_result, base_path):
    segmentation_module.eval()

    present_classes = set()

    for batch_data in loader:
        # process data
        batch_data = batch_data[0]
        seg_label = as_numpy(batch_data['seg_label'][0])
        img_resized_list = batch_data['img_data']

        present_classes.update(np.unique(seg_label))

        with torch.no_grad():
            segSize = (seg_label.shape[0], seg_label.shape[1])
            scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
            scores = async_copy_to(scores, gpu_id)

            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                del feed_dict['img_ori']
                del feed_dict['info']
                feed_dict = async_copy_to(feed_dict, gpu_id)

                # forward pass
                scores_tmp = segmentation_module(feed_dict, segSize=segSize)
                scores = scores + scores_tmp / len(cfg.DATASET.imgSizes)

            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())

        # calculate accuracy and SEND THEM TO MASTER
        acc, pix = accuracy(pred, seg_label)
        intersection, union = intersectionAndUnion(pred, seg_label, cfg.DATASET.num_class)
        result_queue.put_nowait((acc, pix, intersection, union, present_classes))

        # visualization
        if cfg.VAL.visualize:
            visualize_result(
                (batch_data['img_ori'], seg_label, batch_data['info']),
                pred,
                dir_result,
                base_path
            )

def worker(cfg, gpu_id, start_idx, end_idx, result_queue, dir_result, base_path):
    torch.cuda.set_device(gpu_id)

    # Dataset and Loader
    dataset_val = ValDataset(
        cfg.DATASET.root_dataset,
        cfg.DATASET.list_val,
        cfg.DATASET,
        start_idx=start_idx, end_idx=end_idx)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=cfg.VAL.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=2)

    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder,
        use_softmax=True)

    crit = nn.NLLLoss(ignore_index=-1)

    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)

    segmentation_module.cuda()

    # Main loop
    evaluate(segmentation_module, loader_val, cfg, gpu_id, result_queue, dir_result, base_path)

def main(cfg, gpus, dir_result, base_path):
    with open(cfg.DATASET.list_val, 'r') as f:
        lines = f.readlines()
        num_files = len(lines)

    num_files_per_gpu = math.ceil(num_files / len(gpus))

    pbar = tqdm(total=num_files)

    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    result_queue = Queue(500)
    procs = []

    all_present_classes = set()

    for idx, gpu_id in enumerate(gpus):
        start_idx = idx * num_files_per_gpu
        end_idx = min(start_idx + num_files_per_gpu, num_files)
        proc = Process(target=worker, args=(cfg, gpu_id, start_idx, end_idx, result_queue, dir_result, base_path))
        print('gpu:{}, start_idx:{}, end_idx:{}'.format(gpu_id, start_idx, end_idx))
        proc.start()
        procs.append(proc)

    # master fetches results
    processed_counter = 0
    while processed_counter < num_files:
        if result_queue.empty():
            continue
        (acc, pix, intersection, union, present_classes) = result_queue.get()
        acc_meter.update(acc, pix)
        intersection_meter.update(intersection)
        union_meter.update(union)
        all_present_classes.update(present_classes)
        processed_counter += 1
        pbar.update(1)

    for p in procs:
        p.join()

    # summary
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    
    # Filtrar clases no presentes en el conjunto de datos
    iou_filtered = {cls: iou[cls] for cls in all_present_classes if union_meter.sum[cls] > 0}
    
    for cls, _iou in iou_filtered.items():
        print('class [{}], IoU: {:.4f}'.format(cls + 1, _iou))

    mean_iou = np.mean(list(iou_filtered.values()))
    print('[Eval Summary]:')
    print('Mean IoU: {:.4f}, Accuracy: {:.2f}%'
          .format(mean_iou, acc_meter.average()*100))

    print('Evaluation Done!')

if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Validation"
    )
    parser.add_argument(
        "--cfg",
        default="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpus",
        default="0-3",
        help="gpus to use, e.g. 0-3 or 0,1,2,3"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where output images will be saved"
    )
    parser.add_argument(
        "--base_path",
        required=True,
        help="Base path for the ground truth images"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    logger = setup_logger(distributed_rank=0)   # TODO
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    # absolute paths of model weights
    cfg.MODEL.weights_encoder = os.path.join(
        cfg.DIR, 'encoder_' + cfg.VAL.checkpoint)
    cfg.MODEL.weights_decoder = os.path.join(
        cfg.DIR, 'decoder_' + cfg.VAL.checkpoint)
    assert os.path.exists(cfg.MODEL.weights_encoder) and \
        os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    # Parse gpu ids
    gpus = parse_devices(args.gpus)
    gpus = [x.replace('gpu', '') for x in gpus]
    gpus = [int(x) for x in gpus]

    main(cfg, gpus, args.output_dir, args.base_path)
