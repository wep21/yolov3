import argparse 

import torch.distributed as dist 
import torch.optim as optim 
import torch.optim.lr_scheduler as lr_scheduler 
  
import test  # import test.py to get mAP after each epoch 
from models import * 
from utils.datasets import * 
from utils.utils import * 
 
mixed_precision = True 
try:  # Mixed precision training https://github.com/NVIDIA/apex 
    from apex import amp 
except: 
    mixed_precision = False  # not installed 

parser = argparse.ArgumentParser() 
parser.add_argument('--epochs', type=int, default=273)  # 500200 batches at bs 16, 117263 COCO images = 273 epochs 
parser.add_argument('--batch-size', type=int, default=16)  # effective bs = batch_size * accumulate = 16 * 4 = 64 
parser.add_argument('--accumulate', type=int, default=4, help='batches to accumulate before optimizing') 
parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path') 
parser.add_argument('--data', type=str, default='data/coco2017.data', help='*.data path') 
parser.add_argument('--multi-scale', action='store_true', help='adjust (67% - 150%) img_size every 10 batches') 
parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)') 
parser.add_argument('--rect', action='store_true', help='rectangular training') 
parser.add_argument('--resume', action='store_true', help='resume training from last.pt') 
parser.add_argument('--nosave', action='store_true', help='only save final checkpoint') 
parser.add_argument('--notest', action='store_true', help='only test final epoch') 
parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters') 
parser.add_argument('--bucket', type=str, default='', help='gsutil bucket') 
parser.add_argument('--cache-images', action='store_true', help='cache images for faster training') 
parser.add_argument('--weights', type=str, default='weights/ultralytics68.pt', help='initial weights') 
parser.add_argument('--arc', type=str, default='default', help='yolo architecture')  # defaultpw, uCE, uBCE 
parser.add_argument('--prebias', action='store_true', help='pretrain model biases') 
parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied') 
parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1 or cpu)') 
parser.add_argument('--adam', action='store_true', help='use adam optimizer') 
parser.add_argument('--var', type=float, help='debug variable') 
opt = parser.parse_args() 
opt.weights = last if opt.resume else opt.weights 
print(opt) 
device = torch_utils.select_device(opt.device, apex=mixed_precision, batch_size=opt.batch_size) 
if device.type == 'cpu': 
    mixed_precision = False 

model = Darknet(opt.cfg, (opt.img_size, opt.img_size))
if opt.weights.endswith('.pt'):  # pytorch format 
    model.load_state_dict(torch.load(opt.weights, map_location=device)['model']) 
else:  # darknet format 
    _ = load_darknet_weights(model, opt.weights) 
img=np.random.randn(3, opt.img_size, opt.img_size)
img = torch.from_numpy(img).unsqueeze(0).float()
img.to('cpu')
torch.onnx.export(model,                     # model being run 
                  img,                       # model input (or a tuple for multiple inputs) 
                  "weights/yolo3.onnx",      # where to save the model (can be a file or file-like object) 
                  export_params=True,        # store the trained parameter weights inside the model file 
                  opset_version=11,          # the ONNX version to export the model to 
                  do_constant_folding=True,  # wether to execute constant folding for optimization 
                  input_names = ['input'],   # the model's input names 
                  output_names = ['output'], # the model's output names 
                  ) 
