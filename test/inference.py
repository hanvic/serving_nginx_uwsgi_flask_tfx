import time, datetime, argparse, cv2, os

import tensorflow as tf
import warnings , os
from utils import show_results


warnings.filterwarnings(action='ignore')
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


"""
===========================================================
                       configuration
===========================================================
"""



# def generate_expname_automatically():
#     name = "OCR_%s_%02d_%02d_%02d_%-2d_%02d" % (config.model_tag, time_now.month,time_now.day, time_now.hour,
#                                                 time_now.minute, time_now.second)
#     return name

# expname = generate_expname_automatically()
# config.checkpoint_dir += "segmentation_" + config.model_tag; check_folder(config.checkpoint_dir)
# config.summary_dir += expname ; check_folder(config.summary_dir)

# read dataset







def serve_by_tfrecords(config):
    from models.train_model_segmentation import Model2eye
    """
    ===========================================================
                          prepare dataset
    ===========================================================
    """
    list_val_tfrecords = os.listdir(config.val_data_root)
    list_only_img_val_tfrecords = [os.path.join(config.val_data_root, path2tfrecord) for path2tfrecord in
                                   list_val_tfrecords if path2tfrecord.split('_')[0] == 'only']
    list_val_tfrecords = [os.path.join(config.val_data_root, path2tfrecord) for path2tfrecord in list_val_tfrecords if
                          not path2tfrecord.split('_')[0] == 'only']

    # dataset = read_tfrecord(list_tfrecords[:-1], batch_size=config.batch_size)
    validation_dataset = read_record_validation(list_val_tfrecords, batch_size=1)
    """
    ===========================================================
                          build model
    ===========================================================
    """
    model = Model2eye(config)
    model.restore(config.stamp_epoch)

    confusion_matrix = np.zeros((6, 6))
    for i, image_features in enumerate(validation_dataset):
        # print(image_features),

        data = apply_validation(image_features, config)
        predictions, target = model.test_step(data, config.maxClsSize)
        print("i", i)
        confusion_matrix = show_results.visual_result_teeth(i, config.target_features, data['img'], predictions, target,confusion_matrix)
    print("confusion_matrix",confusion_matrix)

def serve_by_image(stamp_epoch,threshold, target_height, target_width, maxClsSize,checkpoint_dir,target_features,image):
    from models.serve_model_segmentation import Model2eye
    """
    ===========================================================
                         build model
    ===========================================================
    """
    image = cv2.resize(image, (target_height, target_width))
    model = Model2eye(maxClsSize,checkpoint_dir)



    # model.restore(115)
    data = image
    predictions = model.single_image_test_step(data, maxClsSize)
    result_dict, result_prob_dict,  segmentation_image = \
        show_results.single_image_visual_result(data, predictions,target_features, threshold)
    return result_dict, result_prob_dict,  segmentation_image


if __name__ == "__main__":
    start = time.time()
    time_now = datetime.datetime.now()
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=1, type=str)
    parser.add_argument("--stamp_epoch", default=425, type=int)
    parser.add_argument("--mode", default='normal', type=str,help="one of ['normal', 'serve']")
    parser.add_argument("--maxClsSize", default=45, type=int)
    parser.add_argument("--target_size", default=250, type=list, nargs="+", help="Image size after crop")
    parser.add_argument("--batch_size", default=32, type=int, help="Minibatch size(global)")
    parser.add_argument("--val_batch_size", default=1, type=int, help="Minibatch size(global)")
    parser.add_argument("--data_root", default='/home/projects/data/train/tfrecords/', type=str,
                        help="Dir to data root")
    parser.add_argument("--val_data_root", default='/home/projects/data_ear/validation/tfrecords/', type=str,
                        help="Dir to val data root")
    parser.add_argument("--target_width", default=512, type=int, help="target width of input image")
    parser.add_argument("--target_height", default=512, type=int, help="target width of input image")
    parser.add_argument("--image_file", default='./dataset/test/176039.jpg', type=str, help="Dir to data root")
    parser.add_argument("--channels", default=1, type=int, help="Channel size")
    parser.add_argument("--color_map", default="RGB", type=str, help="Channel mode. [RGB, YCbCr]")
    parser.add_argument("--model_tag", default="default", type=str, help="Exp name to save logs/checkpoints.")
    parser.add_argument("--checkpoint_dir", default="/home/serving/eye/", type=str, help="Dir for checkpoints")
    parser.add_argument("--summary_dir", default="outputs/summaries/", type=str, help="Dir for tensorboard logs.")
    parser.add_argument("--restore_file", default=None, type=str, help="file for restoration")
    parser.add_argument("--graph_mode", default=False, type=bool, help="use graph mode for training")
    # teeth
    parser.add_argument("--target_features", default=[1,4,5,26,27,28,29,30,31,32,33], type=list, help="use graph mode for training")
    # teeth
    # parser.add_argument("--target_features", default=[1,2,3,4,5], type=list, help="use graph mode for training")
    parser.add_argument("--threshold", default=0.3, type=float, help="threshold for dnn")
    """
    target features are as follows:
    Third_eyelid_protrude	1
    blepharitis_inflammation	4
    blepharitis_inner_inflammation	5
    cataract	32
    cataract_initial	33
    """
    config = parser.parse_args()
    image=cv2.imread('/home/serving/alpha/data/AlphadoPhoto_2020-10-15 13_33_51_931.JPG')
    result_dict, result_prob_dict,  segmentation_image  = serve_by_image(config.stamp_epoch,config.threshold,config.target_height, config.target_width, config.maxClsSize,config.checkpoint_dir,config.target_features,image)
