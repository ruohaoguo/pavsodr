import atexit
import bisect
import multiprocessing as mp
import numpy as np
import torch

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode
from detectron2.structures import Instances, Boxes

from demo_video.visualizer import TrackVisualizer



def process_list(input_list):
    max_value = max(input_list)[0]
    max_index = input_list.index([max_value])
    sorted_list = sorted(input_list, key=lambda x: x[0], reverse=True)
    result = []
    for i, value in enumerate(sorted_list):
        if i == max_index:
            result.append(4)
        else:
            result.append(i + 1)

    return result


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = VideoPredictor(cfg)

    def run_on_video(self, frames, audios, threshold):
        """
        Args:
            frames (List[np.ndarray]): a list of images of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        frames_audios = [frames, audios]
        predictions_all = self.predictor(frames_audios)

        binary_masks = []

        for frame_idx, predictions in enumerate(predictions_all):
            pred_scores = predictions["pred_scores"]
            pred_labels = predictions["pred_labels"]
            pred_masks = predictions["pred_masks"]

            # select high-score masks
            pred_scores_p = []
            pred_labels_p = []
            pred_masks_p = []
            for p in range(len(pred_scores)):
                if pred_scores[p] > threshold:
                    pred_scores_p.append(pred_scores[p])
                    pred_labels_p.append(pred_labels[p])
                    pred_masks_p.append(pred_masks[p])

            frame = frames[frame_idx][:, :, ::-1]

            if len(pred_scores_p) > 0:
                # binary images
                mask_ = torch.stack(pred_masks_p).sum(dim=0).float()
                mask_ = torch.clamp(mask_, max=1)
                for i in range(mask_.shape[0]):
                    img = mask_[i].cpu().detach().numpy() * 255
                    img = img.astype(np.uint8)
                    binary_masks.append(img)
            else:
                img = np.zeros(frame.shape)
                binary_masks.append(img)

        return binary_masks


    def run_on_video_rank(self, frames, audios, threshold, vid_frames_name_new):
        """
        Args:
            frames (List[np.ndarray]): a list of images of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        frames_audios = [frames, audios]
        predictions_all = self.predictor(frames_audios)

        binary_masks = []

        pred_output = []

        for frame_idx, predictions in enumerate(predictions_all):
            pred_scores = predictions["pred_scores"]
            pred_labels = predictions["pred_labels"]
            pred_masks = predictions["pred_masks"]

            pred_rank_scores = predictions["pred_rank_scores"]
            pred_ranks = predictions["pred_ranks"]

            # select high-score masks
            pred_scores_p = []
            pred_labels_p = []
            pred_masks_p = []
            pred_rank_scores_p = []
            pred_ranks_p = []
            for p in range(len(pred_scores)):
                if pred_scores[p] > threshold:
                    pred_scores_p.append(pred_scores[p])
                    pred_labels_p.append(pred_labels[p])
                    pred_masks_p.append(pred_masks[p])
                    pred_rank_scores_p.append(pred_rank_scores[p])
                    pred_ranks_p.append(pred_ranks[p])

            frame = frames[frame_idx][:, :, ::-1]

            if len(pred_scores_p) > 0:
                mask_ = torch.stack(pred_masks_p).sum(dim=0).float()
                mask_ = torch.clamp(mask_, max=1)
                for i in range(mask_.shape[0]):
                    img = mask_[i].cpu().detach().numpy() * 255
                    img = img.astype(np.uint8)
                    binary_masks.append(img)

                if vid_frames_name_new[frame_idx] != "no_image":
                    ins = Instances(predictions["image_size"])
                    for mk in pred_masks_p:
                        mk[mk >= 0.3] = 1.0
                        mk[mk < 0.3] = 0.0
                    ins.pred_masks = torch.cat(pred_masks_p, dim=0)
                    ins.pred_bbox = torch.tensor([0.0, 0.0, 0.0, 0.0]).unsqueeze(0).repeat(len(pred_masks_p), 1)
                    # pred_rank_scores_p = [i[0] for i in pred_rank_scores_p]
                    # sorted_pred_rank_scores_p = sorted(pred_rank_scores_p, reverse=True)
                    # weights = [1.0, 0.9, 0.8, 0.7, 0.6]
                    # result_dict = {element: weight for element, weight in zip(sorted_pred_rank_scores_p, weights)}
                    # ins.pred_ranks = torch.tensor([result_dict[element] for element in pred_rank_scores_p])
                    ins.pred_ranks = torch.tensor([(10-i)/10 if i!=5 else .0 for i in pred_ranks_p])
                    ins.scores = torch.tensor([max(i) for i in pred_rank_scores_p])

                    image_dir = "../datasets/pavsodr/ranking_eval_dataset/images/test/"
                    pred_dict = {
                        "image_id": int(vid_frames_name_new[frame_idx].split(".")[-2].split("_")[-1]),
                        "file_name": image_dir + vid_frames_name_new[frame_idx],
                        "predictions_for_ranking": ins}

                    pred_output.append(pred_dict)
            else:
                img = np.zeros(frame.shape)
                binary_masks.append(img)

                if vid_frames_name_new[frame_idx] != "no_image":
                    ins = Instances(predictions["image_size"])
                    ins.pred_masks = torch.tensor([])
                    ins.pred_bbox = Boxes(torch.tensor([]))
                    ins.pred_ranks = torch.tensor([])
                    ins.scores = torch.tensor([])

                    image_dir = "../datasets/pavsodr/ranking_eval_dataset/images/test/"
                    pred_dict = {
                        "image_id": int(vid_frames_name_new[frame_idx].split(".")[-2].split("_")[-1]),
                        "file_name": image_dir + vid_frames_name_new[frame_idx],
                        "predictions_for_ranking": ins}

                    pred_output.append(pred_dict)

        return pred_output, binary_masks


    def run_on_video_ins(self, frames, audios, threshold):
        """
        Args:
            frames (List[np.ndarray]): a list of images of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        frames_audios = [frames, audios]
        predictions_all = self.predictor(frames_audios)

        total_vis_output = []

        for frame_idx, predictions in enumerate(predictions_all):
            image_size = predictions["image_size"]
            pred_scores = predictions["pred_scores"]
            pred_labels = predictions["pred_labels"]
            pred_masks = predictions["pred_masks"]

            # select high-score masks
            pred_scores_p = []
            pred_labels_p = []
            pred_masks_p = []
            for p in range(len(pred_scores)):
                if pred_scores[p] > threshold:
                    pred_scores_p.append(pred_scores[p])
                    pred_labels_p.append(pred_labels[p])
                    pred_masks_p.append(pred_masks[p])

            for mk in pred_masks_p:
                mk[mk >= 0.3] = 1.0
                mk[mk < 0.3] = 0.0

            frame = frames[frame_idx][:, :, ::-1]
            visualizer = TrackVisualizer(frame, self.metadata, instance_mode=self.instance_mode)
            ins = Instances(image_size)
            if len(pred_scores_p) > 0:
                ins.scores = pred_scores_p
                ins.pred_classes = pred_labels_p
                ins.pred_masks = torch.cat(pred_masks_p, dim=0)

            vis_output = visualizer.draw_instance_predictions(predictions=ins)
            total_vis_output.append(vis_output)

        return total_vis_output

class VideoPredictor(DefaultPredictor):
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.
    Compared to using the model directly, this class does the following additions:
    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take one input image and produce a single output, instead of a batch.
    If you'd like to do anything more fancy, please refer to its source code
    as examples to build and use the model manually.
    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.
    Examples:
    ::
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """
    def __call__(self, frames_audios):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            input_frames = []
            frames = frames_audios[0]
            audios = frames_audios[1]
            for original_image in frames:
                # Apply pre-processing to image.
                if self.input_format == "RGB":
                    # whether the model expects BGR inputs or RGB
                    original_image = original_image[:, :, ::-1]
                height, width = original_image.shape[:2]
                image = self.aug.get_transform(original_image).apply_image(original_image)
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                input_frames.append(image)

            inputs = {"image": input_frames, "height": height, "width": width, "audio": audios}

            # num_params = self.count_parameters(self.model)
            # print(f"Model Parameters: {num_params}")
            # import time
            # from thop import profile
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # input = torch.randn(2, 3, 360, 675)
            # input = input.to(device)
            # flops, params = profile(self.model, inputs=(input,))
            # print(f"FLOPs: {flops}")
            # print(f"Params: {params}")
            # start_time = time.time()
            predictions = self.model([inputs])
            # end_time = time.time()
            #
            # avg_inference_time = (end_time - start_time) / len(input_frames)
            # fps = 1 / avg_inference_time
            # print("FPS: {}  ||  height: {}  ||  width: {}".format(fps, height, width))

            return predictions


    def count_parameters(self, model):
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params_in_mb = total_params * 4 / (1024 ** 2)

        return total_params_in_mb


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = VideoPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5
