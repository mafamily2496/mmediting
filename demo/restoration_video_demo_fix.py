import argparse
from tqdm import tqdm
import mmcv
import numpy as np
import cv2
from tqdm import tqdm
import torch
from mmedit.datasets.pipelines import Compose
from mmedit.apis import init_model
from mmedit.utils import modify_args
import queue
import threading

def parse_args():
    modify_args()
    parser = argparse.ArgumentParser(description='Restoration demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('input_dir', help='directory of the input video')
    parser.add_argument('output_dir', help='directory of the output video')
    parser.add_argument(
        '--max-seq-len',
        type=int,
        default=10,
        help='maximum sequence length if recurrent framework is used')
    parser.add_argument(
        '--window-size',
        type=int,
        default=0,
        help='window size if sliding-window framework is used')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    args = parser.parse_args()
    return args

def read_frame(que, video_reader, max_seq_len, window_size, test_pipeline, device, batch_size):
    frame_count = video_reader.frame_cnt
    list = []
    if window_size > 0:  # sliding window framework
        for i in range(0, frame_count - 2 * (window_size // 2)):
            data = dict(lq=[], lq_path=None, key="")
            frames = video_reader[i: i + window_size]
            data['lq'] = [np.flip(frame, axis=2) for frame in frames]
            data = test_pipeline(data)
            data = data['lq'].unsqueeze(0)
            if data.shape[1] == window_size:
                data = pad_sequence(data, window_size)
                data = data[:, 0: window_size]
                list.append(data)
                if i != 0 and len(list) == batch_size:
                    que.put(torch.cat(list).to(device))
                    list = []
            else:
                que.put(torch.cat(list).to(device))
                list = [data]
                que.put(torch.cat(list).to(device))
    else:  # recurrent framework
        for i in range(0, frame_count, max_seq_len):
            data = dict(lq=[], lq_path=None, key="")
            frames = video_reader[i: i + max_seq_len]
            data['lq'] = [np.flip(frame, axis=2) for frame in frames]
            data = test_pipeline(data)
            data = data['lq'].unsqueeze(0)
            if data.shape[1] == max_seq_len:
                list.append(data)
                if i != 0 and len(list) == batch_size:
                    que.put(torch.cat(list).to(device))
                    list = []
            else:
                que.put(torch.cat(list).to(device))
                list = [data]
                que.put(torch.cat(list).to(device))

def write_frame(que, video_writer, f, min_max, window_size):
    while not que.empty() or f():
        data = que.get()
        with torch.no_grad(), torch.cuda.amp.autocast():
            if window_size == 0:  # recurrent framework
                data = data.flatten(end_dim=1)
            data = data.float().detach().clamp_(*min_max)
            data = (data - min_max[0]) / (min_max[1] - min_max[0])
            data = data.permute(0, 2, 3, 1)
            data = (data * 255.0).round().cpu().numpy()
        for frame in data:
            video_writer.write(frame.astype(np.uint8))

def pad_sequence(data, window_size):
    padding = window_size // 2

    data = torch.cat([
        data[:, 1 + padding:1 + 2 * padding].flip(1), data,
        data[:, -1 - 2 * padding:-1 - padding].flip(1)
    ],
                     dim=1)

    return data

def main():
    """Demo for video restoration models.

    Note that we accept video as input/output, when 'input_dir'/'output_dir' is
    set to the path to the video. But using videos introduces video
    compression, which lowers the visual quality. If you want actual quality,
    please save them as separate images (.png).
    """

    args = parse_args()

    if args.device < 0 or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda', args.device)

    model = init_model(args.config, args.checkpoint, device=device)
    device = next(model.parameters()).device
    
    if model.cfg.get('demo_pipeline', None):
        test_pipeline = model.cfg.demo_pipeline
    elif model.cfg.get('test_pipeline', None):
        test_pipeline = model.cfg.test_pipeline
    else:
        test_pipeline = model.cfg.val_pipeline
    
    tmp_pipeline = []
    for pipeline in test_pipeline:
        if pipeline['type'] not in [
                'GenerateSegmentIndices', 'LoadImageFromFileList'
        ]:
            tmp_pipeline.append(pipeline)
    test_pipeline = tmp_pipeline

    test_pipeline = Compose(test_pipeline)
    
    video_reader = mmcv.VideoReader(args.input_dir)
    frame_count = video_reader.frame_cnt
    
    min_max=(0, 1)
    video_writer = cv2.VideoWriter(args.output_dir, int(video_reader.fourcc), video_reader.fps, (video_reader.width * 4, video_reader.height * 4))
    write_frame_que = queue.Queue(maxsize=1024)
    t_write_frame_f = True
    t_write_frame = threading.Thread(target=write_frame, args=(write_frame_que, video_writer, lambda:t_write_frame_f, min_max, args.window_size,))
    t_write_frame.start()
    read_frame_que = queue.Queue(maxsize=2)
    t_read_frame = threading.Thread(target=read_frame, args=(read_frame_que, video_reader, args.max_seq_len, args.window_size, test_pipeline, device, args.batch_size,))
    t_read_frame.start()
    with torch.no_grad(), torch.cuda.amp.autocast():
        if args.window_size > 0:  # sliding window framework
            len = (frame_count - 2 * (args.window_size // 2)) // args.batch_size
            if (frame_count - 2 * (args.window_size // 2)) % args.batch_size > 0:
                len += 1
        else:  # recurrent framework
            len = frame_count // (args.max_seq_len * args.batch_size)
            if frame_count % (args.max_seq_len * args.batch_size) > 0:
                len += 1
            if frame_count % args.max_seq_len > 0:
                len += 1
        for i in tqdm(range(0, len)):
            data = read_frame_que.get()
            data = model(lq=data, test_mode=True)['output']
            write_frame_que.put(data)

    t_write_frame_f = False

if __name__ == '__main__':
    main()