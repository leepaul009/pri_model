import os
import random
import datetime
import multiprocessing
from multiprocessing import Process
# from torchvision import datasets
# from torchvision.datasets import ImageNet
# from torchvision.datasets.folder import ImageFolder
from webdataset import TarWriter
import pandas as pd
from tqdm import tqdm
import numpy as np




def make_wds_shards(pattern, num_shards, num_workers, samples, map_func, **kwargs):
    '''
    num_shards = 256
    '''
    # random.shuffle(samples)
    samples_per_shards = [samples[i::num_shards] for i in range(num_shards)]
    shard_ids = list(range(num_shards))

    processes = [
        Process(
            target=write_partial_samples,
            args=(
                pattern,
                shard_ids[i::num_workers],
                samples_per_shards[i::num_workers],
                map_func,
                kwargs
            )
        )
        for i in range(num_workers)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()


def write_partial_samples(pattern, shard_ids, samples, map_func, kwargs):
    for shard_id, samples in zip(shard_ids, samples):
        write_samples_into_single_shard(pattern, shard_id, samples, map_func, kwargs)


def write_samples_into_single_shard(pattern, shard_id, samples, map_func, kwargs):
    fname = pattern % shard_id
    print(f"[{datetime.datetime.now()}] start to write samples to shard {fname}")
    stream = TarWriter(fname, **kwargs)
    size = 0
    for item in samples:
        size += stream.write(map_func(item))
    stream.close()
    print(f"[{datetime.datetime.now()}] complete to write samples to shard {fname}")
    return size


if __name__ == "__main__":
    root = "dataset/dna_dataset/_dataset/tmp"
    
    _, _, cur_files = os.walk(root).__next__()
    files = [os.path.join(root, file) for file in cur_files if file.endswith(".csv")]
    print("input files size = {}".format(len(files)))

    ########
    num_files = len(files)
    # for i, f in tqdm(enumerate(files)):
    #     df = pd.read_csv(f, sep='\t')
    #     num_files = num_files + len(df)
    # print("num_files = {}".format(num_files))

    core_num = 4
    queue = multiprocessing.Queue(core_num) # inputs container
    queue_res = multiprocessing.Queue() # outputs container

    def calc_ex_list(queue, queue_res):
        res = []
        dis_list = []
        while True:
            df = queue.get()
            if df is None:
                break
            # for j in range(len(df)):
            #     queue_res.put(df.loc[j])
            prot = df["protein_sequence"].values
            nuc = df["nucleotide_sequence"].values
            dg = df["dG"].values
            res = np.stack((prot, nuc, dg), axis=1)
            queue_res.put(res)


    processes = [Process(target=calc_ex_list, args=(queue, queue_res)) 
                    for _ in range(core_num)]
    for each in processes:
        each.start()

    pbar = tqdm(total=num_files)
    for i, f in tqdm(enumerate(files)):
        df = pd.read_csv(f, sep='\t')
        queue.put(df)
        pbar.update(1)
    
    # wait until all thread finished
    while not queue.empty():
        pass
    pbar.close()
    print("thread finished...")

    items = []
    pbar = tqdm(total=num_files)
    for i in range(num_files):
        t = queue_res.get()
        if t is not None:
            items.append(t)
        pbar.update(1)
    pbar.close()
    pass

    for i in range(core_num):
        queue.put(None)
    for each in processes:
        each.join()

    items = np.concatenate(items)
    ########


    def map_func(item):
        sample = {
            "protein": str(item[0]), # use by esm
            "dna": str(item[1]), # used by dbert
            "label": float(item[2]), # float scalar
        }
        return sample

    make_wds_shards(
        pattern="dataset/dna_dataset/tars/dna-1k-%06d.tar",
        num_shards=256, # 设置分片数量
        num_workers=8, # 设置创建wds数据集的进程数
        samples=items,
        map_func=map_func,
    )
