from torch.utils.data import Dataset, DataLoader
from dataset.joint import PairDataset
from engine.sampling import MixedPairSampler, HomogeneousTaskSampler


class ActiveDataset(Dataset):
    """Local record indices, unchanged global task IDs and original video order."""

    def __init__(self, config, split):
        self.base = PairDataset(config, split)
        self.video_count = self.base.video_count
        self.tasks = self.base.tasks
        self.task_ids = config['ablation']['task_ids']

    def __len__(self):
        return self.video_count * len(self.task_ids)

    def __getitem__(self, index):
        task, video = divmod(index, self.video_count)
        row = dict(self.base[self.task_ids[task] * self.video_count + video])
        row['pair_index'] = index
        return row


def build_loader(config, split):
    dataset = ActiveDataset(config, split)
    opts = config['loader']
    bs = opts['video_batch_size']
    if split == 'train':
        sampler = (HomogeneousTaskSampler(dataset.video_count, len(dataset.task_ids),
                                          opts['sampler_seed'], bs)
                   if 'head_rank' in config else
                   MixedPairSampler(len(dataset), opts['sampler_seed'], bs))
    else:
        n = dataset.video_count
        sampler = [
            [t * n + i for i in range(start, min(start + bs, n))]
            for t in range(len(dataset.task_ids))
            for start in range(0, n, bs)
        ]
    # Match baseline DataLoader RNG behavior; permutations use independent CPU RNG.
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=opts['num_workers'],
        pin_memory=opts['pin_memory'],
        persistent_workers=opts['num_workers'] > 0 and opts['persistent_workers'],
    )
