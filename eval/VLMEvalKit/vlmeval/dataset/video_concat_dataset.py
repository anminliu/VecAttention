from ..smp import *
from ..smp.file import get_intermediate_file_path
from .video_base import VideoBaseDataset


class ConcatVideoDataset(VideoBaseDataset):
    # This dataset takes multiple dataset names as input and aggregate them into a single dataset.
    # Each single dataset should not have a field named `SUB_DATASET`

    DATASET_SETS = {}

    def __init__(self, dataset, **kwargs):
        from . import build_dataset
        datasets = self.DATASET_SETS[dataset]
        self.dataset_map = {}
        # The name of the compliation
        self.dataset_name = dataset
        self.datasets = datasets
        self.nframe = kwargs.get('nframe', 0)
        self.fps = kwargs.get('fps', -1)
        for dname in datasets:
            dataset = build_dataset(dname, **kwargs)
            assert dataset is not None, dataset
            self.dataset_map[dname] = dataset
        TYPES = [x.TYPE for x in self.dataset_map.values()]
        MODALITIES = [x.MODALITY for x in self.dataset_map.values()]
        # assert np.all([x == TYPES[0] for x in TYPES]), (datasets, TYPES)
        assert np.all([x == MODALITIES[0] for x in MODALITIES]), (datasets, MODALITIES)
        self.TYPE = TYPES
        self.MODALITY = MODALITIES[0]
        data_all = []
        for dname in datasets:
            data = self.dataset_map[dname].data
            data['SUB_DATASET'] = [dname] * len(data)
            data_all.append(data)

        data = pd.concat(data_all)
        data['original_index'] = data.pop('index')
        data['index'] = np.arange(len(data))
        self.data = data
        # breakpoint()

    def subdataset_sample(self, num_samples=0, random_seed=42):
        """
        按照子数据集 SUB_DATASET 的 videos 数量比例进行采样。

        :param num_samples: 采样的总数量。如果 <= 0，返回所有数据。
        :param random_seed: 随机种子，用于保证采样结果可复现。
        :return: self（采样后的数据集）。
        """
        # 获取总视频数量
        total_video_nums = len(self.data['video'].unique())
        if num_samples <= 0 or num_samples >= total_video_nums:
            # 如果采样数量无效或大于等于总视频数量，返回所有数据
            return self

        # 获取每个子数据集的 video 数量
        subdataset_groups = self.data.groupby('SUB_DATASET')['video'].nunique()
        subdataset_videos = subdataset_groups.to_dict()  # 转换为字典，键为子数据集名称，值为视频数量

        # 计算每个子数据集的采样数量
        sampled_videos = []
        remaining_samples = num_samples
        for i, (subdataset, video_count) in enumerate(subdataset_videos.items()):
            if remaining_samples <= 0:
                break

            if i == len(subdataset_videos) - 1:  # 最后一个子数据集分配剩余的采样数量
                n = min(remaining_samples, video_count)
            else:
                # 按比例分配采样数量，但不能超过子数据集的视频数量或剩余采样数量
                n = min(int(num_samples * video_count / total_video_nums), video_count)
                n = max(1, n)  # 每个子数据集至少采样 1 个视频
                n = min(n, remaining_samples)  # 不能超过剩余采样数量

            # 从子数据集中随机采样 n 个视频
            sub_data = self.data[self.data['SUB_DATASET'] == subdataset]
            sampled = sub_data.drop_duplicates(subset=['video']).sample(n=n, random_state=random_seed)
            sampled_videos.extend(sampled['video'].tolist())
            remaining_samples -= n

        # 根据采样的视频更新数据
        self.data = self.data[self.data['video'].isin(sampled_videos)]

        # 重新赋值 data['index']
        # self.data['index'] = np.arange(len(self.data))

        return self

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            line = self.data.iloc[line]
        idx = line['original_index']
        dname = line['SUB_DATASET']
        org_data = self.dataset_map[dname].data
        org_line = cp.deepcopy(org_data[org_data['index'] == idx]).iloc[0]
        return self.dataset_map[dname].build_prompt(org_line, video_llm)

    def dump_image(self, line):
        # Assert all images are pre-dumped
        assert 'image' not in line
        assert 'image_path' in line
        tgt_path = toliststr(line['image_path'])
        return tgt_path

    @classmethod
    def supported_datasets(cls):
        return []  # list(cls.DATASET_SETS)

    def evaluate(self, eval_file, profile_metrics, **judge_kwargs):
        suffix = eval_file.split('.')[-1]
        # First, split the eval_file by dataset
        data_all = load(eval_file)
        for dname in self.datasets:
            base_dir, file_name = os.path.split(eval_file)  # 分离路径和文件名
            new_file_name = file_name.replace(self.dataset_name, dname)  # 替换文件名中的 self.dataset_name
            tgt = os.path.join(base_dir, new_file_name)  # 重新组合路径

            data_sub = data_all[data_all['SUB_DATASET'] == dname]
            data_sub.pop('index')
            data_sub['index'] = data_sub.pop('original_index')
            data_sub.pop('SUB_DATASET')
            dump(data_sub, tgt)
        # Then, evaluate each dataset separately
        results_all = {}
        for dname in self.datasets:
            base_dir, file_name = os.path.split(eval_file)  # 分离路径和文件名
            new_file_name = file_name.replace(self.dataset_name, dname)  # 替换文件名中的 self.dataset_name
            tgt = os.path.join(base_dir, new_file_name)  # 重新组合路径
            
            res = self.dataset_map[dname].evaluate(tgt, profile_metrics, **judge_kwargs)
            results_all.update(res)

        result = pd.DataFrame(results_all, index=['success', 'overall'] + profile_metrics)
        result = result.T
        for idx, item in result.iterrows():
            result.loc[idx, 'acc'] = round(item['success'] / item['overall'] * 100, 1)
        score_file = get_intermediate_file_path(eval_file, '_acc', 'csv')
        dump(result, score_file)
        return result
