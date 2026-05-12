import torch
import torch.distributed as dist
from vlmeval.config import supported_VLM
from vlmeval.utils import track_progress_rich
from vlmeval.smp import *
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import repeat_kv,apply_multimodal_rotary_pos_emb,Cache
from typing import Optional, Tuple
from transformers.models.llama.modeling_llama import (
    logging,
)
FAIL_MSG = 'Failed to obtain answer via API.'
from spattn.src.models.config_args import FastPrefillConfig
from spattn.src.models.utils import SpAttnProfiler
logger = logging.get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument('--model', type=str, nargs='+', required=True)
    parser.add_argument('--nproc', type=int, default=4, required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args


# Only API model is accepted
def infer_data_api(model, work_dir, model_name, dataset, samples_dict={}, api_nproc=4):
    rank, world_size = get_rank_and_world_size()
    assert rank == 0 and world_size == 1
    dataset_name = dataset.dataset_name
    model = supported_VLM[model_name]() if isinstance(model, str) else model
    assert getattr(model, 'is_api', False)

    indices = list(samples_dict.keys())
    if getattr(model,'backend', None) == 'genai':
        if dataset.nframe > 0:
            print(
                'Gemini model (with genai backend) does not support nframe, '
                'will set its VIDEO_LLM to False to enable multi-image input for video.'
            )
            setattr(model, 'VIDEO_LLM', False)
        else:
            print('Gemini model (with genai backend) is a video-llm, '
                  'will reset fps setting in model to match the dataset.')
            setattr(model, 'fps', dataset.fps)
            print(f'The fps is set to {dataset.fps} for the model {model_name}.')
    elif getattr(model,'backend', None) == 'vertex':
        print('Gemini model (with vertex backend) does not support video input, '
              'will set its VIDEO_LLM to False to enable multi-image input for video.')
        setattr(model, 'VIDEO_LLM', False)

    packstr = 'pack' if getattr(dataset, 'pack', False) else 'nopack'
    build_prompt_input = [(samples_dict[idx], getattr(model, 'VIDEO_LLM', False)) for idx in indices]
    if dataset.nframe > 0:
        struct_tmp_file = f'{work_dir}/{model_name}_{dataset_name}_{dataset.nframe}frame_{packstr}_structs.pkl'
    else:
        struct_tmp_file = f'{work_dir}/{model_name}_{dataset_name}_{dataset.fps}fps_{packstr}_structs.pkl'
    structs = track_progress_rich(
        dataset.build_prompt,
        tasks=build_prompt_input,
        nproc=api_nproc,
        save=struct_tmp_file,
        keys=indices,
    )

    if dataset.nframe > 0:
        out_file = f'{work_dir}/{model_name}_{dataset_name}_{dataset.nframe}frame_{packstr}_supp.pkl'
    else:
        out_file = f'{work_dir}/{model_name}_{dataset_name}_{dataset.fps}fps_{packstr}_supp.pkl'
    res = load(out_file) if osp.exists(out_file) else {}

    structs = [s for i, s in zip(indices, structs) if i not in res or res[i] == FAIL_MSG]
    structs = [struct for struct in structs if struct is not None]
    indices = [i for i in indices if i not in res or res[i] == FAIL_MSG]

    gen_func = model.generate
    structs = [dict(message=struct, dataset=dataset_name) for struct in structs]

    if len(structs):
        track_progress_rich(gen_func, structs, nproc=api_nproc, chunksize=api_nproc, save=out_file, keys=indices)

    res = load(out_file)
    return res


def infer_data(model, model_name, work_dir, dataset, out_file, 
               prev_result_file, 
               prefill_config, 
               verbose=False, api_nproc=4, use_vllm=False,
            ):
    res = load(out_file) if osp.exists(out_file) else {}
    if osp.exists(prev_result_file):
        prev_res = load(prev_result_file)
        for _, row in prev_res.iterrows():
            idx = row['index']
            if idx not in res:
                # Assume prediction and sparsity fields exist
                res[idx] = {
                    'response': row['prediction'],
                    'input_length': row['input_length'],
                    'request_latency': row['request_latency'],
                    'ttft': row['ttft'],
                    'attn_layer_latency': row['attn_layer_latency'],
                }
    rank, world_size = get_rank_and_world_size()
    dataset_name = dataset.dataset_name
    warmup_iters = 2 if os.getenv("WARMUP_PROFILE") else 0

    sample_indices = list(dataset.videos) if getattr(dataset, 'pack', False) else list(dataset.data['index'])
    samples = list(dataset.videos) if getattr(dataset, 'pack', False) else list(range(len(dataset.data)))
    sample_map = {i: s for i, s in zip(sample_indices, samples)}

    sample_indices_sub = sample_indices[rank::world_size]

    if world_size > 1: # The pop below may cause index assignment issues.
        dist.barrier()

    if np.all([idx in res for idx in sample_indices_sub]):
        return model
    sample_indices_subrem = [x for x in sample_indices_sub if x not in res]

    kwargs = {}
    if model_name is not None and (
        'Llama-4' in model_name
        or 'Qwen2-VL' in model_name
        or 'Qwen2.5-VL' in model_name
        or 'Qwen2.5-Omni' in model_name
    ):
        kwargs = {'use_vllm': use_vllm}

    # (25.06.05) In newer version of transformers (after 4.50), with device_map='auto' and torchrun launcher,
    # Transformers automatically adopt TP parallelism, which leads to compatibility problems with VLMEvalKit
    # (In VLMEvalKit, we use torchrun to launch multiple model instances on a single node).
    # To bypass this problem, we unset `WORLD_SIZE` before building the model to not use TP parallel.
    ws_bak = os.environ.pop('WORLD_SIZE', None)
    model = supported_VLM[model_name](prefill_config=prefill_config, **kwargs) if isinstance(model, str) else model
    if ws_bak:
        os.environ['WORLD_SIZE'] = ws_bak

    is_api = getattr(model, 'is_api', False)
    if is_api:
        assert world_size == 1
        supp = infer_data_api(
            model=model,
            work_dir=work_dir,
            model_name=model_name,
            dataset=dataset,
            samples_dict={k: sample_map[k] for k in sample_indices_subrem},
            api_nproc=api_nproc)
        for k in sample_indices_subrem:
            assert k in supp
        res.update(supp)
        dump(res, out_file)
        return model

    assert not getattr(dataset, 'pack', False), 'Current model not supported pack mode!'
    if 'megabench' in dataset_name.lower() and 'llava_onevision' in model_name:
        print(
            'LLaVA-OneVision does not support Megabench dataset as video dataset, '
            'will set its VIDEO_LLM to False to enable multi-image input for video.'
        )
        setattr(model, 'VIDEO_LLM', False)
    # there are many videos in VCR-Bench, which cannot be decoded by torchcodec
    if 'vcr-bench' in dataset_name.lower() and listinstr(['Qwen2-VL','Qwen2.5-VL','Qwen2.5-Omni'], model_name):
        from qwen_vl_utils import vision_process 
        vision_process.FORCE_QWENVL_VIDEO_READER = "decord"

    for i, idx in tqdm(enumerate(sample_indices_subrem), total=len(sample_indices_subrem)):
        if idx in res:
            continue
        if getattr(model, 'nframe', None) is not None and getattr(model, 'nframe', 0) > 0:
            if dataset.nframe > 0:
                if getattr(model, 'nframe', 0) != dataset.nframe:
                    print(f'{model_name} is a video-llm model, nframe is set to {dataset.nframe}, not using default')
                    setattr(model, 'nframe', dataset.nframe)
            elif getattr(model, 'fps', 0) == 0:
                raise ValueError(f'fps is not suitable for {model_name}')
            else:
                setattr(model, 'nframe', None)
        if getattr(model, 'fps', None) is not None and getattr(model, 'fps', 0) > 0:
            if dataset.fps > 0:
                if getattr(model, 'fps', 0) != dataset.fps:
                    print(f'{model_name} is a video-llm model, fps is set to {dataset.fps}, not using default')
                    setattr(model, 'fps', dataset.fps)
            elif getattr(model, 'nframe', 0) == 0:
                raise ValueError(f'nframe is not suitable for {model_name}')
            else:
                setattr(model, 'fps', None)
        if (
            'Qwen2-VL' in model_name
            or 'Qwen2.5-VL' in model_name
            or 'Qwen2.5-Omni' in model_name
        ):
            if getattr(model, 'nframe', None) is None and dataset.nframe > 0:
                print(f'using {model_name} default setting for video, dataset.nframe is ommitted')
            if getattr(model, 'fps', None) is None and dataset.fps > 0:
                print(f'using {model_name} default setting for video, dataset.fps is ommitted')
        if 'SUB_DATASET' in dataset.data.iloc[sample_map[idx]]:
            dataset_name = dataset.data.iloc[sample_map[idx]]['SUB_DATASET']
        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            if dataset.nframe == 0:
                raise ValueError(f'nframe must be set for custom prompt, fps is not suitable for {model_name}')
            struct = model.build_prompt(
                dataset.data.iloc[sample_map[idx]], dataset=dataset, video_llm=getattr(model, 'VIDEO_LLM', False)
            )
        else:
            struct = dataset.build_prompt(
                sample_map[idx], video_llm=getattr(model, 'VIDEO_LLM', False)
            )
        if struct is None:
            continue
        # for profiling time, we only warmup once per dataset
        if os.getenv("WARMUP_PROFILE") and warmup_iters:
            for i in range(warmup_iters):
                with SpAttnProfiler() as profiler:
                    _ = model.generate(message=struct, dataset=dataset_name)
            warmup_iters = 0
        # If `SKIP_ERR` flag is set, the model will skip the generation if error is encountered
        if os.environ.get('SKIP_ERR', False) == '1':
            FAIL_MSG = 'Failed to obtain answer'
            try:
                with SpAttnProfiler() as profiler:
                    response = model.generate(message=struct, dataset=dataset_name)
            except RuntimeError as err:
                torch.cuda.synchronize()
                warnings.error(f'{type(err)} {str(err)}')
                response = f'{FAIL_MSG}: {type(err)} {str(err)}'
        else:
            with SpAttnProfiler() as profiler:
                response = model.generate(message=struct, dataset=dataset_name)
        torch.cuda.empty_cache()

        if verbose:
            print(response, flush=True)

        res[idx] = {
            'response': response,
            'input_length': getattr(model, 'input_length', -1),
            'request_latency': profiler.latency,
            'ttft': profiler.ttft,
            'attn_layer_latency': profiler.layer_metrics['attention_latency']['mean'],
        }
        if (i + 1) % 20 == 0:
            dump(res, out_file)

    res = {k: res[k] for k in sample_indices_sub}
    dump(res, out_file)
    return model


# A wrapper for infer_data, do the pre & post processing
def infer_data_job_video(
        model,
        work_dir,
        model_name,
        dataset,
        result_file_name,
        prefill_config,
        verbose=False,
        api_nproc=4,
        use_vllm=False):

    dataset_name = dataset.dataset_name
    rank, world_size = get_rank_and_world_size()
    result_file = osp.join(work_dir, result_file_name)
    # Dump Predictions to Prev File if result file exists
    # Comment this line when you want to enlarge the result file
    if osp.exists(result_file) and os.getenv("ENLARGE_RESULT_FILE", "0") == "0":
        return model

    tmpl = osp.join(work_dir, '{}' + f'{world_size}_{osp.splitext(result_file_name)[0]}.pkl')
    out_file = tmpl.format(rank)
    model = infer_data(
        model=model,
        model_name=model_name,
        work_dir=work_dir,
        dataset=dataset,
        out_file=out_file,
        prev_result_file =result_file,
        prefill_config=prefill_config,
        verbose=verbose,
        api_nproc=api_nproc,
        use_vllm=use_vllm)

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        data_all = {}
        for i in range(world_size):
            data_all.update(load(tmpl.format(i)))

        meta = dataset.data
        if dataset_name == 'MMBench-Video' and getattr(dataset, 'pack', False):
            meta, vstats = dataset.load_pack_answers(data_all)
            print(f'Statitics of Pack Video Inference: {vstats}')
        else:
            for x in meta['index']:
                assert x in data_all

            meta['prediction'] = [str(data_all[x]['response']) for x in meta['index']]
            
            # Get all keys from the first data item (except response)
            first_data = next(iter(data_all.values()))
            all_keys = set(first_data.keys()) if isinstance(first_data, dict) else set()
            all_keys.discard('response')  # Already stored as 'prediction'
            
            # Create a column for each key
            for key in all_keys:
                values = []
                for x in meta['index']:
                    value = data_all[x][key]
                    if isinstance(value, (list, tuple)):
                        values.append(json.dumps(value))
                    else:
                        values.append(value)
                meta[key] = values
            
            if 'image' in meta:
                meta.pop('image')

        dump(meta, result_file)
        for i in range(world_size):
            os.remove(tmpl.format(i))
    return model
