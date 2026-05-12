import json
import os
from collections import defaultdict
import sys
import pandas as pd


def xlsx2json(xlsx_file, json_file):
    df = pd.read_excel(xlsx_file)
    df.to_json(json_file, orient='records')


def calculate_accuracy(data, profile_metrics):
    total_correct = 0
    total_items = len(data)

    dimension_stats = defaultdict(lambda: {
        "correct": 0, "total": 0,
        **{f"{metric}_sum": 0 for metric in profile_metrics}
    })
    duration_stats = {
        "0-60": {"correct": 0, "total": 0, **{f"{metric}_sum": 0 for metric in profile_metrics}},
        "60-300": {"correct": 0, "total": 0, **{f"{metric}_sum": 0 for metric in profile_metrics}},
        "300+": {"correct": 0, "total": 0, **{f"{metric}_sum": 0 for metric in profile_metrics}}
    }

    for item in data:
        if item.get("answer_scoring") == '1':
            total_correct += 1

        dimension = item.get("dimension")
        if dimension:
            dimension_stats[dimension]["total"] += 1
            if item.get("answer_scoring") == '1':
                dimension_stats[dimension]["correct"] += 1

            # 统计 profile_metrics
            for metric in profile_metrics:
                dimension_stats[dimension][f"{metric}_sum"] += item.get(metric, 0)

        duration = item.get("duration", 0)
        if duration <= 60:
            key = "0-60"
        elif duration <= 300:
            key = "60-300"
        else:
            key = "300+"

        duration_stats[key]["total"] += 1
        if item.get("answer_scoring") == '1':
            duration_stats[key]["correct"] += 1

        # 统计 profile_metrics
        for metric in profile_metrics:
            duration_stats[key][f"{metric}_sum"] += item.get(metric, 0)

    overall_accuracy = total_correct / total_items if total_items > 0 else 0

    dimension_metrics = {}
    for dimension, stats in dimension_stats.items():
        accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        profile_averages = {
            metric: stats[f"{metric}_sum"] / stats["total"] if stats["total"] > 0 else 0
            for metric in profile_metrics
        }
        dimension_metrics[dimension] = {"accuracy": accuracy, **profile_averages}

    duration_metrics = {}
    for key, stats in duration_stats.items():
        accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        profile_averages = {
            metric: stats[f"{metric}_sum"] / stats["total"] if stats["total"] > 0 else 0
            for metric in profile_metrics
        }
        duration_metrics[key] = {"accuracy": accuracy, **profile_averages}

    overall_metrics = {"overall":{
        "accuracy": overall_accuracy,
        **{
            metric: sum(item.get(metric, 0) for item in data) / total_items if total_items > 0 else 0
            for metric in profile_metrics
        }
        }
    }

    return {
        "overall_metrics": overall_metrics,
        "dimension_metrics": dimension_metrics,
        "duration_metrics": duration_metrics
    }


def format_results(results):
    def format_dict(d):
        return {k: f"{v:.3f}" if isinstance(v, (int, float)) else v for k, v in d.items()}

    formatted_results = {
        "Overall Metrics": format_dict(results["overall_metrics"]['overall']),
        "Metrics by Dimension": {
            dimension: format_dict(metrics) for dimension, metrics in results["dimension_metrics"].items()
        },
        "Metrics by Duration": {
            duration: format_dict(metrics) for duration, metrics in results["duration_metrics"].items()
        }
    }

    return formatted_results


def calu_acc_main(file_path, txt_file, profile_metrics):
    # Load data from the provided file path
    data = json.load(open(file_path, 'r', encoding='utf-8'))
    for item in data:
        item["answer_scoring"] = str(item["answer_scoring"])

    results = calculate_accuracy(data, profile_metrics)
    formatted_results = format_results(results)

    print("===== Statistics =====")
    print("Overall Metrics:")
    for key, value in formatted_results["Overall Metrics"].items():
        print(f"  {key}: {value}")

    print("\nMetrics by Dimension:")
    for dimension, metrics in formatted_results["Metrics by Dimension"].items():
        print(f"  {dimension}:")
        for key, value in metrics.items():
            print(f"    {key}: {value}")

    print("\nMetrics by Duration:")
    for duration, metrics in formatted_results["Metrics by Duration"].items():
        print(f"  {duration}:")
        for key, value in metrics.items():
            print(f"    {key}: {value}")

    print('\n\n')

    with open(txt_file, 'w') as file:
        file.write("===== Statistics =====\n")
        file.write("Overall Metrics:\n")
        for key, value in formatted_results["Overall Metrics"].items():
            file.write(f"  {key}: {value}\n")

        file.write("\nMetrics by Dimension:\n")
        for dimension, metrics in formatted_results["Metrics by Dimension"].items():
            file.write(f"  {dimension}:\n")
            for key, value in metrics.items():
                file.write(f"    {key}: {value}\n")

        file.write("\nMetrics by Duration:\n")
        for duration, metrics in formatted_results["Metrics by Duration"].items():
            file.write(f"  {duration}:\n")
            for key, value in metrics.items():
                file.write(f"    {key}: {value}\n")

        file.write('\n\n')

    # 处理 results，去掉 key，将所有值聚合到一个字典中
    final_results = {}
    for key, value in results.items():
        if key == "overall_metrics":
            value["overall"]["overall"] = value["overall"]["accuracy"]
        final_results.update(value)

    return final_results