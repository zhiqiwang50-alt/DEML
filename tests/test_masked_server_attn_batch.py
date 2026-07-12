import unittest

from pia_masked_server_attn_pia import (
    GpuMemory,
    build_auto_batch_tasks,
    build_child_env,
    build_dev_ablation_tasks,
    choose_best_candidate,
    plan_auto_batch_jobs,
)


class AutoBatchPlanningTests(unittest.TestCase):
    def test_allocates_more_jobs_to_gpus_with_more_free_memory(self):
        gpus = [
            GpuMemory(index=0, free_mb=10644, total_mb=24564),
            GpuMemory(index=1, free_mb=6514, total_mb=24564),
            GpuMemory(index=2, free_mb=5216, total_mb=24564),
            GpuMemory(index=3, free_mb=8278, total_mb=24564),
        ]

        plan = plan_auto_batch_jobs(
            gpus,
            min_free_mb_per_job=4096,
            reserve_free_mb=2048,
            max_jobs_per_gpu=2,
            max_total_jobs=6,
        )

        self.assertEqual(plan.total_jobs, 4)
        self.assertEqual(plan.jobs_by_gpu, {0: 2, 1: 1, 3: 1})
        self.assertNotIn(2, plan.jobs_by_gpu)

    def test_keeps_at_least_one_job_on_the_best_gpu_when_memory_is_tight(self):
        gpus = [
            GpuMemory(index=0, free_mb=2200, total_mb=24564),
            GpuMemory(index=1, free_mb=1800, total_mb=24564),
        ]

        plan = plan_auto_batch_jobs(
            gpus,
            min_free_mb_per_job=4096,
            reserve_free_mb=2048,
            max_jobs_per_gpu=2,
            max_total_jobs=4,
        )

        self.assertEqual(plan.total_jobs, 1)
        self.assertEqual(plan.jobs_by_gpu, {0: 1})

    def test_respects_total_job_cap(self):
        gpus = [
            GpuMemory(index=0, free_mb=22000, total_mb=24564),
            GpuMemory(index=1, free_mb=22000, total_mb=24564),
            GpuMemory(index=2, free_mb=22000, total_mb=24564),
        ]

        plan = plan_auto_batch_jobs(
            gpus,
            min_free_mb_per_job=4096,
            reserve_free_mb=2048,
            max_jobs_per_gpu=4,
            max_total_jobs=5,
        )

        self.assertEqual(plan.total_jobs, 5)
        self.assertEqual(sum(plan.jobs_by_gpu.values()), 5)

    def test_builds_method_layer_tasks_round_robin_over_planned_slots(self):
        plan = plan_auto_batch_jobs(
            [
                GpuMemory(index=0, free_mb=10644, total_mb=24564),
                GpuMemory(index=3, free_mb=8278, total_mb=24564),
            ],
            min_free_mb_per_job=4096,
            reserve_free_mb=2048,
            max_jobs_per_gpu=2,
            max_total_jobs=4,
        )

        tasks = build_auto_batch_tasks(
            methods=["B0", "P2"],
            target_layers=[11, 17],
            jobs_by_gpu=plan.jobs_by_gpu,
        )

        self.assertEqual([task.gpu_index for task in tasks], [0, 0, 3, 0])
        self.assertEqual([(task.method, task.target_layer) for task in tasks], [("B0", 11), ("P2", 11), ("B0", 17), ("P2", 17)])

    def test_child_env_sets_gpu_and_transformers_cache(self):
        env = build_child_env(
            base_env={"PATH": "/bin"},
            gpu_index=3,
            transformers_cache="/home/zhiqi/data/hf_cache/transformers",
        )

        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "3")
        self.assertEqual(env["TRANSFORMERS_CACHE"], "/home/zhiqi/data/hf_cache/transformers")
        self.assertEqual(env["PATH"], "/bin")

    def test_dev_ablation_tasks_keep_baselines_separate_from_mts_grid(self):
        tasks = build_dev_ablation_tasks(
            gpu_slots=[0, 1],
            seed=42,
            target_layer=17,
            epoch=100,
            top_k=10,
            weight_min=0.25,
            betas=[0.25, 0.5],
            powers=[0.25],
            weight_maxes=[2.0, 4.0],
        )

        baseline = [task for task in tasks if task.output_subdir == "dev_ablation_baselines"]
        grid = [task for task in tasks if task.output_subdir == "dev_ablation_runs"]

        self.assertEqual([task.method for task in baseline], ["B0", "B0V", "P1"])
        self.assertEqual(len(grid), 8)
        self.assertEqual({task.method for task in grid}, {"P2", "P3"})
        self.assertEqual(max(tasks, key=lambda task: task.gpu_index).gpu_index, 1)
        self.assertEqual(len({task.run_name for task in tasks}), len(tasks))

    def test_candidate_selection_filters_bad_audits_and_prefers_gentler_tie(self):
        rows = [
            {
                "method": "mts_mean_query",
                "token_accuracy_mean": 0.90,
                "bleu_mean": 0.90,
                "weight_std_mean": 0.2,
                "weight_mean_mean": 1.0,
                "weight_max_max": 1.4,
                "configured_weight_max": 2.0,
                "fixed_weight_sum_max": 0.0,
                "completed_sample_count": 28,
            },
            {
                "method": "mts_mean_query",
                "token_accuracy_mean": 0.95,
                "bleu_mean": 0.91,
                "weight_std_mean": 0.4,
                "weight_mean_mean": 1.0,
                "weight_max_max": 1.8,
                "configured_weight_max": 2.0,
                "fixed_weight_sum_max": 0.0,
                "completed_sample_count": 28,
            },
            {
                "method": "mts_mean_query",
                "token_accuracy_mean": 0.95,
                "bleu_mean": 0.91,
                "weight_std_mean": 0.1,
                "weight_mean_mean": 1.0,
                "weight_max_max": 1.8,
                "configured_weight_max": 2.0,
                "fixed_weight_sum_max": 0.0,
                "completed_sample_count": 28,
            },
            {
                "method": "mts_mean_query",
                "token_accuracy_mean": 0.99,
                "bleu_mean": 0.99,
                "weight_std_mean": 0.1,
                "weight_mean_mean": 1.0,
                "weight_max_max": 3.0,
                "configured_weight_max": 2.0,
                "fixed_weight_sum_max": 0.0,
                "completed_sample_count": 28,
            },
        ]

        best = choose_best_candidate(rows, method="mts_mean_query", b0v_acc=0.90, min_completed=2)

        self.assertIsNotNone(best)
        self.assertEqual(best["weight_std_mean"], 0.1)


if __name__ == "__main__":
    unittest.main()
