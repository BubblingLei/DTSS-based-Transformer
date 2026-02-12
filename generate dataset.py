import os
import multiprocessing
import time
import subprocess
from tqdm import tqdm
from multiprocessing import Manager
import sys
import logging

# 设置日志文件
logging.basicConfig(
    filename="simulation.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# 创建一个全局锁，用于协调 tqdm 的多进程输出
manager = Manager()
tqdm_lock = manager.Lock()


d_values = [0.5, 0.6, 0.7]  # 5 个值
emission_gap_values = [10, 12, 14, 16, 18]  # 6 个值

# 定义参数组
param_groups = [
    {
        # 时间相关参数
        "num_time_slots": 30000,  # 总时隙数量
        "delta_t": 0.001,  # 每个小时间段的长度（秒）
        "emission_gap": eg,  # 发射间隙（每次信号发射后检测的次数）
        "sequence_length": 10,  # 序列长度（每个时隙的信号序列长度）
        "final_gap": 30,  # 发射完成后等待的检测次数

        # 发射器和接收器相关参数
        "num_emitters": 4,  # 发射器数量
        "molecules_per_emitter": 10000,  # 每个发射器发送的分子数量
        "radius_vessel": 5e-6,  # 血管半径（米）
        "blood_flow_speed": 300e-6,  # 中心流速（米/秒）
        "radius_receiver": 1e-6,  # 接收器半径（米）
        "initial_position_receiver": [15e-6, 0.0, 0.0],  # 接收器初始位置（米）

        # 物理参数
        "D_emitters": [120e-12, 120e-12, 120e-12, 120e-12],  # 每个发射器的扩散系数（米^2/秒）

        # 发射器位置参数（自动生成 senders）
        "d": d,  # 相邻发射器在 x 轴方向的距离（单位：米，实际值需乘以 1e-6）
        "r": 0.5,  # yz 坐标的绝对值（单位：米，实际值需乘以 1e-6）
    }
    for d in d_values
    for eg in emission_gap_values
]

# 仿真代码模板，所有参数从外部传入
simulation_code_template = """
import numpy as np
import os
import time
import pandas as pd
import torch
from tqdm import tqdm
import sys
import logging

# 设置日志
logging.basicConfig(
    filename="simulation.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# 仿真参数（从外部传入）
delta_t = {delta_t}
num_emitters = {num_emitters}
molecules_per_emitter = {molecules_per_emitter}
radius_vessel = {radius_vessel}
blood_flow_speed = {blood_flow_speed}
radius_receiver = {radius_receiver}
initial_position_receiver = np.array({initial_position_receiver})
D_emitters = {D_emitters}
num_time_slots = {num_time_slots}
emission_gap = {emission_gap}
sequence_length = {sequence_length}
final_gap = {final_gap}
d = {d}
r = {r}
position = {position}  # 接收 position 参数，用于进度条显示位置

# 初始化固定发射器位置
senders = torch.tensor({senders}) * 1e-6  # 单位：米

def absorb_molecules(
    molecule_positions,
    molecule_ids,
    radius_vessel,
    radius_receiver,
    position_receiver
):
    # 计算分子到接收器的距离
    distances_to_receiver = np.sqrt(
        np.sum((molecule_positions - position_receiver) ** 2, axis=1)
    )
    within_receiver = distances_to_receiver <= radius_receiver

    # 统计每个发射器检测到的分子数量
    detected_molecules = [
        np.sum(molecule_ids[within_receiver] == i) 
        for i in range(num_emitters)
    ]

    # 移除被接收器吸收的分子
    molecule_positions = molecule_positions[~within_receiver]
    molecule_ids = molecule_ids[~within_receiver]

    # 计算剩余分子到血管壁的距离
    distances_to_vessel = np.sqrt(
        np.sum(molecule_positions[:, 1:] ** 2, axis=1)
    )
    within_vessel = distances_to_vessel <= (radius_vessel - 1e-6)

    # 移除被血管壁吸收的分子
    molecule_positions = molecule_positions[within_vessel]
    molecule_ids = molecule_ids[within_vessel]

    return detected_molecules, molecule_positions, molecule_ids

def simulate_step(
    molecule_positions,
    molecule_ids,
    delta_t,
    blood_flow_speed,
    radius_vessel,
    radius_receiver,
    position_receiver
):
    # 扩散
    for i, D in enumerate(D_emitters):
        mask = (molecule_ids == i)
        if mask.sum() > 0:
            step_size = np.sqrt(2 * D * delta_t)
            random_steps = np.random.normal(
                0, step_size, size=(mask.sum(), 3)
            )
            molecule_positions[mask] += random_steps

    # 血液流动效应
    y = molecule_positions[:, 1]
    z = molecule_positions[:, 2]
    r_squared = y ** 2 + z ** 2
    r = np.sqrt(r_squared)
    Vf = 2 * blood_flow_speed * (1 - (r / radius_vessel) ** 2)
    x_flow = Vf * delta_t
    molecule_positions[:, 0] += x_flow

    # 吸收分子
    detected_molecules, molecule_positions, molecule_ids = absorb_molecules(
        molecule_positions,
        molecule_ids,
        radius_vessel,
        radius_receiver,
        position_receiver
    )

    return molecule_positions, molecule_ids, detected_molecules

def simulate_single_data_group():
    emitter_positions = senders.numpy()
    receiver_position = initial_position_receiver
    molecule_positions = np.empty((0, 3))
    molecule_ids = np.empty(0, dtype=int)

    # 生成信号序列
    signal_sequences = np.random.choice(
        [0, 1], size=(num_emitters, num_time_slots, sequence_length)
    )

    all_detections = []
    all_signals = []

    # 在模拟代码模板中修改进度条设置
    progress = tqdm(
        range(num_time_slots),
        desc=f"Task {position}",  # 使用传入的position值
        position=position,  # 固定位置
        leave=False,
        ncols=100,
        dynamic_ncols=False,
        mininterval=0.5,  # 降低刷新频率
        maxinterval=1.0
    )
    for slot in progress:
        slot_detections = []
        for signal_idx in range(sequence_length):
            new_positions = []
            new_ids = []
            for emitter in range(num_emitters):
                if signal_sequences[emitter, slot, signal_idx] == 1:
                    new_positions.append(
                        np.tile(emitter_positions[emitter], (molecules_per_emitter, 1))
                    )
                    new_ids.extend([emitter] * molecules_per_emitter)
            if new_positions:
                new_positions_array = np.vstack(new_positions)
                new_ids = np.array(new_ids)
                molecule_positions = np.vstack([molecule_positions, new_positions_array])
                molecule_ids = np.concatenate([molecule_ids, new_ids])
            for step in range(emission_gap):
                molecule_positions, molecule_ids, detected = simulate_step(
                    molecule_positions,
                    molecule_ids,
                    delta_t,
                    blood_flow_speed,
                    radius_vessel,
                    radius_receiver,
                    receiver_position
                )
                slot_detections.append(detected)
        for step in range(final_gap):
            molecule_positions, molecule_ids, detected = simulate_step(
                molecule_positions,
                molecule_ids,
                delta_t,
                blood_flow_speed,
                radius_vessel,
                radius_receiver,
                receiver_position
            )
            slot_detections.append(detected)
        all_detections.append(slot_detections)
        all_signals.append(signal_sequences[:, slot])
        progress.refresh()
        sys.stdout.flush()

    return np.array(all_detections), np.array(all_signals)

def generate_data():
    detections, signals = simulate_single_data_group()
    data_emitter_0 = []
    data_emitter_1 = []
    data_emitter_2 = []
    data_emitter_3 = []
    data_total = []

    for i in range(len(detections)):
        U_i = detections[i]
        if len(U_i) != final_gap + emission_gap * sequence_length:
            raise ValueError(
                "时隙 " + str(i) + " 的检测次数为 " + str(len(U_i)) + "，应为 " + str(final_gap + emission_gap * sequence_length) + " 次"
            )
        for j in range(len(U_i)):
            data_emitter_0.append(U_i[j][0])
            data_emitter_1.append(U_i[j][1])
            data_emitter_2.append(U_i[j][2])
            data_emitter_3.append(U_i[j][3])
            data_total.append(sum(U_i[j]))

    flat_signals = signals.reshape(signals.shape[0], -1)
    df_signals = pd.DataFrame(
        flat_signals,
        columns=[
            f"Emitter_{{e}}_Signal_{{s}}"
            for e in range(4)
            for s in range(sequence_length)
        ]
    )

    # 修改输出文件夹名称，添加 num_time_slots 和 emission_gap 信息
    output_folder = '/data/output_data_15e-6_t{emission_gap}_{sequence_length}_r{r}_d{d}'.format(
        emission_gap=emission_gap,
        sequence_length=sequence_length,
        r=str(r).replace('e-06', ''),  # 去掉 e-6，简写为 r0.5
        d=str(d).replace('e-06', '')   # 去掉 e-6，简写为 d0.3
    )
    os.makedirs(output_folder, exist_ok=True)

    signal_file = os.path.join(output_folder, 'signal_data.csv')
    df_signals.to_csv(signal_file, index=False)
    logging.info(f"信号文件已保存至: {{signal_file}} (形状: {{df_signals.shape}})")

    def save_emitter_data(data, filename):
        df = pd.DataFrame(
            [
                data[i:i + (final_gap + emission_gap * sequence_length)]
                for i in range(0, len(data), (final_gap + emission_gap * sequence_length))
            ],
            columns=[
                f"D_{{i + 1}}"
                for i in range(final_gap + emission_gap * sequence_length)
            ]
        )
        df.to_csv(os.path.join(output_folder, filename), index=False)

    save_emitter_data(data_emitter_0, "emitter_0_data.csv")
    save_emitter_data(data_emitter_1, "emitter_1_data.csv")
    save_emitter_data(data_emitter_2, "emitter_2_data.csv")
    save_emitter_data(data_emitter_3, "emitter_3_data.csv")
    save_emitter_data(data_total, "emitter_total_data.csv")

    logging.info("所有文件保存完成，数据结构验证：")
    logging.info(f"- 信号数据: {{len(df_signals)}} 个时隙 x {{len(df_signals.columns)}} 个信号")
    logging.info(
        f"- 发射器数据: 每个文件 "
        f"{{len(data_emitter_0) // (final_gap + emission_gap * sequence_length)}} 个时隙 x "
        f"{{final_gap + emission_gap * sequence_length}} 次检测"
    )

if __name__ == "__main__":
    start_time = time.time()
    generate_data()
    logging.info(f"数据生成完成，耗时 {{time.time() - start_time:.2f}} 秒")
"""


# 生成 senders 坐标的函数
def generate_senders(d, r):
    # x 坐标：四个发射器，等间距 d，关于原点对称
    x_coords = [-1.5 * d, -0.5 * d, 0.5 * d, 1.5 * d]

    # yz 坐标：按照 90° 旋转，绝对值等于 r
    yz_coords = [
        [r, -r],  # 第一个发射器
        [-r, -r],  # 第二个发射器
        [-r, r],  # 第三个发射器
        [r, r]  # 第四个发射器
    ]

    # 组合 x 和 yz 坐标
    senders = [[x, yz[0], yz[1]] for x, yz in zip(x_coords, yz_coords)]
    return senders


# 生成 .py 文件并运行
def run_simulation(params, idx):
    logging.info(f"任务 {idx} 的参数: {params}")  # 日志记录
    # 自动生成 senders
    params["senders"] = generate_senders(params["d"], params["r"])

    # 填充参数到代码模板，传递 position 参数
    code = simulation_code_template.format(
        delta_t=params["delta_t"],
        num_emitters=params["num_emitters"],
        molecules_per_emitter=params["molecules_per_emitter"],
        radius_vessel=params["radius_vessel"],
        blood_flow_speed=params["blood_flow_speed"],
        radius_receiver=params["radius_receiver"],
        initial_position_receiver=params["initial_position_receiver"],
        D_emitters=params["D_emitters"],
        num_time_slots=params["num_time_slots"],
        emission_gap=params["emission_gap"],
        sequence_length=params["sequence_length"],
        final_gap=params["final_gap"],
        d=params["d"],
        r=params["r"],
        senders=params["senders"],
        position=idx + 1  # 主进度条在第 0 行，子进度条从第 1 行开始
    )
    # 保存为 .py 文件
    script_path = f"simulate_params_vector_{idx}.py"
    with open(script_path, "w") as f:
        f.write(code)
    # 使用 cpu_py310 环境的 Python 可执行文件运行脚本
    try:
        env_python = "/root/miniconda3/envs/cpu_py310/bin/python"
        # 设置环境变量 PYTHONUNBUFFERED=1，确保输出不被缓冲
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        result = subprocess.run(
            [env_python, script_path],
            check=True,
            env=env
        )
        logging.info(f"子脚本 {script_path} 运行成功")
    except subprocess.CalledProcessError as e:
        logging.error(f"子脚本 {script_path} 运行失败")
        logging.error(f"错误信息 (stderr): {e.stderr if hasattr(e, 'stderr') else '不可用'}")
        logging.error(f"标准输出 (stdout): {e.stdout if hasattr(e, 'stdout') else '不可用'}")
        raise
    return idx  # 返回任务索引，用于更新主进度条


# 定义一个普通函数替代 lambda
def run_simulation_wrapper(task):
    params, idx = task
    return run_simulation(params, idx)


if __name__ == "__main__":
    try:
        # 设置并行任务数量
        max_workers = min(15, len(param_groups))  # 8 个任务并行运行
        logging.info(f"并行运行 {max_workers} 个任务...")
        logging.info(f"总任务数量: {len(param_groups)}")

        # 检查哪些任务已完成
        completed_tasks = set()
        for idx, params in enumerate(param_groups):
            # 构造输出文件夹路径，与子脚本中的 output_folder 一致
            output_folder = f"/data/output_data_15e-6_t{params['emission_gap']}_{params['sequence_length']}_r{str(params['r']).replace('e-06', '')}_d{str(params['d']).replace('e-06', '')}"
            signal_file = os.path.join(output_folder, 'signal_data.csv')
            if os.path.exists(signal_file):
                logging.info(f"任务 {idx} 已完成，跳过...")
                completed_tasks.add(idx)

        # 只运行未完成的任务
        tasks_to_run = [(params, idx) for idx, params in enumerate(param_groups) if idx not in completed_tasks]
        logging.info(f"剩余未完成任务数量: {len(tasks_to_run)}")

        if tasks_to_run:
            # 在主程序开始时初始化
            tqdm.set_lock(multiprocessing.RLock())
            # 创建主进度条
            # 在主程序中修改主进度条的创建方式
            # 修改主进度条设置
            main_progress = tqdm(
                total=len(tasks_to_run),
                desc="Main Progress",
                position=0,  # 固定在0行
                leave=True,
                ncols=100,
                dynamic_ncols=False,
                lock_args=(False,)
            )

            # 修改进程池创建方式
            with multiprocessing.Pool(
                    processes=max_workers,
                    initializer=tqdm.set_lock,
                    initargs=(tqdm.get_lock(),)
            ) as pool:
                results = []
                for idx in pool.imap_unordered(run_simulation_wrapper, tasks_to_run):
                    main_progress.update(1)
                    results.append(idx)
            main_progress.close()

        print("所有仿真任务完成！")
        print("日志已保存至 simulation.log，请查看详细信息。")

    except KeyboardInterrupt:
        print("\n程序被用户中断，优雅退出...")
        logging.info("程序被用户中断，优雅退出...")
        # 确保进程池被正确关闭
        if 'pool' in locals():
            pool.terminate()
            pool.join()
        exit(0)