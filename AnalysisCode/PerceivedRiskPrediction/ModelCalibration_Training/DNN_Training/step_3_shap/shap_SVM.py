import os
import numpy as np
import pandas as pd
import pickle
import torch
import shap
import argparse
import sys 
sys.path.append("../step_2_NN_train/") 
from modules.MC_Dropout_Model import MC_Dropout_Model
from modules.utils import *
from modules.MC_Dropout_Wrapper import MC_Dropout_Wrapper
import matplotlib.pyplot as plt
from scipy.io import savemat


def generate_prediction(input):
    input_tensor = torch.as_tensor(input, dtype=torch.float32)
    global best_model
    with torch.no_grad():
        pred = best_model(input_tensor)
    arr = pred.detach().cpu().numpy()
    # ensure 1D output (use first column if model returns [mean, uncertainty], etc.)
    if arr.ndim == 2:
        arr = arr[:, 0]
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)[:, 0]
    else:
        arr = arr.squeeze()
    return arr


def shap_cal(model_name, data, K, event_id, event_duration, event_numb, feature_names):
    in_dim = data.shape[1] - 1   # input dimension
    x_all, y_all =  data[:, :in_dim], data[:, in_dim:]
    x_all_means, x_all_stds = x_all.mean(axis = 0), x_all.var(axis = 0)**0.5
    x_all_stds[x_all_stds == 0] = 1.0
    x_all = (x_all - x_all_means)/x_all_stds  # standard by means nad stds
    samples_selected = shap.sample(x_all, K) 

    # Ensure output dir exists and compute/load SHAP cache (recompute if K changed)
    os.makedirs(model_name, exist_ok=True)
    pkl_path = os.path.join(model_name, 'shap_data_all_random.pkl')

    def _compute_and_cache(background):
        exp = shap.KernelExplainer(generate_prediction, background)
        inst = background
        vals = exp.shap_values(inst)
        with open(pkl_path, 'wb') as f:
            pickle.dump({"samples_selected": background, "shap_values": vals}, f)
        return exp, inst, vals

    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            shap_data_all_random = pickle.load(f)
        cached_samples = shap_data_all_random.get("samples_selected")
        cached_vals = shap_data_all_random.get("shap_values")
        if cached_samples is not None and getattr(cached_samples, 'shape', (0,))[0] == K:
            samples_selected = cached_samples
            explainer = shap.KernelExplainer(generate_prediction, samples_selected)
            test_instance = samples_selected
            shap_values = cached_vals if cached_vals is not None else explainer.shap_values(test_instance)
        else:
            # K changed → resample and recompute
            samples_selected = shap.sample(x_all, K)
            explainer, test_instance, shap_values = _compute_and_cache(samples_selected)
    else:
        samples_selected = shap.sample(x_all, K)
        explainer, test_instance, shap_values = _compute_and_cache(samples_selected)

    shap.initjs()

    # ----- Local dynamics for the selected event -----
    start = (event_id - 1) * event_duration
    end = start + event_duration
    event_samples = x_all[start:end]  # (T, n_features)

    # Compute SHAP values for this event
    shap_values_event = explainer.shap_values(event_samples)
    sv_event = shap_values_event[0] if isinstance(shap_values_event, list) else shap_values_event
    sv_event = np.asarray(sv_event)
    # reduce any extra output dims until (T, n_features)
    while sv_event.ndim > 2:
        sv_event = sv_event[..., 0]
    if sv_event.ndim == 1:
        sv_event = sv_event.reshape(-1, 1)

    # 1) Heatmap over time (prefer SHAP heatmap, fallback to matplotlib)
    try:
        expl_event = shap.Explanation(
            values=sv_event,
            base_values=float(np.atleast_1d(explainer.expected_value)[0]),
            data=event_samples,
            feature_names=feature_names
        )
        plt.figure(figsize=(8, 4))
        shap.plots.heatmap(
            expl_event,
            instance_order=np.arange(len(event_samples)),
            max_display=20,
            show=False
        )
        plt.savefig(f"{model_name}/shap_ts_heatmap_event{event_id}.pdf", dpi=600, bbox_inches='tight')
        plt.close()
    except Exception:
        imp_evt = np.mean(np.abs(sv_event), axis=0)
        order_evt = np.argsort(imp_evt)[::-1][:20]
        plt.figure(figsize=(8, 4))
        plt.imshow(sv_event[:, order_evt].T, aspect='auto', cmap='coolwarm', interpolation='nearest')
        plt.yticks(range(len(order_evt)), [feature_names[i] for i in order_evt])
        plt.xlabel("time index")
        plt.colorbar(label="SHAP value")
        plt.tight_layout()
        plt.savefig(f"{model_name}/shap_ts_heatmap_event{event_id}.pdf", dpi=600, bbox_inches='tight')
        plt.close()

    # 2) Time-series of Top-k features' SHAP
    topk = 6
    order2 = np.argsort(np.mean(np.abs(sv_event), axis=0))[::-1][:topk]
    plt.figure(figsize=(7, 3))
    for i in order2:
        plt.plot(sv_event[:, i], label=feature_names[i])
    plt.legend(loc="upper right", ncol=2, fontsize=7)
    plt.xlabel("time index"); plt.ylabel("SHAP")
    plt.tight_layout()
    plt.savefig(f"{model_name}/shap_lines_event{event_id}.pdf", dpi=600, bbox_inches='tight')
    plt.close()

    # 3) Local explanation at a single time step (waterfall)
    t = len(event_samples) // 2
    single = event_samples[t:t+1]
    sv_t = explainer.shap_values(single)
    sv_t = sv_t[0] if isinstance(sv_t, list) else sv_t
    sv_t = np.asarray(sv_t).reshape(1, -1)
    expl_t = shap.Explanation(
        values=sv_t,
        base_values=float(np.atleast_1d(explainer.expected_value)[0]),
        data=single,
        feature_names=feature_names
    )
    shap.plots.waterfall(expl_t[0], show=False)
    plt.savefig(f"{model_name}/shap_waterfall_event{event_id}_t{t}.pdf", dpi=600, bbox_inches='tight')
    plt.close()


# #  selected events results
#     for event_id in range(1, event_numb + 1):  
#         start = (event_id - 1) * event_duration
#         end = event_id * event_duration
#         event_samples = x_all[start:end]
#         # test for instance order
#         order_test_output = generate_prediction(event_samples)
#         plt.plot(order_test_output)
#         file_name = f"{model_name}/shap_ts_heatmap_{event_id}_prediction.pdf" 
#         plt.savefig(file_name, dpi=600, bbox_inches='tight')
#         plt.close()
#         # Compute SHAP values for the current event samples
#         shap_values_event = explainer.shap_values(event_samples)
#         shap_data_event = { "event_samples": event_samples,
#                             "explainer": explainer,
#                             "shap_values_event": shap_values_event
#                             }
#         file_name = f"{model_name}/shap_data_event_{event_id}.pkl" 
#         with open(file_name, 'wb') as file:
#             pickle.dump(shap_data_event, file)

#         expl = shap.Explanation(values=shap_values_event[0], feature_names=feature_names)
#         plt.figure(figsize=(10, 5)) 
#         shap.plots.heatmap(expl, max_display= 20, instance_order = np.arange(0, event_duration))
#         file_name = f"{model_name}/shap_ts_heatmap_{event_id}.pdf" 
#         plt.savefig(file_name, dpi=600, bbox_inches='tight')
#         plt.close()

    # Prepare SHAP Explanation with correct shapes
    sv = shap_values[0] if isinstance(shap_values, list) else shap_values
    # ---- make sure SHAP values and background have consistent numeric shapes ----
    sv = np.asarray(sv)
    while sv.ndim > 2:
        sv = sv[..., 0]
    if sv.ndim == 1:
        sv = sv.reshape(-1, 1)
    sv = sv.astype(np.float64, copy=False)

    test_instance = np.asarray(test_instance)
    if test_instance.ndim == 1:
        test_instance = test_instance.reshape(1, -1)
    test_instance = test_instance.astype(np.float64, copy=False)

    # Align feature names length if needed (fallback generic names)
    if sv.shape[1] != len(feature_names):
        feature_names = [f"f{i}" for i in range(sv.shape[1])]

    base_value = float(np.atleast_1d(explainer.expected_value)[0])
    explanation = shap.Explanation(
        values=sv,
        base_values=base_value,
        data=test_instance,
        feature_names=feature_names
    )
    file_name = f"{model_name}/explanation.mat"
    savemat(file_name, {'explanation': explanation.data})

    file_path = f'{model_name}/shap_bar_all_random.pdf'
    try:
        plt.figure(figsize=(6, 4))
        shap.plots.bar(explanation, max_display=10, show=False)
        plt.savefig(file_path, dpi=600, bbox_inches='tight')
        plt.close()
    except Exception as e:
        # Fallback: aggregate mean(|SHAP|) and plot with matplotlib
        imp = np.mean(np.abs(sv), axis=0).astype(np.float64, copy=False).ravel()
        imp = np.nan_to_num(imp, nan=0.0, posinf=0.0, neginf=0.0)
        order = np.argsort(imp)[::-1]
        topk = min(10, imp.shape[0])
        top = order[:topk]
        vals = imp[top][::-1]
        names = [feature_names[i] for i in top][::-1]
        plt.figure(figsize=(6, 4))
        # plt.barh(range(topk), vals)
        plt.barh(range(topk), vals, color="#FF0051")
        plt.yticks(range(topk), names, fontsize=8)
        plt.xlabel("mean(|SHAP|)")
        plt.tight_layout()
        plt.savefig(file_path, dpi=600, bbox_inches='tight')
        plt.close()


# # two feature coupled
#     test_instance_df = pd.DataFrame(data=test_instance, columns=feature_names)
#     plt.figure(figsize=(6, 4))
#     shap.dependence_plot(1, shap_values[0], test_instance_df, interaction_index = "$Iv_{x,n}$", show=False)
#     file_path = f'{model_name}/shap_coupled_all_random.pdf'
#     plt.savefig(file_path ,dpi=600, bbox_inches='tight')
#     plt.close()
    return shap_values


# ---- CLI arguments ----
parser = argparse.ArgumentParser(description='SHAP for trained MC-Dropout model')
parser.add_argument('--scenario', type=str, default='SVM', choices=['HB','MB','SVM','LC_1','LC_2','LC_3'])
parser.add_argument('--experiment_index', type=str, default='2025')
parser.add_argument('--K', type=int, default=1500)
args = parser.parse_args()

SCN = args.scenario
model_name = SCN

path_data = f"../step_2_NN_train/data/{SCN}_feature_reg.npy"
data = np.load(path_data)

K = args.K  # background sample size for SHAP
event_id = 1
event_duration = 301
event_numb = 27

path_best_weight = f"../step_2_NN_train/models/{args.experiment_index}/{SCN}/best_model_{SCN}.pth"
best_model = MC_Dropout_Model(input_dim=data.shape[1]-1, output_dim=1, num_units=500, drop_prob=0.1)
try:
    state = torch.load(path_best_weight, map_location='cpu', weights_only=True)
except TypeError:
    state = torch.load(path_best_weight, map_location='cpu')
if isinstance(state, dict) and 'state_dict' in state:
    state = state['state_dict']
best_model.load_state_dict(state)
best_model.eval()
feature_names = ["$v_{x,s}$",
                    "$v_{x,n}$",
                    "$v_{x,n2}$",
                    "$v_{y,s}$",
                    "$a_{x,s}$",
                    "$a_{x,n}$",
                    "$a_{x,n2}$",
                    "$a_{y,s}$",
                    "$\Delta_x$",
                    "$\Delta_y$",
                    "$\Delta_{x2}$",
                    "$\Delta_{y2}$",
                    "$Iv_{x,n}$",
                    "$Iv_{y,n}$",
                    "$Iv_{x,s}$",
                    "$Iv_{y,s}$",
                    "$Iv_{x,n2}$",
                    "$Iv_{y,n2}$",
                    "$DRAC_{Rx}$",
                    "$DRAC_{Ry}$",
                    "$DRAC_{Ry2}$",
                    "$DRAC_{Ix}$",
                    "$DRAC_{Iy}$",
                    "$DRAC_{Iy2}$",
                    "$\Delta_{vx}$",
                    "$\Delta_{vy}$",
                    "$\Delta_{ax}$",
                    "$\Delta_{ay}$",
                    "$\Delta_{vx2}$",
                    "$\Delta_{vy2}$",
                    "$\Delta_{ax2}$",
                    "$\Delta_{ay2}$",
                    ]
shap_values = shap_cal(model_name = model_name,
                       data=data,
                       K = K,
                       event_id = event_id,
                       event_duration = event_duration,
                       event_numb = event_numb,
                       feature_names = feature_names)
