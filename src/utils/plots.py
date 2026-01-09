import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, precision_recall_curve
import seaborn as sns
import pandas as pd
import torch


def plot_anomaly_detection_analysis(scores, labels, pred_types, figsize=(20, 16)):
    if hasattr(scores, 'cpu'):
        scores = scores.cpu().numpy()
    if hasattr(labels, 'cpu'):
        labels = labels.cpu().numpy()
    if hasattr(pred_types, 'cpu'):
        pred_types = pred_types.cpu().numpy()
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    normal_mask = pred_types == 0
    current_fail_mask = pred_types == 1
    future_fail_mask = pred_types == 2
    
    normal_scores = scores[normal_mask]
    current_fail_scores = scores[current_fail_mask]
    future_fail_scores = scores[future_fail_mask]
    
    overall_roc = roc_auc_score(labels, scores)
    overall_pr = average_precision_score(labels, scores)
    
    if current_fail_mask.sum() > 0:
        current_labels = (pred_types == 1).astype(int)
        current_roc = roc_auc_score(current_labels, scores)
        current_pr = average_precision_score(current_labels, scores)
    
    if future_fail_mask.sum() > 0:
        future_labels = (pred_types == 2).astype(int)
        future_roc = roc_auc_score(future_labels, scores)
        future_pr = average_precision_score(future_labels, scores)
    
    ax1 = fig.add_subplot(gs[0, 0])
    fpr, tpr, _ = roc_curve(labels, scores)
    ax1.plot(fpr, tpr, linewidth=2, label=f'Overall (AUC={overall_roc:.4f})')
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('False Positive Rate', fontsize=11)
    ax1.set_ylabel('True Positive Rate', fontsize=11)
    ax1.set_title('ROC Curve', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 1])
    precision, recall, _ = precision_recall_curve(labels, scores)
    ax2.plot(recall, precision, linewidth=2, label=f'Overall (AP={overall_pr:.4f})')
    ax2.axhline(y=labels.mean(), color='k', linestyle='--', linewidth=1, alpha=0.5, 
                label=f'Baseline ({labels.mean():.4f})')
    ax2.set_xlabel('Recall', fontsize=11)
    ax2.set_ylabel('Precision', fontsize=11)
    ax2.set_title('Precision-Recall Curve', fontsize=13, fontweight='bold')
    ax2.legend(loc='lower left')
    ax2.grid(alpha=0.3)
    
    ax3 = fig.add_subplot(gs[0, 2])
    data_for_box = [normal_scores]
    labels_for_box = ['Normal']
    colors_box = ['green']
    
    if len(current_fail_scores) > 0:
        data_for_box.append(current_fail_scores)
        labels_for_box.append('Current\nFailure')
        colors_box.append('red')
    
    if len(future_fail_scores) > 0:
        data_for_box.append(future_fail_scores)
        labels_for_box.append('Future\nFailure')
        colors_box.append('orange')
    
    bp = ax3.boxplot(data_for_box, labels=labels_for_box, patch_artist=True, 
                     showfliers=False)
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax3.set_ylabel('Anomaly Score', fontsize=11)
    ax3.set_title('Score Distribution (Box Plot)', fontsize=13, fontweight='bold')
    ax3.grid(alpha=0.3, axis='y')
    
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.axis('off')
    stats_data = [['Metric', 'Normal', 'Current Fail', 'Future Fail']]
    groups = {
        "Normal": normal_scores,
        "Current Fail": current_fail_scores,
        "Future Fail": future_fail_scores
    }
    stats_data.append([
        'Count',
        str(len(normal_scores)),
        str(len(current_fail_scores)),
        str(len(future_fail_scores))
    ])
    stats_data.append([
        'Mean',
        fmt(normal_scores.mean()),
        fmt(current_fail_scores.mean()) if len(current_fail_scores) else "N/A",
        fmt(future_fail_scores.mean()) if len(future_fail_scores) else "N/A"
    ])
    stats_data.append([
        'Std',
        fmt(normal_scores.std()),
        fmt(current_fail_scores.std()) if len(current_fail_scores) else "N/A",
        fmt(future_fail_scores.std()) if len(future_fail_scores) else "N/A"
    ])
    robust = {k: robust_stats(v) for k, v in groups.items() if len(v) > 0}
    stats_data.append([
        'Median',
        fmt(robust['Normal']['median']),
        fmt(robust['Current Fail']['median']) if 'Current Fail' in robust else "N/A",
        fmt(robust['Future Fail']['median']) if 'Future Fail' in robust else "N/A"
    ])
    stats_data.append([
        'IQR',
        fmt(robust['Normal']['iqr']),
        fmt(robust['Current Fail']['iqr']) if 'Current Fail' in robust else "N/A",
        fmt(robust['Future Fail']['iqr']) if 'Future Fail' in robust else "N/A"
    ])
    stats_data.append([
        'MAD',
        fmt(robust['Normal']['mad']),
        fmt(robust['Current Fail']['mad']) if 'Current Fail' in robust else "N/A",
        fmt(robust['Future Fail']['mad']) if 'Future Fail' in robust else "N/A"
    ])
    tm_n, ts_n = trimmed_mean_std(normal_scores)
    tm_c, ts_c = trimmed_mean_std(current_fail_scores) if len(current_fail_scores) else (None, None)
    tm_f, ts_f = trimmed_mean_std(future_fail_scores) if len(future_fail_scores) else (None, None)
    stats_data.append([
        'Trimmed Mean',
        fmt(tm_n),
        fmt(tm_c),
        fmt(tm_f)
    ])
    stats_data.append([
        'Trimmed Std',
        fmt(ts_n),
        fmt(ts_c),
        fmt(ts_f)
    ])
    table = ax4.table(cellText=stats_data, cellLoc='center', loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    ax4.set_title('Summary Statistics', fontsize=13, fontweight='bold', pad=20)

    ax5 = fig.add_subplot(gs[1, 1])
    metrics_text = f"Overall Performance:\n"
    metrics_text += f"  ROC AUC: {overall_roc:.4f}\n"
    metrics_text += f"  PR AUC: {overall_pr:.4f}\n\n"
    
    if current_fail_mask.sum() > 0:
        metrics_text += f"Current Failure Detection:\n"
        metrics_text += f"  ROC AUC: {current_roc:.4f}\n"
        metrics_text += f"  PR AUC: {current_pr:.4f}\n\n"
    
    if future_fail_mask.sum() > 0:
        metrics_text += f"Future Failure Prediction:\n"
        metrics_text += f"  ROC AUC: {future_roc:.4f}\n"
        metrics_text += f"  PR AUC: {future_pr:.4f}\n\n"
    
    if len(current_fail_scores) > 0:
        cohen_d_current = (current_fail_scores.mean() - normal_scores.mean()) / \
                         np.sqrt((current_fail_scores.std()**2 + normal_scores.std()**2) / 2)
        metrics_text += f"Cohen's d (Normal vs Current): {cohen_d_current:.4f}\n"
    
    if len(future_fail_scores) > 0:
        cohen_d_future = (future_fail_scores.mean() - normal_scores.mean()) / \
                        np.sqrt((future_fail_scores.std()**2 + normal_scores.std()**2) / 2)
        metrics_text += f"Cohen's d (Normal vs Future): {cohen_d_future:.4f}\n"
    
    ax5.text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax5.axis('off')
    ax5.set_title('Performance Metrics', fontsize=13, fontweight='bold')
    
    ax6 = fig.add_subplot(gs[1, 2])
    counts = [len(normal_scores), len(current_fail_scores), len(future_fail_scores)]
    labels_pie = ['Normal', 'Current Failure', 'Future Failure']
    colors_pie = ['green', 'red', 'orange']
    
    _, _, autotexts = ax6.pie(counts, labels=labels_pie, colors=colors_pie,
                                        autopct='%1.1f%%', startangle=90, labeldistance=1.2)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax6.set_title('Class Distribution', fontsize=13, fontweight='bold')
    
    plt.suptitle('Anomaly Detection Analysis: Current vs Future Failures', 
                fontsize=16, fontweight='bold', y=0.995)
    
    return fig

def robust_stats(x):
    return {
        "median": np.median(x),
        "iqr": np.percentile(x, 75) - np.percentile(x, 25),
        "mad": np.median(np.abs(x - np.median(x)))
    }

def trimmed_mean_std(x, k=1.5):
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    mask = (x >= q1 - k * iqr) & (x <= q3 + k * iqr)
    x_trim = x[mask]
    return x_trim.mean(), x_trim.std()

def fmt(x):
    return f"{x:.4f}" if x is not None else "N/A"

def plot_failure_analysis(scores, labels, pred_types, failure_types, failure_map):
    if isinstance(scores, torch.Tensor): scores = scores.cpu().numpy()
    if isinstance(labels, torch.Tensor): labels = labels.cpu().numpy()
    if isinstance(failure_types, torch.Tensor): failure_types = failure_types.cpu().numpy()
    if isinstance(pred_types, torch.Tensor): pred_types = pred_types.cpu().numpy()
    
    normal_indices = np.where(labels == 0)[0]
    normal_scores = scores[normal_indices]
    
    results = []
    
    results.append({
        'Failure_Name': 'Normal (Baseline)',
        'Score': normal_scores,
        'Type': 'Baseline',
        'ROC_AUC': np.nan,
        'PR_AUC': np.nan
    })
    
    summary_metrics = []

    print(f"Analyzing {failure_types.shape[1]} failure types...")
    
    for idx, name in failure_map.items():
        
        fail_indices = np.where(failure_types[:, idx] == 1)[0]
        
        if len(fail_indices) < 5:
            print(f"Skipping {name}: Too few samples ({len(fail_indices)})")
            continue
            
        fail_scores = scores[fail_indices]
        
        curr_scores = np.concatenate([normal_scores, fail_scores])
        curr_labels = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(fail_scores))])
        
        roc = roc_auc_score(curr_labels, curr_scores)
        pr = average_precision_score(curr_labels, curr_scores)
        
        results.append({
            'Failure_Name': name,
            'Score': fail_scores,
            'Type': 'Failure',
            'ROC_AUC': roc,
            'PR_AUC': pr
        })
        
        summary_metrics.append({
            'Failure_Name': name,
            'Count': len(fail_indices),
            'ROC_AUC': roc,
            'PR_AUC': pr,
            'Mean_Score': np.mean(fail_scores),
            'Median_Score': np.median(fail_scores),
            'Std_Score': np.std(fail_scores)
        })

    df_metrics = pd.DataFrame(summary_metrics).sort_values(by='ROC_AUC', ascending=False)
    
    plot_data = []
    for r in results:
        for s in r['Score']:
            plot_data.append({'Failure_Name': r['Failure_Name'], 'Score': s, 'Type': r['Type']})
    df_plot = pd.DataFrame(plot_data)

    plt.figure(figsize=(20, 12))
    
    plt.subplot(2, 1, 1)    
    sort_order = ['Normal (Baseline)'] + df_metrics.sort_values(by='Median_Score', ascending=False)['Failure_Name'].tolist()
    
    sns.boxplot(
        data=df_plot, 
        x='Failure_Name', 
        y='Score', 
        hue='Type', 
        order=sort_order,
        palette={'Baseline': 'forestgreen', 'Failure': 'firebrick'},
        showfliers=False, # hide outliers
    )
    plt.yscale('symlog', linthresh=100)
    plt.title('Distribution of Anomaly Scores by Failure Type (symlog scale)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    plt.xlabel("")
    
    plt.subplot(2, 1, 2)
    
    df_melted = df_metrics.melt(id_vars="Failure_Name", value_vars=["ROC_AUC", "PR_AUC"], var_name="Metric", value_name="Value")
    
    sns.barplot(
        data=df_melted, 
        x='Failure_Name', 
        y='Value', 
        hue='Metric',
        palette="viridis"
    )
    
    plt.axhline(0.5, color='red', linestyle='--', label='Random Guess')
    
    plt.title('Model Detection Performance (AUC) by Failure Type', fontsize=14, fontweight='bold')
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45, ha='right')
    plt.legend(loc='lower right')
    plt.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()
    
    return df_metrics

def plot_future_failure_analysis(scores, labels, pred_types, failure_types, failure_map):
    if isinstance(scores, torch.Tensor): scores = scores.cpu().numpy()
    if isinstance(labels, torch.Tensor): labels = labels.cpu().numpy()
    if isinstance(pred_types, torch.Tensor): pred_types = pred_types.cpu().numpy()
    if isinstance(failure_types, torch.Tensor): failure_types = failure_types.cpu().numpy()
    
    normal_indices = np.where(pred_types == 0)[0]
    normal_scores = scores[normal_indices]
    
    results = []
    
    results.append({
        'Failure_Name': 'Normal (Baseline)',
        'Score': normal_scores,
        'Type': 'Baseline',
        'ROC_AUC': np.nan,
        'PR_AUC': np.nan
    })
    
    summary_metrics = []

    print(f"Analyzing {failure_types.shape[1]} failure types for PRE-FAILURE (Future) detection...")
    print(f"Baseline samples: {len(normal_indices)}")
    
    for idx, name in failure_map.items():
        is_specific_failure = failure_types[:, idx] == 1
        is_future_prediction = pred_types == 2
        
        fail_indices = np.where(is_specific_failure & is_future_prediction)[0]
        
        skipped_indices = np.where(is_specific_failure & (pred_types == 1))[0]
        
        if len(fail_indices) < 5:
            if len(skipped_indices) > 0:
                 print(f"Skipping {name}: Found {len(skipped_indices)} current failures, but only {len(fail_indices)} future predictions.")
            continue
            
        fail_scores = scores[fail_indices]
        
        curr_scores = np.concatenate([normal_scores, fail_scores])
        curr_labels = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(fail_scores))])
        
        try:
            roc = roc_auc_score(curr_labels, curr_scores)
            pr = average_precision_score(curr_labels, curr_scores)
        except ValueError:
            roc, pr = 0.5, 0.0 
        
        results.append({
            'Failure_Name': name,
            'Score': fail_scores,
            'Type': 'Future Failure',
            'ROC_AUC': roc,
            'PR_AUC': pr
        })
        
        summary_metrics.append({
            'Failure_Name': name,
            'Count': len(fail_indices),
            'Skipped_Current_Failures': len(skipped_indices),
            'ROC_AUC': roc,
            'PR_AUC': pr,
            'Mean_Score': np.mean(fail_scores),
            'Median_Score': np.median(fail_scores),
            'Std_Score': np.std(fail_scores)
        })

    if not summary_metrics:
        print("No failure types had enough future prediction samples to plot.")
        return pd.DataFrame()

    df_metrics = pd.DataFrame(summary_metrics).sort_values(by='ROC_AUC', ascending=False)
    
    plot_data = []
    for r in results:
        for s in r['Score']:
            plot_data.append({'Failure_Name': r['Failure_Name'], 'Score': s, 'Type': r['Type']})
    df_plot = pd.DataFrame(plot_data)

    plt.figure(figsize=(20, 12))
    
    plt.subplot(2, 1, 1)    
    sort_order = ['Normal (Baseline)'] + df_metrics.sort_values(by='Median_Score', ascending=False)['Failure_Name'].tolist()
    
    sns.boxplot(
        data=df_plot, 
        x='Failure_Name', 
        y='Score', 
        hue='Type', 
        order=sort_order,
        palette={'Baseline': 'forestgreen', 'Future Failure': 'darkorange'}, # Changed color to indicate warning/future
        showfliers=False, 
    )
    plt.yscale('symlog', linthresh=100)
    plt.title('Anomaly Scores: Normal vs Future Failures (Predictive Power)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    plt.xlabel("")
    
    plt.subplot(2, 1, 2)
    
    df_melted = df_metrics.melt(id_vars="Failure_Name", value_vars=["ROC_AUC", "PR_AUC"], var_name="Metric", value_name="Value")
    
    sns.barplot(
        data=df_melted, 
        x='Failure_Name', 
        y='Value', 
        hue='Metric',
        palette="viridis"
    )
    
    plt.axhline(0.5, color='red', linestyle='--', label='Random Guess')
    
    plt.title('Predictive Performance (AUC) for Future Failures', fontsize=14, fontweight='bold')
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45, ha='right')
    plt.legend(loc='lower right')
    plt.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()
    
    return df_metrics
