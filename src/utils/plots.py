import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, precision_recall_curve


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
    
    # ROC curve
    ax1 = fig.add_subplot(gs[0, 0])
    fpr, tpr, _ = roc_curve(labels, scores)
    ax1.plot(fpr, tpr, linewidth=2, label=f'Overall (AUC={overall_roc:.4f})')
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('False Positive Rate', fontsize=11)
    ax1.set_ylabel('True Positive Rate', fontsize=11)
    ax1.set_title('ROC Curve', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(alpha=0.3)
    
    # Precision-Recall Curve
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
    
    # Score Distribution by Type
    ax3 = fig.add_subplot(gs[0, 2])
    bins = 50
    ax3.hist(normal_scores, bins=bins, alpha=0.6, label=f'Normal (n={len(normal_scores)})', 
             color='green', density=True)
    if len(current_fail_scores) > 0:
        ax3.hist(current_fail_scores, bins=bins, alpha=0.6, 
                label=f'Current Failure (n={len(current_fail_scores)})', 
                color='red', density=True)
    if len(future_fail_scores) > 0:
        ax3.hist(future_fail_scores, bins=bins, alpha=0.6, 
                label=f'Future Failure (n={len(future_fail_scores)})', 
                color='orange', density=True)
    ax3.set_xlabel('Anomaly Score', fontsize=11)
    ax3.set_ylabel('Density', fontsize=11)
    ax3.set_title('Score Distribution by Type', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # Box Plot
    ax4 = fig.add_subplot(gs[1, 0])
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
    
    bp = ax4.boxplot(data_for_box, labels=labels_for_box, patch_artist=True, 
                     showfliers=False)
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax4.set_ylabel('Anomaly Score', fontsize=11)
    ax4.set_title('Score Distribution (Box Plot)', fontsize=13, fontweight='bold')
    ax4.grid(alpha=0.3, axis='y')
    
    # Cumulative Distribution
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(normal_scores, bins=100, cumulative=True, alpha=0.6, 
            label='Normal', color='green', density=True, histtype='step', linewidth=2)
    if len(current_fail_scores) > 0:
        ax5.hist(current_fail_scores, bins=100, cumulative=True, alpha=0.6, 
                label='Current Failure', color='red', density=True, histtype='step', linewidth=2)
    if len(future_fail_scores) > 0:
        ax5.hist(future_fail_scores, bins=100, cumulative=True, alpha=0.6, 
                label='Future Failure', color='orange', density=True, histtype='step', linewidth=2)
    ax5.set_xlabel('Anomaly Score', fontsize=11)
    ax5.set_ylabel('Cumulative Probability', fontsize=11)
    ax5.set_title('Cumulative Distribution', fontsize=13, fontweight='bold')
    ax5.legend()
    ax5.grid(alpha=0.3)
    
    # Statistics Summary Table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    stats_data = []
    stats_data.append(['Metric', 'Normal', 'Current Fail', 'Future Fail'])
    stats_data.append(['Count', f"{len(normal_scores)}", 
                      f"{len(current_fail_scores)}", f"{len(future_fail_scores)}"])
    stats_data.append(['Mean Score', f"{normal_scores.mean():.4f}", 
                      f"{current_fail_scores.mean():.4f}" if len(current_fail_scores) > 0 else "N/A",
                      f"{future_fail_scores.mean():.4f}" if len(future_fail_scores) > 0 else "N/A"])
    stats_data.append(['Std Dev', f"{normal_scores.std():.4f}", 
                      f"{current_fail_scores.std():.4f}" if len(current_fail_scores) > 0 else "N/A",
                      f"{future_fail_scores.std():.4f}" if len(future_fail_scores) > 0 else "N/A"])
    stats_data.append(['Median', f"{np.median(normal_scores):.4f}", 
                      f"{np.median(current_fail_scores):.4f}" if len(current_fail_scores) > 0 else "N/A",
                      f"{np.median(future_fail_scores):.4f}" if len(future_fail_scores) > 0 else "N/A"])
    
    table = ax6.table(cellText=stats_data, cellLoc='center', loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax6.set_title('Summary Statistics', fontsize=13, fontweight='bold', pad=20)
    
    # Separation Metrics
    ax7 = fig.add_subplot(gs[2, 0])
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
    
    # Calculate separation between distributions
    if len(current_fail_scores) > 0:
        cohen_d_current = (current_fail_scores.mean() - normal_scores.mean()) / \
                         np.sqrt((current_fail_scores.std()**2 + normal_scores.std()**2) / 2)
        metrics_text += f"Cohen's d (Normal vs Current): {cohen_d_current:.4f}\n"
    
    if len(future_fail_scores) > 0:
        cohen_d_future = (future_fail_scores.mean() - normal_scores.mean()) / \
                        np.sqrt((future_fail_scores.std()**2 + normal_scores.std()**2) / 2)
        metrics_text += f"Cohen's d (Normal vs Future): {cohen_d_future:.4f}\n"
    
    ax7.text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax7.axis('off')
    ax7.set_title('Performance Metrics', fontsize=13, fontweight='bold')
    
    # Scatter plot of scores (if we have both types)
    ax8 = fig.add_subplot(gs[2, 1])
    if len(current_fail_scores) > 0 and len(future_fail_scores) > 0:
        sample_size = min(1000, len(scores))
        indices = np.random.choice(len(scores), sample_size, replace=False)
        
        for idx in indices:
            if pred_types[idx] == 0:
                ax8.scatter(idx, scores[idx], c='green', alpha=0.3, s=10)
            elif pred_types[idx] == 1:
                ax8.scatter(idx, scores[idx], c='red', alpha=0.6, s=20)
            elif pred_types[idx] == 2:
                ax8.scatter(idx, scores[idx], c='orange', alpha=0.6, s=20)
        
        ax8.set_xlabel('Sample Index', fontsize=11)
        ax8.set_ylabel('Anomaly Score', fontsize=11)
        ax8.set_title('Score Timeline (Sampled)', fontsize=13, fontweight='bold')
        ax8.legend(['Normal', 'Current Failure', 'Future Failure'])
    else:
        ax8.text(0.5, 0.5, 'Insufficient data\nfor timeline view', 
                ha='center', va='center', fontsize=12)
        ax8.axis('off')
    ax8.grid(alpha=0.3)
    
    # 9. Class Proportions
    ax9 = fig.add_subplot(gs[2, 2])
    counts = [len(normal_scores), len(current_fail_scores), len(future_fail_scores)]
    labels_pie = ['Normal', 'Current Failure', 'Future Failure']
    colors_pie = ['green', 'red', 'orange']
    
    _, _, autotexts = ax9.pie(counts, labels=labels_pie, colors=colors_pie,
                                        autopct='%1.1f%%', startangle=90, labeldistance=1.2)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax9.set_title('Class Distribution', fontsize=13, fontweight='bold')
    
    plt.suptitle('Anomaly Detection Analysis: Current vs Future Failures', 
                fontsize=16, fontweight='bold', y=0.995)
    
    return fig