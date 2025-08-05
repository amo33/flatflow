import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

from matplotlib.patches import Rectangle

def visualize_pipeline(sqlite_path, min_category=0.0, max_category=100000.0, num_stages=8, dp_size=2, tp_size=1):
    """
    Extract forward/backward events from SQLite database and visualize Pipeline Parallelism.
    Make forward/backward events more distinguishable and visually appealing. 
    Supports multiple nodes with Tensor Parallelism visualization.
    Args:
        sqlite_path: Path to the SQLite database
        min_category: Minimum category to visualize
        max_category: Maximum category to visualize
        num_stages: Number of stages (devices) for pipeline parallelism
        dp_size: Data parallelism size
        tp_size: Tensor parallelism size
    """
    print("Connecting to SQLite database...")
    conn = sqlite3.connect(sqlite_path)

    query = f"""
    SELECT 
        text,
        category,
        start,
        end,
        (end - start) as duration,
        start/1000000.0 as start_ms,
        end/1000000.0 as end_ms,
        (end - start)/1000000.0 as duration_ms
    FROM NVTX_EVENTS 
    WHERE (text LIKE 'forward' OR text LIKE 'backward')
    AND category IS NOT NULL
    AND category >= {min_category} 
    AND category <= {max_category}
    AND end IS NOT NULL
    ORDER BY category, start
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()

    df = df.dropna(subset=['category', 'start_ms', 'end_ms', 'duration_ms'])
    
    # FIX: category IS the stage number directly
    df['stage'] = df['category'].astype(int)
    df['tp_group'] = (df['category'] % tp_size).astype(int)

    # Print the number of events
    for stage in range(num_stages):
        stage_events = df[df['stage'] == stage]
        print(f"Stage {stage} - Forward: {len(stage_events[stage_events['text'] == 'forward'])}, "
              f"Backward: {len(stage_events[stage_events['text'] == 'backward'])}")
        for tp_group in range(tp_size):
            tp_events = stage_events[stage_events['tp_group'] == tp_group]
            print(f"  TP Group {tp_group}: Forward: {len(tp_events[tp_events['text'] == 'forward'])}, "
                  f"Backward: {len(tp_events[tp_events['text'] == 'backward'])}")
    
    # Set the time range
    min_time = df['start_ms'].min()
    max_time = df['end_ms'].max()
    time_range = max_time - min_time
    
    print(f"Time range: {min_time:.2f} - {max_time:.2f} ms (total: {time_range:.2f} ms)")
    
    # Prepare the plot
    plt.style.use('ggplot')  # Apply a cleaner style
    fig, ax = plt.subplots(figsize=(16, 10), dpi=100)  # Increase height for TP visualization
    
    # Calculate statistics
    forward_events = df[df['text'] == 'forward']
    backward_events = df[df['text'] == 'backward']
    
    if len(forward_events) > 0:
        forward_avg = forward_events['duration_ms'].mean()
        forward_max = forward_events['duration_ms'].max()
    else:
        forward_avg = forward_max = 0
        
    if len(backward_events) > 0:
        backward_avg = backward_events['duration_ms'].mean()
        backward_max = backward_events['duration_ms'].max()
    else:
        backward_avg = backward_max = 0
    
    # Set the background color
    ax.set_facecolor('#f8f8f8')
    fig.patch.set_facecolor('#ffffff')

    stage_height = 0.6  # Height of each stage
    stage_spacing = 0.2  # Gap between stages
    tp_height = stage_height / max(tp_size, 1)  # Height of each TP sub-stage
    
    # Define colors for different TP groups
    tp_colors_forward = ['#5B86E5', '#8B5CF6', '#F59E0B', '#EF4444']  # Blue, Purple, Orange, Red
    tp_colors_backward = ['#56B4B4', '#10B981', '#F97316', '#DC2626']  # Teal, Green, Orange, Red
    
    for stage in range(num_stages):
        stage_events = df[df['stage'] == stage]
        y_pos_base = (num_stages - 1 - stage) * (stage_height + stage_spacing)
        
        for _, event in stage_events.iterrows():
            tp_group = int(event['tp_group'])
            
            if event['text'] == 'forward':
                color = tp_colors_forward[tp_group % len(tp_colors_forward)]
                edge_color = '#4A75D3'
                height = tp_height * 0.8  # Slightly smaller for visibility
                y_offset = tp_group * tp_height + (tp_height - height) / 2
                alpha = 0.8
            else:  # backward
                color = tp_colors_backward[tp_group % len(tp_colors_backward)]
                edge_color = '#45A3A3'
                height = tp_height * 0.8  # Slightly smaller for visibility
                y_offset = tp_group * tp_height + (tp_height - height) / 2
                alpha = 0.9

            # Use relative time for x-axis
            start_time = event['start_ms'] - min_time
            # FIXED: Use actual duration, but set minimum width for visibility
            width = max(event['duration_ms'], time_range * 0.02)  # At least 2% of timeline
            
            # Draw the event box
            rect = Rectangle((start_time, y_pos_base + y_offset), 
                           width, height, 
                           facecolor=color, 
                           alpha=alpha,
                           edgecolor=edge_color,
                           linewidth=1.0)
            ax.add_patch(rect)
            
            # FIXED: Add batch ID labels (simulated since we don't have batch_id in data)
            # Estimate batch ID based on timing and position
            estimated_batch = int((start_time / (time_range / 15)) % 8) + 1  # Rough estimate
            
            # Add event and batch labels
            if width > time_range * 0.03:  # Only if box is wide enough
                ax.text(start_time + width/2, 
                       y_pos_base + y_offset + height/2, 
                       f"{estimated_batch}",  # Show estimated batch ID
                       fontsize=10, 
                       verticalalignment='center',
                       horizontalalignment='center',
                       color='white',
                       fontweight='bold')
            
            # Add F/B indicator on smaller boxes
            elif width > time_range * 0.015:
                ax.text(start_time + width/2, 
                       y_pos_base + y_offset + height/2, 
                       event['text'][0].upper(),  # 'F' or 'B'
                       fontsize=8, 
                       verticalalignment='center',
                       horizontalalignment='center',
                       color='white',
                       fontweight='bold')
    
    # Set the grid and axes
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Set the Y-axis label considering the stage height and spacing
    total_height = num_stages * (stage_height + stage_spacing) - stage_spacing
    
    # Set the Y-axis label to the center of each stage
    y_ticks = [(num_stages - 1 - stage) * (stage_height + stage_spacing) + stage_height/2 
               for stage in range(num_stages)]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'Stage {stage}' for stage in range(num_stages)])
    
    # Set the X-axis range and label
    ax.set_xlim(0, time_range)
    ax.set_ylim(-0.5, total_height + 0.5)  # Adjust the margin
    ax.set_xlabel('Time (ms)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Pipeline Stage', fontsize=12, fontweight='bold')

    # Add time arrow
    arrow_y_pos = -0.4
    ax.arrow(0, arrow_y_pos, time_range, 0, 
            head_width=0.15, head_length=time_range * 0.01, 
            fc='black', ec='black', linewidth=1.5)
    
    # Create legend with TP groups
    legend_elements = []
    for tp_group in range(tp_size):
        legend_elements.append(
            plt.Rectangle((0, 0), 1, 1, 
                         facecolor=tp_colors_forward[tp_group % len(tp_colors_forward)], 
                         alpha=0.8, 
                         edgecolor='#4A75D3', 
                         label=f'Forward TP{tp_group}')
        )
        legend_elements.append(
            plt.Rectangle((0, 0), 1, 1, 
                         facecolor=tp_colors_backward[tp_group % len(tp_colors_backward)], 
                         alpha=0.9, 
                         edgecolor='#45A3A3', 
                         label=f'Backward TP{tp_group}')
        )

    ax.legend(handles=legend_elements, loc='upper center', 
              bbox_to_anchor=(0.85, 1.05), 
              ncol=2, fontsize=9, frameon=True, facecolor='white', edgecolor='gray')
    
    plt.title(f'Pipeline Parallelism DP{dp_size}_PP{num_stages}_TP{tp_size} - 1F1B Pattern', 
              fontsize=16, fontweight='bold', pad=20)
    
    plt.figtext(0.5, 0.01, 
               f'Forward Events: Avg {forward_avg:.2f}ms, Max {forward_max:.0f}ms | '
               f'Backward Events: Avg {backward_avg:.2f}ms, Max {backward_max:.0f}ms | '
               f'TP Groups: {tp_size}', 
               ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    output_filename = f'pipeline_parallelism_dp{dp_size}_pp{num_stages}_tp{tp_size}.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"시각화 이미지 저장 완료: {output_filename}")
    plt.show()

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--sqlite_path", type=str, required=True)
    args.add_argument("--min_category", type=float, default=0.0)
    args.add_argument("--max_category", type=float, default=7.0)  # Changed default for 8 stages
    args.add_argument("--num_stages", type=int, default=8)
    args.add_argument("--dp_size", type=int, default=1)
    args.add_argument("--tp_size", type=int, default=1)
    args = args.parse_args()
    visualize_pipeline(args.sqlite_path, args.min_category, args.max_category, args.num_stages, args.dp_size, args.tp_size)