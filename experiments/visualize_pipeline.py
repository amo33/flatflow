import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

from matplotlib.patches import Rectangle

def visualize_pipeline(sqlite_path, min_category=0.0, max_category=96.0, num_stages=8):
    """
    Extract forward/backward events from SQLite database and visualize Pipeline Parallelism.
    Make forward/backward events more distinguishable and visually appealing. 
    Supports multiple nodes.
    Args:
        sqlite_path: Path to the SQLite database
        min_category: Minimum category to visualize
        max_category: Maximum category to visualize
        num_stages: Number of stages (devices)
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
        start/1000000 as start_ms,
        end/1000000 as end_ms,
        (end - start)/1000000 as duration_ms
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
    
    # Map forward/backward events to stages for each category
    df['stage'] = None  # Initialize with default value
    
    for category in df['category'].unique():
        cat_events = df[df['category'] == category]
        
        # Forward event
        forward_events = cat_events[cat_events['text'] == 'forward'].sort_values('start_ms')
        forward_count = len(forward_events)
        if forward_count > 0:
            stages_per_forward = num_stages // forward_count if forward_count > 0 else 1
            for i, idx in enumerate(forward_events.index):
                stage = min(i * stages_per_forward, num_stages - 1)
                df.loc[idx, 'stage'] = stage
        
        # Backward event
        backward_events = cat_events[cat_events['text'] == 'backward'].sort_values('start_ms')
        backward_count = len(backward_events)
        if backward_count > 0:
            stages_per_backward = num_stages // backward_count if backward_count > 0 else 1
            for i, idx in enumerate(backward_events.index):
                stage = max(num_stages - 1 - (i * stages_per_backward), 0)
                df.loc[idx, 'stage'] = stage

    df = df.dropna(subset=['stage'])
    df['stage'] = df['stage'].astype(int)

    # Print the number of events
    for stage in range(num_stages):
        stage_events = df[df['stage'] == stage]
        print(f"Stage {stage} - Forward: {len(stage_events[stage_events['text'] == 'forward'])}, "
              f"Backward: {len(stage_events[stage_events['text'] == 'backward'])}")
    
    # Set the time range
    min_time = df['start_ms'].min()
    max_time = df['end_ms'].max()
    time_range = max_time - min_time
    
    # Prepare the plot
    plt.style.use('ggplot')  # Apply a cleaner style
    fig, ax = plt.subplots(figsize=(16, 8), dpi=100)  # Reduce the height to reduce the gap
    
    # Calculate statistics
    forward_avg = df[df['text'] == 'forward']['duration_ms'].mean()
    forward_max = df[df['text'] == 'forward']['duration_ms'].max()
    backward_avg = df[df['text'] == 'backward']['duration_ms'].mean()
    backward_max = df[df['text'] == 'backward']['duration_ms'].max()
    
    # Set the background color
    ax.set_facecolor('#f8f8f8')
    fig.patch.set_facecolor('#ffffff')

    stage_height = 0.75  # Height of each stage
    stage_spacing = 0.25  # Gap between stages
    
    for stage in range(num_stages):
        stage_events = df[df['stage'] == stage]
        y_pos = (num_stages - 1 - stage) * (stage_height + stage_spacing)
        
        for _, event in stage_events.iterrows():
            if event['text'] == 'forward':
                color = '#5B86E5'  # Light blue
                edge_color = '#4A75D3'
                height = stage_height
                y_offset = 0
            else:  # backward
                color = '#56B4B4'  # Teal
                edge_color = '#45A3A3'
                height = stage_height
                y_offset = 0

            width = max(2.0, event['duration_ms'])
            
            # Draw the event box
            rect = Rectangle((event['start_ms'] - min_time, y_pos + y_offset), 
                           width, height, 
                           facecolor=color, 
                           alpha=0.85,
                           edgecolor=edge_color,
                           linewidth=0.8)
            ax.add_patch(rect)
    
    # Set the grid and axes
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Set the Y-axis label considering the stage height and spacing
    stage_height = 0.75
    stage_spacing = 0.25
    total_height = num_stages * (stage_height + stage_spacing) - stage_spacing
    
    # Set the Y-axis label to the center of each stage
    y_ticks = [(num_stages - 1 - stage) * (stage_height + stage_spacing) + stage_height/2 
               for stage in range(num_stages)]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'Stage {num_stages-1-stage}' for stage in range(num_stages)])
    
    # Set the X-axis range and label
    ax.set_xlim(0, time_range)
    ax.set_ylim(-0.5, total_height + 0.5)  # Adjust the margin
    ax.set_xlabel('Time (ms)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Pipeline Stage', fontsize=12, fontweight='bold')

    stage_height = 0.75
    stage_spacing = 0.25
    arrow_y_pos = -0.4
    
    ax.arrow(0, arrow_y_pos, time_range, 0, 
            head_width=0.15, head_length=time_range * 0.01, 
            fc='black', ec='black', linewidth=1.5)
    

    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='#5B86E5', alpha=0.85, edgecolor='#4A75D3', label='Forward (F)'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#56B4B4', alpha=0.85, edgecolor='#45A3A3', label='Backward (B)')
    ]

    arrow_y_pos = -0.4
    ax.legend(handles=legend_elements, loc='upper center', 
              bbox_to_anchor=(0.85, 1.05), 
              ncol=2, fontsize=10, frameon=True, facecolor='white', edgecolor='gray')
    

    plt.title(f'Pipeline Parallelism TP1_PP8_BS16_MBS1', 
              fontsize=16, fontweight='bold', pad=20)
    
    plt.figtext(0.5, 0.01, 
               f'Forward Events: Avg {forward_avg:.2f}ms, Max {forward_max:.0f}ms | '
               f'Backward Events: Avg {backward_avg:.2f}ms, Max {backward_max:.0f}ms', 
               ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    output_filename = 'pipeline_parallelism_improved.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"시각화 이미지 저장 완료: {output_filename}")
    plt.show()

if __name__ == "__main__":
  args = argparse.ArgumentParser()
  args.add_argument("--sqlite_path", type=str, required=True)
  args.add_argument("--min_category", type=float, required=True)
  args.add_argument("--max_category", type=float, required=True)
  args = args.parse_args()
  visualize_pipeline(args.sqlite_path, args.min_category, args.max_category, args.num_stages)
