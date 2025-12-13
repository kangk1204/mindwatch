import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_process_box(ax, x, y, width, height, title, details, color='#E1F5FE', edge_color='#0277BD'):
    rect = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.1", 
                                  linewidth=1.5, edgecolor=edge_color, facecolor=color)
    ax.add_patch(rect)
    
    # Title
    ax.text(x + width/2, y + height - 0.4, title, ha='center', va='center', 
            fontsize=10, fontweight='bold', wrap=True)
    
    # Details
    ax.text(x + width/2, y + height/2 - 0.2, details, ha='center', va='center', 
            fontsize=8, wrap=True, linespacing=1.4)
    
    return x + width/2, y  # Return bottom center for connection

def draw_arrow(ax, x1, y1, x2, y2, color='black'):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.5, color=color))

def main():
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Main Title
    ax.text(8, 11.5, "MindWatch Data Preprocessing Pipeline", 
            ha='center', va='center', fontsize=18, fontweight='bold', color='#333')

    # Lane Headers
    ax.text(3, 11, "SENSOR DATA", fontsize=14, fontweight='bold', color='#1565C0', ha='center')
    ax.text(8, 11, "VOICE DATA", fontsize=14, fontweight='bold', color='#2E7D32', ha='center')
    ax.text(13, 11, "TEXT / SURVEY", fontsize=14, fontweight='bold', color='#EF6C00', ha='center')

    # --- SENSOR LANE ---
    s_x = 1.5
    w = 3.0
    h = 1.8
    gap = 0.8
    y_start = 9.0

    # Step 1: Cleaning
    s1_x, s1_y = draw_process_box(ax, s_x, y_start, w, h, 
                                  "Cleaning & Resampling", 
                                  "- Resample to 1 Hour\n- Forward Fill (ffill)\n- Impute w/ Global Median\n- Diff & Rolling (24h) Baseline",
                                  color='#BBDEFB', edge_color='#1565C0')
    
    # Step 2: Windowing
    s2_x, s2_y = draw_process_box(ax, s_x, y_start - h - gap, w, h, 
                                  "Rolling Window Features", 
                                  "Windows: [6h, 12h, 24h,\n48h, 72h, 10 Days]\n\nStats: Mean, Std, Min, Max, Last",
                                  color='#90CAF9', edge_color='#1565C0')

    # Step 3: Advanced Stats
    s3_x, s3_y = draw_process_box(ax, s_x, y_start - 2*(h + gap), w, h, 
                                  "Advanced Stats & Trends", 
                                  "- Trends (Slope over time)\n- Z-Scores\n- Exponential Moving Avg (EWMA)\n- Log Transforms",
                                  color='#64B5F6', edge_color='#1565C0')

    # Arrows
    draw_arrow(ax, s1_x, s1_y, s2_x, y_start - gap + 0.1) # +0.1 accounts for pad
    draw_arrow(ax, s2_x, s2_y, s3_x, y_start - h - gap - gap + 0.1)


    # --- VOICE LANE ---
    v_x = 6.5
    
    # Step 1: Extraction
    v1_x, v1_y = draw_process_box(ax, v_x, y_start, w, h, 
                                  "Acoustic Extraction (Librosa)", 
                                  "- MFCCs + Deltas\n- Spectral (Centroid, Contrast)\n- Pitch, Zero-Crossing Rate\n- Chroma, Tonnetz",
                                  color='#C8E6C9', edge_color='#2E7D32')

    # Step 2: Selection
    v2_x, v2_y = draw_process_box(ax, v_x, y_start - h - gap, w, h, 
                                  "Clip Selection", 
                                  "- Align with Survey Waves\n- Select max 2 clips\n- Within 7 days prior to survey",
                                  color='#A5D6A7', edge_color='#2E7D32')

    # Step 3: Aggregation
    v3_x, v3_y = draw_process_box(ax, v_x, y_start - 2*(h + gap), w, h, 
                                  "Aggregation", 
                                  "- Compute Mean & Std\nacross selected clips\n- Final Feature Vector per Wave",
                                  color='#81C784', edge_color='#2E7D32')

    # Arrows
    draw_arrow(ax, v1_x, v1_y, v2_x, y_start - gap + 0.1)
    draw_arrow(ax, v2_x, v2_y, v3_x, y_start - h - gap - gap + 0.1)


    # --- TEXT LANE ---
    t_x = 11.5
    
    # Step 1: Cleaning
    t1_x, t1_y = draw_process_box(ax, t_x, y_start, w, h, 
                                  "Cleaning & Handling NaNs", 
                                  "- Fill Categorical NaNs\nwith 'No Response'\n- Blocklist Leakage Cols\n(PHQ-9, GAD-7 scores)",
                                  color='#FFE0B2', edge_color='#EF6C00')

    # Step 2: Encoding
    t2_x, t2_y = draw_process_box(ax, t_x, y_start - h - gap, w, h, 
                                  "Categorical Encoding", 
                                  "- One-Hot: < 13 categories\n- Ordinal: > 13 categories\n- Prevent high-dim sparsity",
                                  color='#FFCC80', edge_color='#EF6C00')

    # Step 3: Specifics
    t3_x, t3_y = draw_process_box(ax, t_x, y_start - 2*(h + gap), w, h, 
                                  "Special Features", 
                                  "- String Length (Text depth)\n- Time Parsing (e.g. Bedtime)\nconverted to minutes",
                                  color='#FFB74D', edge_color='#EF6C00')

    # Arrows
    draw_arrow(ax, t1_x, t1_y, t2_x, y_start - gap + 0.1)
    draw_arrow(ax, t2_x, t2_y, t3_x, y_start - h - gap - gap + 0.1)


    # --- MERGE & FINAL ---
    
    # Big Merge Box
    m_y = 2.0
    m_h = 1.5
    m_w = 10
    m_x = 3.0

    draw_process_box(ax, m_x, m_y, m_w, m_h, 
                     "UNIFIED FEATURE MATRIX", 
                     "Final Concatenation of all processed features",
                     color='#E1BEE7', edge_color='#7B1FA2')

    # Arrows to Merge
    draw_arrow(ax, s3_x, s3_y, s3_x, m_y + m_h + 0.1) # Sensor
    draw_arrow(ax, v3_x, v3_y, v3_x, m_y + m_h + 0.1) # Voice
    draw_arrow(ax, t3_x, t3_y, t3_x, m_y + m_h + 0.1) # Text

    # Final Processing
    final_y = 0.5
    final_w = 12
    final_x = 2.0
    
    # Split & Scale Box
    draw_process_box(ax, final_x, final_y, final_w, 1.0, 
                     "FINAL PREPARATION", 
                     "Stratified GroupKFold (Participant-Aware)  |  Imputation (Median)  |  Scaling (QuantileTransformer / StandardScaler)",
                     color='#CFD8DC', edge_color='#455A64')

    # Arrow to Final
    draw_arrow(ax, 8, m_y, 8, final_y + 1.1)

    plt.tight_layout()
    output_path = 'mindwatch_preprocessing_pipeline.png'
    plt.savefig(output_path, dpi=300)
    print(f"Diagram saved to {output_path}")

if __name__ == "__main__":
    main()
