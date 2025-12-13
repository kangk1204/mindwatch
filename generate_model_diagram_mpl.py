import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_box(ax, x, y, width, height, text, color='#E0E0E0', edge_color='black', fontsize=10):
    rect = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.1", 
                                  linewidth=1.5, edgecolor=edge_color, facecolor=color)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=fontsize, wrap=True)
    return x + width, y + height/2  # Return right connection point

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.5, color='black'))

def main():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(6, 9.5, "MindWatch Multimodal Depression Detection Model", 
            ha='center', va='center', fontsize=16, fontweight='bold')

    # --- Column 1: Raw Data Inputs ---
    input_x = 0.5
    box_w = 2.0
    box_h = 1.0
    
    # Sensor Data
    draw_box(ax, input_x, 7.5, box_w, box_h, "Raw Sensor Data\n(CSV)\n[Heart, Sleep, Activity]", color='#BBDEFB')
    # Voice Data
    draw_box(ax, input_x, 5.0, box_w, box_h, "Raw Voice Data\n(Audio Files)\n[WAV/MP3]", color='#C8E6C9')
    # Text Data
    draw_box(ax, input_x, 2.5, box_w, box_h, "Survey Responses\n(Text/Categorical)\n[Excel/DB]", color='#FFF9C4')

    # --- Column 2: Preprocessing & Feature Engineering ---
    proc_x = 3.5
    
    # Sensor Proc
    draw_box(ax, proc_x, 7.5, box_w, box_h, "Time-Series Processing\n(Rolling Stats, Lags,\nZ-scores, Trends)", color='#90CAF9')
    # Voice Proc
    draw_box(ax, proc_x, 5.0, box_w, box_h, "Audio Analysis (Librosa)\n(MFCCs, Spectral,\nPitch, Aggregation)", color='#A5D6A7')
    # Text Proc
    draw_box(ax, proc_x, 2.5, box_w, box_h, "Text Encoding\n(One-Hot, Ordinal,\nString Length)", color='#FFF59D')

    # Arrows Input -> Proc
    draw_arrow(ax, input_x + box_w + 0.2, 8.0, proc_x, 8.0)
    draw_arrow(ax, input_x + box_w + 0.2, 5.5, proc_x, 5.5)
    draw_arrow(ax, input_x + box_w + 0.2, 3.0, proc_x, 3.0)

    # --- Column 3: Fusion ---
    fusion_x = 6.5
    fusion_w = 2.0
    fusion_h = 6.0
    
    # Early Fusion Block
    draw_box(ax, fusion_x, 2.5, fusion_w, fusion_h, "EARLY FUSION\n\nUnified Wide\nFeature Matrix", color='#E1BEE7')

    # Arrows Proc -> Fusion
    draw_arrow(ax, proc_x + box_w + 0.2, 8.0, fusion_x, 7.5)  # Sensor to top of fusion
    draw_arrow(ax, proc_x + box_w + 0.2, 5.5, fusion_x, 5.5)  # Voice to mid
    draw_arrow(ax, proc_x + box_w + 0.2, 3.0, fusion_x, 3.5)  # Text to bot

    # --- Column 4: Modeling ---
    model_x = 9.5
    model_h = 1.0
    
    # Ensemble Block
    draw_box(ax, model_x, 4.5, box_w, 2.5, "Ensemble Classifier\n\n- XGBoost\n- LightGBM\n- CatBoost\n- HistGradientBoosting", color='#FFCCBC')

    # Arrow Fusion -> Model
    draw_arrow(ax, fusion_x + fusion_w + 0.2, 5.5, model_x, 5.75)

    # --- Column 5: Output ---
    output_x = 9.5
    
    # Output Block
    draw_box(ax, output_x, 2.0, box_w, box_h, "Prediction Output\n\nDepression Risk\n(PHQ-9 >= 10)", color='#FFAB91', edge_color='#D84315')

    # Arrow Model -> Output
    draw_arrow(ax, model_x + box_w/2, 4.5, model_x + box_w/2, 3.2) # Down from model to output? Actually let's just point arrow down
    
    # Adjust arrow to point from model block bottom to output block top
    # Model block y is 4.5, height 2.5 -> top 7.0, bottom 4.5. 
    # Output block y is 2.0, height 1.0 -> top 3.0.
    # Ah, `draw_box` y arg is bottom-left corner.
    # Model box: y=4.5, h=2.5. Top is 7.0.
    # Output box: y=2.0, h=1.0. Top is 3.0.
    # Gap is 4.5 - 3.0 = 1.5.
    
    # Arrow from bottom of Model to top of Output
    ax.annotate("", xy=(model_x + box_w/2, 3.1), xytext=(model_x + box_w/2, 4.5),
                arrowprops=dict(arrowstyle="->", lw=2.0, color='#D84315'))

    plt.tight_layout()
    output_path = 'mindwatch_model_architecture.png'
    plt.savefig(output_path, dpi=300)
    print(f"Diagram saved to {output_path}")

if __name__ == "__main__":
    main()
