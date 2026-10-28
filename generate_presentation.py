#!/usr/bin/env python3
"""
Generate the presentation PPTX for:
Inventory Demand Forecasting via Kalman Filtering and Global DeepAR Models
Matching the design, color hierarchy, and editorial structure of AML_Traffic_Flow_Forecasting.pptx
With dedicated individual slides for each team member:
- Hemamdhar Nath: The Question, Data Scale, EDA Dynamics
- Harshakarthikeya: Leakage Trap, Features & Pipeline, AML Baselines (LightGBM)
- Sameer Alam Khan: Dedicated Structural Kalman Filter per SKU
- Akshay Y: Dedicated Global DeepAR Probabilistic Forecaster
- Abhishek Tripathi: Dedicated Final Exam / Benchmark Scorecard
- All Team Members: Decision Rules, Limitations & Future Hybrid
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

# ── Color Palette ────────────────────────────────────────────────────────────
DARK_BG = RGBColor(11, 19, 37)          # #0B1325 Deep Navy
DARK_CARD = RGBColor(18, 30, 54)        # #121E36 Dark Card
DARK_CARD_ALT = RGBColor(24, 38, 68)    # Slightly lighter dark card
DARK_BORDER = RGBColor(35, 54, 90)      # #23365A Border

LIGHT_BG = RGBColor(245, 240, 235)      # #F5F0EB Warm Beige/Cream
LIGHT_CARD = RGBColor(255, 255, 255)    # #FFFFFF Pure White
LIGHT_BORDER = RGBColor(228, 220, 210)  # #E4DCD2 Card Border

TEXT_WHITE = RGBColor(255, 255, 255)
TEXT_MUTED_DARK = RGBColor(148, 163, 184) # Slate Muted
TEXT_DARK = RGBColor(15, 23, 42)        # #0F172A Slate Dark
TEXT_MUTED_LIGHT = RGBColor(100, 116, 139) # Slate Light

ACCENT_AMBER = RGBColor(245, 158, 11)   # #F59E0B Warning / Stat
ACCENT_AMBER_DARK = RGBColor(194, 100, 20)
ACCENT_BLUE = RGBColor(59, 130, 246)    # #3B82F6 Tech Blue
ACCENT_EMERALD = RGBColor(16, 185, 129) # #10B981 Winner Green
ACCENT_CYAN = RGBColor(6, 182, 212)     # #06B6D4 Probabilistic Cyan
ACCENT_RED = RGBColor(239, 68, 68)

FIGURES_DIR = os.path.abspath("reports/figures")

def create_deck(output_path="Inventory_Demand_Forecasting_Presentation.pptx"):
    prs = Presentation()
    # 16:9 Widescreen dimensions
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]
    TOTAL_SLIDES = 11

    # Helper: set slide solid background
    def set_bg(slide, color):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = color
        bg.line.fill.background()
        return bg

    # Helper: add footer
    def add_footer(slide, current_idx, is_dark=True):
        tx_box = slide.shapes.add_textbox(Inches(0.8), Inches(6.92), Inches(8), Inches(0.4))
        tf = tx_box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = "Inventory Demand Forecasting · CSAI Project 5 · Team AML"
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_MUTED_DARK if is_dark else TEXT_MUTED_LIGHT

        num_box = slide.shapes.add_textbox(Inches(11.0), Inches(6.92), Inches(1.5), Inches(0.4))
        tf_num = num_box.text_frame
        p_num = tf_num.paragraphs[0]
        p_num.text = f"{current_idx:02d} / {TOTAL_SLIDES:02d}"
        p_num.font.size = Pt(11)
        p_num.alignment = PP_ALIGN.RIGHT
        p_num.font.color.rgb = TEXT_MUTED_DARK if is_dark else TEXT_MUTED_LIGHT

    # Helper: add category eyebrow + title + presenter pill
    def add_header(slide, category_text, title_text, presenter_tag=None, is_dark=True):
        # Category eyebrow
        eyebrow_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.55), Inches(7.5), Inches(0.35))
        tf_eye = eyebrow_box.text_frame
        p_eye = tf_eye.paragraphs[0]
        p_eye.text = category_text.upper()
        p_eye.font.size = Pt(11.5)
        p_eye.font.bold = True
        p_eye.font.color.rgb = ACCENT_AMBER if is_dark else ACCENT_AMBER_DARK

        # Presenter Tag Pill (Top Right)
        if presenter_tag:
            tag_box = slide.shapes.add_textbox(Inches(8.0), Inches(0.55), Inches(4.5), Inches(0.35))
            tf_ptag = tag_box.text_frame
            p_ptag = tf_ptag.paragraphs[0]
            p_ptag.text = presenter_tag.upper()
            p_ptag.alignment = PP_ALIGN.RIGHT
            p_ptag.font.size = Pt(11)
            p_ptag.font.bold = True
            p_ptag.font.color.rgb = ACCENT_CYAN if is_dark else ACCENT_BLUE

        # Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.88), Inches(11.7), Inches(0.8))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        p_title = tf_title.paragraphs[0]
        p_title.text = title_text
        p_title.font.size = Pt(27)
        p_title.font.bold = True
        p_title.font.color.rgb = TEXT_WHITE if is_dark else TEXT_DARK

    # ═════════════════════════════════════════════════════════════════════════
    # SLIDE 1: Title Slide (Dark)
    # ═════════════════════════════════════════════════════════════════════════
    s1 = prs.slides.add_slide(blank_layout)
    set_bg(s1, DARK_BG)

    tag_box = s1.shapes.add_textbox(Inches(0.8), Inches(0.75), Inches(11), Inches(0.4))
    tf1_tag = tag_box.text_frame
    p1_tag = tf1_tag.paragraphs[0]
    p1_tag.text = "AML PROJECT  ·  CSAI STUDENT PROJECT 5  ·  SECTION AML"
    p1_tag.font.size = Pt(12)
    p1_tag.font.bold = True
    p1_tag.font.color.rgb = ACCENT_AMBER

    title_box = s1.shapes.add_textbox(Inches(0.8), Inches(1.35), Inches(11.7), Inches(1.8))
    tf1_title = title_box.text_frame
    tf1_title.word_wrap = True
    p1_t = tf1_title.paragraphs[0]
    p1_t.text = "How many units\nwill sell next month?"
    p1_t.font.size = Pt(45)
    p1_t.font.bold = True
    p1_t.font.color.rgb = TEXT_WHITE

    sub_box = s1.shapes.add_textbox(Inches(0.8), Inches(3.45), Inches(11.7), Inches(0.9))
    tf1_sub = sub_box.text_frame
    tf1_sub.word_wrap = True
    p1_sub = tf1_sub.paragraphs[0]
    p1_sub.text = "Predicting 28-day retail inventory demand across intermittent SKU histories\nusing Kalman Filters and Global DeepAR models."
    p1_sub.font.size = Pt(17)
    p1_sub.font.color.rgb = TEXT_MUTED_DARK

    # Authors Grid (5 team members with roles)
    authors = [
        ("Dakarapu Hemamdhar Nath", "2401010137", "Data Cleaning & EDA"),
        ("Palaparthi Harshakarthikeya", "2401010313", "Feature Eng. & Baselines"),
        ("Sameer Alam Khan", "2401020054", "Kalman Filter per SKU"),
        ("Akshay Y", "2401020008", "Global DeepAR Forecaster"),
        ("Abhishek Tripathi", "2401010022", "Benchmark & Validation"),
    ]
    card_w = Inches(2.23)
    card_h = Inches(1.45)
    card_y = Inches(5.0)

    for i, (name, roll, role) in enumerate(authors):
        x = Inches(0.8 + i * 2.38)
        card = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, card_y, card_w, card_h)
        card.fill.solid()
        card.fill.fore_color.rgb = DARK_CARD
        card.line.color.rgb = DARK_BORDER

        tf_c = card.text_frame
        tf_c.word_wrap = True
        tf_c.margin_left = Inches(0.12)
        tf_c.margin_right = Inches(0.12)
        tf_c.margin_top = Inches(0.15)

        p_name = tf_c.paragraphs[0]
        p_name.text = name
        p_name.font.size = Pt(11)
        p_name.font.bold = True
        p_name.font.color.rgb = TEXT_WHITE

        p_roll = tf_c.add_paragraph()
        p_roll.text = roll
        p_roll.font.size = Pt(10)
        p_roll.font.color.rgb = ACCENT_AMBER
        p_roll.space_before = Pt(2)

        p_role = tf_c.add_paragraph()
        p_role.text = role
        p_role.font.size = Pt(9.5)
        p_role.font.bold = True
        p_role.font.color.rgb = ACCENT_CYAN
        p_role.space_before = Pt(4)

    # ═════════════════════════════════════════════════════════════════════════
    # SLIDE 2: 01 · THE QUESTION (Light — Hemamdhar)
    # ═════════════════════════════════════════════════════════════════════════
    s2 = prs.slides.add_slide(blank_layout)
    set_bg(s2, LIGHT_BG)
    add_header(s2, "01 · THE QUESTION", "Can past sales predict demand when 90% of days are zeros?",
               presenter_tag="Presenter: Hemamdhar Nath", is_dark=False)

    cards_data_s2 = [
        ("The target", "Daily Unit Demand (y_t)", "Forecast daily unit sales for individual product-store SKUs across an untouched 28-day forward horizon."),
        ("The cutoff", "Strict 28-Day Horizon", "Predictions originate at Day 1913. No actual sales from the 28-day validation period are ever consumed."),
        ("The rule", "Zero-Leakage Rollout", "Multi-step forecasts must be strictly causal (recursive or state-propagated). True sales cannot enter future lags."),
    ]
    for i, (tag, header, desc) in enumerate(cards_data_s2):
        x = Inches(0.8 + i * 3.95)
        card = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, Inches(1.95), Inches(3.75), Inches(3.0))
        card.fill.solid()
        card.fill.fore_color.rgb = LIGHT_CARD
        card.line.color.rgb = LIGHT_BORDER

        tf = card.text_frame
        tf.word_wrap = True
        tf.margin_top = Inches(0.3)
        tf.margin_left = Inches(0.25)
        tf.margin_right = Inches(0.25)

        p1 = tf.paragraphs[0]
        p1.text = tag.upper()
        p1.font.size = Pt(11)
        p1.font.bold = True
        p1.font.color.rgb = ACCENT_AMBER_DARK

        p2 = tf.add_paragraph()
        p2.text = header
        p2.font.size = Pt(16)
        p2.font.bold = True
        p2.font.color.rgb = TEXT_DARK
        p2.space_before = Pt(8)

        p3 = tf.add_paragraph()
        p3.text = desc
        p3.font.size = Pt(13)
        p3.font.color.rgb = TEXT_MUTED_LIGHT
        p3.space_before = Pt(12)

    banner = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(5.35), Inches(11.7), Inches(1.2))
    banner.fill.solid()
    banner.fill.fore_color.rgb = DARK_CARD
    banner.line.color.rgb = DARK_BORDER

    tf_b = banner.text_frame
    tf_b.word_wrap = True
    tf_b.margin_top = Inches(0.18)
    tf_b.margin_left = Inches(0.3)
    p_b1 = tf_b.paragraphs[0]
    p_b1.text = "CORE RESEARCH QUESTION"
    p_b1.font.size = Pt(10)
    p_b1.font.bold = True
    p_b1.font.color.rgb = ACCENT_AMBER
    p_b2 = tf_b.add_paragraph()
    p_b2.text = "Does a localized state-space model (Kalman Filter per SKU) adapt faster than a global neural network (DeepAR) under severe retail intermittency, and which produces better uncertainty bounds for inventory safety stock?"
    p_b2.font.size = Pt(13)
    p_b2.font.color.rgb = TEXT_WHITE
    p_b2.space_before = Pt(4)

    add_footer(s2, 2, is_dark=False)

    # ═════════════════════════════════════════════════════════════════════════
    # SLIDE 3: 02 · THE DATA (Dark — Hemamdhar)
    # ═════════════════════════════════════════════════════════════════════════
    s3 = prs.slides.add_slide(blank_layout)
    set_bg(s3, DARK_BG)
    add_header(s3, "02 · THE DATA", "58.3 Million records, 1,941 daily timesteps",
               presenter_tag="Presenter: Hemamdhar Nath", is_dark=True)

    stats_s3 = [
        ("SOURCE MATRIX", "30,490", "SKUs across 10 stores (CA, TX, WI)"),
        ("INTERMITTENCY", "90.84%", "Intermittent or lumpy sales regimes"),
        ("EVALUATION HORIZON", "1,400", "Untouched points (50 SKUs × 28 days)"),
    ]
    for i, (lbl, big, desc) in enumerate(stats_s3):
        x = Inches(0.8 + i * 3.95)
        card = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, Inches(1.95), Inches(3.75), Inches(1.85))
        card.fill.solid()
        card.fill.fore_color.rgb = DARK_CARD
        card.line.color.rgb = DARK_BORDER

        tf = card.text_frame
        tf.word_wrap = True
        tf.margin_top = Inches(0.2)
        tf.margin_left = Inches(0.25)

        p1 = tf.paragraphs[0]
        p1.text = lbl
        p1.font.size = Pt(11)
        p1.font.bold = True
        p1.font.color.rgb = ACCENT_AMBER

        p2 = tf.add_paragraph()
        p2.text = big
        p2.font.size = Pt(36)
        p2.font.bold = True
        p2.font.color.rgb = TEXT_WHITE

        p3 = tf.add_paragraph()
        p3.text = desc
        p3.font.size = Pt(11)
        p3.font.color.rgb = TEXT_MUTED_DARK

    bar_bg = s3.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(4.2), Inches(11.7), Inches(0.6))
    bar_bg.fill.solid()
    bar_bg.fill.fore_color.rgb = RGBColor(30, 48, 80)
    bar_bg.line.fill.background()

    bar_fg = s3.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(4.2), Inches(11.7 * 0.9084), Inches(0.6))
    bar_fg.fill.solid()
    bar_fg.fill.fore_color.rgb = ACCENT_AMBER
    bar_fg.line.fill.background()

    dist_box = s3.shapes.add_textbox(Inches(0.8), Inches(4.9), Inches(11.7), Inches(0.4))
    tf_dist = dist_box.text_frame
    p_dist = tf_dist.paragraphs[0]
    p_dist.text = "90.84% Intermittent & Lumpy (ADI ≥ 1.32 or CV² ≥ 0.49)              9.16% Smooth & Erratic"
    p_dist.font.size = Pt(12)
    p_dist.font.bold = True
    p_dist.font.color.rgb = TEXT_WHITE

    note_box = s3.shapes.add_textbox(Inches(0.8), Inches(5.4), Inches(11.7), Inches(1.1))
    tf_note = note_box.text_frame
    tf_note.word_wrap = True
    p_note = tf_note.paragraphs[0]
    p_note.text = "Zero-sales observations on active shelf days are preserved as legitimate zero-demand observations rather than treated as missing data or imputed. 12.38 million pre-launch zeros are isolated via first-sale date indexing to ensure clean, uncorrupted demand signals."
    p_note.font.size = Pt(13)
    p_note.font.color.rgb = TEXT_MUTED_DARK

    add_footer(s3, 3, is_dark=True)

    # ═════════════════════════════════════════════════════════════════════════
    # SLIDE 4: 02 · THE DATA (EDA Patterns) (Light — Hemamdhar)
    # ═════════════════════════════════════════════════════════════════════════
    s4 = prs.slides.add_slide(blank_layout)
    set_bg(s4, LIGHT_BG)
    add_header(s4, "02 · THE DATA", "Weekend surges, holiday shocks, and SNAP policy pulses",
               presenter_tag="Presenter: Hemamdhar Nath", is_dark=False)

    fig_path = os.path.join(FIGURES_DIR, "04_seasonality_and_calendar_effects.png")
    if os.path.exists(fig_path):
        s4.shapes.add_picture(fig_path, Inches(0.8), Inches(2.0), width=Inches(6.2))

    insights = [
        ("+20.7% Weekend Surge", "Saturday & Sunday demand significantly outperforms weekday baseline across all categories. Autocorrelation confirms strong weekly harmonics at lags 7, 14, 21, and 28."),
        ("+21.31% SNAP Uplift", "Government food stamp disbursement in Wisconsin creates sharp demand pulses (and +11.1% in Texas, +7.6% in California), concentrated heavily in the FOODS department."),
        ("Dec 25 Zero Sales", "Total chain sales contract to 0 units every year on Christmas. Stationarity confirmed only after first-differencing (ADF test: t = -23.46, p < 1e-6)."),
    ]
    for i, (title, body) in enumerate(insights):
        y = Inches(2.0 + i * 1.55)
        card = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.3), y, Inches(5.2), Inches(1.35))
        card.fill.solid()
        card.fill.fore_color.rgb = LIGHT_CARD
        card.line.color.rgb = LIGHT_BORDER

        tf = card.text_frame
        tf.word_wrap = True
        tf.margin_top = Inches(0.15)
        tf.margin_left = Inches(0.2)
        tf.margin_right = Inches(0.2)

        p1 = tf.paragraphs[0]
        p1.text = title
        p1.font.size = Pt(13.5)
        p1.font.bold = True
        p1.font.color.rgb = ACCENT_AMBER_DARK

        p2 = tf.add_paragraph()
        p2.text = body
        p2.font.size = Pt(10.5)
        p2.font.color.rgb = TEXT_MUTED_LIGHT
        p2.space_before = Pt(4)

    add_footer(s4, 4, is_dark=False)

    # ═════════════════════════════════════════════════════════════════════════
    # SLIDE 5: 03 · THE LEAKAGE TRAP (Dark — Harsha)
    # ═════════════════════════════════════════════════════════════════════════
    s5 = prs.slides.add_slide(blank_layout)
    set_bg(s5, DARK_BG)
    add_header(s5, "03 · THE LEAKAGE TRAP", "The biggest trap in forecasting is future data leakage",
               presenter_tag="Presenter: Harshakarthikeya", is_dark=True)

    card_l = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.95), Inches(3.7), Inches(4.35))
    card_l.fill.solid()
    card_l.fill.fore_color.rgb = DARK_CARD
    card_l.line.color.rgb = DARK_BORDER

    tf_l = card_l.text_frame
    tf_l.word_wrap = True
    tf_l.margin_top = Inches(0.3)
    tf_l.margin_left = Inches(0.25)
    tf_l.margin_right = Inches(0.25)

    p_l1 = tf_l.paragraphs[0]
    p_l1.text = "THE GOLDEN RULE"
    p_l1.font.size = Pt(11)
    p_l1.font.bold = True
    p_l1.font.color.rgb = ACCENT_AMBER

    p_l2 = tf_l.add_paragraph()
    p_l2.text = "Shift(1)\nPrior to Rolling"
    p_l2.font.size = Pt(24)
    p_l2.font.bold = True
    p_l2.font.color.rgb = TEXT_WHITE
    p_l2.space_before = Pt(8)

    p_l3 = tf_l.add_paragraph()
    p_l3.text = "Computing rolling statistics using day t sales introduces direct target leakage. In naive setups this artificially inflates notebook scores by 40%+ while failing completely in production.\n\nEvery rolling metric in our pipeline enforces a strict shift(1) operation."
    p_l3.font.size = Pt(12)
    p_l3.font.color.rgb = TEXT_MUTED_DARK
    p_l3.space_before = Pt(14)

    card_b = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(4.8), Inches(1.95), Inches(3.7), Inches(4.35))
    card_b.fill.solid()
    card_b.fill.fore_color.rgb = RGBColor(35, 20, 25)
    card_b.line.color.rgb = RGBColor(180, 50, 50)

    tf_b = card_b.text_frame
    tf_b.word_wrap = True
    tf_b.margin_top = Inches(0.3)
    tf_b.margin_left = Inches(0.25)
    tf_b.margin_right = Inches(0.25)

    p_b1 = tf_b.paragraphs[0]
    p_b1.text = "BANNED IN PRODUCTION"
    p_b1.font.size = Pt(11)
    p_b1.font.bold = True
    p_b1.font.color.rgb = ACCENT_RED

    items_b = [
        "Unshifted rolling means (using day t)",
        "Consuming actual validation sales in the 28-day rollout",
        "Random temporal cross-validation shuffling",
        "Imputing missing values with global future statistics",
    ]
    for it in items_b:
        p = tf_b.add_paragraph()
        p.text = f"✕  {it}"
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_WHITE
        p.space_before = Pt(10)

    card_a = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.8), Inches(1.95), Inches(3.7), Inches(4.35))
    card_a.fill.solid()
    card_a.fill.fore_color.rgb = RGBColor(15, 30, 30)
    card_a.line.color.rgb = ACCENT_EMERALD

    tf_a = card_a.text_frame
    tf_a.word_wrap = True
    tf_a.margin_top = Inches(0.3)
    tf_a.margin_left = Inches(0.25)
    tf_a.margin_right = Inches(0.25)

    p_a1 = tf_a.paragraphs[0]
    p_a1.text = "ALLOWED & ENFORCED"
    p_a1.font.size = Pt(11)
    p_a1.font.bold = True
    p_a1.font.color.rgb = ACCENT_EMERALD

    items_a = [
        "Causal lags strictly from past: lag 1, 7, 28",
        "Shift(1) rolling mean & std for 7 and 28 days",
        "Known future calendar & SNAP indicators",
        "Pure recursive multi-step rollout using model's own predictions",
    ]
    for it in items_a:
        p = tf_a.add_paragraph()
        p.text = f"✓  {it}"
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_WHITE
        p.space_before = Pt(10)

    add_footer(s5, 5, is_dark=True)

    # ═════════════════════════════════════════════════════════════════════════
    # SLIDE 6: 04 · THE FEATURES & PIPELINE (Light — Harsha)
    # ═════════════════════════════════════════════════════════════════════════
    s6 = prs.slides.add_slide(blank_layout)
    set_bg(s6, LIGHT_BG)
    add_header(s6, "04 · THE FEATURES", "Causal lags, rolling statistics, and calendar signals",
               presenter_tag="Presenter: Harshakarthikeya", is_dark=False)

    cat_feats = [
        ("CATEGORICAL", "Static Hierarchy", "item_id, dept_id, cat_id, store_id, state_id preserved for multi-level hierarchical indexing and static entity embeddings."),
        ("NUMERIC & LAGS", "Causal Autoregressive", "lag_1, lag_7, lag_28, rolling_mean_7, rolling_mean_28, rolling_std_7, rolling_std_28 strictly computed via shift(1)."),
        ("CYCLICAL & POLICY", "Temporal & Policy", "day_sin, day_cos, month_sin, month_cos, is_weekend, holiday markers, and dynamic state-specific snap_active indicators."),
    ]
    for i, (tag, h, desc) in enumerate(cat_feats):
        x = Inches(0.8 + i * 3.95)
        card = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, Inches(1.95), Inches(3.75), Inches(2.5))
        card.fill.solid()
        card.fill.fore_color.rgb = LIGHT_CARD
        card.line.color.rgb = LIGHT_BORDER

        tf = card.text_frame
        tf.word_wrap = True
        tf.margin_top = Inches(0.2)
        tf.margin_left = Inches(0.2)
        tf.margin_right = Inches(0.2)

        p1 = tf.paragraphs[0]
        p1.text = tag
        p1.font.size = Pt(11)
        p1.font.bold = True
        p1.font.color.rgb = ACCENT_AMBER_DARK

        p2 = tf.add_paragraph()
        p2.text = h
        p2.font.size = Pt(15)
        p2.font.bold = True
        p2.font.color.rgb = TEXT_DARK
        p2.space_before = Pt(4)

        p3 = tf.add_paragraph()
        p3.text = desc
        p3.font.size = Pt(11)
        p3.font.color.rgb = TEXT_MUTED_LIGHT
        p3.space_before = Pt(8)

    steps = [
        ("1 · Tidy Reshaping", "Melt wide sales to long panel"),
        ("2 · Date Reindexing", "Verify daily continuity"),
        ("3 · Shift(1) Features", "Causal lags & rolling stats"),
        ("4 · Warmup Trimming", "Drop uncomputable 28 days"),
    ]
    pipe_box = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(4.75), Inches(11.7), Inches(1.7))
    pipe_box.fill.solid()
    pipe_box.fill.fore_color.rgb = DARK_CARD
    pipe_box.line.color.rgb = DARK_BORDER

    tf_p = pipe_box.text_frame
    tf_p.word_wrap = True
    tf_p.margin_top = Inches(0.15)
    tf_p.margin_left = Inches(0.25)
    p_ptit = tf_p.paragraphs[0]
    p_ptit.text = "LEAK-FREE PREPROCESSING PIPELINE (FITTED ON TRAINING HISTORY ONLY)"
    p_ptit.font.size = Pt(10)
    p_ptit.font.bold = True
    p_ptit.font.color.rgb = ACCENT_AMBER

    for i, (st, sub) in enumerate(steps):
        bx = Inches(1.0 + i * 2.85)
        step_card = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, bx, Inches(5.2), Inches(2.6), Inches(1.0))
        step_card.fill.solid()
        step_card.fill.fore_color.rgb = RGBColor(28, 44, 76)
        step_card.line.color.rgb = ACCENT_BLUE

        tf_s = step_card.text_frame
        tf_s.word_wrap = True
        tf_s.margin_top = Inches(0.12)
        tf_s.margin_left = Inches(0.12)
        p1 = tf_s.paragraphs[0]
        p1.text = st
        p1.font.size = Pt(11)
        p1.font.bold = True
        p1.font.color.rgb = TEXT_WHITE
        p2 = tf_s.add_paragraph()
        p2.text = sub
        p2.font.size = Pt(9.5)
        p2.font.color.rgb = TEXT_MUTED_DARK

    add_footer(s6, 6, is_dark=False)

    # ═════════════════════════════════════════════════════════════════════════
    # SLIDE 7: 05 · AML BASELINES & BENCHMARK (Dark — Harsha)
    # ═════════════════════════════════════════════════════════════════════════
    s7 = prs.slides.add_slide(blank_layout)
    set_bg(s7, DARK_BG)
    add_header(s7, "05 · AML BASELINES", "Seasonal-Naive vs Recursive LightGBM Multi-Step Rollout",
               presenter_tag="Presenter: Harshakarthikeya", is_dark=True)

    fig_base = os.path.join(FIGURES_DIR, "10_baseline_aggregate_comparison.png")
    if os.path.exists(fig_base):
        s7.shapes.add_picture(fig_base, Inches(0.8), Inches(2.0), width=Inches(5.7))

    # Right side 2 cards for baselines
    card_s7_1 = s7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(2.0), Inches(5.7), Inches(2.1))
    card_s7_1.fill.solid()
    card_s7_1.fill.fore_color.rgb = DARK_CARD
    card_s7_1.line.color.rgb = DARK_BORDER
    tf7_1 = card_s7_1.text_frame
    tf7_1.word_wrap = True
    tf7_1.margin_top = Inches(0.18)
    tf7_1.margin_left = Inches(0.2)
    p7_1a = tf7_1.paragraphs[0]
    p7_1a.text = "1 · SEASONAL-NAIVE (7-DAY PERSISTENCE)"
    p7_1a.font.size = Pt(11)
    p7_1a.font.bold = True
    p7_1a.font.color.rgb = ACCENT_AMBER
    p7_1b = tf7_1.add_paragraph()
    p7_1b.text = "Formula: ŷ_{T+h} = y_{T+h-7} (repeats same weekday last week)\nZero-parameter deterministic benchmark. Establishes baseline error floor:\nMAE: 1.1143 | RMSE: 2.6030 | WAPE: 95.59%."
    p7_1b.font.size = Pt(11.5)
    p7_1b.font.color.rgb = TEXT_MUTED_DARK
    p7_1b.space_before = Pt(6)

    card_s7_2 = s7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(4.3), Inches(5.7), Inches(2.1))
    card_s7_2.fill.solid()
    card_s7_2.fill.fore_color.rgb = DARK_CARD
    card_s7_2.line.color.rgb = DARK_BORDER
    tf7_2 = card_s7_2.text_frame
    tf7_2.word_wrap = True
    tf7_2.margin_top = Inches(0.18)
    tf7_2.margin_left = Inches(0.2)
    p7_2a = tf7_2.paragraphs[0]
    p7_2a.text = "2 · GBDT (LIGHTGBM RECURSIVE ROLLOUT)"
    p7_2a.font.size = Pt(11)
    p7_2a.font.bold = True
    p7_2a.font.color.rgb = ACCENT_CYAN
    p7_2b = tf7_2.add_paragraph()
    p7_2b.text = "100 boosted trees with strictly recursive multi-step forecasting.\nTop feature: rolling_mean_28 (12.82% importance), followed by rolling_mean_7.\nAchieves 14.64% reduction in WAPE over Seasonal-Naive (81.60%)."
    p7_2b.font.size = Pt(11.5)
    p7_2b.font.color.rgb = TEXT_MUTED_DARK
    p7_2b.space_before = Pt(6)

    add_footer(s7, 7, is_dark=True)

    # ═════════════════════════════════════════════════════════════════════════
    # SLIDE 8: 06 · KALMAN FILTER PER SKU (Light — Sameer)
    # ═════════════════════════════════════════════════════════════════════════
    s8 = prs.slides.add_slide(blank_layout)
    set_bg(s8, LIGHT_BG)
    add_header(s8, "06 · STATE-SPACE MODELING", "Structural Kalman Filter per SKU (Local Level + Seasonality)",
               presenter_tag="Presenter: Sameer Alam Khan", is_dark=False)

    fig_kalman = os.path.join(FIGURES_DIR, "13_kalman_aggregate_tracking.png")
    if os.path.exists(fig_kalman):
        s8.shapes.add_picture(fig_kalman, Inches(0.8), Inches(2.0), width=Inches(5.7))

    card_s8_1 = s8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(2.0), Inches(5.7), Inches(2.1))
    card_s8_1.fill.solid()
    card_s8_1.fill.fore_color.rgb = LIGHT_CARD
    card_s8_1.line.color.rgb = LIGHT_BORDER
    tf8_1 = card_s8_1.text_frame
    tf8_1.word_wrap = True
    tf8_1.margin_top = Inches(0.18)
    tf8_1.margin_left = Inches(0.2)
    p8_1a = tf8_1.paragraphs[0]
    p8_1a.text = "MATHEMATICAL FORMULATION"
    p8_1a.font.size = Pt(11)
    p8_1a.font.bold = True
    p8_1a.font.color.rgb = ACCENT_AMBER_DARK
    p8_1b = tf8_1.add_paragraph()
    p8_1b.text = "State Vector: x_t = [μ_t, s_{1,t}, ..., s_{6,t}]ᵀ ∈ ℝ⁷\nTransition: x_t = F x_{t-1} + w_t,  w_t ~ N(0, Q)\nObservation: y_t = H x_t + v_t,  v_t ~ N(0, R)\nTracks latent consumption level μ_t and 7-day cyclical harmonic in closed form."
    p8_1b.font.size = Pt(11)
    p8_1b.font.color.rgb = TEXT_DARK
    p8_1b.space_before = Pt(4)

    card_s8_2 = s8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(4.3), Inches(5.7), Inches(2.1))
    card_s8_2.fill.solid()
    card_s8_2.fill.fore_color.rgb = LIGHT_CARD
    card_s8_2.line.color.rgb = LIGHT_BORDER
    tf8_2 = card_s8_2.text_frame
    tf8_2.word_wrap = True
    tf8_2.margin_top = Inches(0.18)
    tf8_2.margin_left = Inches(0.2)
    p8_2a = tf8_2.paragraphs[0]
    p8_2a.text = "BENCHMARK PERFORMANCE & ANALYTICAL UNCERTAINTY"
    p8_2a.font.size = Pt(11)
    p8_2a.font.bold = True
    p8_2a.font.color.rgb = ACCENT_EMERALD
    p8_2b = tf8_2.add_paragraph()
    p8_2b.text = "🏆 Lowest MAE (0.9333) and lowest WAPE (80.06%) across all models.\nExact analytical covariance propagation P_t yields 80.93% coverage at 80% CI (width 2.30 units) and 90.07% at 95% CI.\nAdapts to individual SKU step-changes without recursive compounding."
    p8_2b.font.size = Pt(11)
    p8_2b.font.color.rgb = TEXT_DARK
    p8_2b.space_before = Pt(4)

    add_footer(s8, 8, is_dark=False)

    # ═════════════════════════════════════════════════════════════════════════
    # SLIDE 9: 07 · GLOBAL DEEPAR FORECASTER (Dark — Akshay)
    # ═════════════════════════════════════════════════════════════════════════
    s9 = prs.slides.add_slide(blank_layout)
    set_bg(s9, DARK_BG)
    add_header(s9, "07 · PROBABILISTIC DEEP LEARNING", "Global DeepAR Autoregressive LSTM Network",
               presenter_tag="Presenter: Akshay Y", is_dark=True)

    fig_deepar = os.path.join(FIGURES_DIR, "deepar_forecasts.png")
    if os.path.exists(fig_deepar):
        s9.shapes.add_picture(fig_deepar, Inches(0.8), Inches(2.0), width=Inches(5.7))

    card_s9_1 = s9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(2.0), Inches(5.7), Inches(2.1))
    card_s9_1.fill.solid()
    card_s9_1.fill.fore_color.rgb = DARK_CARD
    card_s9_1.line.color.rgb = DARK_BORDER
    tf9_1 = card_s9_1.text_frame
    tf9_1.word_wrap = True
    tf9_1.margin_top = Inches(0.18)
    tf9_1.margin_left = Inches(0.2)
    p9_1a = tf9_1.paragraphs[0]
    p9_1a.text = "UNIFIED GLOBAL RECURRENT ARCHITECTURE"
    p9_1a.font.size = Pt(11)
    p9_1a.font.bold = True
    p9_1a.font.color.rgb = ACCENT_AMBER
    p9_1b = tf9_1.add_paragraph()
    p9_1b.text = "Single LSTM network trained across all 50 SKUs simultaneously.\nScale normalization: ν_i = 1 + (1/T) ∑ y_{i,t} ensures scale invariance.\nOutputs parameters of Gaussian likelihood: y_{i,t} ~ N(μ_{i,t} ν_i, (σ_{i,t} ν_i)²).\nShares representations across sparse series via entity embeddings."
    p9_1b.font.size = Pt(11)
    p9_1b.font.color.rgb = TEXT_MUTED_DARK
    p9_1b.space_before = Pt(4)

    card_s9_2 = s9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(4.3), Inches(5.7), Inches(2.1))
    card_s9_2.fill.solid()
    card_s9_2.fill.fore_color.rgb = DARK_CARD
    card_s9_2.line.color.rgb = DARK_BORDER
    tf9_2 = card_s9_2.text_frame
    tf9_2.word_wrap = True
    tf9_2.margin_top = Inches(0.18)
    tf9_2.margin_left = Inches(0.2)
    p9_2a = tf9_2.paragraphs[0]
    p9_2a.text = "MONTE CARLO QUANTILE ROLLOUT & RMSE SUPREMACY"
    p9_2a.font.size = Pt(11)
    p9_2a.font.bold = True
    p9_2a.font.color.rgb = ACCENT_CYAN
    p9_2b = tf9_2.add_paragraph()
    p9_2b.text = "🏆 Lowest RMSE (2.2062) across all models (−15.24% vs baseline).\n200 Monte Carlo sample paths drawn autoregressively for empirical quantiles.\n80% Prediction Interval Coverage: 91.79% (width 3.20 units).\nMean Pinball Loss: 0.3075 (P10: 0.1116, P50: 0.5208, P90: 0.3258)."
    p9_2b.font.size = Pt(11)
    p9_2b.font.color.rgb = TEXT_MUTED_DARK
    p9_2b.space_before = Pt(4)

    add_footer(s9, 9, is_dark=True)

    # ═════════════════════════════════════════════════════════════════════════
    # SLIDE 10: 08 · THE FINAL EXAM / BENCHMARK (Light — Abhishek)
    # ═════════════════════════════════════════════════════════════════════════
    s10 = prs.slides.add_slide(blank_layout)
    set_bg(s10, LIGHT_BG)
    add_header(s10, "08 · THE FINAL EXAM", "Four-Way Benchmark Scorecard & Error Trade-Offs",
               presenter_tag="Presenter: Abhishek Tripathi", is_dark=False)

    score_cards = [
        ("MEAN ABSOLUTE ERROR (MAE)", "0.9333", "🏆 Kalman Filter", "GBDT: 0.9512 | DeepAR: 0.9748 | SNaive: 1.1143\n16.25% lower error than baseline."),
        ("ROOT MEAN SQ ERROR (RMSE)", "2.2062", "🏆 Global DeepAR", "GBDT: 2.2679 | Kalman: 2.3303 | SNaive: 2.6030\n15.24% lower error than baseline."),
        ("WEIGHTED ABS % ERROR (WAPE)", "80.06%", "🏆 Kalman Filter", "GBDT: 81.60% | DeepAR: 83.63% | SNaive: 95.59%\nLowest overall portfolio error."),
        ("80% INTERVAL COVERAGE", "91.79%", "🏆 Global DeepAR", "Pinball loss: 0.3075 (width 3.20)\nKalman: 80.93% coverage (width 2.30)."),
    ]
    for i, (metric, val, winner, details) in enumerate(score_cards):
        x = Inches(0.8 + i * 2.95)
        card = s10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, Inches(1.95), Inches(2.8), Inches(2.35))
        card.fill.solid()
        card.fill.fore_color.rgb = LIGHT_CARD
        card.line.color.rgb = LIGHT_BORDER

        tf = card.text_frame
        tf.word_wrap = True
        tf.margin_top = Inches(0.18)
        tf.margin_left = Inches(0.18)
        tf.margin_right = Inches(0.18)

        p1 = tf.paragraphs[0]
        p1.text = metric
        p1.font.size = Pt(9.5)
        p1.font.bold = True
        p1.font.color.rgb = ACCENT_AMBER_DARK

        p2 = tf.add_paragraph()
        p2.text = val
        p2.font.size = Pt(30)
        p2.font.bold = True
        p2.font.color.rgb = TEXT_DARK
        p2.space_before = Pt(4)

        p3 = tf.add_paragraph()
        p3.text = winner
        p3.font.size = Pt(11)
        p3.font.bold = True
        p3.font.color.rgb = ACCENT_EMERALD
        p3.space_before = Pt(2)

        p4 = tf.add_paragraph()
        p4.text = details
        p4.font.size = Pt(9.5)
        p4.font.color.rgb = TEXT_MUTED_LIGHT
        p4.space_before = Pt(4)

    # Diagnostic explanations below
    diag1 = s10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(4.55), Inches(5.75), Inches(2.1))
    diag1.fill.solid()
    diag1.fill.fore_color.rgb = LIGHT_CARD
    diag1.line.color.rgb = LIGHT_BORDER
    tf_d1 = diag1.text_frame
    tf_d1.word_wrap = True
    tf_d1.margin_top = Inches(0.18)
    tf_d1.margin_left = Inches(0.2)
    p_d1_h = tf_d1.paragraphs[0]
    p_d1_h.text = "WHY KALMAN FILTER WON MAE & WAPE (SAMEER'S MODEL)"
    p_d1_h.font.size = Pt(11)
    p_d1_h.font.bold = True
    p_d1_h.font.color.rgb = ACCENT_AMBER_DARK
    p_d1_b = tf_d1.add_paragraph()
    p_d1_b.text = "Structural decomposition into local level and 7-day seasonality updates per SKU in closed form. It avoids recursive error compounding and adapts instantly to individual product step-changes without cross-series distortion."
    p_d1_b.font.size = Pt(11.5)
    p_d1_b.font.color.rgb = TEXT_MUTED_LIGHT
    p_d1_b.space_before = Pt(6)

    diag2 = s10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.75), Inches(4.55), Inches(5.75), Inches(2.1))
    diag2.fill.solid()
    diag2.fill.fore_color.rgb = LIGHT_CARD
    diag2.line.color.rgb = LIGHT_BORDER
    tf_d2 = diag2.text_frame
    tf_d2.word_wrap = True
    tf_d2.margin_top = Inches(0.18)
    tf_d2.margin_left = Inches(0.2)
    p_d2_h = tf_d2.paragraphs[0]
    p_d2_h.text = "WHY GLOBAL DEEPAR WON RMSE (AKSHAY'S MODEL)"
    p_d2_h.font.size = Pt(11)
    p_d2_h.font.bold = True
    p_d2_h.font.color.rgb = ACCENT_BLUE
    p_d2_b = tf_d2.add_paragraph()
    p_d2_b.text = "DeepAR's shared recurrent representation across 50 SKUs pools statistical strength. This acts as a strong regularizer that severely penalizes extreme outlier forecasts, achieving the lowest quadratic loss."
    p_d2_b.font.size = Pt(11.5)
    p_d2_b.font.color.rgb = TEXT_MUTED_LIGHT
    p_d2_b.space_before = Pt(6)

    add_footer(s10, 10, is_dark=False)

    # ═════════════════════════════════════════════════════════════════════════
    # SLIDE 11: 09 · THE TAKEAWAY (Dark — All Team Members)
    # ═════════════════════════════════════════════════════════════════════════
    s11 = prs.slides.add_slide(blank_layout)
    set_bg(s11, DARK_BG)
    add_header(s11, "09 · THE TAKEAWAY", "No single model dominates: Align model with inventory goal",
               presenter_tag="Presenters: All Team Members", is_dark=True)

    c_left = s11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.95), Inches(5.75), Inches(3.45))
    c_left.fill.solid()
    c_left.fill.fore_color.rgb = DARK_CARD
    c_left.line.color.rgb = DARK_BORDER
    tf_cl = c_left.text_frame
    tf_cl.word_wrap = True
    tf_cl.margin_top = Inches(0.2)
    tf_cl.margin_left = Inches(0.25)
    tf_cl.margin_right = Inches(0.25)

    p_cl_h = tf_cl.paragraphs[0]
    p_cl_h.text = "OPERATIONAL DECISION RULES"
    p_cl_h.font.size = Pt(11)
    p_cl_h.font.bold = True
    p_cl_h.font.color.rgb = ACCENT_AMBER

    rules = [
        ("Routine Daily Replenishment", "Kalman Filter: Lowest absolute units off-target (0.9333 MAE) and lightweight per-SKU execution."),
        ("Safety Stock & Risk Mitigation", "Global DeepAR: Calibrated prediction intervals (91.8% at 80% CI) protect against expensive stockout spikes."),
        ("Fast Tabular Operations", "LightGBM: Fast training, handles tabular calendar features well, highly interpretable via feature importance."),
    ]
    for rh, rd in rules:
        p = tf_cl.add_paragraph()
        p.text = f"• {rh}: {rd}"
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_WHITE
        p.space_before = Pt(8)

    c_right = s11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.75), Inches(1.95), Inches(5.75), Inches(3.45))
    c_right.fill.solid()
    c_right.fill.fore_color.rgb = DARK_CARD
    c_right.line.color.rgb = DARK_BORDER
    tf_cr = c_right.text_frame
    tf_cr.word_wrap = True
    tf_cr.margin_top = Inches(0.2)
    tf_cr.margin_left = Inches(0.25)
    tf_cr.margin_right = Inches(0.25)

    p_cr_h = tf_cr.paragraphs[0]
    p_cr_h.text = "LIMITATIONS & FUTURE HORIZONS"
    p_cr_h.font.size = Pt(11)
    p_cr_h.font.bold = True
    p_cr_h.font.color.rgb = ACCENT_CYAN

    limits = [
        ("Evaluation Scale", "Tested on 50 representative series; next phase scales to all 30,490 SKUs."),
        ("DeepAR Interval Sharpness", "80% CI coverage reached 91.79%, indicating conservative intervals that can be tightened with hyperparameter tuning."),
        ("Hybrid Architecture", "Future work: Feed Kalman Filter latent states (μ_t, s_t) directly as features into DeepAR to combine local tracking with global sequence learning."),
    ]
    for lh, ld in limits:
        p = tf_cr.add_paragraph()
        p.text = f"• {lh}: {ld}"
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_WHITE
        p.space_before = Pt(8)

    verdict = s11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(5.6), Inches(11.7), Inches(1.0))
    verdict.fill.solid()
    verdict.fill.fore_color.rgb = RGBColor(18, 30, 54)
    verdict.line.color.rgb = DARK_BORDER
    tf_v = verdict.text_frame
    tf_v.word_wrap = True
    tf_v.margin_top = Inches(0.18)
    tf_v.margin_left = Inches(0.3)
    p_vh = tf_v.paragraphs[0]
    p_vh.text = "FINAL VERDICT"
    p_vh.font.size = Pt(10)
    p_vh.font.bold = True
    p_vh.font.color.rgb = ACCENT_AMBER
    p_vb = tf_v.add_paragraph()
    p_vb.text = "Kalman Filter minimizes average units lost; Global DeepAR mitigates extreme inventory risk."
    p_vb.font.size = Pt(13)
    p_vb.font.bold = True
    p_vb.font.color.rgb = TEXT_WHITE
    p_vb.space_before = Pt(2)

    add_footer(s11, 11, is_dark=True)

    prs.save(output_path)
    print(f"Presentation saved successfully to {output_path}")

if __name__ == "__main__":
    create_deck()
