#!/usr/bin/env python3
"""
Benchmark comparison: Boolean SHA-256 vs Tropical SHA-256 (Round 16)
Generates a PDF report with formatted benchmark results.
"""

import time
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# Import both implementations
from sha256_bool import compute_sha256_bool_round16
from sha256_shortcut import compute_sha256_trop_round16


def benchmark_function(func, message, iterations=100):
    """Benchmark a function with given message and iterations."""
    # Warmup
    for _ in range(10):
        func(message)
    
    # Actual benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        func(message)
    end = time.perf_counter()
    
    total_time = (end - start) * 1000  # Convert to milliseconds
    avg_time = total_time / iterations
    hashes_per_sec = iterations / (end - start)
    
    return avg_time, hashes_per_sec, total_time


def run_benchmarks():
    """Run benchmarks for both implementations."""
    test_messages = [
        (b"", "Empty"),
        (b"abc", "Short (3 bytes)"),
        (b"The quick brown fox jumps over the lazy dog", "Medium (43 bytes)"),
        (b"a" * 55, "Max single-block (55 bytes)"),
    ]
    
    iterations = 100
    
    results = {
        'Boolean': [],
        'Tropical': []
    }
    
    print("Running benchmarks...")
    print("=" * 70)
    
    for msg, desc in test_messages:
        print(f"\nTesting: {desc}")
        
        # Benchmark Boolean version
        try:
            bool_avg, bool_hps, bool_total = benchmark_function(
                compute_sha256_bool_round16, msg, iterations
            )
            results['Boolean'].append({
                'desc': desc,
                'msg_len': len(msg),
                'avg_ms': bool_avg,
                'hps': bool_hps,
                'total_ms': bool_total
            })
            print(f"  Boolean:  {bool_avg:.4f} ms/hash, {bool_hps:.1f} hashes/sec")
        except Exception as e:
            print(f"  Boolean: ERROR - {e}")
            results['Boolean'].append(None)
        
        # Benchmark Tropical version
        try:
            trop_avg, trop_hps, trop_total = benchmark_function(
                compute_sha256_trop_round16, msg, iterations
            )
            results['Tropical'].append({
                'desc': desc,
                'msg_len': len(msg),
                'avg_ms': trop_avg,
                'hps': trop_hps,
                'total_ms': trop_total
            })
            print(f"  Tropical: {trop_avg:.4f} ms/hash, {trop_hps:.1f} hashes/sec")
        except Exception as e:
            print(f"  Tropical: ERROR - {e}")
            results['Tropical'].append(None)
    
    return results, test_messages


def generate_pdf(results, test_messages, filename="benchmark_report.pdf"):
    """Generate a beautifully formatted PDF report."""
    
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )
    
    elements = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.darkblue,
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.navy,
        spaceAfter=12,
        spaceBefore=12
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        leading=14,
        alignment=TA_LEFT
    )
    
    # Title
    elements.append(Paragraph("SHA-256 Round 16: Boolean vs Tropical Algebra", title_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Introduction
    intro_text = """
    This report compares the performance of two SHA-256 implementations that execute 
    exactly 16 rounds of the compression function:<br/><br/>
    
    <b>1. Boolean Implementation:</b> Standard SHA-256 using native bitwise operations 
    (AND, OR, XOR, NOT) on 32-bit integers, JIT-compiled with Numba.<br/><br/>
    
    <b>2. Tropical Implementation:</b> Tropicalized SHA-256 using Min-Plus semiring 
    arithmetic on floating-point vectors. Each 32-bit word is represented as 32 floats, 
    with boolean operations mapped to tropical algebra:<br/>
    • True → 0.0, False → 10.0<br/>
    • AND → addition, OR → min, NOT → C-a<br/>
    • XOR → min(a+C-b, C-a+b)<br/><br/>
    
    All functions are JIT-compiled with Numba for fair comparison.
    """
    elements.append(Paragraph(intro_text, normal_style))
    elements.append(Spacer(1, 0.3*inch))
    
    # Benchmark Results Section
    elements.append(Paragraph("Benchmark Results", heading_style))
    
    # Summary table
    summary_data = [['Test Case', 'Message Length', 'Boolean (ms)', 'Tropical (ms)', 
                     'Slowdown Factor', 'Speed Ratio (B/T)']]
    
    for i, (msg, desc) in enumerate(test_messages):
        bool_res = results['Boolean'][i]
        trop_res = results['Tropical'][i]
        
        if bool_res and trop_res:
            slowdown = trop_res['avg_ms'] / bool_res['avg_ms']
            speed_ratio = bool_res['hps'] / trop_res['hps']
            summary_data.append([
                desc,
                f"{bool_res['msg_len']} bytes",
                f"{bool_res['avg_ms']:.4f}",
                f"{trop_res['avg_ms']:.4f}",
                f"{slowdown:.2f}x",
                f"{speed_ratio:.2f}x"
            ])
    
    summary_table = Table(summary_data, colWidths=[1.8*inch, 1.2*inch, 1*inch, 1*inch, 1*inch, 1*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
    ]))
    
    elements.append(summary_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Detailed Boolean Results
    elements.append(Paragraph("Detailed Boolean Implementation Results", heading_style))
    
    bool_data = [['Test Case', 'Avg Time (ms)', 'Hashes/sec', 'Total Time (100 iter, ms)']]
    for i, res in enumerate(results['Boolean']):
        if res:
            bool_data.append([
                res['desc'],
                f"{res['avg_ms']:.4f}",
                f"{res['hps']:.1f}",
                f"{res['total_ms']:.2f}"
            ])
    
    bool_table = Table(bool_data, colWidths=[2.5*inch, 1.2*inch, 1.2*inch, 1.5*inch])
    bool_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.navy),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
    ]))
    
    elements.append(bool_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Detailed Tropical Results
    elements.append(Paragraph("Detailed Tropical Implementation Results", heading_style))
    
    trop_data = [['Test Case', 'Avg Time (ms)', 'Hashes/sec', 'Total Time (100 iter, ms)']]
    for i, res in enumerate(results['Tropical']):
        if res:
            trop_data.append([
                res['desc'],
                f"{res['avg_ms']:.4f}",
                f"{res['hps']:.1f}",
                f"{res['total_ms']:.2f}"
            ])
    
    trop_table = Table(trop_data, colWidths=[2.5*inch, 1.2*inch, 1.2*inch, 1.5*inch])
    trop_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.purple),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lavender),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
    ]))
    
    elements.append(trop_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Analysis section
    elements.append(Paragraph("Performance Analysis", heading_style))
    
    # Calculate overall statistics
    bool_avg_overall = np.mean([r['avg_ms'] for r in results['Boolean'] if r])
    trop_avg_overall = np.mean([r['avg_ms'] for r in results['Tropical'] if r])
    overall_slowdown = trop_avg_overall / bool_avg_overall
    
    analysis_text = f"""
    <b>Key Findings:</b><br/><br/>
    
    1. <b>Average Performance:</b><br/>
       • Boolean: {bool_avg_overall:.4f} ms/hash<br/>
       • Tropical: {trop_avg_overall:.4f} ms/hash<br/>
       • Overall Slowdown: {overall_slowdown:.2f}x<br/><br/>
    
    2. <b>Why Tropical is Slower:</b><br/>
       • Each 32-bit word requires 32 float operations instead of 1 integer op<br/>
       • Tropical XOR requires 2 subtractions, 2 additions, and 1 comparison<br/>
       • Vector operations (rotation, shifting) require element-wise processing<br/>
       • Memory overhead: 32 floats (256 bytes) vs 1 uint32 (4 bytes) per word<br/><br/>
    
    3. <b>Trade-offs:</b><br/>
       • Boolean: Fast, standard, production-ready<br/>
       • Tropical: Research-oriented, enables SAT solver integration,<br/>
         cryptanalysis applications, and tropical geometry exploration<br/><br/>
    
    4. <b>Use Cases:</b><br/>
       • Boolean: General-purpose hashing, verification<br/>
       • Tropical: Academic research, formal verification, constraint solving
    """
    
    elements.append(Paragraph(analysis_text, normal_style))
    elements.append(Spacer(1, 0.3*inch))
    
    # Technical Details
    elements.append(Paragraph("Technical Implementation Details", heading_style))
    
    tech_text = """
    <b>Boolean Implementation:</b><br/>
    • Native 32-bit integer arithmetic<br/>
    • Bitwise operators: & (AND), | (OR), ^ (XOR), ~ (NOT)<br/>
    • Rotation: ((x >> n) | (x << (32-n))) & 0xFFFFFFFF<br/>
    • Single value per word, direct CPU instruction mapping<br/><br/>
    
    <b>Tropical Implementation:</b><br/>
    • Min-Plus semiring: (ℝ ∪ {∞}, min, +)<br/>
    • Encoding: bit=1 → 0.0 (True), bit=0 → 10.0 (False)<br/>
    • Tropical NOT: C - a<br/>
    • Tropical AND: a + b<br/>
    • Tropical OR: min(a, b)<br/>
    • Tropical XOR: min(a + (C-b), (C-a) + b)<br/>
    • Tropical Ch: min(x + y, (C-x) + z)<br/>
    • Tropical Maj: min(x+y, x+z, y+z)<br/>
    • Vector representation: 32 floats per 32-bit word
    """
    
    elements.append(Paragraph(tech_text, normal_style))
    elements.append(Spacer(1, 0.3*inch))
    
    # Footer
    footer_text = """
    <i>Note: Benchmarks performed with Numba JIT compilation enabled. 
    First execution includes compilation overhead (excluded from measurements via warmup). 
    Test environment: Python with NumPy and Numba acceleration.</i>
    """
    
    elements.append(Paragraph(footer_text, normal_style))
    
    # Build PDF
    doc.build(elements)
    print(f"\nPDF report generated: {filename}")


if __name__ == "__main__":
    # Run benchmarks
    results, test_messages = run_benchmarks()
    
    # Generate PDF
    generate_pdf(results, test_messages)
    
    print("\n" + "=" * 70)
    print("Benchmark complete! Check benchmark_report.pdf for detailed results.")
