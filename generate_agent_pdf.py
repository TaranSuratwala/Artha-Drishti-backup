import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak
)
from reportlab.lib import colors

# Drawing imports for Graph
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon, Group
from reportlab.graphics import renderPDF

OUTPUT = os.path.join(os.path.dirname(__file__), "Artha_Drishti_AI_Agent_Documentation.pdf")

# Colors
C1 = HexColor("#1a73e8")
C2 = HexColor("#6366f1")
C3 = HexColor("#0f172a")
C4 = HexColor("#222222")
BG = HexColor("#f1f5f9")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle("TT", parent=styles["Title"], fontSize=24, leading=30, textColor=C1, spaceAfter=6, alignment=TA_CENTER, fontName="Helvetica-Bold"))
styles.add(ParagraphStyle("ST", parent=styles["Normal"], fontSize=12, leading=16, textColor=HexColor("#475569"), spaceAfter=20, alignment=TA_CENTER))
styles.add(ParagraphStyle("S1", parent=styles["Heading1"], fontSize=18, leading=22, textColor=C1, spaceBefore=18, spaceAfter=8, fontName="Helvetica-Bold"))
styles.add(ParagraphStyle("S2", parent=styles["Heading2"], fontSize=14, leading=18, textColor=C3, spaceBefore=12, spaceAfter=6, fontName="Helvetica-Bold"))
styles.add(ParagraphStyle("B", parent=styles["Normal"], fontSize=10, leading=14, textColor=C4, spaceAfter=6, alignment=TA_JUSTIFY))
styles.add(ParagraphStyle("BL", parent=styles["Normal"], fontSize=10, leading=13, textColor=C4, leftIndent=18, bulletIndent=6, spaceAfter=3))
styles.add(ParagraphStyle("TH", parent=styles["Normal"], fontSize=10, leading=12, textColor=colors.white, fontName="Helvetica-Bold", alignment=TA_CENTER))
styles.add(ParagraphStyle("TC", parent=styles["Normal"], fontSize=9, leading=12, textColor=C4))

def T(headers, rows, cw=None):
    d = [[Paragraph(h, styles["TH"]) for h in headers]]
    for r in rows:
        d.append([Paragraph(str(c), styles["TC"]) for c in r])
    t = Table(d, colWidths=cw, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), C1), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,0), 10),
        ("BOTTOMPADDING", (0,0), (-1,0), 6), ("TOPPADDING", (0,0), (-1,0), 6),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, BG]),
        ("GRID", (0,0), (-1,-1), 0.5, HexColor("#cbd5e1")),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 6), ("RIGHTPADDING", (0,0), (-1,-1), 6),
    ]))
    return t

def create_agent_graph():
    """Create a LangGraph architecture diagram using ReportLab graphics."""
    d = Drawing(400, 200)
    
    # Nodes
    d.add(Rect(150, 150, 100, 40, rx=5, ry=5, fillColor=HexColor("#e2e8f0"), strokeColor=C1, strokeWidth=2))
    d.add(String(200, 165, "START", fontName="Helvetica-Bold", fontSize=12, textAnchor="middle"))
    
    d.add(Rect(150, 80, 100, 40, rx=5, ry=5, fillColor=HexColor("#dbeafe"), strokeColor=C1, strokeWidth=2))
    d.add(String(200, 95, "Agent (LLM)", fontName="Helvetica-Bold", fontSize=12, textAnchor="middle"))
    
    d.add(Rect(300, 80, 100, 40, rx=5, ry=5, fillColor=HexColor("#fef08a"), strokeColor=HexColor("#ca8a04"), strokeWidth=2))
    d.add(String(350, 95, "Tools", fontName="Helvetica-Bold", fontSize=12, textAnchor="middle"))
    
    d.add(Rect(150, 10, 100, 40, rx=5, ry=5, fillColor=HexColor("#e2e8f0"), strokeColor=C1, strokeWidth=2))
    d.add(String(200, 25, "END", fontName="Helvetica-Bold", fontSize=12, textAnchor="middle"))

    # Edges
    # Start -> Agent
    d.add(Line(200, 150, 200, 120, strokeColor=C3, strokeWidth=1.5))
    d.add(Polygon([196, 126, 204, 126, 200, 120], fillColor=C3, strokeColor=C3))

    # Agent -> Tools
    d.add(Line(250, 105, 300, 105, strokeColor=C3, strokeWidth=1.5))
    d.add(Polygon([294, 109, 294, 101, 300, 105], fillColor=C3, strokeColor=C3))
    d.add(String(275, 110, "tool_call", fontName="Helvetica", fontSize=8, textAnchor="middle"))

    # Tools -> Agent
    d.add(Line(300, 95, 250, 95, strokeColor=C3, strokeWidth=1.5))
    d.add(Polygon([256, 99, 256, 91, 250, 95], fillColor=C3, strokeColor=C3))
    
    # Agent -> END
    d.add(Line(200, 80, 200, 50, strokeColor=C3, strokeWidth=1.5))
    d.add(Polygon([196, 56, 204, 56, 200, 50], fillColor=C3, strokeColor=C3))
    d.add(String(215, 65, "no tools", fontName="Helvetica", fontSize=8, textAnchor="start"))

    return d

def build_doc():
    doc = SimpleDocTemplate(OUTPUT, pagesize=A4, rightMargin=inch, leftMargin=inch, topMargin=inch, bottomMargin=inch)
    story = []

    # Title
    story.append(Paragraph("Artha Drishti AI Agent", styles["TT"]))
    story.append(Paragraph("Detailed Architecture & Capability Report", styles["ST"]))
    story.append(Spacer(1, 0.2 * inch))

    # 1. Introduction
    story.append(Paragraph("1. Introduction", styles["S1"]))
    story.append(Paragraph(
        "The Artha Drishti AI Agent is a professional Stock Market Assistant designed for the Indian financial markets (NSE/BSE). "
        "Built on top of LangChain and LangGraph, the agent harnesses the power of Google's Gemini LLMs to "
        "provide real-time market data, run trading strategies, analyze sentiment, and deliver AI-driven price predictions.",
        styles["B"]
    ))
    story.append(Spacer(1, 0.1 * inch))

    # 2. Core Architecture
    story.append(Paragraph("2. Core Architecture (LangGraph)", styles["S1"]))
    story.append(Paragraph(
        "The agent is implemented as a state machine using LangGraph. This design enables deterministic cycles "
        "where the LLM decides whether to invoke tools, and loops back upon tool completion. State persistence "
        "is managed via PostgreSQL (using PostgresSaver), ensuring conversation history across sessions.",
        styles["B"]
    ))
    
    # Graph depiction
    story.append(Spacer(1, 0.2 * inch))
    story.append(create_agent_graph())
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Figure 1: LangGraph State Machine Architecture of the AI Agent", styles["ST"]))

    story.append(PageBreak())

    # 3. Agent Tools and Capabilities
    story.append(Paragraph("3. Available Tools & Capabilities", styles["S1"]))
    story.append(Paragraph(
        "To prevent hallucination, the agent strictly relies on specialized tools rather than its parametric memory "
        "for financial data. Below is the list of integrated tools:",
        styles["B"]
    ))
    story.append(Spacer(1, 0.1 * inch))

    tools_headers = ["Tool Name", "Description", "Primary Use Case"]
    tools_rows = [
        ["screen_stocks", "Runs quantitative screening strategies (Piotroski, Momentum, Swing, Breakout, Value).", "Finding stocks that match specific technical or fundamental criteria."],
        ["predict_stock", "Fetches AI price predictions and technical analysis signals for a specific NSE ticker.", "Forecasting price movements and assessing buy/sell signals."],
        ["get_market_overview", "Retrieves the latest market snapshot with price and volume data for top stocks.", "Providing a quick summary of current market conditions."],
        ["get_stock_history", "Gets recent OHLCV (Open, High, Low, Close, Volume) history for the last 10 days.", "Analyzing recent price action for a specific company."],
        ["get_sentiment", "Analyzes news sentiment, calculating a score (-1 to 1) based on recent articles.", "Gauging market mood and evaluating news impact on a stock."],
        ["generate_pattern_chart", "Generates interactive Plotly candlestick charts with detected patterns.", "Visualizing stock data and technical patterns directly in the chat UI."],
        ["search_web", "Performs live web searches using Google Search API for grounded answers.", "Fetching breaking news, IPO dates, or live prices not covered by other tools."]
    ]
    story.append(T(tools_headers, tools_rows, cw=[1.2*inch, 3.2*inch, 1.8*inch]))
    story.append(Spacer(1, 0.3 * inch))

    # 4. Prompt Engineering & Guardrails
    story.append(Paragraph("4. Prompt Engineering & Guardrails", styles["S1"]))
    story.append(Paragraph(
        "The agent relies on a strict System Prompt with rules designed to ensure safety, accuracy, and professional conduct:",
        styles["B"]
    ))
    story.append(Paragraph("• <font color='#1a73e8'><b>Anti-Hallucination:</b></font> The agent is instructed NEVER to fabricate prices, percentages, or dates. All numerical data must stem from a tool result.", styles["BL"]))
    story.append(Paragraph("• <font color='#1a73e8'><b>Honesty:</b></font> If data is missing or a tool fails, the agent honestly admits its inability to fetch the information.", styles["BL"]))
    story.append(Paragraph("• <font color='#1a73e8'><b>Mandatory Disclaimer:</b></font> The agent automatically appends a financial disclaimer to its responses: 'This is AI-generated analysis... not financial advice.'", styles["BL"]))
    story.append(Paragraph("• <font color='#1a73e8'><b>UI Integration:</b></font> Special markdown tags (e.g., chart rendering tags) returned by tools are passed exactly as-is to ensure proper frontend rendering.", styles["BL"]))

    # 5. Data Flow and Persistence
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph("5. Data Flow and Persistence", styles["S1"]))
    story.append(Paragraph(
        "The system employs a <b>streaming architecture</b>. As the LangGraph agent executes, chunks of text and "
        "tool invocation statuses are streamed immediately to the user via Server-Sent Events (SSE). The "
        "<b>PostgresSaver</b> checkpointer automatically saves the `MessagesState` to a PostgreSQL database at "
        "the end of every execution node. This allows a user to refresh the page and continue their session seamlessly "
        "while capping the max history at a configurable limit (e.g., 20 messages) to save context window tokens.",
        styles["B"]
    ))

    # Build PDF
    doc.build(story)

if __name__ == "__main__":
    build_doc()
    print(f"Documentation generated at: {OUTPUT}")
