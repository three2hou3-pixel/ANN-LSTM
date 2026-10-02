from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

root = Path(r'D:\Project Code\ANN-LSTM-Exam')
out_path = root / 'outputs' / 'reports' / 'L1_L6_exam_answer.docx'

# ==========================================
# Word 基础设置：正式考试答案风格
# ==========================================
doc = Document()
section = doc.sections[0]
section.top_margin = Inches(1)
section.bottom_margin = Inches(1)
section.left_margin = Inches(1)
section.right_margin = Inches(1)
section.header_distance = Inches(0.492)
section.footer_distance = Inches(0.492)

styles = doc.styles
normal = styles['Normal']
normal.font.name = 'Calibri'
normal._element.rPr.rFonts.set(qn('w:eastAsia'), 'Microsoft YaHei')
normal.font.size = Pt(11)
normal.paragraph_format.space_after = Pt(6)
normal.paragraph_format.line_spacing = 1.10

for style_name, size, color, before, after in [
    ('Heading 1', 16, '2E74B5', 16, 8),
    ('Heading 2', 13, '2E74B5', 12, 6),
    ('Heading 3', 12, '1F4D78', 8, 4),
]:
    st = styles[style_name]
    st.font.name = 'Calibri'
    st._element.rPr.rFonts.set(qn('w:eastAsia'), 'Microsoft YaHei')
    st.font.size = Pt(size)
    st.font.color.rgb = RGBColor.from_string(color)
    st.paragraph_format.space_before = Pt(before)
    st.paragraph_format.space_after = Pt(after)
    st.paragraph_format.line_spacing = 1.10

# ==========================================
# 常用写入函数
# ==========================================
def set_font(run, size=11, bold=False, color=None, mono=False):
    run.font.name = 'Consolas' if mono else 'Calibri'
    run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Microsoft YaHei')
    run.font.size = Pt(size)
    run.bold = bold
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def add_title(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(12)
    run = p.add_run(text)
    set_font(run, size=18, bold=True, color='1F4D78')


def add_subtitle(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(14)
    run = p.add_run(text)
    set_font(run, size=10, color='555555')


def add_h1(text):
    doc.add_paragraph(text, style='Heading 1')


def add_h2(text):
    doc.add_paragraph(text, style='Heading 2')


def add_p(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run(text)
    set_font(run)


def add_bullet(text):
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(text)
    set_font(run)


def add_code(text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.25)
    p.paragraph_format.space_after = Pt(8)
    run = p.add_run(text)
    set_font(run, size=9, mono=True, color='1F4D78')


def shade_cell(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:fill'), fill)
    tc_pr.append(shd)


def set_cell_text(cell, text, header=False):
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER if header or len(str(text)) < 18 else WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    run = p.add_run(str(text))
    set_font(run, size=9, bold=header)


def add_table(headers, rows):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = 'Table Grid'
    table.autofit = True
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        set_cell_text(cell, h, header=True)
        shade_cell(cell, 'F2F4F7')
    for i, row in enumerate(rows, start=1):
        for j, value in enumerate(row):
            set_cell_text(table.cell(i, j), value)
    doc.add_paragraph()

# 页脚页码
footer = section.footer.paragraphs[0]
footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
r = footer.add_run('第 ')
set_font(r, size=9)
field_begin = OxmlElement('w:fldChar')
field_begin.set(qn('w:fldCharType'), 'begin')
instr = OxmlElement('w:instrText')
instr.set(qn('xml:space'), 'preserve')
instr.text = 'PAGE'
field_end = OxmlElement('w:fldChar')
field_end.set(qn('w:fldCharType'), 'end')
footer._p.append(field_begin)
footer._p.append(instr)
footer._p.append(field_end)
r = footer.add_run(' 页')
set_font(r, size=9)

# ==========================================
# 正文内容：六道考试题答案
# ==========================================
add_title('AI 增强型量化选股与回测系统考试答案')
add_subtitle('基于 ANN-LSTM-Exam 项目 run_full_pipeline.py 的实测输出整理；未输出的内容不做虚构。')

add_h1('L1 数据获取与预处理')
add_p('本题选择标的为 300502.SZ。数据来源为本地已下载的 Tushare 日频 CSV 文件，其中个股数据为 data/stock_300502.csv，基准指数数据为 data/benchmark_300.csv。实测读取到的个股数据区间为 2016-03-03 至 2026-05-29，共 2485 条日频样本，覆盖时间超过 10 年，满足题目要求的“不少于 5 年，且截至 2026 年 5 月”。')
add_p('数据预处理时，首先按交易日期升序排列，并删除重复日期；随后检查缺失值，实测原始缺失单元格数量为 0。项目仍保留前向填充和后向填充作为兜底逻辑，以防后续更换数据源后出现少量缺失。异常值处理采用 1% 和 99% 分位数缩尾，目的是降低极端异常值或异常行情对统计检验和模型训练的影响。')
add_p('复权方式采用前复权。前复权可以保持最新价格不变，同时修正历史价格中的分红、送转影响，因此更适合构造收益率和技术指标。项目构造的核心收益率为：')
add_code('r_t = ln(Close_t / Close_{t-1})\ntarget_h1  = ln(Close_{t+1}  / Close_t)\ntarget_h5  = ln(Close_{t+5}  / Close_t)\ntarget_h20 = ln(Close_{t+20} / Close_t)')
add_p('日历效应按星期几分组统计。实测结果如下：')
add_table(['星期', '平均收益', '标准差', '样本数', '正收益比例'], [
    ['Monday', '0.005600', '0.044051', '457', '56.67%'],
    ['Tuesday', '0.003729', '0.037497', '473', '52.22%'],
    ['Wednesday', '-0.000966', '0.037373', '476', '47.69%'],
    ['Thursday', '0.001308', '0.039668', '475', '50.11%'],
    ['Friday', '-0.000796', '0.038327', '466', '46.78%'],
])
add_p('从结果看，Monday 和 Tuesday 的平均收益与正收益比例相对较高，Wednesday 和 Friday 相对偏弱，说明样本内存在一定星期效应。但当前流程只做了描述性统计，尚未进一步做显著性检验，因此考试作答中应谨慎表述为“存在描述性差异”。')

add_h1('L2 探索性分析与平稳性检验')
add_p('收益率序列的描述统计结果如下：')
add_table(['指标', '实测值'], [
    ['均值', '0.001753'],
    ['标准差', '0.039477'],
    ['偏度', '0.275234'],
    ['峰度', '2.687883'],
    ['JB 统计量', '731.7661'],
    ['JB p 值', '1.26e-159'],
])
add_p('从分布形态看，收益率均值接近 0，但标准差较大，说明该标的日收益波动较高；偏度为正，说明收益率分布略右偏；JB 检验 p 值极小，显著拒绝正态分布假设，符合金融收益率序列常见的非正态、高噪声特征。')
add_p('平稳性检验方面，ADF 统计量为 -46.0753，p 值为 0.0，强烈拒绝单位根原假设，说明收益率序列支持平稳。KPSS 统计量为 0.4671，p 值约 0.0491，在 5% 水平附近拒绝平稳性原假设，提示序列可能存在结构变化或状态切换。')
add_p('综合 ADF 与 KPSS 结果，本文在 ARIMA 建模中取差分阶数 d = 0。理由是建模对象已经是对数收益率，而不是价格水平序列；ADF 明确支持收益率平稳。但 KPSS 的边界结果提示后续分析需要关注非平稳市场状态和时变波动。')

add_h1('L3 ARMA/ARIMA 基准建模')
add_p('本题在 p,q ∈ [0,5] 的范围内进行 ARIMA 网格搜索，并以 AIC/BIC 作为选择依据。实测前三个候选模型如下：')
add_table(['排名', '模型', 'AIC', 'BIC'], [
    ['1', 'ARIMA(1,0,0)', '-6697.8196', '-6681.3547'],
    ['2', 'ARIMA(0,0,1)', '-6697.7045', '-6681.2397'],
    ['3', 'ARIMA(0,0,0)', '-6697.5143', '-6686.5377'],
])
add_p('因此，本文选择 ARIMA(1,0,0) 作为传统时间序列基准模型。残差诊断中，Ljung-Box 检验 lag 10 的 p 值为 0.4177，lag 20 的 p 值为 0.6487，均大于 0.05，说明残差没有显著线性自相关，ARIMA 对均值方程中的线性相关结构刻画基本充分。')
add_p('但是 ARCH-LM 检验统计量为 84.6770，p 值约 6.04e-14，显著拒绝无 ARCH 效应的原假设，说明残差存在明显条件异方差和波动聚集。因此，若继续改进传统模型，应在 ARIMA 均值方程基础上引入 GARCH(1,1)、EGARCH 或 GJR-GARCH 等波动率模型。')
add_p('滚动窗口一步外推预测结果为：方向准确率 50.00%，RMSE 为 0.04798，MAE 为 0.03408。该结果说明 ARIMA 作为基准模型误差规模较稳定，但方向预测能力基本接近随机猜测。')

add_h1('L4 VAR/VECM 多元扩展')
add_p('多元扩展中，项目使用个股收益率、沪深 300 收益率以及若干市场状态变量进行 VAR 相关分析。核心经济逻辑是：沪深 300 收益率代表市场系统性风险，成交量变化代表资金活跃度，滚动波动率代表风险状态和波动聚集特征。这些变量能够从市场、资金和风险三个角度补充单变量 ARIMA 模型。')
add_p('Johansen 检验中，trace statistic 为 1389.2753，大于 5% 临界值 15.4943。不过需要注意，当前流程使用的是收益率变量，而不是价格水平变量，因此不能过度解释为价格之间存在严格协整关系。更稳妥的表述是：收益率和市场变量之间存在较强共同动态变化特征。')
add_p('当前实测流程完成了 Johansen 检验、VAR 滚动预测、Granger 因果检验和 IRF 图输出，但没有完整输出 VECM 的协整向量与调整系数。因此在考试答案中必须如实说明：本项目完成了 VECM 判断所需的协整检验，但未完整建立和报告 VECM 参数，严格来说 L4 的 VECM 部分仍有不足。')
add_p('Granger 因果检验结果如下：')
add_table(['变量', 'p 值', '结论'], [
    ['hs300_ret', '0.6983', '5% 水平不显著'],
    ['volume_change', '0.3163', '5% 水平不显著'],
    ['volatility_20', '0.1014', '5% 水平不显著'],
])
add_p('以上变量在 5% 显著性水平下均未表现出显著 Granger 因果关系。其中 volatility_20 的 p 值相对较低，说明波动率状态可能仍具有一定预测信息，因此后续 AI 特征矩阵中继续保留该类变量。')

add_h1('L5 AI 增强建模与集成')
add_p('本题是系统核心。项目构造了 41 维监督学习特征矩阵，满足题目要求的“不少于 20 维”。特征包括滞后收益、技术指标、波动率与量价特征、市场因子，以及 ARIMA/VAR 残差滞后项。')
add_bullet('滞后收益：ret_lag_1 到 ret_lag_10。')
add_bullet('技术指标：MA 偏离、MACD、RSI、ATR、OBV、布林带宽度等。')
add_bullet('波动率与量价：滚动波动率、日内收益、振幅、成交量变化、成交额变化等。')
add_bullet('市场因子：沪深 300 收益率、市场波动率、超额收益、滚动 beta、滚动相关性等。')
add_bullet('传统模型残差：arima_resid_lag_1、arima_resid_lag_2、var_resid_lag_1、var_resid_lag_2。')
add_p('这正是“AI 增强传统模型”的关键：不是只把 ARIMA、VAR 和 LSTM 放在一起比较，而是把 ARIMA/VAR 没有解释掉的残差信息作为 AI 模型输入，让 LSTM 继续学习传统线性模型未捕捉到的非线性关系。')
add_p('特征选择使用了两种方法：互信息法和随机森林重要性。互信息结果中，intraday_return_scaled、MACD_HIST_scaled、arima_resid_lag_2_scaled、var_resid_lag_2_scaled 等排名靠前；随机森林结果中，ret_lag_7_scaled、hs300_ret_scaled、hs300_volatility_20_scaled、arima_resid_lag_2_scaled、var_resid_lag_2_scaled 等排名靠前。由此可见，ARIMA/VAR 残差滞后项确实提供了一定信息增量。')
add_p('AI 模型采用扩展后的 ANN-LSTM，并基于 41 维新特征重新训练。旧的 10 维 best_model.pth 权重没有覆盖，只作为旧版基线保留。由于新旧输入维度不同，直接复用旧输入层权重并不严谨，因此主实验采用重新训练。')
add_p('测试集模型对比结果如下：')
add_table(['模型', '方向准确率', 'RMSE', 'MAE'], [
    ['LSTM', '51.89%', '0.04807', '0.03397'],
    ['ARIMA', '50.00%', '0.04785', '0.03401'],
    ['VAR', '48.74%', '0.04819', '0.03441'],
    ['Ensemble', '47.80%', '0.04796', '0.03393'],
])
add_p('从结果看，LSTM 的方向准确率最高，为 51.89%；ARIMA 的 RMSE 最低；集成模型的 MAE 最低，但方向准确率不如 LSTM。因此不能夸大为 AI 模型全面优于传统模型，只能说 AI 在方向识别上略有优势，集成模型在平均绝对误差上略有优势。')
add_p('Diebold-Mariano 检验结果显示，LSTM vs ARIMA 的 p 值为 0.1490，LSTM vs VAR 的 p 值为 0.7083，LSTM vs Ensemble 的 p 值为 0.3318，均大于 0.05。因此，不同模型的预测误差差异没有达到统计显著水平。考试答案中应谨慎表述为：AI 增强模型有一定改进迹象，但 DM 检验未证明其显著优于传统模型。')

add_h1('L6 策略回测与最终交付')
add_p('策略使用模型预测的未来收益率生成交易信号。若预测收益率大于阈值，则持有；否则空仓或降低仓位。交易成本按照 A 股规则设置：印花税 0.05%、佣金 0.025%、滑点 0.1%。样本外测试期为 2025-01-01 至 2026-05-31，与训练期不重叠，且超过 12 个月，满足题目要求。')
add_p('样本外回测结果如下：')
add_table(['指标', '实测结果'], [
    ['总收益率', '270.05%'],
    ['年化收益率', '170.68%'],
    ['年化波动率', '55.52%'],
    ['Sharpe', '2.0292'],
    ['最大回撤 MDD', '31.85%'],
    ['Calmar', '5.3590'],
    ['胜率', '30.82%'],
    ['盈亏比', '2.4507'],
    ['换手率', '39.75%'],
])
add_p('该结果说明策略胜率并不高，但盈亏比较高，主要依靠少数较大的盈利交易贡献收益。策略样本外收益较强，但最大回撤达到 31.85%，说明风险仍然较高，实盘应用中需要进一步加强仓位控制和止损机制。')
add_p('稳健性检验中，对关键信号阈值和仓位方式进行扰动，Sharpe 大致位于 1.96 到 2.35 区间。说明策略对参数变化具有一定稳健性。波动率降权方案可以明显降低最大回撤，但也会牺牲年化收益率。')
add_p('分市态表现方面，牛市总收益约 218.98%，Sharpe 为 2.7130；震荡市总收益仅 3.40%，Sharpe 为 0.3399；熊市样本天数较少，仅 19 天，虽然统计结果较好，但不能过度解释。整体看，策略更适合趋势行情，在震荡行情中表现明显变弱。')
add_p('最终结论是：本项目在 ARIMA/VAR 传统时间序列模型基础上，引入 41 维监督学习特征矩阵，并将 ARIMA/VAR 残差作为 AI 输入特征，实现了“传统模型骨架 + AI 非线性增强”的量化建模框架。实测显示，LSTM 在方向准确率上略优于 ARIMA 和 VAR，集成模型在 MAE 上略优，但 DM 检验未显示统计显著优势。因此，本项目可以称为 AI 增强型量化系统，但不能夸大为 AI 显著战胜传统模型。')
add_p('需要如实说明的不足包括：VECM 未完整输出协整向量与调整系数；日历效应只做了描述性统计；模型预测差异未通过 DM 显著性检验；样本外回测收益较高但回撤较大，震荡市表现较弱。后续可从 GARCH 波动率建模、更严格的市态识别、更多外部因子和更稳健的仓位风控继续改进。')

# ==========================================
# 保存文档
# ==========================================
doc.core_properties.title = 'AI 增强型量化选股与回测系统考试答案'
doc.core_properties.subject = 'L1-L6 六道考试题实测答案'
doc.core_properties.author = 'Codex'
doc.save(out_path)
print(out_path)
