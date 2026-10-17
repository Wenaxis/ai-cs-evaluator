"""生成自托管的 SVG 徽章（替代 shields.io）。

为什么不用 shields.io：国内网络对 shields.io 时通时不通，HR 打开仓库看到五个破图，
比没有徽章更糟。把徽章当静态文件放在仓库里，永远显示得出来。

用法：python make_badges.py <输出目录>
"""
import os
import sys

# (文件名, 左标签, 右值, 右侧颜色)
BADGES = [
    ("python.svg", "Python", "3.13", "#3776AB"),
    ("tests.svg", "tests", "37 passed", "#2EA043"),
    ("kappa.svg", "Cohen's Kappa", "0.9231", "#1A5C9E"),
    ("license.svg", "license", "MIT", "#111111"),
    ("docs.svg", "docs", "中文", "#C2410C"),
]

CHAR_W = 8.4   # 11px 下各字体里最宽的那个，宁可留白也别挤出去
PAD = 14
H = 20


def measure(s):
    """按字符估宽：中文/全角字符约为西文的两倍，不然「中文」两个字会溢出。"""
    w = 0.0
    for ch in s:
        w += CHAR_W * (1.9 if ord(ch) > 0x2E80 else 1.0)
    return w + PAD


SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" role="img" aria-label="{label}: {value}">
  <title>{label}: {value}</title>
  <rect width="{lw}" height="{h}" rx="3" ry="3" fill="#555555"/>
  <rect x="{lw}" width="{vw}" height="{h}" rx="3" ry="3" fill="{color}"/>
  <rect x="{lw}" width="4" height="{h}" fill="{color}"/>
  <g fill="#ffffff" text-anchor="middle"
     font-family="Verdana,Geneva,DejaVu Sans,sans-serif" font-size="11">
    <text x="{lx}" y="14">{label}</text>
    <text x="{vx}" y="14">{value}</text>
  </g>
</svg>
"""


def main(outdir):
    os.makedirs(outdir, exist_ok=True)
    for name, label, value, color in BADGES:
        lw, vw = round(measure(label), 1), round(measure(value), 1)
        w = round(lw + vw, 1)
        svg = SVG.format(w=w, h=H, lw=lw, vw=vw, color=color,
                         lx=lw / 2, vx=lw + vw / 2,
                         label=label, value=value)
        path = os.path.join(outdir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(svg)
        print(f"[badge] {path}  {w}x{H}")


if __name__ == "__main__":
    main(sys.argv[1])
