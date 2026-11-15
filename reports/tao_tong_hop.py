"""Tổng hợp kết quả đã lưu; không chạy lại hay thay đổi thực nghiệm.

Chạy từ thư mục bất kỳ: python <đường_dẫn>/reports/tao_tong_hop.py
Phụ thuộc vẽ hình: matplotlib, numpy.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
COLORS = ("#3168a8", "#d37a36", "#8d99a8")


def number(x, places=0):
    if x is None:
        return "—"
    return f"{x:,.{places}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def percent(x, places=2):
    return number(100 * x, places) + "%"


def duration(seconds):
    sec = round(seconds)
    h, sec = divmod(sec, 3600)
    m, s = divmod(sec, 60)
    return f"{h:02}:{m:02}:{s:02}"


def link(path, label=None):
    path = Path(path)
    return f"[{label or path.name}](<{path.resolve().as_posix()}>)"


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(v) for v in row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


records = []
for folder in (ROOT / "checkpoints", ROOT / "src/checkpoints"):
    for path in folder.glob("*/metrics.json"):
        raw = path.read_bytes()
        d = json.loads(raw)
        d["source_path"] = path.relative_to(ROOT).as_posix()
        d["source_sha256"] = hashlib.sha256(raw).hexdigest()
        d["plot_path"] = next((p.relative_to(ROOT).as_posix() for p in sorted(path.parent.glob("training_log.*"))
                               if p.suffix.lower() in (".jpg", ".png")), None)
        records.append(d)
records.sort(key=lambda d: d["started_at"])
BY_ID = {}
for i, d in enumerate(records, 1):
    d["report_id"] = f"E{i:02}"
    tr = d["training"]
    n = tr["games_played"]
    assert sum(tr["wins"].values()) == n, d["experiment"]
    if d.get("metrics"):
        assert sum(d["metrics"]["draw_reasons"].values()) == tr["wins"]["Draw"], d["experiment"]
        assert d["metrics"]["throughput"]["total_games"] == n, d["experiment"]
    d["derived"] = {
        "overall_rates": {k: v / n for k, v in tr["wins"].items()},
        "average_games_per_second": n / d["elapsed_seconds"],
        "white_score_rate": (tr["wins"]["White"] + 0.5 * tr["wins"]["Draw"]) / n,
    }
    BY_ID[d["experiment"]] = d


def get(suffix):
    return BY_ID["exp_" + suffix]


def algo(d, side):
    key = d.get("algorithms", {}).get(side)
    return {"a2c": "A2C", "ppo": "PPO", "dqn": "DQN", "alphazero": "AlphaZero"}.get(key, "Baseline*")


def pair(d):
    return f"{algo(d, 'White')} / {algo(d, 'Black')}"


NOTES = {
    "20261108_175144": "Phiên nền tảng dài đầu tiên; hòa chiếm 92,47%. Chưa có nhóm metrics chi tiết nên không xác định được cơ cấu nguyên nhân hòa từ file này.",
    "20261108_222012": "Huấn luyện tiếp từ E01; hòa còn 88,62% trong phiên mới. Đây là mô tả hai phiên nối tiếp, chưa chứng minh tác động của một thay đổi riêng lẻ.",
    "20261109_004149": "Chỉ 28 ván trong 16,7 giây; mức hòa 50% không đủ đại diện để chọn mô hình tốt nhất. Các bàn ngắn kết thúc trước cũng có thể ảnh hưởng mẫu đầu phiên.",
    "20261109_140833": "Phiên nối tiếp ngắn, 440 ván; hòa 85,00%, phần lớn ván hòa do giới hạn lượt hoặc lặp.",
    "20261109_231239": "Phiên nhiều ván nhất: 225.447 ván. Trắng thắng 25.718, Đen thắng 8.060; hòa vẫn 85,02%. Biểu đồ loss có đỉnh rất lớn, entropy Đen thường gần 0: dấu hiệu cần xem xét về độ ổn định và độ đa dạng hành động.",
    "20261111_095430": "Tiếp tục E05 trong thời gian ngắn; Trắng thắng nhiều hơn Đen, nhưng hòa 82,21%. Biểu đồ actor loss Đen dao động lớn và entropy hai bên thấp.",
    "20261111_135357": "A2C từ đầu, chạy ngắn; 572/575 ván hòa là do chạm giới hạn lượt. Chưa thể đánh giá khả năng kết thúc ván.",
    "20261111_140057": "A2C từ đầu; 2.842/2.866 ván hòa do chạm giới hạn lượt. Độ dài gần 160 ply ở cửa sổ cuối cho thấy nhiều ván bị cắt.",
    "20261111_141426": "A2C từ đầu, 651 ván; 586/590 ván hòa do giới hạn lượt, hòa ở cửa sổ cuối lên 97%.",
    "20261111_143632": "A2C chạy dài theo đồng hồ: 20 giờ 28 phút, 23.559 ván, hòa 86,88%. Không coi thời gian này là thời gian tính toán GPU thuần; file không tách thời gian tạm dừng.",
    "20261112_113115": "A2C Trắng nạp từ E10, PPO Đen không có đường dẫn nạp. Trắng thắng 316 so với 129: so sánh chịu ảnh hưởng lịch sử huấn luyện không cân bằng.",
    "20261112_115622": "Cả hai tiếp tục E11; PPO Đen thắng 93 so với 72 của A2C. Kết quả đảo chiều so với phiên trước nên chưa có thứ hạng A2C/PPO ổn định.",
    "20261112_121441": "A2C và DQN từ đầu; DQN Đen thắng 89 so với 16 của A2C trong 638 ván. Phiên còn ngắn và là huấn luyện đồng thời.",
    "20261112_121856": "AlphaZero Đen thắng 32 so với 6 của A2C Trắng; chỉ 145 ván, A2C nạp từ E13. Kết quả thăm dò, chưa đủ kết luận tổng quát.",
    "20261112_132532": "A2C Trắng nạp từ E14, DQN Đen từ E13; DQN thắng 952 so với 67 trong 4.409 ván. Đây là ưu thế quan sát được trong cấu hình cụ thể.",
    "20261112_134110": "DQN tự đấu; cả hai màu nạp cùng checkpoint DQN Đen của E15. Hai bên thắng gần cân bằng (1.260/1.342), hòa 89,48%; 14.117 ván bị cắt theo giới hạn lượt.",
    "20261112_153443": "AlphaZero Trắng và DQN Đen từ đầu; AlphaZero thắng 73 so với 26, hòa 76,48%. Chỉ 421 ván và DQN có giai đoạn warm-up, nên không dùng làm xếp hạng cuối.",
    "20261112_154125": "AlphaZero tự đấu 34.219 ván; hòa 91,35%. Lặp thế cờ gây 30.193 ván hòa, bằng 96,59% số ván hòa. Loss giảm nhưng khả năng tạo kết quả thắng/thua chưa cải thiện tương ứng.",
    "20261112_232706": "Tiếp tục E18 chỉ 50 ván; cả 45 ván hòa đều do lặp. Reward trung bình −1,7879, thành phần phạt lặp −1,7071.",
    "20261112_232956": "Tiếp tục E19: 268 ván hòa đều do lặp. Thành phần mang tên repetition_penalty_mean lại bằng +1,66, reward trung bình +1,58. Cần kiểm tra dấu/hệ số reward của phiên lịch sử; không dùng reward dương này làm bằng chứng chơi tốt hơn.",
    "20261114_081112": "PPO Trắng / DQN Đen từ đầu; DQN thắng 497 so với 139, hòa 85,03%. Cấu hình gốc max_plies=160, draw_reward=−0,1.",
    "20261114_084332": "Đổi màu hai checkpoint E21, đồng thời tăng max_plies lên 200 và phạt hòa xuống −5. DQN thắng 91 so với 50; hòa 85,14%. Đổi nhiều yếu tố cùng lúc nên không tách được tác động từng yếu tố.",
    "20261114_084944": "Tiếp tục E22, max_plies=600 và draw_reward=−10; hòa 80,76%, chỉ 4 ván hòa do chạm giới hạn lượt. Learning rate cấu hình là 0,003 nhưng cả hai mô hình ghi 0,0003; chưa có bằng chứng LR thực tế tăng 10 lần.",
    "20261114_145819": "A2C tự đấu trên CPU, từ đầu: 45.204 ván, hòa 85,13%. Cuối phiên entropy 0,0274/0,1364, explained variance −0,23/−0,4966 và gradient clipping 100% ở cửa sổ cập nhật: dấu hiệu chính sách tập trung và critic cần cải thiện.",
    "20261114_164954": "PPO Trắng / DQN Đen, nạp đổi màu từ E23; mạng tăng lên 6.385.348 tham số mỗi bên, có dữ liệu strategy và curriculum. PPO thắng 4.003 so với 950; hòa 85,47%. Kết quả bị ảnh hưởng đồng thời bởi checkpoint, màu, mạng, reward và dữ liệu khởi tạo.",
    "20261114_190522": "PPO tự đấu CPU, từ đầu: 75.686 ván, hòa 85,42%. Entropy cuối 1,5031/1,6845, giữ đa dạng hơn phiên A2C CPU theo chỉ số này; 48.488 ván hòa do giới hạn lượt nên vẫn khó kết thúc ván.",
    "20261114_235757": "DQN tự đấu CPU, từ đầu, mạng 6.385.348 tham số: 11.077 ván, hòa 91,80%. Có 7.775 ván hòa do lặp; cửa sổ cuối hòa 96%. Explained variance Q-target cao không đồng nghĩa sức chơi cao.",
}

PLOT_NOTES = {
    "20261109_004149": "Biểu đồ dưới 0,3 phút, ít điểm và chỉ 28 ván; chỉ dùng minh họa thử chạy/ghi log.",
    "20261109_231239": "Hòa chiếm ưu thế; actor loss có các đỉnh rất lớn, entropy Đen thường sát 0. Có đoạn đường nối dài, chưa đủ dữ liệu thô để giải thích khoảng trống.",
    "20261111_095430": "Phiên khoảng 1,2 phút; actor loss Đen dao động mạnh, entropy thấp. Không diễn giải đường giảm critic loss thành tiến bộ sức chơi.",
    "20261112_115622": "PPO Đen có critic loss thấp hơn A2C Trắng trong hình; tỷ lệ hòa giảm ở cuối ảnh. Không so độ lớn hai hàm loss để kết luận thuật toán tốt hơn.",
    "20261112_121441": "DQN Đen tăng tỷ lệ thắng trong phần sau ảnh; A2C Trắng gần 0 ở cửa sổ cuối. Actor loss DQN bằng 0 theo cách ghi chỉ số.",
    "20261112_154125": "Loss AlphaZero giảm dần trong khi hòa vẫn rất cao; cơ cấu hòa do lặp tăng, độ dài ván giảm. Đây là hình rõ nhất để bàn về mục tiêu tối ưu và hành vi thực tế.",
    "20261112_232706": "Mẫu rất nhỏ; hòa do lặp tăng nhanh, reward giảm sâu theo thành phần phạt lặp.",
    "20261112_232956": "Đường thành phần lặp mang dấu dương, cùng reward dương, trong khi hòa do lặp vẫn áp đảo. Hình củng cố việc cần kiểm tra reward lịch sử.",
    "20261114_081112": "DQN Đen thường có tỷ lệ thắng lớn hơn PPO Trắng; hòa chủ yếu do giới hạn lượt. PPO có một số điểm explained variance âm sâu.",
    "20261114_084332": "Reward bị chi phối bởi phạt hòa và lặp; ảnh chỉ bao phủ khoảng 2 phút trong khi metrics ghi 4,82 phút. Không xem mép phải ảnh là snapshot cuối JSON.",
    "20261114_084944": "Reward âm lớn, ván dài hơn, hòa do giới hạn lượt gần biến mất nhưng các nguyên nhân hòa khác còn cao. Ảnh khoảng 50 phút, metrics ghi 58,52 phút; hai artifact có thể khác thời điểm lưu.",
    "20261114_145819": "Entropy giảm về rất thấp ở cuối, gradient norm tăng mạnh và explained variance âm; loss không ổn định hoàn toàn. Hòa vẫn chiếm phần lớn.",
    "20261114_164954": "PPO Trắng thắng nhiều hơn DQN Đen; loss giảm sau đầu phiên nhưng hòa vẫn cao. Entropy tăng dần trong phiên này, khác nhóm PPO tự đấu CPU.",
    "20261114_190522": "Entropy giảm từ khoảng 3,4 rồi duy trì trên 1 ở phần sau; hòa vẫn quanh mức cao. Có đỉnh explained variance âm rất sâu, không nên chỉ báo cáo giá trị cuối.",
    "20261114_235757": "Hòa thường trên 90%; loss DQN tăng trong giai đoạn đầu rồi dao động. Cột CPU/GPU trống vì thiếu dữ liệu, không phải mức sử dụng bằng 0%.",
}


def make_figures():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    # All runs: descriptive outcomes, explicitly not a ranking.
    fig, ax = plt.subplots(figsize=(12, 13))
    y = np.arange(len(records))
    left = np.zeros(len(records))
    for key, label, color in zip(("White", "Black", "Draw"), ("Trắng thắng", "Đen thắng", "Hòa"), COLORS):
        values = np.array([d["derived"]["overall_rates"][key] * 100 for d in records])
        ax.barh(y, values, left=left, label=label, color=color, height=.73)
        if key == "Draw":
            for yy, v, ll in zip(y, values, left):
                ax.text(ll + v / 2, yy, f"{v:.1f}%", ha="center", va="center", fontsize=8, color="white")
        left += values
    ax.set_yticks(y, [f"{d['report_id']}  {pair(d)}  |  n={d['training']['games_played']:,}" for d in records], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Tỷ lệ trên toàn bộ ván của từng phiên (%)")
    ax.set_title("Kết quả 27 phiên huấn luyện cờ vua", loc="left", fontsize=16, fontweight="bold", pad=84)
    ax.text(0, 1.055, "Cấu hình và lịch sử huấn luyện khác nhau; biểu đồ mô tả kết quả, không xếp hạng sức chơi.",
            transform=ax.transAxes, fontsize=9)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(.5, 1.003), frameon=False)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"01_ket_qua_27_phien.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    # Main runs, denominator is all games, so the result sums to the draw rate.
    chosen = [get(k) for k in ("20261112_134110", "20261112_154125", "20261114_145819",
                                "20261114_190522", "20261114_164954", "20261114_235757")]
    fig, ax = plt.subplots(figsize=(12, 5.4))
    left = np.zeros(len(chosen))
    reasons = [("repetition", "Lặp", "#785ab5"), ("max_plies", "Giới hạn ply", "#4c88aa"),
               ("insufficient_material", "Thiếu lực lượng", "#83aa79"), ("stalemate", "Hết nước hợp lệ", "#d3a05a"),
               ("fifty_moves", "Luật 50 nước", "#c77075"), ("other", "Khác", "#bbbbbb")]
    for key, label, color in reasons:
        values = np.array([d["metrics"]["draw_reasons"]["draw_by_" + key] / d["training"]["games_played"] * 100 for d in chosen])
        ax.barh(np.arange(len(chosen)), values, left=left, label=label, color=color, height=.65)
        for yy, v, ll in zip(range(len(chosen)), values, left):
            if v >= 7:
                ax.text(ll + v / 2, yy, f"{v:.1f}", ha="center", va="center", fontsize=9, color="white")
        left += values
    ax.set_yticks(range(len(chosen)), [f"{d['report_id']}  {pair(d)}" for d in chosen])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Tỷ lệ nguyên nhân hòa trên toàn bộ ván của phiên (%)")
    ax.set_title("Vì sao các ván kết thúc hòa?", loc="left", fontsize=16, fontweight="bold", pad=38)
    ax.text(0, 1.04, "Phần trống đến 100% là các ván có bên thắng. Các phiên có cấu hình khác nhau.", transform=ax.transAxes, fontsize=9)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(.5, -.2), frameon=False, fontsize=9)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"02_nguyen_nhan_hoa.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_report():
    total_games = sum(d["training"]["games_played"] for d in records)
    total_seconds = sum(d["elapsed_seconds"] for d in records)
    nplots = sum(bool(d["plot_path"]) for d in records)
    p = ["# TỔNG HỢP THỰC NGHIỆM — HỌC TĂNG CƯỜNG CHO CỜ VUA\n",
         "Ngày tổng hợp: **15/11/2026**. Nguồn: dữ liệu trong repository tại thời điểm tổng hợp; không chạy thêm huấn luyện hoặc đánh giá mô hình.\n",
         "## 1. Phạm vi và kết luận chính\n",
         f"Đã tổng hợp **{len(records)} file `metrics.json` và {nplots} ảnh `training_log`**, gồm 25 phiên trong `checkpoints/` và 2 phiên trong `src/checkpoints/`. Các phiên bắt đầu từ 08–14/11/2026; snapshot cuối có thời điểm lưu tới 15/11/2026. Tổng số ván ghi nhận trong các phiên là **{number(total_games)}**, tổng `elapsed_seconds` là **{number(total_seconds / 3600, 2)} giờ**. Tổng này cộng các phiên, có huấn luyện nối tiếp và phiên có thể chạy đồng thời; không phải 27 phép thử độc lập, cũng không phải thời gian tính toán thuần.\n",
         "Các kết quả cho thấy hệ thống đã thu thập được dữ liệu huấn luyện cho A2C, PPO, DQN và AlphaZero cùng nhóm Actor–Critic nền tảng. Hạn chế nổi bật là **tỷ lệ hòa cao**, thường khoảng 80–92% ở các phiên lớn; hai nguyên nhân cần phân biệt là chạm giới hạn lượt và lặp thế cờ. Dữ liệu hiện tại chưa đủ để xếp hạng sức chơi tổng quát hoặc công bố Elo.\n",
         "Các điểm nên đưa vào phần nhận xét:\n\n- DQN có ưu thế quan sát được trước A2C trong E13/E15 và trước PPO trong E21–E23; PPO lại thắng nhiều hơn DQN trong E25. Điều kiện của các nhóm khác nhau nên chưa có thuật toán thắng tuyệt đối.\n- AlphaZero E18 giảm loss nhưng 91,35% ván hòa; 96,59% số ván hòa do lặp. Đây là ví dụ rõ về việc loss giảm chưa bảo đảm sức chơi tăng.\n- A2C CPU E24 và PPO CPU E26 có tỷ lệ hòa gần nhau (85,13% và 85,42%). PPO giữ entropy cuối cao hơn; điều này chỉ hỗ trợ nhận xét về đa dạng chính sách trong hai phiên.\n- E23 có metadata learning rate không thống nhất; E20 có thành phần “phạt lặp” dương. Cần công bố các điểm này khi sử dụng số liệu.\n",
         "## 2. Thiết kế hệ thống và cấu hình thực nghiệm\n",
         "### 2.1. Bài toán và biểu diễn\n",
         "Hai tác nhân Trắng/Đen chơi và cập nhật mô hình trong quá trình huấn luyện. Nước đi được lọc bằng tập nước hợp lệ. Theo mã nguồn hiện tại, trạng thái là tensor **16 × 8 × 8**: 12 mặt phẳng quân cờ theo góc nhìn tác nhân và 4 mặt phẳng quyền nhập thành. Không gian hành động có **4.096** cặp ô đi–ô đến; phần sinh nước đi chỉ giữ phong cấp Hậu. Vì vậy mô tả thí nghiệm nên ghi rõ giới hạn phong cấp này.\n",
         "Mạng hiện tại dùng 3 lớp tích chập 64 kênh, lớp fully connected 512, đầu policy/Q và đầu value; phiên mới bổ sung đầu dự đoán nước đối phương và đầu phân loại giai đoạn. Metadata ghi hai quy mô: **4.282.561** tham số/mô hình ở 25 phiên và **6.385.348** ở E25/E27. Không gán kiến trúc hiện tại nguyên vẹn cho toàn bộ phiên lịch sử.\n",
         "### 2.2. Môi trường chạy được ghi nhận\n",
         table(["Nhóm", "Số phiên", "Thiết bị / phần mềm được ghi trong JSON"], [
             ["GPU", 24, "NVIDIA GeForce RTX 3050 Laptop GPU; CUDA; Python 3.10.18; PyTorch 2.5.1+cu121"],
             ["CPU A2C/PPO", 2, "E24/E26; Python 3.13.5; PyTorch 2.15.0+cpu"],
             ["CPU DQN", 1, "E27; Python 3.12.10; PyTorch 2.15.0+cpu"],
         ]),
         "Đây là thông tin lấy nguyên từ log, không phải môi trường Python đang dùng để tổng hợp. Chưa có tên CPU, RAM tổng, seed huấn luyện hoặc cấu hình số luồng đầy đủ để tái lập hiệu năng. Toàn bộ phiên ghi `parallel_games=128`, `episodes_per_update=4` và `gamma=0,99`.\n",
         "### 2.3. Các nhóm siêu tham số chung\n",
         table(["Nhóm", "Phiên", "LR cấu hình", "Shaping", "Reward hòa", "Max ply", "Tham số/mô hình"], [
             ["Gốc", "E01–E21, E24, E26", "0,0003", "0,05", "−0,1", 160, "4.282.561"],
             ["Phạt hòa mạnh", "E22", "0,0003", "0,05", "−5", 200, "4.282.561"],
             ["Kéo dài ván", "E23", "0,003*", "0,05", "−10", 600, "4.282.561"],
             ["Mạng mở rộng", "E25, E27", "0,0003", "0,01", "0", 600, "6.385.348"],
         ]),
         "*E23: `models.White.learning_rate` và `models.Black.learning_rate` đều là **0,0003**. Giá trị 0,003 chỉ nằm trong cấu hình chung. Không viết rằng LR thực tế đã tăng 10 lần. Theo mã hiện tại, optimizer có thể được khôi phục từ checkpoint; đây là một khả năng giải thích, chưa phải nguyên nhân lịch sử đã được xác nhận. `max_plies` là lượt của một bên (ply), nên 160 ply tương đương tối đa khoảng 80 lượt đôi từ bàn khởi đầu chuẩn.\n",
         "### 2.4. Siêu tham số theo thuật toán\n",
         table(["Thuật toán", "Cơ chế được cài đặt", "Giá trị có trong metadata"], [
             ["A2C", "Actor–Critic, return chiết khấu, chuẩn hóa advantage, entropy bonus", "gamma=0,99; entropy_coef=0,01; episodes_per_update=4"],
             ["PPO", "Clipped surrogate và GAE; nhiều vòng cập nhật theo minibatch", "lambda=0,95; clip=0,2; epochs=4; minibatch=256; entropy_coef=0,01"],
             ["DQN", "Double DQN, replay buffer, target network, epsilon-greedy", "buffer=20.000; batch=64; warmup=1.000; target_sync=500; train_every=8; epsilon 1→0,05 / 20.000 bước"],
             ["AlphaZero", "Policy/value + MCTS và dữ liệu visit count", "25 simulations tự đấu; 50 khi chơi; c_puct=1,5; alpha=0,3; noise_frac=0,25; temp_plies=30; buffer=20.000; batch=128; train_steps=20"],
         ]),
         "Siêu tham số AlphaZero ở trên là **giá trị thực nghiệm đã lưu**. File cấu hình hiện tại đã đổi thành 128/256 simulations và buffer 40.000; không dùng các giá trị hiện tại để mô tả các lần chạy cũ. Sáu phiên E01–E06 không có trường `algorithms`/`algorithm_hyperparameters`; gọi là **Baseline Actor–Critic** theo mã lịch sử Git, và giữ dấu * để phân biệt với A2C có nhãn xác nhận.\n",
         "### 2.5. Reward và dữ liệu khởi tạo\n",
         "Theo mã hiện tại, thưởng kết thúc gồm thắng +1, thua −1, hòa theo cấu hình, cộng phạt lặp; shaping dùng biến thiên chênh lệch giá trị quân nhân hệ số. Mỗi ván tạo tối đa hai episode của hai màu. AlphaZero hiện tại dùng mục tiêu value từ tổng reward kết thúc đã chặn về [−1,1], còn shaping vẫn được ghi để theo dõi. Đây là giải thích mã hiện tại, không phải xác nhận mọi bản lịch sử có cùng công thức.\n",
         "E25/E27 có dữ liệu curriculum: khoảng 50% ván khởi tạo nhóm opening, 20% middlegame và 30% endgame. Mã hiện tại khởi tạo opening sau 2–8 ply ngẫu nhiên, middlegame sau 10–24 ply, endgame từ mẫu thế cờ; giới hạn endgame riêng là 100 ply. Hai nhóm khác không xuất trường curriculum nên không suy diễn chúng đã dùng cùng bộ khởi tạo. Hệ số phạt lặp lịch sử không nằm trong `hyperparameters` đã lưu.\n",
         "## 3. Cách đọc chỉ số\n",
         table(["Chỉ số", "Ý nghĩa và phạm vi"], [
             ["Tỷ lệ toàn phiên", "Tự tính từ training.wins / training.games_played; dùng để lập bảng kết quả chính."],
             ["recent_rates", "Cửa sổ tối đa 100 ván cuối; E03 chỉ 28, E19 chỉ 50. Khác mẫu với tỷ lệ toàn phiên."],
             ["avg_game_length_plies", "Trung bình của cửa sổ ván gần nhất; không phải trung bình toàn phiên."],
             ["game_quality", "Trung bình đặc trưng trên tối đa 100 ván gần nhất; repetition_rate là tỷ phần vị trí lặp trong mỗi ván, không phải tỷ lệ ván hòa do lặp."],
             ["draw_reasons", "Số ván hòa tích lũy theo nguyên nhân trong phiên. Chia cho tổng ván và chia cho số ván hòa cho hai tỷ lệ khác nhau."],
             ["mean_episode_reward", "Trung bình tối đa 200 episode gần nhất, gộp hai màu; không tương đương reward riêng bên thắng."],
             ["optimization", "Theo mã hiện tại, trung bình tối đa 50 lần cập nhật mỗi màu. last_loss là lần cập nhật cuối, có thể khác total_loss trung bình."],
             ["games_per_second / plies_per_second", "Chỉ số log lấy cửa sổ khoảng 30 giây. Tốc độ trung bình trong báo cáo được tính riêng bằng games_played / elapsed_seconds."],
             ["policy_entropy", "Độ phân tán hành động; DQN ghi entropy của softmax(Q/T), không hoàn toàn cùng ý nghĩa với policy A2C/PPO."],
             ["value_explained_variance", "Mức giải thích biến thiên target trong batch; ở DQN là Q-target. Giá trị cao chưa đủ kết luận chơi mạnh; giá trị âm cho thấy dự đoán kém theo chỉ số này."],
         ]),
         "Loss có thể âm do thành phần actor/entropy. Không so trực tiếp trị tuyệt đối loss A2C, PPO, DQN và AlphaZero vì chúng tối ưu các mục tiêu khác nhau. Số `training_updates` cũng không cùng đơn vị công việc giữa các thuật toán.\n",
         "## 4. Toàn bộ 27 phiên thực nghiệm\n",
         "Thứ tự cặp thuật toán luôn là **Trắng / Đen**. Tỷ lệ T/Đ/H trong bảng được tính từ số đếm toàn phiên. Thời gian định dạng giờ:phút:giây. Mã E24/E26 lấy từ `src/checkpoints`; các mã còn lại từ `checkpoints`.\n"]
    rows = []
    for d in records:
        tr = d["training"]
        rates = d["derived"]["overall_rates"]
        rows.append([d["report_id"], link(ROOT / d["source_path"], d["experiment"].removeprefix("exp_")), pair(d),
                     duration(d["elapsed_seconds"]), number(tr["games_played"]),
                     " / ".join(percent(rates[k]) for k in ("White", "Black", "Draw")),
                     percent(tr["recent_rates"]["Draw"], 1), d["system"]["device"].upper()])
    p.append(table(["Mã", "Phiên / nguồn JSON", "T / Đ", "Thời gian", "Ván", "T thắng / Đ thắng / Hòa", "Hòa gần nhất", "Thiết bị"], rows))
    p.append("\n![Kết quả 27 phiên](figures/01_ket_qua_27_phien.png)\n\nBiểu đồ 1. Phân bố kết quả toàn phiên; cấu hình khác nhau nên không dùng biểu đồ này để xếp hạng sức chơi. Bản SVG cùng tên dùng để chèn vào báo cáo mà không giảm chất lượng.\n")
    p.append("## 5. Phân tích theo nhóm thực nghiệm\n\n### 5.1. Baseline và A2C\n\nChuỗi baseline E01 → E02 → E03 → E04 → E05 → E06 là huấn luyện nối tiếp, không phải sáu lần khởi tạo độc lập. E05 có nhiều ván nhất nhưng Trắng thắng nhiều hơn Đen và loss/entropy có dấu hiệu mất ổn định. Sáu file cũ chưa ghi nhãn thuật toán, nên chỉ nên dùng làm phần phát triển mô hình nền.\n\nE07–E09 là ba phiên A2C khởi tạo mới, cùng cấu hình và đều bị giới hạn lượt chi phối: lần lượt 572/575, 2.842/2.866 và 586/590 ván hòa do max ply. E10 dài hơn và hòa giảm còn 86,88%, nhưng khác thời lượng và quá trình học nên chưa chứng minh hội tụ.\n\nE24 là phiên A2C CPU đáng đưa vào báo cáo chính: 45.204 ván, 3.792 Trắng thắng, 2.930 Đen thắng và 38.482 hòa. Cuối phiên, entropy rất thấp, explained variance âm và toàn bộ cửa sổ cập nhật bị clipping. Có cơ sở nêu rằng chính sách trở nên tập trung, còn chất lượng critic và khả năng kết thúc ván vẫn hạn chế.\n\n### 5.2. A2C đối đầu PPO, DQN và AlphaZero\n\nE11: A2C Trắng thắng 316, PPO Đen thắng 129; E12 tiếp tục cả hai mô hình nhưng PPO thắng 93, A2C thắng 72. A2C ở E11 đã được huấn luyện trước còn PPO không có checkpoint nạp, do đó ưu thế ở E11 không thể quy hoàn toàn cho thuật toán.\n\nDQN Đen thắng nhiều hơn A2C Trắng trong E13 (89 so với 16) và E15 (952 so với 67). AlphaZero Đen thắng nhiều hơn A2C trong E14 (32 so với 6), nhưng chỉ có 145 ván. Các kết quả này là bằng chứng thăm dò trong khi hai bên vẫn đang học; cần đánh giá mô hình cố định và đổi màu trước khi xếp hạng.\n\n### 5.3. DQN và AlphaZero tự đấu\n\nE16 có 24.727 ván DQN tự đấu và 89,48% hòa. Hai màu bắt đầu từ cùng checkpoint DQN của E15 nên tỷ lệ thắng gần nhau không phải phép đo sức chơi độc lập.\n\nE17: AlphaZero Trắng thắng 73 so với 26 của DQN; chỉ 421 ván. E18 AlphaZero tự đấu 34.219 ván trong khoảng 7,42 giờ, hòa 31.260 ván (91,35%). Có tới 30.193 ván hòa do lặp, bằng 88,23% toàn bộ ván và 96,59% số ván hòa. Độ dài cửa sổ cuối chỉ 58,3 ply; ván ngắn đi chủ yếu đi kèm hòa do lặp, chưa cho thấy kết thúc bằng chiếu hết hiệu quả hơn.\n\nE19/E20 là chuỗi huấn luyện tiếp của E18, tổng mẫu nhỏ; tất cả ván hòa trong từng phiên đều do lặp. E20 có reward lặp dương nên cần kiểm tra trước khi dùng để bàn về hiệu quả reward.\n\nE27 là DQN CPU từ đầu, dùng mạng mở rộng và curriculum: 11.077 ván, hòa 91,80%; 70,19% toàn bộ ván kết thúc hòa do lặp. Explained variance cuối khoảng 0,94 nhưng cửa sổ cuối hòa 96%, cho thấy chỉ số khớp target tốt chưa đủ phản ánh khả năng thắng.\n\n### 5.4. Chuỗi thay đổi PPO–DQN\n")
    ablation = []
    for key in ("20261114_081112", "20261114_084332", "20261114_084944", "20261114_164954"):
        d = get(key)
        h = d["hyperparameters"]
        ablation.append([d["report_id"], pair(d), h["max_plies"], number(h["draw_reward"], 1), number(h["reward_shaping"], 2),
                         percent(d["derived"]["overall_rates"]["Draw"]),
                         number(d["metrics"]["draw_reasons"]["draw_by_max_plies"]), number(d["training"]["avg_game_length_plies"], 1)])
    p.append(table(["Mã", "T / Đ", "Max ply", "Reward hòa", "Shaping", "Hòa toàn phiên", "Hòa do max ply", "Ply gần nhất"], ablation))
    p.append("E21 DQN Đen thắng 497 so với 139 của PPO; E22/E23 đổi DQN sang Trắng và vẫn thắng nhiều hơn PPO. Sang E25, PPO Trắng thắng 4.003 so với 950 của DQN Đen.\n\nTăng giới hạn từ 160 lên 600 đi kèm việc giảm mạnh số ván bị cắt ở E23, nhưng hòa tổng vẫn 80,76%. Đồng thời đã đổi reward, checkpoint, màu và thời lượng, nên đây là chuỗi điều chỉnh hệ thống; chưa phải thí nghiệm loại bỏ từng thành phần có kiểm soát. E25 còn thay quy mô mạng, shaping, reward hòa và phân phối khởi tạo. Không viết rằng một thay đổi riêng lẻ đã gây ra đảo chiều PPO/DQN.\n\n### 5.5. A2C và PPO trên CPU\n")
    cpu_rows = []
    for key in ("20261114_145819", "20261114_190522"):
        d = get(key)
        m = d["metrics"]
        op = m["optimization"]
        cpu_rows.append([d["report_id"], pair(d), number(d["training"]["games_played"]), duration(d["elapsed_seconds"]),
                         number(d["derived"]["average_games_per_second"], 3), percent(d["derived"]["overall_rates"]["Draw"]),
                         " / ".join(number(op[k]["policy_entropy"], 4) for k in ("White", "Black")),
                         " / ".join(number(op[k]["value_explained_variance"], 4) for k in ("White", "Black"))])
    p.append(table(["Mã", "Thuật toán", "Ván", "Thời gian", "Ván/giây TB", "Hòa", "Entropy T/Đ", "Explained variance T/Đ"], cpu_rows))
    p.append("Hai phiên cùng cấu hình gốc và cùng số tham số, nhưng khác thời lượng/ngân sách dữ liệu, không phải phép so sánh hiệu năng đã kiểm soát. Tỷ lệ hòa chênh khoảng 0,29 điểm phần trăm, chưa đủ để nói A2C chơi mạnh hơn. PPO giữ entropy cuối cao hơn và biến động gradient trong hình ít cực đoan hơn A2C; cả hai vẫn có hạn chế ở dự đoán value và kết thúc ván. Tốc độ ván/giây là mô tả từng phiên, không phải kết luận CPU tốt hơn GPU hay thuật toán nhanh hơn trong mọi điều kiện.\n\n## 6. Nguyên nhân hòa, curriculum và học chiến lược\n\n### 6.1. Cơ cấu nguyên nhân hòa ở các phiên tiêu biểu\n\nMỗi ô dưới đây là **số ván (tỷ lệ trên toàn bộ ván của phiên)**, không phải tỷ lệ trong riêng tập ván hòa.\n")
    drawrows = []
    for key in ("20261112_134110", "20261112_154125", "20261114_145819", "20261114_190522", "20261114_164954", "20261114_235757"):
        d = get(key)
        n = d["training"]["games_played"]
        r = d["metrics"]["draw_reasons"]
        drawrows.append([d["report_id"]] + [f"{number(r['draw_by_' + k])} ({percent(r['draw_by_' + k] / n)})"
                                             for k in ("repetition", "max_plies", "insufficient_material", "stalemate", "fifty_moves")])
    p.append(table(["Mã", "Lặp thế", "Giới hạn ply", "Thiếu lực lượng", "Stalemate", "50 nước"], drawrows))
    p.append("\n![Cơ cấu nguyên nhân hòa](figures/02_nguyen_nhan_hoa.png)\n\nBiểu đồ 2. Cần phân biệt việc hết ván sớm do lặp với khả năng tạo kết quả thắng/thua. Giới hạn ply lớn hơn làm giảm một nguyên nhân hòa nhưng không tự giải quyết các nguyên nhân khác.\n\n### 6.2. Curriculum chỉ có ở E25/E27\n")
    currrows = []
    for key in ("20261114_164954", "20261114_235757"):
        d = get(key)
        for kind, v in d["metrics"]["curriculum"].items():
            currrows.append([d["report_id"], kind, number(v["games"]), number(v["White"]), number(v["Black"]),
                             number(v["Draw"]), percent(v["Draw"] / v["games"]), number(v["avg_plies"], 1)])
    p.append(table(["Mã", "Nhóm khởi tạo", "Ván", "Trắng thắng", "Đen thắng", "Hòa", "Tỷ lệ hòa", "Ply TB nhóm"], currrows))
    p.append("Tàn cuộc có tỷ lệ hòa 98,43% ở E25 và 98,63% ở E27, cao hơn nhóm opening/middlegame. Điều này cho thấy kết quả nhóm tàn cuộc còn hạn chế. Các mẫu tàn cuộc không phải tất cả đều bảo đảm thắng, nên không diễn giải trực tiếp tỷ lệ thắng thấp thành tỷ lệ chuyển hóa tàn cuộc có lợi. Muốn đo chuyển hóa cần dùng bài kiểm tra KQvK/KRvK riêng.\n\n### 6.3. Đầu dự đoán đối phương và nhận diện giai đoạn\n")
    strategyrows = []
    for key in ("20261114_164954", "20261114_235757"):
        d = get(key)
        for side, v in d["metrics"]["strategy"].items():
            strategyrows.append([d["report_id"], side, algo(d, side), percent(v["opp_top1"]), percent(v["opp_top3"]), percent(v["phase_accuracy"])])
    p.append(table(["Mã", "Màu", "Thuật toán", "Dự đoán Top-1", "Top-3", "Nhận diện giai đoạn"], strategyrows))
    p.append("Các đầu phụ đã có dữ liệu dự đoán; accuracy nhận diện giai đoạn cao hơn accuracy dự đoán nước đối phương. Nhãn giai đoạn trong mã là heuristic từ số nước/lực lượng, nên không gọi accuracy này là độ chính xác chiến thuật hoặc khả năng chơi cờ 95%. Chưa có phép thử tắt/bật đầu phụ hay lookahead để xác định tác động lên sức chơi.\n\n## 7. Nhận xét toàn bộ 15 ảnh huấn luyện\n\nCác nhận xét xu hướng dưới đây dựa trên xem ảnh. Số liệu chính xác lấy từ JSON. Không có `training_log.json` trong bản dữ liệu hiện tại để tính độ dốc, cực trị hoặc thời điểm hội tụ chính xác. Ảnh có thể được xuất trước lần lưu metrics cuối.\n")
    plotrows = []
    for d in records:
        if d["plot_path"]:
            key = d["experiment"].removeprefix("exp_")
            plotrows.append([d["report_id"], link(ROOT / d["plot_path"], Path(d["plot_path"]).name), PLOT_NOTES[key]])
    p.append(table(["Mã", "Ảnh nguồn", "Nhận xét để viết báo cáo"], plotrows))
    p.append("## 8. Các giới hạn và điểm dữ liệu cần công bố\n\n1. **Đây là log huấn luyện.** Hai bên vẫn cập nhật mô hình, chưa phải giải đấu đánh giá checkpoint cố định. Tỷ lệ thắng trong huấn luyện không phải win rate trước đối thủ chuẩn.\n2. **Lịch sử nạp mô hình khác nhau.** Có từ đầu, nạp một bên, nạp cả hai bên và đổi màu. Không gộp toàn bộ ván để tạo bảng xếp hạng thuật toán.\n3. **Không đồng nhất cấu hình.** Khác reward hòa, shaping, max ply, quy mô mạng, thiết bị, thời gian và curriculum.\n4. **Thiếu kiểm soát ngẫu nhiên.** Không lưu seed huấn luyện hoặc thiết kế đánh giá lặp nhiều seed có kiểm soát. E07–E09 cùng cấu hình nhưng đều ngắn và không ghi seed; chưa có khoảng tin cậy cho đánh giá độc lập.\n5. **Thiếu file mô hình và log thời gian thô.** Các file `.pt` được nhắc trong metadata nhưng không có trong hai thư mục thực nghiệm; `training_log.json` cũng vắng mặt. `.gitignore` có `*.pt` và `training_log.json`. Chưa thể nạp lại model để kiểm thử hoặc tái tính đường cong từ repository này.\n6. **E23 không nhất quán LR.** Cấu hình 0,003; optimizer trong thông tin cả hai mô hình 0,0003.\n7. **E20 có “phạt” mang dấu dương.** `repetition_penalty_mean=+1,66`, `mean_episode_reward=+1,58`, trong khi tất cả 268 ván hòa do lặp. Đây là điểm cần kiểm tra reward lịch sử, chưa xác định là chủ ý hay lỗi.\n8. **Hệ số phạt lặp chưa được lưu trong hyperparameters.** Không thể khẳng định các phiên đã dùng cùng hệ số chỉ từ bảng cấu hình.\n9. **Không suy diễn từ threefold=0.** Các snapshot chi tiết đều ghi `threefold_repetition_rate=0` dù có hòa do lặp. Theo mã hiện tại, `claim_draw=True` có thể nhận hòa khi một nước hợp lệ tiếp theo sẽ tạo lần lặp thứ ba; bộ đặc trưng chỉ đếm vị trí thực sự đã xuất hiện trong move stack. Hai chỉ số không nhất thiết mâu thuẫn, nhưng cần ghi rõ định nghĩa.\n10. **Thiếu số liệu là thiếu số liệu.** CPU/GPU utilization hoặc memory `null` không phải 0. Process CPU có thể vượt 100% theo cách cộng nhiều lõi.\n11. **Ảnh không luôn trùng snapshot cuối.** E22/E23 có phạm vi thời gian ảnh ngắn hơn `elapsed_seconds` của JSON.\n12. **Có mã kiểm tra kỹ năng nhưng chưa có kết quả được lưu.** `skill_eval.py` có chiếu hết một nước, tránh chiếu hết, ăn quân miễn phí và chuyển hóa KQvK/KRvK. Không tìm thấy artifact kết quả để báo cáo tỷ lệ thành công hay Elo.\n\n## 9. Đoạn văn có thể dùng trong báo cáo\n\n### 9.1. Mô tả tổ chức thực nghiệm\n\n> Hệ thống cờ vua học tăng cường được thực nghiệm với các thuật toán A2C, PPO, DQN và AlphaZero, bên cạnh mô hình Actor–Critic nền tảng. Dữ liệu thu thập gồm 27 phiên huấn luyện, ghi nhận tổng cộng 556.228 ván. Các phiên sử dụng 128 môi trường bàn cờ đồng thời và hệ số chiết khấu 0,99. Phần lớn phiên được chạy trên GPU NVIDIA GeForce RTX 3050 Laptop GPU; ba phiên chạy trên CPU. Các cấu hình được thay đổi theo giai đoạn phát triển, gồm giới hạn lượt, cơ chế reward và quy mô mạng. Vì vậy, kết quả được phân tích theo từng nhóm cấu hình và lịch sử khởi tạo thay vì xem toàn bộ phiên là các phép so sánh độc lập.\n\n### 9.2. Nhận xét kết quả\n\n> Các phiên huấn luyện lớn vẫn có tỷ lệ hòa cao. Ở phiên A2C CPU, hệ thống ghi nhận 45.204 ván với tỷ lệ hòa 85,13%; phiên PPO CPU ghi nhận 75.686 ván với tỷ lệ hòa 85,42%. PPO duy trì entropy cuối cao hơn trong hai phiên này, trong khi A2C có entropy thấp và explained variance âm ở snapshot cuối. AlphaZero tự đấu 34.219 ván đạt tỷ lệ hòa 91,35%, trong đó hòa do lặp chiếm 96,59% số ván hòa. Kết quả cho thấy việc giảm loss hoặc kéo dài huấn luyện chưa bảo đảm tác nhân học được cách kết thúc ván hiệu quả.\n\n### 9.3. So sánh thuật toán\n\n> DQN có số ván thắng cao hơn A2C và PPO trong một số cấu hình ban đầu; tuy nhiên, PPO lại thắng nhiều hơn DQN ở phiên sử dụng mạng mở rộng và dữ liệu curriculum. Do các phiên khác nhau về checkpoint khởi tạo, màu quân, reward và thời lượng huấn luyện, kết quả này chỉ phản ánh ưu thế trong từng cấu hình cụ thể. Chưa đủ cơ sở xác lập thứ hạng sức chơi tổng quát giữa bốn thuật toán.\n\n### 9.4. Hạn chế và hướng phát triển\n\n> Hạn chế chính của hệ thống là hòa do lặp và hòa do giới hạn lượt còn phổ biến, đồng thời một số chỉ số tối ưu hóa chưa ổn định. Các bước đánh giá tiếp theo cần cố định checkpoint, đổi màu đối xứng, sử dụng cùng bộ vị trí khởi đầu và ngân sách tính toán, đồng thời lặp lại với nhiều seed. Cần bổ sung đối thủ chuẩn và bài kiểm tra kỹ năng để phân biệt khả năng khớp mục tiêu huấn luyện với năng lực chơi cờ thực tế. Dữ liệu cũng cần lưu đầy đủ hệ số reward, trạng thái optimizer và log thời gian nhằm tăng khả năng tái lập.\n\n## 10. Dàn ý đề xuất cho phần thực nghiệm\n\n1. Mục tiêu: kiểm tra khả năng học tự đấu, độ ổn định cập nhật và hiện tượng hòa/lặp.\n2. Môi trường và thiết lập: encoding, mạng, reward, phần cứng, 128 bàn đồng thời.\n3. Nhóm thử nghiệm: baseline → A2C → đối đầu thuật toán → AlphaZero → điều chỉnh PPO/DQN → curriculum/đầu phụ.\n4. Kết quả định lượng: bảng 27 phiên, chọn E16/E18/E24/E25/E26/E27 để phân tích sâu.\n5. Biểu đồ: dùng hai hình tổng hợp cùng ảnh E18, E24, E26; ảnh E20 để minh họa điểm reward cần kiểm tra.\n6. Thảo luận: hòa do giới hạn và lặp, entropy, value, tác động của điều kiện khởi tạo.\n7. Hạn chế và thực nghiệm tiếp theo: mô hình cố định, đổi màu, nhiều seed, kiểm tra kỹ năng và đối thủ chuẩn.\n\nPhần thực nghiệm tiếp theo được đề xuất ở đây **chưa được thực hiện**. Không ghi các đề xuất thành kết quả đã có.\n\n## Phụ lục A. Hồ sơ chi tiết từng phiên\n")
    for d in records:
        tr, h, m = d["training"], d["hyperparameters"], d.get("metrics", {})
        key = d["experiment"].removeprefix("exp_")
        p.append(f"### {d['report_id']} — {d['experiment']}\n\n")
        p.append(f"- Nguồn: {link(ROOT / d['source_path'])}; cặp **{pair(d)}**; thiết bị **{d['system']['device'].upper()}**.\n")
        p.append(f"- Bắt đầu `{d['started_at']}`, lưu `{d['saved_at']}`; thời gian {duration(d['elapsed_seconds'])}; số lần lưu {d['save_count']}. Timestamp được giữ nguyên từ log, không có múi giờ trong chuỗi.\n")
        p.append(f"- Số ván: {number(tr['games_played'])}; Trắng thắng / Đen thắng / hòa: **" + " / ".join(number(tr['wins'][k]) for k in ("White", "Black", "Draw")) + "**.\n")
        p.append(f"- Cửa sổ cuối {tr['recent_window']} ván, T/Đ/H: " + " / ".join(percent(tr['recent_rates'][k], 1) for k in ("White", "Black", "Draw")) + f"; độ dài trung bình cửa sổ {number(tr['avg_game_length_plies'], 1)} ply.\n")
        p.append(f"- Cấu hình: LR chung {h['learning_rate']}; LR mô hình T/Đ {d['models']['White']['learning_rate']}/{d['models']['Black']['learning_rate']}; gamma {h['gamma']}; shaping {h['reward_shaping']}; hòa {h['draw_reward']}; max ply {h['max_plies']}. Tốc độ trung bình theo đồng hồ {number(d['derived']['average_games_per_second'], 3)} ván/giây.\n")
        resume = d["resumed_from"]
        if not resume:
            p.append("- Khởi tạo: không có checkpoint nạp trong `resumed_from` (ghi là từ đầu theo metadata).\n")
        elif isinstance(resume, str):
            p.append(f"- Huấn luyện tiếp từ `{resume}`.\n")
        else:
            parts = []
            for side, path in resume.items():
                normalized = str(path).replace("\\", "/")
                match = re.search(r"exp_\d{8}_\d{6}/[^/]+", normalized)
                parts.append(f"{side}: `{match.group(0) if match else normalized}`")
            p.append("- Checkpoint nạp: " + "; ".join(parts) + ". Các đường dẫn máy gốc không bảo đảm tồn tại trong repository.\n")
        p.append("- Loss lần cuối T/Đ: " + " / ".join(number(d['models'][k]['last_loss'], 4) for k in ("White", "Black")) + ".\n")
        if m:
            g = m["game_quality"]
            p.append(f"- Chất lượng cửa sổ: vị trí lặp {percent(g['repetition_rate'])}; đảo nước {percent(g['reversal_move_rate'])}; trì trệ {percent(g['stagnation_rate'])}; tỷ lệ vị trí duy nhất {percent(g['unique_position_ratio'])}. Reward episode trung bình {number(m['rewards']['mean_episode_reward'], 4)}.\n")
            reason_names = {"repetition": "lặp", "fifty_moves": "50 nước", "stalemate": "stalemate", "insufficient_material": "thiếu lực lượng", "max_plies": "max ply", "other": "khác"}
            p.append("- Nguyên nhân hòa (số ván): " + "; ".join(f"{reason_names[k]} {number(m['draw_reasons']['draw_by_' + k])}" for k in reason_names) + ".\n")
        else:
            p.append("- Chỉ số chi tiết: file chưa có nhóm `metrics`.\n")
        p.append(f"- Nhận xét: {NOTES[key]}\n")
        p.append("- Ảnh: " + (link(ROOT / d["plot_path"]) if d["plot_path"] else "không có ảnh huấn luyện trong thư mục phiên") + ".\n\n")
    p.append("## Phụ lục B. Nguồn mã và cách cập nhật bản tổng hợp\n\n")
    for rel, desc in [
        ("README.md", "mục tiêu và tính năng hệ thống"),
        ("src/encoding.py", "mã hóa trạng thái, hành động và giá trị quân"),
        ("src/model.py", "kiến trúc mạng hiện tại"),
        ("src/algorithms/algo_a2c.py", "cập nhật A2C"),
        ("src/algorithms/algo_ppo.py", "cập nhật PPO"),
        ("src/algorithms/algo_dqn.py", "cập nhật DQN"),
        ("src/algorithms/algo_alphazero.py", "MCTS và mục tiêu AlphaZero"),
        ("src/metrics.py", "định nghĩa chỉ số/cửa sổ và nguyên nhân hòa"),
        ("src/checkpoints.py", "cách lưu metadata"),
        ("src/concurrent_mode/concurrent_game.py", "thu thập dữ liệu song song và reset thống kê mỗi phiên"),
        ("src/plot_logging/training_log.py", "cách ghi chuỗi thời gian"),
        ("src/curriculum.py", "khởi tạo các nhóm thế cờ"),
        ("src/strategy.py", "đầu phụ và cách tính accuracy"),
        ("src/strategy_config.py", "cấu hình strategy/curriculum hiện tại"),
        ("src/skill_eval.py", "bài kiểm tra kỹ năng hiện có trong mã"),
    ]:
        p.append(f"- {link(ROOT / rel, rel)}: {desc}.\n")
    p.append("\nMã lịch sử Git `391632b` (`src/agent.py`) và `63fafb9` (`src/concurrent_mode/concurrent_agent.py`) được dùng để nhận diện nhóm baseline Actor–Critic; nhãn này vẫn được đánh dấu suy luận từ mã lịch sử, không bổ sung giả vào JSON nguồn.\n\n")
    p.append("Dữ liệu gộp kèm theo: `du_lieu_thuc_nghiem.json`, giữ nguyên các trường gốc và bổ sung đường dẫn nguồn, SHA-256 cùng các tỷ lệ tính lại. Script `tao_tong_hop.py` tạo bảng và hai hình từ dữ liệu gốc; các phần nhận xét lịch sử được biên soạn cho bộ 27 phiên hiện tại, cần rà soát lại nếu thêm phiên mới. Các kiểm tra tự động đối chiếu tổng thắng/thua/hòa, tổng nguyên nhân hòa và số ván giữa hai nhóm thống kê.\n")
    return "\n".join(p)


if __name__ == "__main__":
    make_figures()
    (OUT / "du_lieu_thuc_nghiem.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    report = build_report()
    (OUT / "TONG_HOP_THUC_NGHIEM.md").write_text(report, encoding="utf-8")
    print(f"Created report for {len(records)} experiments, {sum(bool(d['plot_path']) for d in records)} original plots.")
    print(f"Recorded games: {sum(d['training']['games_played'] for d in records):,}.")
