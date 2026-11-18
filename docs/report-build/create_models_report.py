"""Build the tutorial manuscript; code excerpts come from the tested modules."""
import ast
import json
import textwrap
from pathlib import Path

workspace = Path(__file__).resolve().parents[2]
out = Path(__file__).resolve().parent / 'models-report'
out.mkdir(exist_ok=True)


def excerpt(file, name):
    source = (workspace / file).read_text(encoding='utf-8')
    tree = ast.parse(source)
    node = next(n for n in ast.walk(tree)
                if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
    return textwrap.dedent('\n'.join(source.splitlines()[start - 1:node.end_lineno]))


def code(caption, text):
    return {'caption': caption, 'text': text.strip('\n')}


def table(caption, headers, widths, rows):
    return {'caption': caption, 'headers': headers, 'widths': widths, 'rows': rows}


def section(title, paragraphs, **kwargs):
    return dict(title=title, paragraphs=paragraphs, **kwargs)


report = {
    'cover_title': 'BÁO CÁO TRIỂN KHAI MÔ HÌNH',
    'cover_topic1': 'XÂY DỰNG VÀ HUẤN LUYỆN AI',
    'cover_topic2': 'CHO GAME CỜ VUA',
    'document_title': 'Triển khai và huấn luyện các mô hình AI Cờ Vua',
    'document_subject': 'Mã hóa bàn cờ, MLP, CNN, huấn luyện có giám sát, TD(0) và tìm kiếm Alpha-beta',
    'document_description': 'Báo cáo mã nguồn và quy trình thực nghiệm AI cho dự án Cờ Vua',
    'running_header': 'TRIỂN KHAI MÔ HÌNH AI CỜ VUA',
    'personal_pages': False,
    'keep_tables': True,
    'chapters': [],
    'references': [
        '[1] PyTorch, Get Started Locally. https://pytorch.org/get-started/locally/. Truy cập ngày 02/11/2026.',
        '[2] PyTorch, Saving and Loading Models. https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html. Truy cập ngày 02/11/2026.',
        '[3] python-chess, UCI/XBoard engine communication. https://python-chess.readthedocs.io/en/latest/engine.html. Truy cập ngày 02/11/2026.',
        '[4] R. S. Sutton, Learning to predict by the methods of temporal differences, Machine Learning, tập 3, tr. 9–44, 1988. https://doi.org/10.1007/BF00115009.',
        '[5] D. E. Knuth và R. W. Moore, An analysis of alpha-beta pruning, Artificial Intelligence, tập 6, tr. 293–326, 1975. https://doi.org/10.1016/0004-3702(75)90019-3.',
        '[6] Mã nguồn và nhật ký kiểm tra của dự án Cờ Vua, thư mục model/learning, tests/test_learning.py và test-results/ai-models-smoke, bản chạy ngày 02/11/2026.'
    ]
}

report['chapters'].append({'title': 'PHẠM VI VÀ CẤU TRÚC BỘ MÃ', 'sections': [
    section('Các thành phần đã triển khai', [
        'Bộ mã bổ sung khả năng học từ dữ liệu vào dự án Cờ Vua đang sử dụng FastAPI và React. Hai kiến trúc được cài đặt là MLP và CNN, cùng dự đoán chính sách lựa chọn nước đi và giá trị thế cờ. Quy trình gồm sinh nhãn từ tác tử, chia dữ liệu, huấn luyện, lưu trọng số và sử dụng mô hình trong ván đấu.',
        'Ngoài học có giám sát, bộ mã có TD(0) để tinh chỉnh đầu giá trị qua tự chơi. Đây là các phương pháp được cài đặt và chạy kiểm tra trong tài liệu. MCTS, PPO, DQN và vòng lặp AlphaZero chưa thuộc bộ mã này; Alpha-beta là thuật toán tìm kiếm sử dụng đầu giá trị đã học, không phải một kiến trúc mạng riêng.'
    ], table=table('Bảng 1.1 Kiến trúc và phương thức sử dụng', ['Thành phần', 'Vai trò', 'Đầu ra'], [27,43,30], [
        ['MLP', 'Mô hình cơ sở trên đặc trưng bàn cờ trải phẳng', 'Logits chính sách và value'],
        ['CNN', 'Học đặc trưng không gian của bàn cờ 8 × 8', 'Logits chính sách và value'],
        ['NeuralAgent', 'Chọn nước theo policy hoặc value', 'Nước UCI hợp lệ'],
        ['Alpha-beta + mạng', 'Tìm kiếm với mạng value tại lá', 'Nước UCI và số đo tìm kiếm'],
        ['TD(0)', 'Tinh chỉnh thân mạng và đầu value', 'Checkpoint td-final.pt']
    ])),
    section('Tổ chức mã nguồn', [
        'Các mô đun học máy nằm trong model/learning để tránh đưa PyTorch vào luồng chạy mặc định của game. CAI_DAT_AI.ps1 tạo môi trường .venv-ai riêng; requirements-ai.txt cố định PyTorch 2.10.0 và dùng các phụ thuộc nền của dự án. Các lệnh chạy từ thư mục gốc bằng cú pháp python -m model.learning.ten_mo_dun.',
        'encoding.py xác lập đặc trưng và mã hành động; models.py định nghĩa mạng và checkpoint. data.py tạo mẫu, train.py huấn luyện có giám sát, agents.py chọn nước. selfplay.py thực hiện TD và arena.py chạy đối kháng. predict.py cho phép kiểm tra một quyết định mà không khởi động web server.'
    ]),
    section('Quy ước xuyên suốt chương trình', [
        'Luật cờ và tập nước hợp lệ do python-chess xác định. Mạng không học lại quy tắc đi quân; logits của nước bất hợp lệ bị loại trước khi tính loss hoặc chọn hành động. Trạng thái bàn cờ của ứng dụng vẫn do ChessEnv quản lý.',
        'Value luôn được diễn giải theo phía đến lượt tại trạng thái đầu vào. Trong học có giám sát, nhãn là tanh(cp/600), với cp là đánh giá teacher theo phía đó. Nhãn này là lợi thế chuẩn hóa, không phải xác suất thắng đã hiệu chỉnh. Khi sang trạng thái của đối thủ, việc so sánh value phải đổi dấu.'
    ])
]})

report['chapters'].append({'title': 'MÃ HÓA TRẠNG THÁI VÀ HÀNH ĐỘNG', 'sections': [
    section('Tensor đầu vào hai mươi kênh', [
        'Encoder mới trả tensor float32 có kích thước 20 × 8 × 8. Hàng 0 tương ứng hàng 1 của bàn cờ, cột 0 là cột a; bàn cờ không bị lật khi đổi lượt. Một batch có dạng B × 20 × 8 × 8. Các mặt phẳng quân giữ màu tuyệt đối, còn kênh lượt đi cho biết phía đang ra quyết định.',
        'Thông tin nhập thành, bắt tốt qua đường và bộ đếm hòa được thêm vào để giảm nhập nhằng giữa các thế có cùng bố trí quân. Kênh lặp lại là đặc trưng tóm tắt; lịch sử đầy đủ vẫn được giữ trong board để xử lý luật. Encoder mang phiên bản planes20-v1 và độc lập với encoder 14 kênh có sẵn trong ChessEnv.'
    ], table=table('Bảng 2.1 Ý nghĩa các kênh đầu vào', ['Kênh', 'Nội dung', 'Giá trị'], [14,53,33], [
        ['0–11', 'Sáu loại quân theo hai màu', '0 hoặc 1 tại từng ô'],
        ['12', 'Bên đến lượt', '+1 Trắng; −1 Đen'],
        ['13–16', 'Quyền nhập thành hai phía của mỗi màu', '0 hoặc 1 trên toàn kênh'],
        ['17', 'Ô bắt tốt qua đường khi nước bắt hợp lệ', '1 tại ô tương ứng'],
        ['18', 'Halfmove clock chuẩn hóa', 'min(clock/150, 1)'],
        ['19', 'Số lần lặp lại hiện tại được tóm tắt', '0; 0,5 khi hai lần; 1 khi ba lần']
    ])),
    section('Mã hành động và phong cấp', [
        'Không gian đầu ra gồm 64 × 64 × 5 = 20.480 mã. Hai chỉ số đầu là ô đi và ô đến. Chỉ số cuối lần lượt biểu diễn không phong cấp, mã, tượng, xe và hậu. Các mã không thể xảy ra được giữ trong từ vựng cố định nhưng luôn bị che bởi mặt nạ hợp lệ.',
        'Cách mã hóa này tách được a7a8n, a7a8b, a7a8r và a7a8q. Nếu chỉ dự đoán hai ô, bốn hành động trên sẽ trùng nhau. move_id và id_move thực hiện chuyển đổi hai chiều; encoder không hỗ trợ biến thể có thả quân.'
    ], codeblocks=[code('Mã 2.1 Mã hóa hành động trong encoding.py', excerpt('model/learning/encoding.py', 'move_id')),
                   code('Mã 2.2 Khôi phục nước đi từ chỉ số', excerpt('model/learning/encoding.py', 'id_move'))]),
    section('Mặt nạ và thông tin lịch sử', [
        'Dataset giữ danh sách mã nước hợp lệ cho từng mẫu. Hàm collate tạo mặt nạ Boolean theo batch; cách này tránh lưu một mảng 20.480 phần tử cho từng thế cờ trên đĩa. Mẫu không có nước hợp lệ bị từ chối khi dùng để học chính sách.',
        'FEN không chứa lịch sử lặp lại. Vì vậy mỗi dòng JSONL lưu thêm repetition đã được tính từ board gốc. Dataset dùng giá trị này khi mã hóa lại; suy luận trên ván đang chơi tính từ ngăn xếp nước đi. Ô en passant chỉ được mã hóa khi có nước bắt hợp lệ để đầu vào gốc khớp đầu vào tái dựng từ FEN.'
    ], codeblocks=[code('Mã 2.3 Che logits bất hợp lệ', excerpt('model/learning/models.py', 'masked_logits'))])
]})

report['chapters'].append({'title': 'CÀI ĐẶT CÁC KIẾN TRÚC MẠNG', 'sections': [
    section('Cấu hình và mô hình MLP', [
        'MLP trải phẳng 20 mặt phẳng thành 1.280 phần tử, qua hai lớp Linear có ReLU rồi đưa đặc trưng đến hai đầu ra. hidden mặc định bằng 128. Kiến trúc này dùng làm đối chứng để xác định lợi ích của lớp tích chập trên cùng bộ dữ liệu.',
        'ModelConfig được ghi cùng checkpoint. Với hidden = 128, MLP có 2.822.529 tham số. Phần lớn nằm ở lớp chính sách 128 → 20.480; vì vậy số tham số lớn chưa đồng nghĩa với thân mạng có khả năng học biểu diễn sâu.'
    ], codeblocks=[code('Mã 3.1 Cấu hình dùng chung trong models.py', 'from dataclasses import dataclass\n\n' + excerpt('model/learning/models.py', 'ModelConfig')),
                   code('Mã 3.2 Khởi tạo hai kiến trúc qua cùng giao diện', '''from model.learning.models import ModelConfig, PolicyValueNet

mlp = PolicyValueNet(ModelConfig(kind='mlp', hidden=128))
cnn = PolicyValueNet(ModelConfig(kind='cnn', width=32,
                                hidden=128))''')]),
    section('Thân CNN và các đầu dự đoán', [
        'CNN dùng hai lớp Conv2d 3 × 3 với padding = 1, số kênh mặc định 32 rồi 64. Sau hai lớp ReLU, đặc trưng 64 × 8 × 8 được trải phẳng và chiếu về 128 phần tử. Lớp chính sách trả 20.480 logits; lớp giá trị trả một số trong [−1, 1] qua Tanh.',
        'CNN mặc định có 3.190.753 tham số. Kiến trúc không dùng BatchNorm nên không phụ thuộc thống kê batch khi suy luận từng thế hoặc cập nhật TD với batch một. Cả MLP và CNN được cài trong PolicyValueNet; nhánh config.kind chọn thân mạng.'
    ], codeblocks=[code('Mã 3.3 Hàm khởi tạo PolicyValueNet lấy từ mã nguồn', excerpt('model/learning/models.py', '__init__'))]),
    section('Forward và lưu trọng số', [
        'forward trả logits và value để huấn luyện chung. Phương thức value chỉ tính thân và đầu giá trị, bỏ phép chiếu chính sách 20.480 chiều. Alpha-beta và tác tử xét trạng thái con dùng phương thức này để giảm chi phí tại mỗi lá.',
        'Checkpoint lưu state_dict cùng cấu hình, phiên bản encoder và metadata thay vì đóng gói toàn bộ đối tượng Python [2]. load_model kiểm tra schema và encoder, nạp bằng weights_only=True rồi chuyển sang chế độ eval. Không thể dùng trực tiếp checkpoint có số kênh hoặc cấu trúc đầu ra khác.'
    ], codeblocks=[code('Mã 3.4 Hai đường suy luận của mạng', excerpt('model/learning/models.py', 'forward') + '\n\n' + excerpt('model/learning/models.py', 'value')),
                   code('Mã 3.5 Nạp mô hình đã huấn luyện', '''from model.learning.models import load_model

model, metadata = load_model('experiments/cnn-01/best.pt')
model.eval()''')])
]})

training_code = excerpt('model/learning/train.py', 'epoch_pass')
training_code = textwrap.dedent(training_code[training_code.index('            logits, value = model(x)'):training_code.index('            n = len(x)')])
sample = json.loads((workspace / 'test-results/ai-models-smoke/samples.jsonl').read_text(encoding='utf-8').splitlines()[0])
report['chapters'].append({'title': 'SINH DỮ LIỆU VÀ HUẤN LUYỆN CÓ GIÁM SÁT', 'sections': [
    section('Nguồn nhãn và cấu trúc mẫu', [
        'learning.data lấy nhãn từ Alpha-beta có sẵn, tối đa độ sâu 2, hoặc engine UCI qua python-chess [3]. Khi dùng Stockfish, cần cung cấp đường dẫn tệp thực thi. Cấu hình teacher được lưu cùng dữ liệu để truy nguyên chất lượng nhãn.',
        'Sau một mở đầu ngẫu nhiên ngắn, teacher đề xuất target_uci. Với xác suất khám phá, chương trình thực hiện nước khác và ghi vào played_uci. Nhãn giá trị là tanh(cp/600) theo phía đến lượt.',
        'Khi chạm max_plies, ván được ghi truncated với result bằng null. Nhãn đánh giá teacher vẫn dùng để học có giám sát; kết quả thắng, hòa, thua chỉ được ghi khi ván kết thúc đúng luật.'
    ], codeblocks=[code('Mã 4.1 Một bản ghi thực tế của đợt chạy kỹ thuật', json.dumps(sample, indent=2, ensure_ascii=False))]),
    section('Chia tập theo ván và lọc vị trí trùng', [
        'split_records chia theo mã ván với tỷ lệ gần 70/15/15 và seed cố định; mỗi tập có ít nhất một ván. Việc chia theo ván giữ các vị trí liên tiếp có quan hệ gần nhau trong cùng một tập.',
        'Validation được lọc các vị trí đã có trong train; test được lọc theo cả hai tập trước. Khóa so trùng dùng bốn trường đầu của FEN, bỏ bộ đếm. Nếu tập rỗng sau lọc, chương trình yêu cầu tăng độ đa dạng dữ liệu. Manifest lưu mã ván, số mẫu và số vị trí bị loại.'
    ]),
    section('Hàm mất mát và cập nhật tham số', [
        'Hàm mất mát gồm cross-entropy của chính sách cộng value_weight lần MSE của đầu giá trị. Cross-entropy nhận logits đã che, không nhận xác suất softmax. Target chính sách có kiểu long; target value là float32. Cả hai đầu và phần thân được cập nhật bằng AdamW với gradient clipping.',
        'epoch_pass chuyển mô hình sang train khi có optimizer và eval khi đánh giá. Trong vòng huấn luyện, gradient được xóa trước backward. Khi đánh giá validation hoặc test, torch.set_grad_enabled(False) ngăn tạo đồ thị gradient. Các độ đo được lấy trung bình theo số mẫu, không lấy trung bình đơn giản giữa các batch khác kích thước.'
    ], codeblocks=[code('Mã 4.2 Phần tính loss và cập nhật trong epoch_pass', training_code)]),
    section('Chọn checkpoint và đọc các tệp đầu ra', [
        'Validation loss quyết định best.pt; last.pt giữ trạng thái cuối. Test được đo một lần sau khi nạp lại best.pt và ghi vào test.json. Việc chọn mô hình không dùng kết quả test.',
        'run.json lưu cấu hình, thiết bị, phiên bản, số tham số, SHA256 dữ liệu và cách chia tập. metrics.csv lưu các loss và policy accuracy theo epoch. Accuracy đo mức khớp nhãn teacher; sức mạnh chơi cần được đo riêng bằng đối kháng.'
    ], table=table('Bảng 4.1 Các tệp đầu ra của supervised training', ['Tệp', 'Nội dung'], [25,75], [
        ['best.pt', 'Trọng số và cấu hình tại validation loss thấp nhất'],
        ['last.pt', 'Trọng số ở epoch cuối'],
        ['run.json', 'Tham số chạy; phiên bản; SHA256; mã ván các tập'],
        ['metrics.csv', 'Độ đo train và validation theo epoch'],
        ['test.json', 'Kết quả kiểm tra của checkpoint đã chọn']
    ]))
]})

report['chapters'].append({'title': 'SUY LUẬN VÀ HỌC QUA TỰ CHƠI', 'sections': [
    section('Chọn nước bằng policy hoặc value', [
        'NeuralAgent ở chế độ policy chỉ lấy logits tương ứng với danh sách nước hợp lệ, softmax trên danh sách này và chọn phần tử lớn nhất. Vì vậy một logit rất cao của nước bất hợp lệ không ảnh hưởng quyết định. Các lựa chọn phong cấp vẫn là các phần tử khác nhau.',
        'Ở chế độ value, tác tử xét các bàn cờ sau nước đi. Mạng đánh giá phía đến lượt ở trạng thái con, tức đối thủ, nên giá trị được đổi dấu trước khi chọn. Nếu trạng thái con kết thúc ván, kết quả luật được dùng trực tiếp. Các giá trị mạng được giới hạn về [−0,95, 0,95] khi lựa chọn để một dự đoán gần 1 không lấn át chiếu hết đã xác định.'
    ], codeblocks=[code('Mã 5.1 Đổi góc nhìn khi đánh giá các trạng thái con', '''batch = torch.from_numpy(np.stack(features)).to(device)
values = model.value(batch).clamp(-.95, .95).cpu().tolist()
scores.update((move, -v) for move, v in zip(pending, values))''')]),
    section('Alpha-beta sử dụng hàm đánh giá học được', [
        'LearnedAlphaBetaAgent thay hàm chấm điểm thủ công tại lá bằng model.value. Tác tử vẫn dùng tìm kiếm sâu dần, sắp xếp nước ăn quân và Alpha-beta. Mỗi khi đổi bên, thuật toán negamax đổi dấu giá trị trả về [5]. Trạng thái kết thúc được xử lý trước khi gọi mạng; điểm chiếu hết có điều chỉnh nhỏ theo ply để ưu tiên kết thúc sớm.',
        'Mô hình được nạp và làm nóng một lần trong hàm khởi tạo, không đọc checkpoint ở mỗi nước. Thời gian quyết định tính cả mã hóa và suy luận. Cơ chế dừng kiểm tra thời gian giữa các bước nên một lượt suy luận đang chạy có thể làm vượt hạn; telemetry ghi latency_ms và budget_exceeded. Khi chưa hoàn tất độ sâu nào, tác tử vẫn trả một nước dự phòng hợp lệ.'
    ], codeblocks=[code('Mã 5.2 Khởi tạo tác tử theo giao diện của dự án', '''from model.learning.agents import LearnedAlphaBetaAgent

agent = LearnedAlphaBetaAgent(
    'experiments/cnn-01/best.pt', depth=2, device='cpu'
)
uci, telemetry = agent.select_action(env, time_limit_ms=100)''')]),
    section('Tinh chỉnh đầu giá trị bằng TD(0)', [
        'learning.selfplay thực hiện TD(0) từ checkpoint đã học có giám sát. Nếu nước vừa đi kết thúc ván, target là kết quả theo phía vừa đi. Nếu chưa kết thúc, target bằng −gamma nhân giá trị trạng thái kế tiếp của target network. Dấu âm xuất phát từ việc lượt chơi đã chuyển sang đối thủ, không phải hình phạt cho mọi nước [4].',
        'Target được tách gradient. Một bản sao mạng được dùng để tính target và đồng bộ sau target_sync bước, mặc định 32. Khi chạm trần ply mà ván chưa kết thúc, bước cuối vẫn bootstrap từ trạng thái kế tiếp; chương trình không gán target hòa bằng 0.',
        'TD chỉ tối ưu thân mạng và đầu value. Chính sách cũ có thể thay đổi do phần thân dùng chung nên checkpoint được gắn policy_trained=False. Tệp đầu ra td-final.pt dùng cho value hoặc search và chưa được gọi là best vì chưa qua chọn mô hình bằng đối kháng. Phương pháp này học V(s), không phải DQN học Q(s,a).'
    ], codeblocks=[code('Mã 5.3 Target TD trong selfplay.py', excerpt('model/learning/selfplay.py', 'td_target'))])
]})

report['chapters'].append({'title': 'CÀI ĐẶT VÀ CHẠY CHƯƠNG TRÌNH', 'sections': [
    section('Chuẩn bị môi trường CPU hoặc CUDA', [
        'Mã đã chạy trên Windows với Python 3.12.14 và PyTorch 2.10.0+cpu. CAI_DAT_AI.ps1 tạo .venv-ai, tìm Python 3.12 từ uv hoặc launcher py; tham số -PythonExe cho phép chỉ định đường dẫn khác.',
        'Tùy chọn -Device cuda cài wheel cu128 từ PyTorch [1]. Cần kiểm tra torch.cuda.is_available() trước khi train bằng GPU. Máy có RTX 3050 Laptop 4 GB nhưng thực nghiệm trong báo cáo chỉ chạy CPU.'
    ], codeblocks=[code('Lệnh 6.1 Cài CPU và xác định Python cho các lệnh sau', r'''& .\CAI_DAT_AI.ps1 -Device cpu
$py = '.\.venv-ai\Scripts\python.exe'
& $py -X utf8 -c 'import torch; print(torch.__version__)' '''),
                   code('Lệnh 6.2 Chuyển sang bản NVIDIA khi cần', r'''& .\CAI_DAT_AI.ps1 -Device cuda
& $py -X utf8 -c 'import torch; print(torch.cuda.is_available())' ''')]),
    section('Sinh dữ liệu và huấn luyện hai kiến trúc', [
        'Lệnh dưới tạo tập thử bằng Alpha-beta. Để đổi sang Stockfish, dùng --teacher stockfish và --engine-path trỏ đến engine UCI. MLP và CNN dùng cùng dữ liệu, cùng seed; mặc định hidden = 128, CNN width = 32 và learning rate 0,001.',
        'Batch 64 dùng làm cấu hình thử ban đầu, sau đó điều chỉnh theo bộ nhớ thực tế. Mỗi lần chạy dùng thư mục đầu ra mới để lưu riêng cấu hình và kết quả.'
    ], codeblocks=[code('Lệnh 6.3 Sinh dữ liệu có nhãn', r'''& $py -X utf8 -m model.learning.data `
  --out data/training/pilot.jsonl --games 100 `
  --max-plies 80 --teacher alphabeta --teacher-ms 30 `
  --seed 42'''), code('Lệnh 6.4 Huấn luyện MLP trên CPU', r'''& $py -X utf8 -m model.learning.train `
  --data data/training/pilot.jsonl `
  --out experiments/mlp-01 --model mlp `
  --epochs 10 --batch-size 64 --device cpu --seed 42'''),
                   code('Lệnh 6.5 Huấn luyện CNN trên CPU', r'''& $py -X utf8 -m model.learning.train `
  --data data/training/pilot.jsonl `
  --out experiments/cnn-01 --model cnn `
  --epochs 10 --batch-size 64 --device cpu --seed 42''')]),
    section('Kiểm tra nước đi, tự chơi và đối kháng', [
        'learning.predict nạp checkpoint và trả một quyết định JSON. Tham số --agent chọn policy, value hoặc search; --fen cho phép nhập một thế cụ thể. Sau khi xác nhận checkpoint hoạt động, selfplay có thể tinh chỉnh value và arena thực hiện đấu đổi màu.',
        'arena dùng cùng mở đầu cho hai ván của một cặp, đổi màu tác tử thử nghiệm và lưu PGN cùng nhật ký từng nước. Kết quả truncated tách khỏi hòa. completed_score chỉ tính trên các ván kết thúc đúng luật và bằng null nếu chưa có ván nào kết thúc; tỷ lệ truncated cao làm kết quả chưa đại diện cho sức mạnh chơi.'
    ], codeblocks=[code('Lệnh 6.6 Kiểm tra quyết định và fine-tune TD', r'''& $py -X utf8 -m model.learning.predict `
  --checkpoint experiments/cnn-01/best.pt `
  --agent search --depth 2 --time-ms 100 --device cpu

& $py -X utf8 -m model.learning.selfplay `
  --init experiments/cnn-01/best.pt `
  --out experiments/td-01 --episodes 100 `
  --max-plies 200 --device cpu --seed 42'''), code('Lệnh 6.7 Đấu với tác tử Alpha-beta cũ', r'''& $py -X utf8 -m model.learning.arena `
  --checkpoint experiments/cnn-01/best.pt `
  --agent search --opponent alphabeta --games 20 `
  --out experiments/arena-cnn-ab --device cpu''')]),
    section('Điểm kết nối với backend', [
        'Các tác tử mới giữ giao diện select_action(env, time_limit_ms) của BaseAgent. Điểm kết nối là GameService.play trong api/games/service.py: khởi tạo tác tử một lần tại worker, sau đó truyền trạng thái để nhận nước UCI.',
        'Web hiện dùng Random và SearchAgent; mô hình mới chưa gắn vào mức độ khó. Telemetry trả value theo phía đến lượt và eval_score bằng null vì giá trị chuẩn hóa không có đơn vị centipawn.'
    ])
]})

run = json.loads((workspace / 'test-results/ai-models-smoke/cnn/run.json').read_text(encoding='utf-8'))
cnn_test = json.loads((workspace / 'test-results/ai-models-smoke/cnn/test.json').read_text(encoding='utf-8'))
mlp_test = json.loads((workspace / 'test-results/ai-models-smoke/mlp/test.json').read_text(encoding='utf-8'))
report['chapters'].append({'title': 'KIỂM THỬ VÀ ĐỌC KẾT QUẢ', 'sections': [
    section('Các kiểm tra bắt buộc trước thực nghiệm', [
        'Bộ kiểm thử mới kiểm tra đặc trưng nhập thành và en passant, thông tin lặp lại, mã phong cấp, mặt nạ nước hợp lệ và việc lưu/nạp trọng số. Các thế chiếu hết cho cả hai màu được dùng để kiểm tra dấu của value và tác tử tìm kiếm; trạng thái môi trường phải giữ nguyên sau khi tác tử phân tích.',
        'Kiểm thử TD xác nhận target đổi dấu ở bước chưa kết thúc và dùng kết quả thật ở bước terminal. Kiểm thử dữ liệu xác nhận các tập không chung mã ván hoặc khóa vị trí. Kết quả chạy python -m pytest tests -q trong .venv-ai là 31 ca đạt, gồm 10 ca của phần học máy và 21 ca hiện có.'
    ], codeblocks=[code('Lệnh 7.1 Chạy kiểm thử', r'''& $py -X utf8 -m pytest tests -q''')]),
    section('Kết quả của đợt chạy kỹ thuật nhỏ', [
        'Đợt kiểm tra sinh 276 mẫu từ 16 ván, giới hạn 20 ply và teacher 8 ms. Sau chia và lọc trùng, train có 194 mẫu, validation 33 và test 45. MLP và CNN dùng hidden = 32; CNN width = 8; cả hai chạy hai epoch với batch 16 trên CPU. Đây là cấu hình nhỏ hơn cấu hình mặc định để kiểm tra toàn bộ luồng mã.',
        f'Policy accuracy trên 45 mẫu test của checkpoint được chọn là {mlp_test["policy_accuracy"] * 100:.2f}% với MLP và {cnn_test["policy_accuracy"] * 100:.2f}% với CNN. Số mẫu nhỏ và nguồn nhãn yếu nên các số này chỉ minh họa cách đọc test.json; chúng không xác lập ưu thế kiến trúc hoặc sức mạnh đối kháng.',
        'TD chạy hai tập, mỗi tập 8 ply, thực hiện 16 cập nhật và ghi cả hai tập là truncated. Đấu thử hai ván ở trần 20 ply cũng ghi hai truncated, không có thắng/hòa/thua hoàn tất và completed_score bằng null. Nhánh CUDA và nguồn nhãn Stockfish chưa được chạy kiểm tra trong đợt này.'
    ], table=table('Bảng 7.1 Các minh chứng chạy được lưu trong dự án', ['Thành phần', 'Vị trí kết quả'], [31,69], [
        ['Dữ liệu và metadata', 'test-results/ai-models-smoke/samples.jsonl và sidecar'],
        ['Supervised MLP/CNN', 'Các thư mục mlp và cnn: run, metrics, checkpoint, test'],
        ['TD(0)', 'Thư mục td: episodes.jsonl và td-final.pt'],
        ['Đối kháng', 'Thư mục arena: summary.json, games.jsonl và games.pgn']
    ])),
    section('Cách mở rộng sau khi luồng mã ổn định', [
        'Khi tăng quy mô, trước hết đo thời gian sinh nhãn và bộ nhớ train trên một phần dữ liệu. Teacher mạnh hơn hoặc có thêm ngân sách có thể tạo nhãn tốt hơn nhưng làm tăng chi phí. Cần giữ cùng dữ liệu và cách chia tập khi so sánh MLP với CNN; số tham số và thời gian suy luận được xem xét cùng accuracy.',
        'Đánh giá chơi cờ cần đủ ván kết thúc, đổi màu và mở đầu được cố định. Bộ arena hiện cung cấp thống kê mô tả và dữ liệu thô, chưa tính khoảng tin cậy, Elo hoặc tự chọn checkpoint theo đấu kiểm tra. Các chức năng này có thể bổ sung trên nhật ký đã lưu; kết luận về sức mạnh phải dựa trên đợt đối kháng riêng.'
    ])
]})

(out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print('Manuscript:', out / 'report.json')
print('Chapters:', len(report['chapters']))
print('Code blocks:', sum(len(s.get('codeblocks', [])) for c in report['chapters'] for s in c['sections']))
