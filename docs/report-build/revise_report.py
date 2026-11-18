import copy
import json
from pathlib import Path

base = Path(__file__).resolve().parent
report = copy.deepcopy(json.loads((base / 'report.json').read_text(encoding='utf-8')))

def section(ch, sec, title, paragraphs):
    item = report['chapters'][ch - 1]['sections'][sec - 1]
    item['title'] = title
    item['paragraphs'] = paragraphs
    item.pop('afterparagraphs', None)
    return item

report['chapters'][0]['title'] = 'TỔNG QUAN VÀ PHẠM VI NGHIÊN CỨU'
section(1, 1, 'Định hướng đề tài', [
    'Đề tài xây dựng hệ thống chơi cờ vua trên trình duyệt và khảo sát các phương pháp lựa chọn nước đi trong điều kiện giới hạn thời gian tính toán. Hệ thống sử dụng FastAPI để xử lý ván đấu, React để hiển thị bàn cờ và một giao diện tác tử thống nhất để thay thế thuật toán mà không thay đổi luật chơi.',
    'Đối tượng nghiên cứu là chất lượng quyết định của tác tử: một phương pháp sử dụng thêm thời gian tìm kiếm hoặc dữ liệu huấn luyện có cải thiện kết quả đối kháng hay không, và mức cải thiện đó đi kèm chi phí nào. Random được dùng làm đường cơ sở. Các phương pháp còn lại được đánh giá trên cùng môi trường và tập thế cờ mở đầu.'
])
section(1, 2, 'Vấn đề nghiên cứu', [
    'Cờ vua là trò chơi đối kháng hai người, có thông tin đầy đủ và không có yếu tố ngẫu nhiên trong quy tắc chuyển trạng thái. Tuy vậy, khả năng quan sát toàn bộ bàn cờ không đồng nghĩa với khả năng xác định ngay nước đi tốt nhất. Số nhánh cần xét tăng nhanh theo độ sâu; chương trình phải dừng tìm kiếm trước khi biết kết quả cuối cùng của phần lớn biến thể [1].',
    'Khó khăn nằm ở cách sử dụng tài nguyên hữu hạn. Tìm kiếm nông có thể bỏ sót một đòn chiến thuật, trong khi tìm kiếm sâu tiêu tốn thời gian và phụ thuộc vào hàm đánh giá. Mạng học giảm nhu cầu xây dựng quy tắc bằng tay nhưng chịu ảnh hưởng của chất lượng nhãn và phân bố thế cờ huấn luyện. Vì vậy, kết quả đối kháng cần được phân tích cùng độ trễ và cấu hình của từng phương pháp.'
])
section(1, 3, 'Mục tiêu và sản phẩm nghiên cứu', [
    'Hệ thống phải duy trì được một ván cờ hợp lệ từ khi khởi tạo đến khi kết thúc, bao gồm nhập thành, bắt tốt qua đường và phong cấp. Giao diện thể hiện lượt chơi, các nước hợp lệ và diễn biến vừa xảy ra. Hoạt ảnh quân cờ được xử lý trên trạng thái đã xác nhận bởi backend để tránh sai lệch giữa hình ảnh và kết quả ván đấu.',
    'Phần thực nghiệm tập trung vào quan hệ giữa chất lượng nước đi và chi phí tính toán. Sản phẩm nghiên cứu gồm các tác tử dùng chung giao diện, bộ dữ liệu có nguồn gốc xác định và bộ chạy đối kháng lưu được toàn bộ cấu hình. Các phép so sánh thành phần được dùng để xác định đóng góp của cắt tỉa, sắp xếp nước và hướng dẫn tìm kiếm bằng mạng học.'
])
item = section(1, 4, 'Phạm vi thực hiện', [
    'Phạm vi cơ bản gồm môi trường luật cờ, ứng dụng chơi cờ và bốn tác tử Random, Rule-based, Alpha-beta, MCTS. Đây là nền tảng để thực hiện đối kháng và khảo sát tìm kiếm. Phần mở rộng gồm CNN học có giám sát, tác tử học giá trị bằng TD và hai hướng kết hợp mạng với tìm kiếm cây.',
    'Các hướng học máy được triển khai sau khi môi trường và bộ đo ổn định. Quy mô dữ liệu, số vòng tự chơi và số ván đối kháng được xác định qua đợt chạy thử ở tuần 4. Phạm vi nghiên cứu giới hạn ở mô hình nhỏ trên tài nguyên của nhóm; việc so sánh tập trung vào các thành phần thuật toán trong hệ thống này.'
])
item['table']['caption'] = 'Bảng 1.1 Phạm vi và sản phẩm của đề tài'
item['table']['headers'] = ['Hạng mục', 'Nội dung thực hiện', 'Sản phẩm']
item['table']['rows'] = [
    ['Ứng dụng', 'Luật cờ; điều khiển ván đấu; hiển thị diễn biến', 'Game chơi được từ đầu đến cuối'],
    ['Thuật toán cơ bản', 'Random; Rule-based; Alpha-beta; MCTS', 'Tác tử và cấu hình đối kháng'],
    ['Dữ liệu và mô hình', 'Ván tự sinh; CNN; học giá trị bằng TD', 'Tập dữ liệu; mô hình; nhật ký học'],
    ['Tìm kiếm có mạng học', 'Chính sách và giá trị hướng dẫn cây tìm kiếm', 'Tác tử kết hợp; kết quả so sánh']
]

report['chapters'][1]['title'] = 'KHẢO SÁT VÀ CƠ SỞ LỰA CHỌN PHƯƠNG PHÁP'
section(2, 1, 'Cơ sở khảo sát', [
    'Khảo sát tập trung vào hai cách giải quyết giới hạn tìm kiếm: dùng tri thức để đánh giá các nhánh chưa xét hết và dùng kết quả lấy mẫu để phân bổ việc duyệt cây. Bài báo của Shannon đặt cơ sở cho cách đánh giá thế cờ thay cho việc duyệt đến trạng thái kết thúc [1]. UCT đưa cơ chế lựa chọn nhánh dựa trên mức tin cậy vào tìm kiếm Monte Carlo [2]. AlphaZero sử dụng mạng chính sách và giá trị để hướng dẫn tìm kiếm qua tự chơi [3].',
    'Các công trình trên khác nhau về tài nguyên và quy trình đánh giá. Kết quả công bố của chúng không thể chuyển trực tiếp thành thứ hạng cho các bản cài đặt trong đề tài. Phần khảo sát được sử dụng để lựa chọn cơ chế cần cài đặt và biến số cần kiểm soát trong thực nghiệm.'
])
section(2, 2, 'Đặc điểm của các phương pháp', [
    'Random chọn đều một nước trong tập hợp lệ. Rule-based bổ sung các ưu tiên như chiếu hết ngay, ăn quân và cải thiện vị trí. Cả hai có chi phí thấp, phù hợp làm đối chứng khi khảo sát lợi ích của tìm kiếm. Hạn chế của Rule-based là quyết định dựa trên đặc điểm cục bộ: một nước ăn quân có thể được đánh giá cao dù dẫn đến mất quân ở lượt sau.',
    'Minimax đánh giá một nước bằng phản ứng tốt nhất của đối thủ trong phạm vi độ sâu đã xét. Alpha-beta giữ kết quả Minimax ở cùng cây và hàm đánh giá, đồng thời loại những nhánh không còn ảnh hưởng đến quyết định [7]. Sắp xếp nước tốt trước giúp xuất hiện cắt tỉa sớm hơn. Tuy nhiên, tăng hiệu quả duyệt cây không khắc phục hoàn toàn sai số của hàm đánh giá tại lá.',
    'MCTS xây dựng cây từng bước qua lựa chọn, mở rộng, mô phỏng và cập nhật giá trị. UCT dành lượt thăm cho cả nhánh có kết quả tốt và nhánh còn ít thông tin [2]. Trong cờ vua, một nước thua có thể chỉ bị bác bỏ bởi một phản ứng hiếm của đối thủ. Mô phỏng ngẫu nhiên dễ bỏ qua phản ứng đó, nên số mô phỏng và cách đánh giá lá là hai biến số cần khảo sát.',
    'Mạng CNN học dự đoán từ dữ liệu thế cờ, nhưng độ chính xác bắt chước nước đi chỉ phản ánh mức khớp với tác tử sinh nhãn. Khả năng chơi phải được đo bằng đối kháng. Khi ghép mạng vào cây, phân bố chính sách định hướng mở rộng còn giá trị dự đoán thay thế hoặc bổ sung mô phỏng. Hướng AlphaZero-inspired trong đề tài sử dụng nguyên lý này ở quy mô nhỏ [3].'
])
item = section(2, 3, 'Lựa chọn phương pháp triển khai', [
    'Thứ tự triển khai đi từ các tác tử không cần dữ liệu đến các tác tử có huấn luyện. Cách tổ chức này tạo ra đối chứng ở từng giai đoạn: Rule-based được so với Random, Alpha-beta được kiểm tra với Minimax, còn tìm kiếm có mạng học được so với chính mạng đó khi chọn nước trực tiếp.'
])
item['table']['headers'] = ['Phương pháp', 'Nguồn đánh giá', 'Cơ sở lựa chọn', 'Hạn chế chính']
item['table']['rows'] = [
    ['Random', 'Lấy mẫu nước hợp lệ', 'Đường cơ sở độc lập dữ liệu', 'Không xét lợi thế thế cờ'],
    ['Rule-based', 'Quy tắc và trọng số', 'Đối chứng cho tìm kiếm', 'Dễ bỏ sót phản đòn'],
    ['Alpha-beta', 'Hàm đánh giá ở lá', 'Khảo sát cắt tỉa và độ sâu', 'Sai số tại điểm dừng tìm kiếm'],
    ['MCTS và UCT', 'Mô phỏng; đánh giá lá', 'Khảo sát phân bổ lượt thăm', 'Nhạy với chính sách mô phỏng'],
    ['CNN và TD', 'Dữ liệu; giá trị học được', 'Khảo sát khả năng học đánh giá', 'Phụ thuộc dữ liệu và mục tiêu học'],
    ['Mạng kết hợp cây', 'Chính sách và giá trị', 'Đo lợi ích của tìm kiếm có hướng dẫn', 'Tăng chi phí suy luận và tự chơi']
]
section(2, 4, 'Câu hỏi nghiên cứu', [
    'RQ1: Alpha-beta và MCTS khác nhau thế nào về Score, độ trễ p95 và tỷ lệ quá hạn khi cùng ngân sách 100 ms mỗi nước?',
    'RQ2: Alpha-beta và MVV-LVA giảm bao nhiêu nút duyệt so với Minimax ở độ sâu 2, 3 và 4 khi dùng cùng hàm đánh giá?',
    'RQ3: Ghép cùng mạng chính sách–giá trị vào cây có cải thiện Score so với mạng chọn trực tiếp khi tính cả thời gian suy luận vào ngân sách quyết định?'
])

report['chapters'][2]['title'] = 'MÔ HÌNH BÀI TOÁN VÀ TỔ CHỨC DỮ LIỆU'
section(3, 1, 'Đầu vào và đầu ra của tác tử', [
    'Tác tử nhận trạng thái ván và giới hạn thời gian, sau đó trả về một nước hợp lệ theo UCI, chẳng hạn e2e4 hoặc e7e8q khi phong cấp thành hậu. Kết quả kèm thời gian xử lý, số nút hoặc số mô phỏng.',
    'Kết quả cuối ván được quy ước theo phía Trắng: thắng nhận +1, hòa nhận 0 và thua nhận −1. Các tác tử có thể dùng giá trị theo phía đang đi trong nội bộ, nhưng phải chuyển đổi nhất quán khi trả kết quả. Một ply là một lượt đi của một bên; độ sâu 4 ply bao gồm tổng cộng bốn lượt, không phải bốn cặp nước.'
])
section(3, 2, 'Biểu diễn trạng thái và luật kết thúc', [
    'Trạng thái ván đấu được lưu bằng FEN cùng lịch sử nước đi. FEN chứa bố trí quân, bên đến lượt, quyền nhập thành, ô bắt tốt qua đường và các bộ đếm. Hai bàn cờ có bố trí quân giống nhau vẫn có thể có tập nước hợp lệ khác nhau nếu quyền nhập thành khác nhau. Lịch sử còn cần thiết để xác định lặp lại thế cờ; vì vậy việc sao chép môi trường phải giữ thông tin này [5].',
    'python-chess đảm nhiệm sinh nước hợp lệ và xác định trạng thái kết thúc. Lớp ChessEnv bao bọc thư viện để cung cấp cùng một môi trường cho ứng dụng, sinh dữ liệu và đối kháng. Nhập thành, phong cấp, bắt tốt qua đường và các điều kiện hòa được kiểm tra trước khi đưa một tác tử vào thực nghiệm. Vua không phải quân có thể bị bắt; chiếu hết là trạng thái kết thúc của luật chơi.',
    'Bộ đối kháng rút gọn áp dụng trần 80 ply theo đặc tả ban đầu [6]. Ván chưa kết thúc tại trần được phân xử bằng chênh lệch vật chất, với giá trị tốt 100, mã 320, tượng 330, xe 500 và hậu 900 centipawn; vua không tham gia tổng này. Chênh lệch ≥ +300 được xử Trắng thắng, ≤ −300 xử Đen thắng, còn lại hòa. Quy tắc này có thể ưu tiên tác tử tích lũy vật chất hơn tác tử đang có đòn chiếu hết dài, nên kết quả phân xử được tách khỏi kết thúc tự nhiên và kiểm tra độ nhạy theo ngưỡng.'
])
section(3, 3, 'Nguồn và cấu trúc dữ liệu', [
    'Dữ liệu kiểm thử luật gồm FEN và chuỗi nước cho các tình huống xác định trước. Dữ liệu thực nghiệm gồm 10 chuỗi mở đầu để tổ chức đấu đổi màu và 50 thế trung cuộc để đo hiệu quả tìm kiếm. Mỗi mẫu có mã định danh, nguồn hoặc cách sinh và cấu hình liên quan; các thế trung cuộc không được chọn chỉ từ những ván một tác tử chơi tốt.',
    'Tập học có giám sát dự kiến được sinh từ 800 ván giữa Alpha-beta độ sâu 2 và Rule-based, với tỷ lệ khám phá ban đầu 15%. Mỗi bản ghi lưu mã ván, ply, FEN, tác tử sinh nhãn, nước đề xuất, nước thực hiện và kết quả. Nước khám phá được lưu riêng với nhãn đề xuất để phân biệt hành động làm đa dạng dữ liệu với hành động dùng làm mục tiêu học.',
    'Đầu vào CNN ban đầu là tensor 14 × 8 × 8, gồm 12 kênh quân theo loại và màu, một kênh lượt đi và một kênh ply chuẩn hóa. Biểu diễn này thiếu quyền nhập thành, bắt tốt qua đường và lịch sử hòa. Mặt nạ hành động vẫn được sinh từ trạng thái đầy đủ; ảnh hưởng của các đặc trưng bị thiếu được khảo sát bằng một cấu hình mã hóa bổ sung.',
    'Ảnh nền và chuỗi khung hình quân cờ thuộc tài nguyên giao diện, được quản lý riêng với dữ liệu học. Danh mục tài nguyên lưu loại quân, màu, trạng thái chuyển động, kích thước và thứ tự khung hình. Việc kiểm kê theo cấu trúc này giúp phát hiện chuỗi ảnh sai trước khi tích hợp vào luồng ván đấu.'
])
section(3, 4, 'Chia tập và kiểm soát sai lệch dữ liệu', [
    'Tập 800 ván được chia theo mã ván thành 560 ván huấn luyện, 120 ván xác thực và 120 ván kiểm tra. Chia theo từng thế cờ sẽ đưa các trạng thái liên tiếp của cùng một ván sang nhiều tập, làm đánh giá quá lạc quan. Tập xác thực dùng để chọn cấu hình và mô hình; tập kiểm tra được giữ cho lần đánh giá cuối.',
    'Chia theo ván chưa loại hết khả năng trùng thế cờ. Khóa so trùng sử dụng bố trí quân, bên đến lượt, quyền nhập thành và ô bắt tốt qua đường. Các vị trí kiểm tra đã xuất hiện trong tập huấn luyện được loại khỏi phép đo dự đoán độc lập, đồng thời thống kê tỷ lệ loại. Bộ mở đầu dùng cho đối kháng được quản lý tách với dữ liệu sinh nhãn.',
    'Accuracy chính sách phản ánh mức khớp với tác tử sinh nhãn; sức mạnh của mô hình được đo bằng đối kháng. Mỗi đợt dữ liệu tự chơi được gắn phiên bản tác tử để theo dõi phân bố.'
])

report['chapters'][3]['title'] = 'PHƯƠNG PHÁP ĐÁNH GIÁ VÀ THỰC NGHIỆM'
item = section(4, 1, 'Độ đo đánh giá', [
    'Kết quả đối kháng được tổng hợp theo số ván thắng W, hòa D và thua L. Điểm đối kháng Score = (W + 0,5 × D) / N, với N là tổng số ván, phản ánh cả kết quả hòa. Điểm này được báo cáo cùng số ván và nguyên nhân kết thúc để phân biệt lợi thế thi đấu với ảnh hưởng của quy tắc phân xử.',
    'Thời gian lựa chọn nước đi được đo trên toàn bộ quá trình quyết định. Trung bình mô tả chi phí chung, còn phân vị p95 cho biết độ trễ của những lượt xử lý lâu. Số nút và số mô phỏng giúp giải thích hành vi tìm kiếm, nhưng không được xem là đơn vị chi phí tương đương giữa Alpha-beta và MCTS vì mỗi nút có thể cần lượng tính toán khác nhau.'
])
item['table']['caption'] = 'Bảng 4.1 Độ đo sử dụng trong thực nghiệm'
item['table']['rows'] = [
    ['Kết quả ván', 'W, D, L và nguyên nhân kết thúc', 'Phân biệt thắng tự nhiên và phân xử'],
    ['Điểm đối kháng', 'Score = (W + 0,5 × D) / N', 'Kết quả tổng hợp của tác tử'],
    ['Thời gian', 'Trung bình, trung vị, p95; tỷ lệ quá hạn', 'Chi phí và khả năng đáp ứng'],
    ['Tìm kiếm', 'Nút duyệt; độ sâu hoàn tất; lượt mô phỏng', 'Ảnh hưởng của cấu hình cây'],
    ['Tài nguyên', 'RAM đỉnh; VRAM đỉnh khi dùng GPU', 'Yêu cầu phần cứng thực tế'],
    ['Mô hình', 'Accuracy chính sách; sai số dự đoán giá trị', 'Mức khớp trên tập kiểm tra'],
    ['Ứng dụng', 'Ca kiểm thử đạt; lỗi đồng bộ; lỗi hoạt ảnh', 'Tính đúng luật và trạng thái hiển thị']
]
section(4, 2, 'Thiết lập đối kháng', [
    'Mỗi cặp tác tử thi đấu từ cùng chuỗi mở đầu và đổi màu ở ván kế tiếp. Cách ghép cặp này giảm ảnh hưởng của lợi thế đi trước và đặc điểm riêng của thế xuất phát. Với tác tử ngẫu nhiên, seed được cố định và lưu trong cấu hình. Phần cứng, phiên bản thư viện và mức song song được giữ ổn định trong một đợt so sánh.',
    'Thí nghiệm cùng ngân sách giới hạn mỗi quyết định ở 100 ms, bao gồm mã hóa trạng thái, tìm kiếm và suy luận mạng. Thời gian hiển thị hoạt ảnh được loại khỏi phép đo. Tìm kiếm dừng tại các điểm kiểm tra thời gian và trả phương án tốt nhất đã có; mọi lần vượt hạn đều được thống kê. Thí nghiệm theo cấu hình mặc định của từng tác tử được tổng hợp riêng.',
    'Đợt chạy thử gồm 20 ván cho mỗi cặp. Đợt chính dự kiến 100–200 ván cho các cặp trả lời câu hỏi nghiên cứu. Nếu mở rộng đủ tám tác tử, 28 cặp với 200 ván mỗi cặp tạo thành 5.600 ván. Với trần 80 ply và 100 ms mỗi lượt, riêng thời gian quyết định tuần tự có thể đạt khoảng 12,4 giờ. Quy mô chạy chính được xác định từ thời gian đo ở đợt thử và tài nguyên còn lại.'
])
section(4, 3, 'Thiết kế thí nghiệm thành phần', [
    'Thí nghiệm tìm kiếm sử dụng 50 thế trung cuộc ở độ sâu 2, 3 và 4. Ba cấu hình là Minimax, Alpha-beta và Alpha-beta có sắp xếp nước MVV-LVA. Các cấu hình dùng cùng hàm đánh giá và quy tắc kết thúc. Chỉ những lượt hoàn tất cùng độ sâu mới được dùng để tính tỷ lệ giảm nút; điểm đánh giá phải nhất quán, dù nước được chọn có thể khác nhau khi nhiều nước đồng điểm.',
    'Thí nghiệm MCTS thay đổi số mô phỏng ở các mức 25, 50, 100 và 200. Chính sách mô phỏng ngẫu nhiên được so với mô phỏng ngắn 8 ply có ưu tiên ăn quân, sau đó dùng hàm đánh giá tại lá. Mục đích là phân biệt lợi ích của tăng số mẫu với lợi ích của chính sách mô phỏng. Giá trị lan truyền được kiểm tra trên thế cờ có phản ứng thắng hoặc thua xác định.',
    'Thí nghiệm kết hợp mạng giữ nguyên một mô hình để so sánh chọn nước trực tiếp và chọn nước qua cây tìm kiếm. Số lần gọi mạng và thời gian suy luận được ghi riêng. Trong giai đoạn tự chơi, mô hình mới đấu đổi màu với mô hình đang sử dụng; quy tắc thay thế được cố định trước mỗi đợt để tránh chọn mô hình theo những kết quả thuận lợi ngẫu nhiên.'
])
section(4, 4, 'Tổng hợp kết quả và sai số ước lượng', [
    'Mỗi bản ghi nước đi lưu mã tác tử, seed, FEN trước nước, UCI được chọn, thời gian, thông số tìm kiếm và cờ quá hạn. Bản ghi ván lưu kết quả, nguyên nhân kết thúc, cấu hình phân xử và PGN. Các bảng tổng hợp được tính từ dữ liệu thô này, thay vì nhập kết quả thủ công.',
    'Khoảng tin cậy 95% của Score được ước lượng bằng bootstrap theo cụm gồm cặp ván cùng mở đầu, seed và đổi màu. Hai ván trong cặp có liên hệ nên được lấy mẫu cùng nhau. Kết quả kết thúc tự nhiên và kết quả phân xử tại trần ply được phân tích riêng; việc thay ngưỡng vật chất hoặc giới hạn ply cho biết kết luận có phụ thuộc mạnh vào quy ước rút gọn hay không.',
    'Một chênh lệch điểm nhỏ với khoảng tin cậy rộng chưa đủ để xác lập thứ hạng. Khi các phép đo cho kết quả không thống nhất, phân tích sẽ xem xét độ trễ, tỷ lệ quá hạn, loại thế cờ và nguyên nhân kết thúc. Phạm vi kết luận gắn với các tác tử, cấu hình và số ván đã thực hiện.'
])

report['chapters'][4]['title'] = 'THIẾT KẾ HỆ THỐNG'
item = section(5, 1, 'Kiến trúc và quản lý trạng thái', [
    'Hệ thống gồm giao diện React, dịch vụ FastAPI, môi trường ChessEnv và các tác tử. Backend giữ trạng thái chính thức của ván đấu. Frontend gửi nước đi theo UCI và nhận trạng thái mới sau khi backend kiểm tra lượt, tính hợp lệ và điều kiện kết thúc. Node.js phục vụ công cụ phát triển và đóng gói frontend.',
    'Mỗi ván có mã định danh và số phiên bản trạng thái. Yêu cầu đi quân mang phiên bản mà người chơi đang thấy; yêu cầu cũ bị từ chối nếu ván đã thay đổi. Cơ chế này xử lý trường hợp nhấn hai lần hoặc gửi thao tác trong lúc lượt AI chưa kết thúc. Tác vụ tìm kiếm chạy qua tiến trình xử lý riêng để không chiếm vòng lặp phục vụ HTTP.'
])
item['table']['caption'] = 'Bảng 5.1 Thành phần và dữ liệu trao đổi'
item['table']['rows'] = [
    ['Frontend', 'Bàn cờ; thao tác; lịch sử; hoạt ảnh', 'Trạng thái và sự kiện ván'],
    ['API', 'Xác thực lượt; điều phối; quản lý phiên bản', 'Mã ván; phiên bản; UCI'],
    ['ChessEnv', 'Luật; lịch sử; chuyển trạng thái; mã hóa', 'FEN; tập nước hợp lệ; kết quả'],
    ['Agent', 'Lựa chọn nước trong giới hạn thời gian', 'UCI; số đo quyết định'],
    ['Lưu trữ', 'Ván đấu; dữ liệu học; mô hình', 'SQLite; PGN; JSON/CSV'],
    ['Bộ thực nghiệm', 'Ghép cặp; chạy ván; tổng hợp số liệu', 'Cấu hình; seed; nhật ký ván']
]
section(5, 2, 'Môi trường và giao diện tác tử', [
    'ChessEnv cung cấp khởi tạo ván, liệt kê nước hợp lệ, thực hiện và hoàn tác nước, sao chép trạng thái, kiểm tra kết thúc và mã hóa đầu vào mạng. Các tác tử cùng sử dụng select_action(env, time_limit_ms), trả về nước UCI và thông tin đo. Logic luật được dùng chung giữa ứng dụng và thực nghiệm; khác biệt chỉ nằm ở cấu hình lựa chọn nước và quy tắc giới hạn ván đối kháng.',
    'Danh mục triển khai gồm RandomAgent, RuleBasedAgent, AlphaBetaAgent, MCTSAgent, SupervisedNNAgent, TDValueAgent, AlphaGoInspiredAgent và AlphaZeroInspiredAgent. AlphaGo-inspired kết hợp chính sách, giá trị mạng và mô phỏng ngắn; AlphaZero-inspired dùng PUCT, đánh giá lá bằng mạng và cập nhật qua tự chơi [3]. Hai hướng sử dụng chung phần quản lý cây, khác nhau ở đánh giá lá và quy trình học.',
    'Mức Dễ sử dụng Random. Ba mức Bình thường, Khó và Ác mộng được cấu hình bằng phương pháp đánh giá hoặc tìm kiếm với giới hạn khác nhau. Mã cấu hình được lưu cùng ván để liên hệ hành vi của chế độ chơi với tác tử tương ứng. Việc tăng độ sâu chỉ tạo thêm khả năng tìm kiếm; mức chênh lệch sức mạnh được xác định qua đối kháng.',
    'CNN gồm đầu chính sách chấm điểm nước đi và đầu giá trị dự đoán kết quả. Khi tách chính sách thành ô đi và ô đến, mô hình bổ sung thành phần phong cấp để phân biệt các hành động có cùng hai ô. TDValueAgent học V(s) từ sai khác giữa các dự đoán liên tiếp [8], còn DQN học Q(s,a) [4]. Phương án của đề tài dùng TD(0) với giá trị trạng thái, sau đó đánh giá các trạng thái kế tiếp hợp lệ để lựa chọn nước.'
])
section(5, 3, 'Đồng bộ diễn biến và hoạt ảnh', [
    'Sau khi chấp nhận nước đi, backend trả trạng thái mới và sự kiện gồm ô xuất phát, ô đích, quân bị bắt, ô bắt thực tế và thông tin nhập thành hoặc phong cấp. Frontend dùng sự kiện để phát hoạt ảnh, sau đó kết thúc tại trạng thái server đã xác nhận. Dữ liệu sự kiện được giữ riêng với hình ảnh để có thể bỏ qua hoạt ảnh khi tải lại ván mà không làm thay đổi kết quả.',
    'Chuỗi chuyển động gồm đứng yên, di chuyển, tấn công và bị loại khỏi bàn. Với nước ăn quân, đối tượng bị bắt còn được hiển thị trong giai đoạn tác động rồi mới biến mất. Bắt tốt qua đường cần dùng ô của tốt bị bắt thay vì ô đích; nhập thành tạo chuyển động đồng thời cho vua và xe. Đây là các trường hợp mà suy diễn sự kiện chỉ từ hình ảnh bàn cờ cuối dễ gây sai.',
    'Tài nguyên được tra qua danh mục theo loại quân, màu và trạng thái. Một chuỗi khung hình thiếu hoặc sai kích thước được thay bằng ảnh đứng yên và dịch chuyển vị trí. Cách xử lý này giữ luồng chơi hoạt động khi tài nguyên chưa đầy đủ. Kiểm thử giao diện đối chiếu vị trí sau hoạt ảnh với trạng thái backend ở cả 12 tổ hợp loại quân và màu.'
])

report['chapters'][5]['title'] = 'PHÂN CÔNG VÀ TỔ CHỨC THỰC HIỆN'
section(6, 1, 'Phân chia trách nhiệm', [
    'Công việc được chia theo các thành phần có thể phát triển và kiểm tra độc lập. Thành viên 1 phụ trách môi trường và backend; Thành viên 2 phụ trách tìm kiếm; Thành viên 3 phụ trách dữ liệu và mô hình; Thành viên 4 phụ trách giao diện và tổng hợp thực nghiệm. Các bảng sử dụng ký hiệu TV1 đến TV4 cho bốn thành viên.',
    'Phần kết hợp mạng với cây được phân chia theo ranh giới kỹ thuật: Thành viên 2 xây dựng cây và PUCT, Thành viên 3 cung cấp hàm suy luận chính sách–giá trị. Thành viên 1 tích hợp cấu hình và nhật ký, Thành viên 4 tổ chức chạy đối kháng và phân tích. Khối lượng thực tế được theo dõi theo sản phẩm hoàn thành và thời gian thực hiện để điều chỉnh phần hỗ trợ giữa các mảng.'
])
item = section(6, 2, 'Thành viên 1 phụ trách môi trường và backend', [
    'Họ tên: ..................................................    Mã số sinh viên: ..........................',
    'Mảng công việc này xác lập nguồn trạng thái chung cho ứng dụng và thực nghiệm. Trách nhiệm trọng tâm là giữ tính đúng luật, lịch sử ván và giao diện tác tử ổn định để các thuật toán có thể được tích hợp mà không sửa đổi logic ván đấu.'
])
item['table']['rows'] = [
    ['CV01', 'Tuần 1–2: xây dựng ChessEnv và kiểm tra luật đặc biệt', 'Môi trường; bộ thế cờ kiểm thử', 'TV2'],
    ['CV02', 'Tuần 2–3: API ván đấu và kiểm soát phiên bản trạng thái', 'API; cấu trúc JSON; ca lỗi đồng thời', 'TV4'],
    ['CV03', 'Tuần 3–5: giao diện tác tử, Random và lưu nhật ký', 'Random có seed; xuất PGN và số đo', 'TV2'],
    ['CV04', 'Tuần 5–7: tích hợp tác tử và bộ chạy đối kháng', 'Bản tích hợp; cấu hình chạy chung', 'TV3'],
    ['CV05', 'Tuần 8: kiểm tra cài đặt và mô tả kiến trúc', 'Hướng dẫn chạy; phần thiết kế hệ thống', 'TV4']
]
item = section(6, 3, 'Thành viên 2 phụ trách thuật toán tìm kiếm', [
    'Họ tên: ..................................................    Mã số sinh viên: ..........................',
    'Mảng tìm kiếm bao gồm các tác tử dùng hàm đánh giá thủ công và phần duyệt cây của tác tử có mạng học. Thành viên 2 kiểm soát quy ước giá trị, điều kiện dừng và thông số tìm kiếm; đồng thời thực hiện thí nghiệm tách ảnh hưởng của cắt tỉa và thứ tự nước.'
])
item['table']['rows'] = [
    ['CV06', 'Tuần 1–3: Rule-based, Minimax và Alpha-beta với MVV-LVA', 'Tác tử; hàm đánh giá; cấu hình độ sâu', 'TV1'],
    ['CV07', 'Tuần 3–4: MCTS/UCT và chính sách mô phỏng', 'MCTS; kiểm tra lan truyền giá trị', 'TV3'],
    ['CV08', 'Tuần 4–6: cây có prior và PUCT kết nối mạng', 'Mô đun cây; số đo lượt gọi mạng', 'TV3'],
    ['CV09', 'Tuần 6–7: so sánh tìm kiếm trên 50 thế trung cuộc', 'Dữ liệu nút và thời gian; phân tích RQ2', 'TV4'],
    ['CV10', 'Tuần 8: phân tích các biến thể tìm kiếm', 'Phần thuật toán; ví dụ giải thích quyết định', 'TV1']
]
item['afterparagraphs'] = [
    'Điều kiện bàn giao là tác tử trả nước hợp lệ và ghi đủ thông số tìm kiếm. Minimax và Alpha-beta phải trả điểm nhất quán khi hoàn tất cùng độ sâu. Các lượt bị ngắt được phân biệt với lượt tìm kiếm đầy đủ; phần đối kháng sử dụng cấu hình đã cố định để kiểm tra lại kết quả.'
]
item = section(6, 4, 'Thành viên 3 phụ trách dữ liệu và mô hình học', [
    'Họ tên: ..................................................    Mã số sinh viên: ..........................',
    'Mảng học máy chịu trách nhiệm từ sinh dữ liệu đến đánh giá mô hình. Thành viên 3 xác định nhãn, cách chia tập và biểu diễn trạng thái, sau đó huấn luyện CNN và tác tử TD. Hàm suy luận được tách khỏi quy trình huấn luyện để Thành viên 2 sử dụng trong cây tìm kiếm.'
])
item['table']['rows'] = [
    ['CV11', 'Tuần 1–3: cấu trúc mẫu, mã hóa trạng thái và chia tập', 'Bộ sinh; danh mục dữ liệu; thống kê trùng', 'TV1'],
    ['CV12', 'Tuần 3–5: CNN chính sách–giá trị và xử lý phong cấp', 'Mã huấn luyện; mô hình; đường học', 'TV2'],
    ['CV13', 'Tuần 5–6: TD(0) tự chơi với giá trị trạng thái', 'Tác tử TD; cấu hình; nhật ký tự chơi', 'TV1'],
    ['CV14', 'Tuần 6–7: ghép mạng với cây và đánh giá mô hình', 'Hàm suy luận; số liệu và phân tích RQ3', 'TV2'],
    ['CV15', 'Tuần 8: phân tích sai số và phạm vi áp dụng', 'Phần dữ liệu và mô hình; cấu hình học', 'TV4']
]
item['afterparagraphs'] = [
    'Dữ liệu bàn giao gồm các tập tách theo ván và báo cáo vị trí trùng. Mô hình đi kèm cấu hình, seed và nhật ký huấn luyện. Kiểm tra suy luận bao gồm mặt nạ nước hợp lệ và phân biệt phong cấp; đánh giá cuối sử dụng tập kiểm tra đã giữ riêng.'
]
item = section(6, 5, 'Thành viên 4 phụ trách giao diện và phân tích thực nghiệm', [
    'Họ tên: ..................................................    Mã số sinh viên: ..........................',
    'Mảng giao diện xử lý thao tác người chơi và biểu diễn sự kiện ván đấu. Trong thực nghiệm, Thành viên 4 tổng hợp nhật ký do các mô đun xuất ra, kiểm tra số lượng ván và tính các độ đo. Phân tích từng thuật toán được thực hiện cùng thành viên phụ trách thuật toán đó.'
])
item['table']['rows'] = [
    ['CV16', 'Tuần 1–3: bàn cờ, điều khiển ván và kết nối API', 'Giao diện React; luồng chơi hoàn chỉnh', 'TV1'],
    ['CV17', 'Tuần 3–5: tài nguyên quân cờ và hoạt ảnh theo sự kiện', 'Danh mục ảnh; ca di chuyển và ăn quân', 'TV1'],
    ['CV18', 'Tuần 5–6: tổng hợp số đo và khoảng tin cậy', 'Chương trình thống kê; bảng kết quả', 'TV3'],
    ['CV19', 'Tuần 6–7: chạy đối kháng và kiểm tra luồng giao diện', 'Nhật ký ván; phân tích RQ1; lỗi giao diện', 'TV2'],
    ['CV20', 'Tuần 8: tổng hợp kết quả và chuẩn bị trình bày', 'Báo cáo tổng hợp; slide; kịch bản demo', 'TV1–3']
]
section(6, 6, 'Tích hợp và kiểm tra chéo', [
    'Mỗi thành phần được bàn giao cùng cấu hình và ca kiểm tra tái hiện hành vi chính. Người kiểm tra chéo xác nhận kết quả trước khi tích hợp. Thay đổi về cấu trúc trạng thái, quy ước giá trị hoặc định dạng nhật ký được cập nhật đồng thời tại các mô đun sử dụng dữ liệu đó.',
    'Mã nguồn và cấu hình được quản lý bằng Git. Dữ liệu thí nghiệm và mô hình có mã phiên bản liên kết với mã nguồn sinh ra chúng. Hằng tuần, nhóm đối chiếu công việc hoàn thành với các mốc triển khai và phân công người xử lý những phụ thuộc đang làm chậm bước tiếp theo.'
])

report['chapters'][6]['title'] = 'KẾ HOẠCH TRIỂN KHAI VÀ KIỂM CHỨNG'
item = section(7, 1, 'Tiến độ thực hiện', [
    'Kế hoạch triển khai kéo dài tám tuần. Hai tuần đầu hoàn thành môi trường và luồng chơi cơ bản. Các thuật toán và dữ liệu được phát triển trên nền này, sau đó dành tuần 7 cho thực nghiệm chính và tuần 8 cho phân tích, kiểm chứng và hoàn thiện sản phẩm.'
])
item['table']['caption'] = 'Bảng 7.1 Tiến độ triển khai'
item['table']['rows'] = [
    ['1', 'Khảo sát tài liệu; thống nhất trạng thái và giao diện tác tử', 'Đặc tả và phân công được thống nhất', 'Cả nhóm'],
    ['2', 'Môi trường luật; API; Random; bàn cờ', 'M1: Hoàn thành ván với Random', 'TV1, TV4'],
    ['3', 'Rule-based; Alpha-beta; sinh dữ liệu thử; hoạt ảnh', 'M2: Tác tử cơ bản và dữ liệu mẫu', 'TV2, TV4'],
    ['4', 'MCTS; chia tập; đo thời gian và tài nguyên', 'M3: Kết quả chạy thử và quy mô học', 'TV2, TV3'],
    ['5', 'CNN; kết nối mạng vào cây; kiểm tra sự kiện giao diện', 'M4: Mô hình đầu và cây có prior', 'TV3, TV2'],
    ['6', 'TD tự chơi; PUCT; hoàn thiện bộ đối kháng', 'M5: Cố định cấu hình thực nghiệm', 'TV1–3'],
    ['7', 'Đối kháng chính; thí nghiệm thành phần; tổng hợp', 'M6: Dữ liệu thô và bảng đánh giá', 'Cả nhóm'],
    ['8', 'Phân tích kết quả; kiểm chứng; đóng gói và trình bày', 'M7: Báo cáo và sản phẩm hoàn chỉnh', 'Cả nhóm']
]
section(7, 2, 'Phụ thuộc và điều chỉnh quy mô', [
    'Sinh dữ liệu phụ thuộc vào môi trường đúng luật và tác tử nguồn ổn định. Huấn luyện bắt đầu sau khi tập dữ liệu đã được chia và kiểm tra. Tích hợp tìm kiếm có mạng học phụ thuộc vào giao diện suy luận; thực nghiệm chính bắt đầu khi tác tử tham gia đã vượt kiểm tra nước hợp lệ và điều kiện dừng.',
    'Cuối tuần 4, số đo chạy thử được dùng để ước lượng thời gian sinh 800 ván, huấn luyện và đối kháng. Mục tiêu 1.500 tập tự chơi cùng tối đa ba chu kỳ tự chơi–huấn luyện–đấu kiểm tra được xem xét trên cơ sở chi phí thực tế [6]. Nếu nguồn lực không đáp ứng toàn bộ, số tác tử học mở rộng hoặc số chu kỳ được giảm; môi trường và các phép so sánh cốt lõi vẫn được hoàn thành.'
])
item = section(7, 3, 'Rủi ro phương pháp và triển khai', [])
item['paragraphs'] = [
    'Rủi ro có thể làm sai kết luận được xử lý trước rủi ro làm chậm tiến độ. Sai luật, sai dấu giá trị hoặc trùng dữ liệu có thể tạo kết quả đối kháng và đánh giá mô hình không còn ý nghĩa, ngay cả khi chương trình vẫn chạy. Các biện pháp kiểm soát được gắn với mốc kiểm tra trước khi chạy chính.'
]
item['table']['caption'] = 'Bảng 7.2 Rủi ro và biện pháp kiểm soát'
item['table']['rows'] = [
    ['Sai luật hoặc lịch sử hòa', 'Thế cờ kiểm thử; giữ lịch sử khi sao chép', 'TV1; trước tích hợp'],
    ['Sai dấu giá trị tìm kiếm', 'Thế thắng/thua xác định; kiểm tra đổi lượt', 'TV2; trước đối kháng'],
    ['Trùng dữ liệu và nhãn nhiễu', 'Chia theo ván; so trùng; tách nước khám phá', 'TV3; trước huấn luyện'],
    ['Chi phí huấn luyện vượt dự kiến', 'Đo đợt thử; lưu mô hình; điều chỉnh số chu kỳ', 'TV3; tuần 4–6'],
    ['Hình ảnh không khớp sự kiện', 'Đối chiếu ô bắt thực tế và trạng thái cuối', 'TV4; khi tích hợp'],
    ['Kết quả không tái lập', 'Cố định seed; lưu cấu hình và dữ liệu thô', 'Cả nhóm; mỗi đợt đo'],
    ['Tích hợp chậm do phụ thuộc', 'Kiểm tra giao diện chung; điều phối hỗ trợ', 'Cả nhóm; hằng tuần']
]
section(7, 4, 'Điều kiện nghiệm thu', [
    'Ứng dụng được nghiệm thu khi hoàn thành luồng tạo ván, thực hiện nước, xử lý lượt AI, hoàn tác và kết thúc ván; các ca luật đặc biệt đều đạt. Hoạt ảnh kết thúc đúng vị trí do backend xác nhận, kể cả khi xảy ra nhập thành, phong cấp hoặc bắt tốt qua đường. Lỗi làm thay đổi luật hoặc gây mất đồng bộ trạng thái phải được xử lý trước bàn giao.',
    'Phần nghiên cứu được nghiệm thu theo dữ liệu và phép đo đã thực hiện. Mỗi tác tử có cấu hình, mỗi đợt đối kháng có seed và nhật ký ván, mỗi bảng kết quả liên kết được với dữ liệu thô. Phân tích trả lời các câu hỏi nghiên cứu trong giới hạn số mẫu và tài nguyên sử dụng; phần chưa thực hiện được tách khỏi kết quả đã kiểm chứng.',
    'Sản phẩm bàn giao gồm mã nguồn, hướng dẫn chạy, dữ liệu và cấu hình thực nghiệm, mô hình đã huấn luyện nếu có, cùng báo cáo phân tích kết quả. Một đợt chạy mẫu trên môi trường cài đặt lại được dùng để kiểm tra khả năng tái lập trước khi kết thúc đề tài.'
])

report['references'][5] = '[6] Báo Cáo Đề Tài Cờ Vua AI – TTNT Nâng Cao, tài liệu đặc tả đề tài, PDF 5 trang, bản lưu của nhóm nghiên cứu.'
report['references'].extend([
    '[7] D. E. Knuth và R. W. Moore, An Analysis of Alpha-Beta Pruning, Artificial Intelligence, tập 6, số 4, tr. 293–326, 1975. DOI: https://doi.org/10.1016/0004-3702(75)90019-3',
    '[8] R. S. Sutton, Learning to predict by the methods of temporal differences, Machine Learning, tập 3, tr. 9–44, 1988. DOI: https://doi.org/10.1007/BF00115009'
])

output_dir = base / 'revision-v2'
output_dir.mkdir(exist_ok=True)
out = output_dir / 'report.json'
out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(out)
print('Paragraphs:', sum(len(s.get('paragraphs', [])) + len(s.get('afterparagraphs', [])) for c in report['chapters'] for s in c['sections']))
