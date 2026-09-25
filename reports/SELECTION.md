# Vì sao chọn lô này?

Trong 50 dòng đứng đầu `outputs/selection_round1.csv`, nếu chỉ có ngân sách rà 5 ảnh, tôi ưu tiên chọn 5 frame sau:
1. **`frame_0182.jpg`** (Rank 1, Score = 0.9591, t = 72.8s, U = 0.9182, A = 1.0000, 18 box mơ hồ): Frame có điểm tổng hợp và số box mơ hồ cao nhất toàn bộ tập pool; chứa nhiều tình huống đặc thù như đèn pha rọi sáng loang mặt đường gây box lệch, xe sedan tối ở góc dưới bị bỏ sót.
2. **`frame_0369.jpg`** (Rank 2, Score = 0.9322, t = 147.6s, U = 0.9311, A = 0.8889, 43 box, 16 box mơ hồ): Mật độ giao thông cực kỳ dày đặc ở cuối video, có sự đan xen giữa xe chạy ngược chiều (đèn pha trắng) và cùng chiều (đèn hậu đỏ), cách `frame_0182.jpg` tới 74.8 giây nên tính đa dạng thời gian rất cao.
3. **`frame_0099.jpg`** (Rank 8, Score = 0.9060, t = 39.6s, U = 0.9453, A = 0.7778, 14 box mơ hồ): Đại diện cho đoạn đầu video (t = 39.6s); U rất cao (0.9453). AI bỏ sót xe sedan lớn ở giữa và xe sát dải phân cách do phản xạ mặt đường ướt.
4. **`frame_0392.jpg`** (Rank 13, Score = 0.8878, t = 156.8s, U = 0.9756, A = 0.6667, 12 box mơ hồ): Có độ bất định U cao kỷ lục (0.9756); đặc biệt xuất hiện hai xe buýt lớn (`bus`) ở làn phải phía xa mà mô hình COCO bỏ sót hoàn toàn. Rà frame này cung cấp mẫu học xe cỡ lớn cực kỳ giá trị cho lớp `car`.
5. **`frame_0270.jpg`** (Rank 15, Score = 0.8874, t = 108.0s, U = 0.9082, A = 0.7778, 14 box mơ hồ): Nằm ở mốc thời gian trung chuyển (t = 108.0s), lấp đầy khoảng cách thời gian giữa cụm giây 74 và giây 130.

*Xét ảnh gần trùng và trường hợp frame trống*:
- `frame_0372.jpg` (Rank 7, Score = 0.9102, t = 148.8s) có điểm rất cao trong top 10 nhưng không được chọn vì chỉ cách `frame_0369.jpg` (t = 147.6s) đúng 1.2 giây (< MIN_GAP_S 2.0s). Do camera cố định, hai ảnh này gần như trùng lặp cảnh, gán cả hai sẽ lãng phí công sức mà không đem lại tri thức mới.
- Nếu có frame mô hình dự đoán 0 box (`empty = True`), frame đó được cộng thêm `EMPTY_BONUS = 0.5`. Với góc quay cao tốc ban đêm luôn có phương tiện, frame trống đồng nghĩa với việc AI bị mù hoàn toàn (False Negative toàn phần), cần ưu tiên rà để khắc phục lỗi nghiêm trọng này.

Ba frame thuộc lô 12 ảnh model chọn và bằng chứng trong CSV/ảnh contact sheet:
1. **`frame_0182.jpg`**: Trong CSV xếp Rank 1 (Score 0.9591, A = 1.0000 với 18 box mơ hồ). Trên contact sheet, ảnh thể hiện rõ luồng xe đối diện với đèn pha cực sáng chiếu xuống mặt đường gây lóa, nhiều xe bị khuất ở dải phân cách giữa.
2. **`frame_0331.jpg`**: Trong CSV xếp Rank 5 (Score 0.9153, t = 132.4s, 47 boxes, A = 1.0000). Trên contact sheet, mật độ xe rất lớn ở cả hai chiều, xuất hiện vệt đèn pha phản chiếu trên mặt đường giữa hai xe khiến AI nhận nhầm thành xe (box giả).
3. **`frame_0392.jpg`**: Trong CSV xếp Rank 13 (Score 0.8878, t = 156.8s, U = 0.9756). Trên contact sheet, xuất hiện hai thân xe buýt dài màu trắng đỏ ở làn phải phía xa bị AI bỏ sót hoàn toàn cùng một xe SUV ở làn giữa.

Một frame có điểm cao nhưng không chọn hoặc một frame có điểm thấp vẫn nên xem, và lý do:
- **`frame_0372.jpg`** (Rank 7, Score = 0.9102, t = 148.8s): Điểm nằm trong top 7 cao nhất toàn pool, nhưng **không được chọn** vì vi phạm ràng buộc khoảng cách thời gian `MIN_GAP_S = 2.0s` so với `frame_0369.jpg` (t = 147.6s, chênh lệch chỉ 1.2s). Tương tự, `frame_0330.jpg` (Rank 12, Score 0.8902, t = 132.0s) bị loại vì chỉ cách `frame_0331.jpg` 0.4s. Đây là cơ chế đúng đắn để tối ưu chi phí gán nhãn, ngăn ngừa tình trạng tập huấn luyện bị thiên lệch vào một khoảnh khắc video ngắn.

Điều phép chọn này chưa chứng minh về chất lượng mô hình:
- Phép chọn chỉ đo lường **sự bất định nội tại** của mô hình (khi xác suất dự đoán nằm quanh ngưỡng 0.5 hoặc nhiều box nằm trong dải [0.15, 0.50]), hoàn toàn **không chứng minh được mô hình đúng hay sai ngoài thực tế**.
- Mô hình có thể mắc lỗi tự tin sai (overconfident False Positive), ví dụ nhận nhầm biển báo phát sáng hoặc ánh đèn cao áp thành xe với confidence > 0.95; khi đó $u \approx 0$ và thuật toán sẽ bỏ qua frame đó.
- Ngược lại, điểm cao có thể rơi vào các frame có nhiễu quá nặng (xe bị che khuất 95%, sương mù, nhòe mờ do rung lắc), nơi chính người gán nhãn cũng không thể xác định ranh giới thân xe, khiến việc đưa vào huấn luyện có nguy cơ làm mô hình học thêm nhiễu thay vì cải thiện độ chính xác.
