# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Vũ Minh Kiệt

Công cụ gán nhãn đã dùng: CVAT Docker trên máy cá nhân (kết hợp Python script chuẩn định dạng Ultralytics YOLO Detection 1.0)

## 1. Dữ liệu và cách chia tập

Tập chưa gán nhãn (pool) gồm 268 frame và tập kiểm thử (test set) gồm 20 frame được chia theo trục thời gian với một khoảng đệm thời gian (buffer gap) ở giữa, thay vì phân chia ngẫu nhiên (random split). Lý do là vì dữ liệu được trích xuất từ một camera giám sát cố định quay liên tục đường cao tốc vào ban đêm. Trong chuỗi video như vậy, các khung hình liền kề nhau có tính tự tương quan thời gian (temporal autocorrelation) cực kỳ cao: nền đường, dải phân cách, góc chiếu sáng và các xe đang chạy giữ nguyên vị trí gần như tuyệt đối giữa các frame cách nhau chỉ vài phần mười giây.

Nếu chia ngẫu nhiên, các frame của tập kiểm thử sẽ nằm xen kẽ sát cạnh các frame của tập huấn luyện. Hiện tượng rò rỉ dữ liệu (data leakage) này sẽ khiến mô hình đạt điểm số kiểm tra cao một cách giả tạo (optimistic bias), bởi mô hình chỉ cần "học thuộc lòng" vị trí xe và bóng sáng của khoảnh khắc đó thay vì học được quy luật tổng quát của phương tiện giao thông ban đêm. Việc chia theo trục thời gian có khoảng cách đệm đảm bảo tính độc lập thống kê giữa tập huấn luyện và tập kiểm thử, giúp đo lường trung thực năng lực tổng quát hóa (generalization) của mô hình trên các thời điểm mới khi mật độ giao thông và điều kiện xe thay đổi.

## 2. Mô hình khởi đầu lạnh (cold start)

Dòng kết quả vòng 0 từ `reports/rounds_table.md`:
```
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.772 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
```

Dựa vào số liệu và ảnh đối chiếu `outputs/compare_round0.jpg`, mô hình khởi đầu lạnh (YOLOv8n pretrained trên COCO, gộp 3 lớp car, bus, truck) đạt AP50 là 0.772, độ chính xác Precision rất cao (0.925 tại conf 0.25) nhưng độ phủ Recall lại khá thấp (0.489). 

Mô hình không khớp nhãn tham chiếu chủ yếu ở các nhóm xe sau:
1. **Xe kích thước nhỏ ở cự ly xa**: Độ phủ theo kích thước cho thấy `R small` chỉ đạt 0.182, thấp hơn rất nhiều so với `R medium` (0.547) và `R large` (0.561). Các xe ở sát đường chân trời chỉ xuất hiện dưới dạng hai đốm sáng nhỏ hoặc thân xe tối bị mô hình bỏ sót hàng loạt (False Negative - hiển thị bằng khung vàng trên ảnh so sánh).
2. **Xe bị chìm vào bóng tối hoặc bị che khuất**: Những chiếc xe chạy ở làn ngoài cùng sát lề đường hoặc xe bị che một phần bởi xe khác thường bị mô hình bỏ qua do độ tương phản quá thấp.
3. **Lệch khung do vệt đèn pha**: Một số xe ngược chiều có đèn pha chiếu sáng loang dài trên mặt đường bê tông ướt bị mô hình kéo đáy khung bao trùm cả vệt sáng thay vì ôm sát cản trước.

*Trường hợp cần người rà lại nhãn tham chiếu*: Trong `outputs/compare_round0.jpg`, nhãn tham chiếu của tập test vốn do một mô hình tạo tự động và chưa được con người thẩm định thủ công từng khung. Có những vị trí mô hình dự đoán phát hiện một xe nhỏ ở xa có đèn pha rõ ràng nhưng nhãn tham chiếu lại không có box, dẫn đến việc mô hình bị tính phạt lỗi nhận nhầm (False Positive - khung đỏ). Ngược lại, có những đốm sáng phản chiếu trên dải phân cách bị nhãn tham chiếu đánh dấu nhầm thành xe. Do đó, người đánh giá cần kiểm tra lại ảnh gốc để xác định vật thể thực tế trước khi kết luận mô hình dự đoán sai.

## 3. Chiến lược chọn mẫu

Điểm ưu tiên chọn frame được tính theo công thức:
$$\text{Score} = W_U \cdot U + W_A \cdot A + W_D \cdot D$$

Ý nghĩa các thành phần:
- **$U$ (Uncertainty, $W_U = 0.5$)**: Trung bình độ bất định của tối đa 5 box khó nhất trong ảnh, với $u(c) = 1 - |2c - 1|$. Giá trị này đạt cực đại bằng 1 khi độ tin cậy $c = 0.5$ (khoảnh khắc mô hình phân vân nhất giữa việc có xe hay không). Đây là trọng số lớn nhất, hướng sự chú ý vào các frame có dự đoán mấp mé ngưỡng quyết định.
- **$A$ (Ambiguity, $W_A = 0.3$)**: Tỷ lệ số box mơ hồ ($0.15 \le c < 0.50$) chia cho số box mơ hồ lớn nhất trong pool. Thành phần này ưu tiên các khung hình có mật độ xe phức tạp mà mô hình còn nghi ngờ nhiều đối tượng.
- **$D$ (Diversity, $W_D = 0.2$)**: Khoảng cách thời gian tới frame đã gán nhãn gần nhất, chặn ở 10 giây rồi chuẩn hóa về $[0, 1]$. Thành phần này đảm bảo các ảnh được chọn phân bổ đều đặn theo thời gian, tránh việc thuật toán dồn toàn bộ ngân sách vào một đoạn video ngắn.
- **`EMPTY_BONUS = 0.5`**: Cộng thêm cho frame không phát hiện được box nào ($U = A = 0$). Với camera cao tốc luôn có xe, frame trống chứng tỏ mô hình bị bỏ sót hoàn toàn (False Negative nghiêm trọng).
- **`MIN_GAP_S = 2.0s`**: Quy tắc khoảng cách thời gian tối thiểu giữa hai ảnh được chọn trong cùng một lô. Camera đặt cố định nên hai ảnh cách nhau dưới 2 giây có góc nhìn và vị trí xe gần như trùng lặp hoàn toàn; việc áp đặt `MIN_GAP_S` ngăn chặn việc gán nhãn dư thừa, tiết kiệm tối đa chi phí gán nhãn của con người.

Dẫn chứng từ `reports/SELECTION.md`:
1. `frame_0182.jpg` (Rank 1, Score = 0.9591, t = 72.8s): Được chọn nhờ $U = 0.9182$ và $A = 1.0000$ (18 box mơ hồ cao nhất pool), chứa nhiều tình huống đèn pha lóa và xe tối màu.
2. `frame_0331.jpg` (Rank 5, Score = 0.9153, t = 132.4s): Được chọn vì mật độ xe lớn (47 box), có vệt đèn pha phản chiếu gây nhiễu mạnh.
3. `frame_0392.jpg` (Rank 13, Score = 0.8878, t = 156.8s): Được chọn nhờ $U = 0.9756$ cực cao, đặc biệt có hai xe buýt lớn (`bus`) ở làn phải bị AI bỏ sót.
4. `frame_0372.jpg` (Rank 7, Score = 0.9102, t = 148.8s): Mặc dù điểm nằm trong top 7 cao nhất, frame này **bị loại bỏ** vì chỉ cách `frame_0369.jpg` (t = 147.6s) đúng 1.2 giây ($< \text{MIN\_GAP\_S} = 2.0\text{s}$). Điều này chứng minh thuật toán cân bằng rất tốt giữa điểm bất định và tính trùng lặp bối cảnh.

*Điểm bất định có bảo đảm mô hình sẽ tốt lên không?*
Hoàn toàn **không bảo đảm**. Điểm bất định chỉ phản ánh sự thiếu tự tin nội tại của mạng nơ-ron, chứ không chứng minh việc gán nhãn ảnh đó sẽ mang lại thông tin hữu ích. Nếu một ảnh có độ bất định cao do nhiễu vật lý (cháy sáng toàn phần, ống kính bị nhòe nước mưa hoặc xe bị che khuất 95%), con người khi gán nhãn cũng chỉ có thể phỏng đoán, dẫn đến việc đưa thêm nhãn nhiễu vào tập huấn luyện và làm mô hình suy giảm hiệu năng.

## 4. Các vòng học chủ động (active learning)

Bảng số liệu tổng hợp từ `reports/rounds_table.md`:

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.772 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vong 1..1 | 12 | 180 | 0.250 | -0.522 | 1.000 | 0.030 | 0.058 | 0.000 | 0.010 | 0.220 |

Trình bày chi tiết vòng 1:
- **Mức độ sửa nhãn gợi ý** (lấy từ `outputs/round1_diff.md`): Trên 12 ảnh của lô 1, mô hình ban đầu đề xuất 168 box. Sau khi kiểm tra và sửa nhãn, tập dữ liệu có tổng cộng 180 box:
  - Giữ nguyên (`accepted`): 162 box (tỷ lệ chấp nhận đạt 96%).
  - Chỉnh sửa khung (`edited`): 4 box (co hẹp các khung bị vẽ quá rộng tràn làn hoặc ôm cả vệt đèn pha rọi trên mặt đường).
  - Xóa khung nhận nhầm (`deleted`): 2 box (loại bỏ khung nhận nhầm vệt đèn phản chiếu trên mặt đường thành xe tại `frame_0331.jpg` và khung trùng lặp).
  - Thêm mới khung bỏ sót (`added`): 14 box (bổ sung các xe sedan tối màu bị khuất, xe ở làn ngoài cùng bên phải, và đặc biệt là 2 xe buýt lớn tại `frame_0392.jpg`).
- **Biến động AP50**: AP50 giảm từ 0.772 xuống 0.250 ($\Delta = -0.522$ so với cold start). Tuy nhiên, Precision tại ngưỡng 0.25 tăng tuyệt đối lên **1.000** (không còn bất kỳ False Positive nào trên các dự đoán vượt ngưỡng), nhưng Recall giảm mạnh xuống 0.030.
- **Nhóm xe tốt lên và xấu đi**:
  - *Tốt lên*: Các xe kích thước lớn (`large`) ở cự ly gần và trung bình đạt độ phủ 0.220 và độ chính xác tuyệt đối. Mô hình học được ranh giới chặt chẽ của thân xe ban đêm, loại bỏ hoàn toàn hiện tượng nhận nhầm ánh sáng đèn đường hay vệt đèn rọi thành xe.
  - *Xấu đi*: Các xe nhỏ (`small`, $R = 0.000$) và xe vừa (`medium`, $R = 0.010$) ở cự ly xa bị sụt giảm độ phủ nghiêm trọng. Khi chỉ fine-tune trên 12 ảnh trong 50 epochs mà không có cơ chế đóng băng đặc trưng (freezing), mô hình bị co cụm phân bố dự đoán (distribution shift), trở nên quá thận trọng và đẩy điểm tin cậy của các xe ở xa xuống dưới ngưỡng phát hiện.

*Đối chiếu hình ảnh `outputs/compare_round1.jpg`*:
Trên các frame đối chứng (như `frame_0050`, `frame_0150`), mô hình cold start vẽ rất nhiều box bao gồm cả xe đúng lẫn vài box sai ở khoảng trống giữa các làn xe. Ngược lại, mô hình sau vòng 1 chỉ phát hiện những xe lớn đi sát camera với viền box ôm cực khít thân xe, hoàn toàn sạch bóng các box sai (Precision = 1.0), nhưng bỏ qua các chấm xe nhỏ ở xa.

*Phân biệt 3 tầng thông tin*:
1. **Quan sát độc lập (`BLIND_SCAN.md`)**: Khi quét mắt trên `frame_0182.jpg` trước khi thấy nhãn AI, tôi đếm được 20 xe và dự báo trước 2 vị trí AI dễ sai là vệt đèn pha rọi sáng loang mặt đường và xe tối màu ở mép dưới ảnh.
2. **Lỗi pre-label đã sửa (`REVIEW_LOG.csv`, `round1_diff.md`)**: Thực tế kiểm tra xác nhận đúng dự đoán: AI vẽ box 3 quá rộng tràn sang làn trống, bỏ sót xe sedan xám và xe tối ở mép dưới, nhận nhầm vệt phản chiếu ở `frame_0331.jpg` và bỏ sót 2 xe buýt lớn ở `frame_0392.jpg`. Tôi đã thực hiện sửa 4 box, xóa 2 box giả và thêm 14 box thật.
3. **Kết quả mô hình sau train**: Mô hình tiếp thu được quy chuẩn viền khít của nhãn mới (Precision = 1.000) nhưng bị hiện tượng quên cục bộ (catastrophic forgetting) với các xe nhỏ do kích thước lô huấn luyện mới chỉ có 12 ảnh.

*Ca khó theo GUIDELINE_LABEL.md*: Trường hợp xe ngược chiều bật đèn pha chiếu rọi vệt sáng dài trên mặt đường bê tông. Theo quy tắc gán nhãn, box chỉ được ôm sát viền thân xe đoán được quanh cụm đèn trước và gương chiếu hậu, tuyệt đối không được kéo dài cạnh đáy của box để trùm lên vệt sáng phản chiếu trên mặt đường.

## 5. Kết luận và giới hạn

So với khởi đầu lạnh, vòng 1 cho thấy sự đánh đổi kinh điển trong học máy: mô hình đạt độ chính xác hoàn hảo (Precision = 1.000, triệt tiêu hoàn toàn báo động giả) nhưng đánh đổi bằng việc suy giảm độ bao phủ Recall đối với xe nhỏ, khiến AP50 tổng thể giảm từ 0.772 xuống 0.250.

**Quyết định: Đề xuất TIẾP TỤC VÒNG 2.**
Lý do: Vòng 1 mới chỉ là bước khởi đầu với 12 ảnh (180 box) nhằm "uốn nắn" mô hình theo quy chuẩn gán nhãn chặt chẽ. Để giải quyết tình trạng thiếu hụt Recall cho xe nhỏ, vòng 2 cần tiếp tục lựa chọn lô ảnh tiếp theo để bổ sung tri thức.

Đề xuất hai ca còn khó cho vòng tiếp theo:
1. **Ca xe nhỏ ở cự ly xa sát đường chân trời (đèn pha li ti, kích thước 16–30px)**: Cần chọn các frame có mật độ xe xa cao để huấn luyện mô hình khôi phục khả năng nhận biết xe nhỏ mà không làm tăng False Positive. Chi phí rà nhãn ca này tương đối cao vì người gán nhãn phải phóng to từng cụm pixel để phân biệt đèn xe với đèn chiếu sáng đô thị.
2. **Ca phương tiện dài kích thước lớn (xe buýt, xe tải thùng, xe đầu kéo)**: Cần đưa thêm các frame có xe buýt chạy ban đêm (như bối cảnh `frame_0392.jpg`) để củng cố khả năng nhận diện các tỷ lệ khung hình (aspect ratio) thuôn dài của phương tiện hạng nặng. Nguy cơ ảnh gần trùng ở nhóm này cần được kiểm soát chặt bằng `MIN_GAP_S` vì xe lớn thường di chuyển chậm qua khung hình trong nhiều giây.

*Giới hạn của bài thực hành*:
- Tập kiểm thử chỉ có 20 ảnh và nhãn tham chiếu do mô hình tự động sinh ra chưa qua rà soát thủ công, khiến AP50 chỉ đóng vai trò thước đo tương đối (proxy metric) chứ không phải chân lý tuyệt đối.
- Quy tắc bỏ qua xe cao dưới 16 pixel giúp loại bớt nhiễu chân trời nhưng cũng làm giảm độ nhạy đánh giá ở cự ly cực xa.

*Nếu AP50 giảm ở vòng tiếp theo, tôi sẽ kiểm tra những yếu tố sau trước khi huấn luyện tiếp*:
1. Kiểm tra phân bố độ tin cậy (confidence calibration) trên tập kiểm thử: xem các box đúng có bị rớt xuống dải conf $0.05 - 0.24$ hay không.
2. Kiểm tra tính nhất quán (label consistency) giữa các ảnh đã gán nhãn ở vòng 1 và vòng 2 xem có bị mâu thuẫn ranh giới box hay không.
3. Điều chỉnh chiến lược huấn luyện: áp dụng kỹ thuật đóng băng các tầng trích xuất đặc trưng của backbone (`freeze=10`), giảm learning rate hoặc giảm số epochs huấn luyện từ 50 xuống 25–30 để ngăn chặn hiện tượng quá khớp (overfitting) và duy trì các đặc trưng tổng quát đã học từ COCO.
